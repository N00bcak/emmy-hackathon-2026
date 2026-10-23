#!/usr/bin/env python3
"""Small direct typed Kernel IR -> PTX compiler, intentionally conservative.

No emitter, renderer, or external compiler performs the IR lowering. ptxas may
assemble the resulting PTX separately. Unsupported constructs fail explicitly.
"""
from __future__ import annotations
import json
import math
import re
import struct
import sys
from pathlib import Path
from emmy.compiler.graph import Graph
from emmy.compiler.ir.elementwise import reduce_canon
from tf32_emitter import PTX_HEADER


class Unsupported(ValueError):
    pass


def ident(value):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", value):
        raise Unsupported(f"Unsupported PTX identifier: {value!r}")
    return value


def static(value):
    """Only static integer dimensions are supported, never body-string parsing."""
    if isinstance(value, int):
        return value
    if hasattr(value, 'expr'):
        return static(value.expr)
    if type(value).__name__ == 'Literal' and isinstance(value.value, int):
        return value.value
    raise Unsupported(f"Expected static integer, got {value!r}")


def f32(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    return f"0f{bits:08x}"


def f64(value):
    bits = struct.unpack('<Q', struct.pack('<d', value))[0]
    return f"0d{bits:016x}"


DTYPES = {
    'f32': (4, 'f32', 'f32'),
    'f64': (8, 'f64', 'f64'),
    # f16 is a storage/transport type. Arithmetic is evaluated in f32 and
    # rounded once at each explicit SSA/store boundary, matching the typed
    # IR's promotion rule. Keep the carrier in a b32 register so the generic
    # value map does not need a second register class for half values.
    'f16': (2, 'b16', 'f16'),
    'i32': (4, 's32', 's32'),
    'u32': (4, 'u32', 'u32'),
    'i64': (8, 's64', 's64'),
    'u64': (8, 'u64', 'u64'),
}
FLOAT_DTYPES = frozenset(('f16', 'f32', 'f64'))
INT_DTYPES = frozenset(('i32', 'u32', 'i64', 'u64'))


def dtype_name(dtype):
    name = getattr(dtype, 'name', dtype)
    if name not in DTYPES:
        raise Unsupported(f'Unsupported buffer dtype {name}')
    return name


def dtype_info(dtype):
    return DTYPES[dtype_name(dtype)]


def literal(value, dtype):
    if dtype == 'f16':
        return '0x%04x' % struct.unpack('<H', struct.pack('<e', float(value)))[0]
    if dtype == 'f32':
        return f32(float(value))
    if dtype == 'f64':
        return f64(float(value))
    if dtype in INT_DTYPES:
        return str(int(value))
    raise Unsupported(f'Unsupported literal dtype {dtype}')


def reg_suffix(dtype):
    if dtype == 'f16':
        return 'rh'
    if dtype == 'f32':
        return 'f'
    if dtype == 'f64':
        return 'fd'
    if dtype in ('i64', 'u64'):
        return 'rd'
    return 'r'


def ptx_compare_type(dtype):
    if dtype in ('f32', 'f64', 'u32', 'u64'):
        return dtype
    if dtype == 'i64':
        return 's64'
    return 's32'


def select_type(dtype):
    if dtype == 'f16':
        return 'b16'
    if dtype == 'f32':
        return 'f32'
    if dtype == 'f64':
        return 'f64'
    if dtype in ('i64', 'u64'):
        return 'b64'
    return 'b32'


def convert_register(emitter, value, source, target):
    """Convert one typed scalar register to ``target`` and return its name."""
    if source == target:
        return value
    if target == 'f16':
        reg = emitter.reg('rh')
        if source in ('f32', 'f64'):
            emitter.emit(f'cvt.rn.f16.{source} {reg}, {value};')
        elif source in INT_DTYPES:
            kind = 's' if source in ('i32', 'i64') else 'u'
            emitter.emit(f'cvt.rn.f16.{kind}{dtype_info(source)[0] * 8} {reg}, {value};')
        else:
            raise Unsupported(f'Unsupported scalar conversion {source} -> {target}')
        return reg
    if source == 'f16':
        reg = emitter.reg(reg_suffix(target))
        if target in ('f32', 'f64'):
            emitter.emit(f'cvt.{target}.f16 {reg}, {value};')
        elif target in INT_DTYPES:
            kind = 's' if target in ('i32', 'i64') else 'u'
            emitter.emit(f'cvt.rzi.{kind}{dtype_info(target)[0] * 8}.f16 {reg}, {value};')
        else:
            raise Unsupported(f'Unsupported scalar conversion {source} -> {target}')
        return reg
    if source in FLOAT_DTYPES and target in FLOAT_DTYPES:
        reg = emitter.reg(reg_suffix(target))
        emitter.emit(f'cvt.{target}.{source} {reg}, {value};')
        return reg
    if source in INT_DTYPES and target in INT_DTYPES:
        kind = 's' if source in ('i32', 'i64') else 'u'
        reg = emitter.reg(reg_suffix(target))
        emitter.emit(f'cvt.{kind}{dtype_info(target)[0] * 8}.{kind}{dtype_info(source)[0] * 8} {reg}, {value};')
        return reg
    if source in INT_DTYPES and target in FLOAT_DTYPES:
        kind = 's' if source in ('i32', 'i64') else 'u'
        reg = emitter.reg(reg_suffix(target))
        emitter.emit(f'cvt.{target}.{kind}{dtype_info(source)[0] * 8} {reg}, {value};')
        return reg
    if source in FLOAT_DTYPES and target in INT_DTYPES:
        kind = 's' if target in ('i32', 'i64') else 'u'
        reg = emitter.reg(reg_suffix(target))
        emitter.emit(f'cvt.rzi.{kind}{dtype_info(target)[0] * 8}.{source} {reg}, {value};')
        return reg
    raise Unsupported(f'Unsupported scalar conversion {source} -> {target}')


COMPARISONS = {
    'equal': 'eq', 'not_equal': 'ne', 'less': 'lt', 'less_equal': 'le',
    'greater': 'gt', 'greater_equal': 'ge',
}
INTEGER_ELEMENTWISE = frozenset({
    'add', 'subtract', 'multiply', 'floor_divide', 'divide', 'remainder',
    'mod', 'left_shift', 'right_shift', 'bitwise_and', 'bitwise_or',
    'bitwise_xor',
})


def as_predicate(emitter, value, dtype):
    """Return a predicate for a typed boolean/mask SSA value.

    Emmy stamps comparisons as f32 (or an integer carrier), not as a separate
    boolean dtype, so a mask reaching ``where`` is usually a numeric zero/one
    value. Keep an existing predicate and materialize one otherwise.
    """
    if dtype == 'pred':
        return value
    if dtype not in DTYPES:
        raise Unsupported(f'Mask has unsupported dtype {dtype}')
    p = emitter.reg('p')
    if dtype in FLOAT_DTYPES:
        if dtype == 'f16':
            value = convert_register(emitter, value, 'f16', 'f32')
            dtype = 'f32'
        emitter.emit(f'setp.ne.{dtype} {p}, {value}, {literal(0, dtype)};')
    else:
        emitter.emit(f'setp.ne.{ptx_compare_type(dtype)} {p}, {value}, 0;')
    return p


def emit_integer_binary(emitter, op, result, a, b):
    """Emit one typed integer operation, preserving Emmy floor semantics."""
    if result not in INT_DTYPES:
        raise Unsupported(f'Integer operation requires integer result, got {result}')
    width = dtype_info(result)[0] * 8
    kind = 'u' if result in ('u32', 'u64') else 's'
    suffix = 'rd' if width == 64 else 'r'
    out = emitter.reg(suffix)
    direct = {
        'add': f'add.{kind}{width}',
        'subtract': f'sub.{kind}{width}',
        'multiply': f'mul.lo.{kind}{width}',
        'bitwise_and': f'and.b{width}',
        'bitwise_or': f'or.b{width}',
        'bitwise_xor': f'xor.b{width}',
        'left_shift': f'shl.b{width}',
        'right_shift': f'shr.{kind}{width}',
    }
    if op in direct:
        emitter.emit(f'{direct[op]} {out}, {a}, {b};')
        return out
    if op not in ('divide', 'floor_divide', 'remainder', 'mod'):
        raise Unsupported(f'Unsupported integer elementwise operation {op}')
    q = emitter.reg(suffix)
    rem = emitter.reg(suffix)
    emitter.emit(f'div.{kind}{width} {q}, {a}, {b};')
    emitter.emit(f'rem.{kind}{width} {rem}, {a}, {b};')
    if kind == 'u':
        chosen = rem if op in ('remainder', 'mod') else q
        emitter.emit(f'mov.{dtype_info(result)[1]} {out}, {chosen};')
        return out
    # PTX signed div/rem truncate toward zero. Adjust only when the
    # remainder's sign differs from the divisor's sign and is nonzero.
    negative_rem = emitter.reg('p')
    negative_divisor = emitter.reg('p')
    mismatch = emitter.reg('p')
    nonzero = emitter.reg('p')
    fix = emitter.reg('p')
    adjusted = emitter.reg(suffix)
    emitter.emit(f'setp.lt.s{width} {negative_rem}, {rem}, 0;')
    emitter.emit(f'setp.lt.s{width} {negative_divisor}, {b}, 0;')
    emitter.emit(f'xor.pred {mismatch}, {negative_rem}, {negative_divisor};')
    emitter.emit(f'setp.ne.s{width} {nonzero}, {rem}, 0;')
    emitter.emit(f'and.pred {fix}, {mismatch}, {nonzero};')
    if op in ('remainder', 'mod'):
        emitter.emit(f'add.s{width} {adjusted}, {rem}, {b};')
        emitter.emit(f'selp.s{width} {out}, {adjusted}, {rem}, {fix};')
    else:
        emitter.emit(f'add.s{width} {adjusted}, {q}, -1;')
        emitter.emit(f'selp.s{width} {out}, {adjusted}, {q}, {fix};')
    return out


def residue(expr, modulus, variables=None):
    """Prove ``expr % modulus`` for integer index ASTs.

    Returns the residue when it is independent of runtime variables, else
    ``None``. This is intentionally conservative and covers the arithmetic
    shapes emitted by tiled copy schedules; it never guesses. A ``%`` by a
    multiple of ``modulus`` preserves the dividend's residue, and a product
    containing a factor that is itself a proven multiple of ``modulus``.
    """
    if modulus <= 0:
        raise Unsupported('Residue modulus must be positive')
    kind = type(expr).__name__
    if kind == 'Literal':
        if isinstance(expr.value, int) and not isinstance(expr.value, bool):
            return int(expr.value) % modulus
        return None
    if kind == 'Var':
        return None if variables is None else variables.get(expr.name)
    if kind == 'BinaryExpr':
        left = residue(expr.left, modulus, variables)
        right = residue(expr.right, modulus, variables)
        if expr.op == '*':
            # A congruence alone does not prove divisibility: (x * 12) % 8 is
            # not zero for all x even though 12 % 8 is not relevant to x.
            # Only a literal multiple of the modulus is unconditionally safe.
            for factor in (expr.left, expr.right):
                if (type(factor).__name__ == 'Literal'
                        and isinstance(factor.value, int)
                        and factor.value % modulus == 0):
                    return 0
        if expr.op in ('+', '-', '*') and left is not None and right is not None:
            if expr.op == '+':
                return (left + right) % modulus
            if expr.op == '-':
                return (left - right) % modulus
            return (left * right) % modulus
        if expr.op in ('%', '/', '//'):
            divisor = expr.right
            if type(divisor).__name__ == 'Literal' and isinstance(divisor.value, int) and divisor.value > 0:
                if expr.op == '%' and left is not None and divisor.value % modulus == 0:
                    # With a positive divisor divisible by the modulus, floor
                    # remainder preserves the dividend's residue.
                    return left % modulus
                if expr.op in ('/', '//'):
                    # Emmy uses floor division. For a positive constant c,
                    # (x * c) // c == x exactly, including negative x.
                    if type(expr.left).__name__ == 'BinaryExpr' and expr.left.op == '*':
                        factors = (expr.left.left, expr.left.right)
                        for other in factors:
                            if type(other).__name__ == 'Literal' and other.value == divisor.value:
                                operand = factors[1] if other is factors[0] else factors[0]
                                return residue(operand, modulus)
                    if type(expr.left).__name__ == 'Literal':
                        return (expr.left.value // divisor.value) % modulus
        return None
    return None


def flat_residue(shape, index, modulus, variables=None):
    """Residue of a row-major flattened index, or None when unproven."""
    if len(index) != len(shape):
        return None
    total = 0
    for axis, coordinate in enumerate(index):
        stride = math.prod(shape[axis + 1:]) % modulus
        if stride == 0:
            continue
        part = residue(coordinate, modulus, variables)
        if part is None:
            return None
        total = (total + part * stride) % modulus
    return total


def has_collective(statements):
    """A divergent tile exit cannot bypass any nested collective."""
    for statement in statements:
        if type(statement).__name__ in ("WarpShuffle", "TreeHalve", "Sync"):
            return True
        for field in ("body", "else_body", "combine_states"):
            nested = getattr(statement, field, ())
            if nested and has_collective(nested):
                return True
    return False


def expr_names(expr):
    """Free value names for a pure index expression, or None if unsupported."""
    kind = type(expr).__name__
    if kind == 'Var':
        return {expr.name}
    if kind == 'Literal':
        return set()
    if kind == 'BinaryExpr':
        left, right = expr_names(expr.left), expr_names(expr.right)
        return None if left is None or right is None else left | right
    if kind == 'TernaryExpr':
        cond = expr_names(expr.cond)
        left = expr_names(expr.if_true)
        right = expr_names(expr.if_false)
        return None if cond is None or left is None or right is None else cond | left | right
    return None


def defined_names(statements):
    """Every SSA/carried name written anywhere in this statement tree."""
    found = set()
    for statement in statements:
        kind = type(statement).__name__
        if kind == 'Load':
            found.update(statement.names)
        elif kind in ('Assign', 'Accum', 'Init', 'Let', 'Carry'):
            found.add(statement.name)
        elif kind in ('WarpShuffle', 'TreeHalve'):
            found.update(statement.state)
            found.update(statement.state_b)
        elif kind in ('Loop', 'StridedLoop'):
            found.add(statement.axis.name)
        # Iterate the IR's own nesting protocol, including Cond bodies.
        # Enumerating a hand-picked field list silently missed Cond and let
        # a branch-local rebind be hoisted as if it were loop-invariant.
        for nested in statement_bodies(statement):
            found.update(defined_names(nested))
    return found


def statement_bodies(statement):
    """All child bodies, including a shuffle/tree's combine statements.

    Relying only on the wrapper's field list missed Cond; relying only on
    ``nested()`` misses ``combine_states`` on WarpShuffle/TreeHalve.
    """
    bodies = list(statement.nested())
    combine = getattr(statement, 'combine_states', ())
    if combine and all(combine is not body for body in bodies):
        bodies.append(combine)
    return bodies


STRAIGHT_LINE_KINDS = frozenset(('Load', 'Let', 'Assign', 'Accum'))
# Group sizes for thread-strided loops, largest first. On the RTX 5090, 32
# halved r02's latency against 8; 64 measured the same as 32.
STRIDED_UNROLL = (32, 8)


def straight_line(statements):
    """Loads, pure bindings and folds only: no memory writes, collectives or
    control flow, so consecutive iterations may be emitted back to back."""
    return all(type(s).__name__ in STRAIGHT_LINE_KINDS for s in statements)


def static_or_none(value):
    try:
        return static(value)
    except Unsupported:
        return None


# Shared bytes one Tile may spend on cooperatively hoisted values.
HOIST_SHARED_LIMIT = 32 * 1024
# Work items a thread evaluates together in one hoisted reduction trip.
HOIST_REDUCTION_BATCH = 4
# Most elements per thread a staged copy issues straight-line.
STAGE_STRAIGHT_LINE = 64
# 128-bit loads a vectorized strided reduction keeps in flight per thread,
# largest first. Bytes in flight, not instruction count, bound a
# single-CTA reduction streaming from DRAM.
STRIDED_VECTOR_BATCH = (32, 8, 1)


def touched_names(statements):
    """Every name a statement tree reads or binds, carried folds included."""
    found = set(defined_names(statements))
    for s in statements:
        found.update(s.deps())
        if type(s).__name__ == 'Accum':
            found.add(s.name)
        for nested in statement_bodies(s):
            found.update(touched_names(nested))
    return found


def written_buffers(statements):
    found = set()
    for s in statements:
        kind = type(s).__name__
        if kind == 'Write':
            found.add(s.output)
        elif kind == 'CpAsyncCopy':
            found.add(s.smem)
        for nested in statement_bodies(s):
            found.update(written_buffers(nested))
    return found


# Copies per trip when a loop's shared loads combine into vector loads.
VECTOR_UNROLL = 4


def affine_form(expr):
    """An integer index AST as (constant, {term: coefficient}), or None.

    Terms are variable names or the structural keys of non-affine
    subexpressions, which stand for arbitrary runtime integers.
    """
    kind = type(expr).__name__
    if kind == 'Literal':
        if isinstance(expr.value, int) and not isinstance(expr.value, bool):
            return expr.value, {}
        return None
    if kind == 'Var':
        return 0, {('var', expr.name): 1}
    if kind == 'BinaryExpr' and expr.op in ('+', '-', '*'):
        left, right = affine_form(expr.left), affine_form(expr.right)
        if left is not None and right is not None:
            if expr.op in ('+', '-'):
                sign = 1 if expr.op == '+' else -1
                terms = dict(left[1])
                for term, c in right[1].items():
                    terms[term] = terms.get(term, 0) + sign * c
                return left[0] + sign * right[0], {k: c for k, c in terms.items() if c}
            for scale, other in ((left, right), (right, left)):
                if not scale[1]:
                    return (scale[0] * other[0],
                            {k: scale[0] * c for k, c in other[1].items() if scale[0] * c})
    key = structural_key(expr)
    return None if key is None else (0, {('opaque', key): 1})


def flat_affine(shape, index):
    """Row-major flattened element offset of ``index`` in affine form."""
    if len(index) != len(shape):
        return None
    constant, terms = 0, {}
    for axis, coord in enumerate(index):
        form = affine_form(coord)
        if form is None:
            return None
        stride = math.prod(shape[axis + 1:])
        constant += stride * form[0]
        for term, c in form[1].items():
            terms[term] = terms.get(term, 0) + stride * c
    return constant, {k: c for k, c in terms.items() if c}


def online_max_carriers(body, opaque):
    """Running maxima of an online softmax whose final value can be known
    before the loop.

    Matches ``acc <- maximum(acc, x)`` where ``x`` depends only on this
    iteration (its definitions in the body reach no carried fold), and
    ``acc`` is read only by ``gn = maximum(acc, x)`` and
    ``dg = subtract(acc, gn)``, typically followed by ``alpha = exp(dg)``.
    Seeding ``acc`` with max over the loop of ``x`` makes every
    ``gn`` that maximum, ``dg`` zero and ``alpha`` one, which turns the
    online recurrence into the two-pass softmax. Returns, per carrier, the
    fold, ``x``, ``gn``, ``dg``, ``alpha`` (or None) and the statements
    that compute ``x``, in body order.
    """
    folds = [s for s in body if type(s).__name__ == 'Accum']
    carried = {s.name for s in folds}
    definer = {}
    for s in body:
        for name in defined_names([s]):
            definer.setdefault(name, s)
    plans = []
    for fold in folds:
        if (reduce_canon(fold.op.name) != 'maximum' or fold.base not in (None, fold.name)
                or sum(s.name == fold.name for s in folds) != 1):
            continue
        x = fold.value
        needed, stack, ok = set(), [x], True
        while stack and ok:
            name = stack.pop()
            if name in carried:
                ok = False
            elif name in definer and id(definer[name]) not in needed:
                s = definer[name]
                needed.add(id(s))
                if id(s) in opaque:
                    continue
                if type(s).__name__ not in ('Load', 'Let', 'Assign'):
                    ok = False
                stack.extend(s.deps())
        if not ok:
            continue
        readers = [s for s in body if s is not fold and fold.name in s.deps()]
        if len(readers) != 2 or any(type(s).__name__ != 'Assign' for s in readers):
            continue
        gn = next((s for s in readers if s.op.name == 'maximum'
                   and sorted(s.args) == sorted((fold.name, x))), None)
        dg = next((s for s in readers if s.op.name == 'subtract'
                   and gn is not None and tuple(s.args) == (fold.name, gn.name)), None)
        if gn is None or dg is None:
            continue
        users = [s for s in body if dg.name in s.deps()]
        alpha = users[0] if (len(users) == 1 and type(users[0]).__name__ == 'Assign'
                             and users[0].op.name == 'exp') else None
        plans.append((fold, x, gn, dg, alpha, [s for s in body if id(s) in needed]))
    return plans


VN_KINDS = frozenset(('Load', 'Let', 'Assign', 'Accum'))


def vn_expr(expr, vn):
    kind = type(expr).__name__
    if kind == 'Var':
        return vn(expr.name)
    if kind == 'Literal':
        return ('lit', repr(expr.value), str(getattr(expr, 'dtype', None)))
    if kind == 'BinaryExpr':
        return ('bin', expr.op, vn_expr(expr.left, vn), vn_expr(expr.right, vn))
    if kind == 'TernaryExpr':
        return ('sel', vn_expr(expr.cond, vn), vn_expr(expr.if_true, vn), vn_expr(expr.if_false, vn))
    return ('opaque', id(expr))


def loop_value_numbering(body, opaque, fixed=None):
    """Optimistic value numbering of one loop body, carried folds included.

    Seeded folds with the same op start congruent; the body is numbered under
    that assumption and classes whose end-of-iteration values differ are
    split until stable (optimistic congruence, as in SSA GVN). Returns
    (skip, record, carried): ``skip`` maps a redundant top-level statement to
    the (name, value number) pairs it would define, ``record`` maps first
    occurrences to theirs, and ``carried`` aliases folds to a representative.
    Statements of other kinds, and ids in ``opaque``, define fresh values;
    ``fixed`` maps a statement id to a function of the numbering that gives
    the (name, value number) pairs of a statement with a known meaning.
    """
    fixed = fixed or {}
    pure = all(type(s).__name__ in VN_KINDS or id(s) in opaque for s in body)
    folds = list(dict.fromkeys(s.name for s in body if type(s).__name__ == 'Accum'))
    ops = {s.name: (reduce_canon(s.op.name), str(s.dtype)) for s in body
           if type(s).__name__ == 'Accum'}
    classes = {name: ops[name] for name in folds}

    def number(classes):
        state = {name: ('carried', classes[name]) for name in folds}
        values, keys = {}, []
        vn = lambda name: state.get(name, values.get(name, ('ext', name)))
        for s in body:
            kind = type(s).__name__
            if id(s) in fixed:
                pairs = fixed[id(s)](vn)
                values.update(pairs)
                keys.append(pairs)
            elif kind not in VN_KINDS or id(s) in opaque:
                for name in defined_names([s]):
                    fresh = ('opaque', id(s), name)
                    if name in state:
                        state[name] = fresh
                    else:
                        values[name] = fresh
                keys.append(None)
            elif kind == 'Load':
                index = tuple(vn_expr(c, vn) for c in s.index)
                base = ('load', s.input, index) if pure else ('opaque', id(s))
                pairs = [(name, (*base, i)) for i, name in enumerate(s.names)]
                values.update(pairs)
                keys.append(pairs)
            elif kind == 'Let':
                pairs = [(s.name, ('let', str(s.dtype), vn_expr(s.value, vn)))]
                values.update(pairs)
                keys.append(pairs)
            elif kind == 'Assign':
                args = tuple(vn(a) if isinstance(a, str) else ('arg', repr(a)) for a in s.args)
                pairs = [(s.name, ('assign', s.op.name, str(s.dtype), args))]
                values.update(pairs)
                keys.append(pairs)
            else:
                state[s.name] = ('fold', ops[s.name], vn(s.base or s.name), vn(s.value))
                keys.append(None)
        return state, keys

    while True:
        state, keys = number(classes)
        refined = {name: (classes[name], state[name]) for name in folds}
        if len(set(refined.values())) == len(set(classes.values())):
            break
        classes = refined
    rep = {}
    carried = {}
    for name in folds:
        carried[name] = rep.setdefault(classes[name], name)
    carried = {name: r for name, r in carried.items() if name != r}
    skip, record, seen = {}, {}, set()
    for s, pairs in zip(body, keys):
        if type(s).__name__ == 'Accum' and s.name in carried:
            skip[id(s)] = []
        elif pairs is not None:
            if all(key in seen for _, key in pairs):
                skip[id(s)] = pairs
            else:
                record[id(s)] = pairs
                seen.update(key for _, key in pairs)
    # A surviving statement must never read an aliased fold mid-iteration:
    # the representative's register may already hold this trip's update.
    live = [s for s in body if id(s) not in skip]
    read = set()
    for s in live:
        read.update(s.deps())
        for nested in statement_bodies(s):
            read.update(touched_names(nested))
    for name in list(carried):
        if name in read:
            del carried[name]
            for s in body:
                if type(s).__name__ == 'Accum' and s.name == name:
                    skip.pop(id(s), None)
    return skip, record, carried


class HoistedValue:
    """A pure statement in a serial loop (a nested reduction loop or a global
    Load) evaluated once per distinct (serial index, used thread-axis values)
    by the whole CTA before the loop.

    ``used`` lists the CTA-varying axes the statement reads, row-major; the
    remaining CTA-varying axes only replicated the same value across threads.
    Results live in shared rows of ``extent + 1`` words (padded so that
    threads reading different rows at one serial index hit distinct banks).
    ``serial_fastest`` orders work items so that the global index the
    statement reads contiguously varies across adjacent threads.
    """

    def __init__(self, statement, exports, used, configs, extent, symbol, serial_fastest):
        self.statement, self.exports, self.used = statement, exports, used
        self.configs, self.extent, self.symbol = configs, extent, symbol
        self.serial_fastest = serial_fastest
        self.row_words = extent + 1
        self.nbytes = len(exports) * configs * self.row_words * 4


def index_addresses(statements):
    """Collect addressable index tuples used by memory statements."""
    found = []
    for statement in statements:
        kind = type(statement).__name__
        if kind == 'Load':
            found.append((statement.input, statement.index, tuple(range(len(statement.names)))))
        elif kind == 'Write':
            found.append((statement.output, statement.index, tuple(range(len(statement.values)))))
        elif kind == 'CpAsyncCopy':
            destination = statement.smem_index
            if statement.lane_index is not None:
                from emmy.compiler.ir.expr import BinaryExpr
                destination = tuple(BinaryExpr('+', a, b) for a, b in
                                    zip(destination, statement.lane_index, strict=True))
            found.append((statement.src, statement.src_index, (0,)))
            found.append((statement.smem, destination, (0,)))
        for field in ('body', 'else_body'):
            nested = getattr(statement, field, ())
            if nested:
                found.extend(index_addresses(nested))
    return found


def provisional_versions(statements, versions=None):
    """Binding versions after the first textual pass over a loop body."""
    versions = dict(versions or {})
    for statement in statements:
        kind = type(statement).__name__
        if kind in ('Assign', 'Accum', 'Init', 'Let', 'Carry'):
            versions[statement.name] = versions.get(statement.name, 0) + 1
        elif kind in ('Loop', 'StridedLoop'):
            versions[statement.axis.name] = versions.get(statement.axis.name, 0) + 1
        for nested in statement_bodies(statement):
            versions = provisional_versions(nested, versions)
    return versions


def invariant_nodes(expressions, names):
    """Maximal pure integer subtrees independent of the loop bindings."""
    found = []
    def walk(expr):
        kind = type(expr).__name__
        if kind in ('Var', 'Literal'):
            return
        free = expr_names(expr)
        if free is not None and not free & names:
            found.append(expr)
            return
        if kind == 'BinaryExpr':
            walk(expr.left)
            walk(expr.right)
        elif kind == 'TernaryExpr':
            walk(expr.cond)
            walk(expr.if_true)
            walk(expr.if_false)
    for expr in expressions:
        walk(expr)
    return found


def structural_key(expr):
    """Version-independent identity of a pure index expression."""
    kind = type(expr).__name__
    if kind == 'Var':
        return ('var', expr.name)
    if kind == 'Literal':
        return ('literal', repr(expr.value), getattr(expr, 'dtype', None))
    if kind == 'BinaryExpr':
        return ('binary', expr.op, structural_key(expr.left), structural_key(expr.right))
    if kind == 'TernaryExpr':
        return ('ternary', structural_key(expr.cond), structural_key(expr.if_true),
                structural_key(expr.if_false))
    return None


def invariant_occurrences(expressions, names):
    """Every maximal pure subtree independent of the loop bindings.

    The cost filter in ``hoist_invariants`` needs occurrence counts: a value
    used twice in the body is worth a preheader register, while a unique value
    is usually better left at its use site.
    """
    found = []

    def walk(expr):
        free = expr_names(expr)
        if free is not None and not free & names:
            found.append(expr)
            return
        if type(expr).__name__ == 'BinaryExpr':
            walk(expr.left)
            walk(expr.right)
        elif type(expr).__name__ == 'TernaryExpr':
            walk(expr.cond)
            walk(expr.if_true)
            walk(expr.if_false)

    for expr in expressions:
        walk(expr)
    return found


def statement_nbytes(statement):
    """Bytes occupied by a typed smem declaration."""
    dtype = smem_dtype(statement)
    return dtype_info(dtype)[0] * math.prod(static(v) for v in statement.extents)


def smem_dtype(statement):
    dtype = getattr(statement, 'dtype', None)
    return {'float': 'f32', 'double': 'f64', 'half': 'f16', 'int': 'i32'}.get(dtype, dtype)


def smem_alignment(statement):
    natural = dtype_info(smem_dtype(statement))[0]
    return max(natural, int(getattr(statement, 'align', 0) or 0))


class Emitter:
    def __init__(self, graph, node, meta):
        self.graph, self.node, self.meta = graph, node, meta
        self.lines, self.declarations = [], []
        self.externals = []
        self.counts = {'r': 0, 'rh': 0, 'rd': 0, 'f': 0, 'fd': 0, 'p': 0}
        self.values, self.buffers, self.shared = {}, {}, {}
        self.label_count = 0
        self.accum_ops = {}
        self.mutable = {}
        self.inactive = []
        self.shared_offsets = {}
        self.address_cache = {}
        self.expr_cache = {}
        self.loop_invariants = []
        self.loop_expressions = []
        self.versions = {}
        self.var_residue = {}
        self.shared_statements = {}
        self.shared_bytes = 0
        self.has_cp_async = False
        self.cp_async_base = '__emmy_smem'
        self.hoist_plans = {}
        self.hoisted = {}
        self.hoist_bytes = 0
        self.vn_skip, self.vn_record, self.vn_registers = {}, {}, {}
        self.vector_loops = 0
        self.tile_extents = {}
        self.subst = {}
        def shared_layout(statements):
            for statement in statements:
                if type(statement).__name__ == 'Smem':
                    if statement.name in self.shared_offsets:
                        raise Unsupported('Duplicate shared buffer name')
                    align = smem_alignment(statement)
                    self.shared_bytes = (self.shared_bytes + align - 1) // align * align
                    self.shared_offsets[statement.name] = self.shared_bytes
                    self.shared_bytes += statement_nbytes(statement)
                    dtype = smem_dtype(statement)
                    ident(statement.name)
                    self.buffers[statement.name] = (tuple(static(v) for v in statement.extents), dtype)
                    self.shared[statement.name] = statement
                    self.shared_statements[statement.name] = statement
                for field in ('body', 'else_body', 'combine_states'):
                    nested = getattr(statement, field, ())
                    if nested:
                        shared_layout(nested)
        shared_layout(node.op.body)
        def find_cp_async(statements):
            for statement in statements:
                if type(statement).__name__ == 'CpAsyncCopy':
                    return True
                for field in ('body', 'else_body'):
                    nested = getattr(statement, field, ())
                    if nested and find_cp_async(nested):
                        return True
            return False
        self.has_cp_async = find_cp_async(node.op.body)
        self.dynamic_shared = bool(self.shared_bytes and meta.get('smem_bytes', 0) >= self.shared_bytes)
        if self.shared_bytes > 48 * 1024 and not self.dynamic_shared:
            raise Unsupported('Shared buffers exceed static limit and metadata dynamic allocation')
        if self.dynamic_shared:
            self.externals.append('.extern .shared .align 16 .b8 __emmy_smem[];')
        else:
            for name, statement in self.shared_statements.items():
                self.declarations.append(
                    f'.shared .align {smem_alignment(statement)} .b8 {name}[{statement_nbytes(statement)}];')
        for n in graph.nodes.values():
            for tensor in n.outputs:
                self.buffers[tensor.name] = (tuple(static(d) for d in tensor.shape), tensor.dtype.name)
        arg_info = {}
        for arg in meta['args']:
            arg_info[arg['name']] = arg
            # A by-value scalar is not addressable storage. Registering it as a
            # buffer would let a Load of the same name bypass the parameter.
            if not arg.get('scalar'):
                self.buffers[arg['name']] = (tuple(arg.get('shape', (1,))), arg['dtype'])
        if node.op.serial:
            raise Unsupported('Serial runtime axes are not supported')
        self.scalar_args = {}
        self.scalar_constants = {}
        self.ptrs = {}
        for i, name in enumerate(meta['arg_order']):
            ident(name)
            arg = arg_info.get(name)
            if arg is None:
                raise Unsupported(f'Argument {name!r} absent from metadata args')
            dtype = dtype_name(arg['dtype'])
            if arg.get('scalar'):
                phys = dtype_info(dtype)[1]
                reg = self.reg(reg_suffix(dtype))
                self.emit(f'ld.param.{phys} {reg}, [param_{i}];')
                self.scalar_args[name] = (reg, dtype)
                if 'value' in arg:
                    self.scalar_constants[name] = arg['value']
                continue
            if 'shape' not in arg:
                raise Unsupported(f'Buffer {name!r} lacks metadata shape')
            ptr = self.reg('rd')
            self.emit(f'ld.param.u64 {ptr}, [param_{i}];')
            self.emit(f'cvta.to.global.u64 {ptr}, {ptr};')
            self.ptrs[name] = ptr
        block_dims = tuple(static(v) for v in self.meta['block'])
        grid_dims = tuple(static(v) for v in self.meta['grid'])
        if len(block_dims) != 3 or len(grid_dims) != 3 or min(*block_dims, *grid_dims) <= 0:
            raise Unsupported('Launch geometry must be three positive dimensions')
        block_size = math.prod(block_dims)
        if block_size > 1024 or block_size >= (1 << 31):
            raise Unsupported('Launch block size outside supported range')
        thread_xyz = []
        for axis, suffix in enumerate('xyz'):
            reg = self.reg('r')
            self.emit(f'mov.u32 {reg}, %tid.{suffix};')
            thread_xyz.append(reg)
            if axis:
                self.values[f'tid_{suffix}'] = (reg, 'i32')
        block_xyz = []
        for axis, suffix in enumerate('xyz'):
            reg = self.reg('r')
            self.emit(f'mov.u32 {reg}, %ctaid.{suffix};')
            block_xyz.append(reg)
            if axis:
                self.values[f'bid_{suffix}'] = (reg, 'i32')
        def linear(parts, dims):
            value = parts[0]
            for axis in range(1, 3):
                reg = self.reg('r')
                self.emit(f'mad.lo.u32 {reg}, {value}, {dims[axis]}, {parts[axis]};')
                value = reg
            return value
        tid = linear(thread_xyz, block_dims)
        bid = linear(block_xyz, grid_dims)
        if tid != thread_xyz[0]:
            self.emit(f'mov.u32 {thread_xyz[0]}, {tid};')
        if bid != block_xyz[0]:
            self.emit(f'mov.u32 {block_xyz[0]}, {bid};')
        self.values['tid'] = (thread_xyz[0], 'i32')
        self.values['bid'] = (block_xyz[0], 'i32')
        ntid = self.reg('r')
        self.emit(f'mov.u32 {ntid}, {block_size};')
        self.values['ntid'] = (ntid, 'i32')
        gid = self.reg('r')
        self.emit(f'mad.lo.u32 {gid}, {bid}, {ntid}, {tid};')
        self.values['_gid'] = (gid, 'i32')
        for name, instruction, rhs in [('lane', 'and.b32', 31), ('warp', 'shr.u32', 5)]:
            reg = self.reg('r')
            self.emit(f'{instruction} {reg}, {tid}, {rhs};')
            self.values[name] = (reg, 'i32')
        # A by-value parameter is an SSA binding; it is never a buffer name.
        for name, (reg, dtype) in self.scalar_args.items():
            self.values[name] = (reg, dtype)

    def reg(self, kind):
        self.counts[kind] += 1
        return f'%{kind}{self.counts[kind]}'

    def emit(self, instruction):
        self.lines.append('    ' + instruction)

    def label(self, prefix='L'):
        self.label_count += 1
        return f'{prefix}_{self.label_count}'

    def mark(self, label):
        # A label may be reached from a path that never evaluated a cached
        # expression. Drop ordinary block temporaries but retain addresses
        # that were emitted into a loop preheader.
        self.address_cache.clear()
        self.expr_cache.clear()
        self.address_cache.update(self.loop_invariants[-1] if self.loop_invariants else {})
        self.expr_cache.update(self.loop_expressions[-1] if self.loop_expressions else {})
        self.lines.append(label + ':')

    def rebind(self, name):
        """Invalidate address expressions that depend on a rebound value."""
        self.versions[name] = self.versions.get(name, 0) + 1
        self.var_residue.pop(name, None)
        self.address_cache.clear()
        self.expr_cache.clear()
        self.address_cache.update(self.loop_invariants[-1] if self.loop_invariants else {})
        self.expr_cache.update(self.loop_expressions[-1] if self.loop_expressions else {})

    def expr_key(self, x):
        return self.expr_key_at(x)

    def hoist_invariants(self, body, names):
        """Emit addresses proven independent of this loop's bindings."""
        versions = provisional_versions(body, self.versions)
        original = dict(self.versions)
        hoisted_expressions = {}
        expressions = [coord for _buffer, index, _offsets in index_addresses(body)
                       for coord in index]
        counts = {}
        for expression in invariant_occurrences(expressions, names):
            key = structural_key(expression)
            counts[key] = counts.get(key, 0) + 1
        for expr in invariant_nodes(expressions, names):
            # The preheader is only a win when the value is consumed repeatedly
            # in the body. Hoisting a one-use expression only lengthens live
            # ranges and can turn a latency-hiding kernel into a slower one.
            if counts.get(structural_key(expr), 0) < 2:
                continue
            before = (dict(self.address_cache), dict(self.expr_cache),
                      dict(self.versions), list(self.loop_invariants))
            try:
                value = self.expr(expr)
            except Unsupported:
                self.address_cache, self.expr_cache, self.versions, self.loop_invariants = before
                continue
            self.address_cache, self.expr_cache, self.versions, self.loop_invariants = before
            if value[1] != 'i32':
                continue
            current = self.expr_key_at(expr, original)
            provisional = self.expr_key_at(expr, versions)
            if current is not None:
                self.expr_cache[current] = value
                hoisted_expressions[current] = value
            if provisional is not None:
                self.expr_cache[provisional] = value
                hoisted_expressions[provisional] = value
        self.loop_expressions.append(hoisted_expressions)
        invariant = {}
        for buffer, index, offsets in index_addresses(body):
            free = set()
            for coord in index:
                part = expr_names(coord)
                if part is None:
                    free = None
                    break
                free |= part
            if free is None or free & names:
                continue
            for element_offset in offsets:
                before = (dict(self.address_cache), dict(self.expr_cache),
                          dict(self.versions), list(self.loop_invariants))
                try:
                    address = self.address(buffer, index, element_offset)
                except Unsupported:
                    self.address_cache, self.expr_cache, self.versions, self.loop_invariants = before
                    continue
                self.address_cache, self.expr_cache, self.versions, self.loop_invariants = before
                parts = tuple(self.expr_key_at(coord, versions) for coord in index)
                current = tuple(self.expr_key_at(coord, original) for coord in index)
                if all(part is not None for part in parts) and all(part is not None for part in current):
                    invariant[(buffer, parts, element_offset)] = address
                    if current != parts:
                        self.address_cache[(buffer, current, element_offset)] = address
        self.loop_invariants.append(invariant)
        self.address_cache.update(invariant)

    def expr_key_at(self, x, versions=None):
        """Stable value identity for pure index expressions.

        The textual index repeats across a loop body while its induction
        register is updated at the backedge. Including the register's
        version in this key lets those addresses share one computation
        without ever reusing a value from a previous iteration.
        """
        versions = self.versions if versions is None else versions
        kind = type(x).__name__
        if kind == 'Var':
            return x.name, versions.get(x.name, 0)
        if kind == 'Literal':
            return 'literal', repr(x.value), getattr(x, 'dtype', None)
        if kind == 'BinaryExpr':
            return 'binary', x.op, self.expr_key_at(x.left, versions), self.expr_key_at(x.right, versions)
        if kind == 'TernaryExpr':
            return ('ternary', self.expr_key_at(x.cond, versions),
                    self.expr_key_at(x.if_true, versions), self.expr_key_at(x.if_false, versions))
        return None

    def expr(self, x):
        value = self.expr_uncached(x)
        if type(x).__name__ == 'BinaryExpr' and x.op in ('/', '//', '%'):
            # Floor division and remainder cost several instructions each;
            # cache them under the same versioned key as the cheaper ops.
            key = self.expr_key(x)
            if key is not None:
                self.expr_cache[key] = value
        return value

    def expr_uncached(self, x):
        kind = type(x).__name__
        key = self.expr_key(x)
        if key is not None and key in self.expr_cache:
            return self.expr_cache[key]
        if kind == 'Var':
            if x.name not in self.values:
                raise Unsupported(f'Unknown IR value {x.name}')
            return self.values[x.name]
        if kind == 'Literal':
            if isinstance(x.value, (bool, int)):
                if not -(1 << 31) <= int(x.value) < (1 << 31):
                    raise Unsupported('Integer literal exceeds signed32 bounds')
                dtype = x.dtype if x.dtype in INT_DTYPES else 'i32'
                if dtype in ('i64', 'u64') and not -(1 << 63) <= int(x.value) < (1 << 63):
                    raise Unsupported('Integer literal exceeds signed64 bounds')
                return str(int(x.value)), dtype
            # Emmy's default float literal dtype is ``float``; the declared
            # dtype selects f32 or f64, never the operand's incidental spelling.
            dtype = getattr(x, 'dtype', 'float')
            if dtype in ('f64', 'double'):
                return f64(x.value), 'f64'
            return f32(x.value), 'f32'
        if kind == 'BinaryExpr':
            a, at = self.expr(x.left)
            b, bt = self.expr(x.right)
            comparisons = {'==':'eq', '!=':'ne', '<':'lt', '<=':'le', '>':'gt', '>=':'ge'}
            if x.op in comparisons:
                if at != bt:
                    target = 'f64' if 'f64' in (at, bt) else 'f32'
                    a = convert_register(self, a, at, target)
                    b = convert_register(self, b, bt, target)
                    at = bt = target
                p = self.reg('p')
                self.emit(f'setp.{comparisons[x.op]}.{ptx_compare_type(at)} {p}, {a}, {b};')
                return p, 'pred'
            if at != bt:
                target = 'f64' if 'f64' in (at, bt) else 'f32'
                a = convert_register(self, a, at, target)
                b = convert_register(self, b, bt, target)
                at = bt = target
            ops = {'+':'add.s32', '-':'sub.s32', '*':'mul.lo.s32', '&':'and.b32', '|':'or.b32', '>>':'shr.s32', '<<':'shl.b32'}
            if x.op in ('&&', '||'):
                if at != 'pred' or bt != 'pred':
                    raise Unsupported(f'Logical {x.op} requires predicates')
                p = self.reg('p')
                self.emit(f'{"and.pred" if x.op == "&&" else "or.pred"} {p}, {a}, {b};')
                return p, 'pred'
            if x.op in ('/', '//', '%'):
                if at not in INT_DTYPES:
                    raise Unsupported(f'Unsupported expression operator {x.op} for {at}')
                # Emmy's typed binary ops use floor division and Python floor
                # remainder. PTX div.s32/rem.s32 truncate toward zero, which
                # differs for a negative dividend. Adjust only that case.
                positive_constant = (type(x.right).__name__ == 'Literal'
                                     and isinstance(x.right.value, int) and x.right.value > 0)
                if positive_constant and at == 'i32':
                    divisor = int(x.right.value)
                    if divisor & (divisor - 1) == 0 and x.op in ('/', '//'):
                        # Arithmetic right shift rounds toward negative
                        # infinity, exactly floor division by a power of two.
                        r = self.reg('r')
                        self.emit(f'shr.s32 {r}, {a}, {divisor.bit_length() - 1};')
                        return r, 'i32'
                    if x.op == '%' and divisor & (divisor - 1) == 0:
                        r = self.reg('r')
                        self.emit(f'and.b32 {r}, {a}, {divisor - 1};')
                        return r, 'i32'
                if at in ('i64', 'u64'):
                    kind = 'u' if at == 'u64' else 's'
                    q = self.reg('rd')
                    self.emit(f'div.{kind}64 {q}, {a}, {b};')
                    rem = self.reg('rd')
                    self.emit(f'rem.{kind}64 {rem}, {a}, {b};')
                    if at == 'u64':
                        return (rem if x.op == '%' else q), at
                else:
                    kind = 'u' if at == 'u32' else 's'
                    q = self.reg('r')
                    self.emit(f'div.{kind}32 {q}, {a}, {b};')
                    rem = self.reg('r')
                    self.emit(f'rem.{kind}32 {rem}, {a}, {b};')
                    if at == 'u32':
                        return (rem if x.op == '%' else q), at
                negative_rem = self.reg('p')
                negative_divisor = self.reg('p')
                mismatch = self.reg('p')
                fix = self.reg('p')
                width = 64 if at == 'i64' else 32
                adjust = self.reg('rd' if width == 64 else 'r')
                self.emit(f'setp.lt.s{width} {negative_rem}, {rem}, 0;')
                self.emit(f'setp.lt.s{width} {negative_divisor}, {b}, 0;')
                self.emit(f'xor.pred {mismatch}, {negative_rem}, {negative_divisor};')
                self.emit(f'setp.ne.s{width} {fix}, {rem}, 0;')
                self.emit(f'and.pred {fix}, {fix}, {mismatch};')
                if x.op == '%':
                    self.emit(f'add.s{width} {adjust}, {rem}, {b};')
                else:
                    self.emit(f'add.s{width} {adjust}, {q}, -1;')
                result = self.reg('rd' if at == 'i64' else 'r')
                self.emit(f'selp.s{width} {result}, {adjust}, {rem if x.op == "%" else q}, {fix};')
                return result, at
            if at not in INT_DTYPES or x.op not in ops:
                raise Unsupported(f'Unsupported expression operator {x.op} for {at}')
            if at in ('i64', 'u64'):
                wide = {'+': 'add', '-': 'sub', '*': 'mul.lo', '&': 'and', '|': 'or', '>>': 'shr', '<<': 'shl'}
                kind = 'u' if at == 'u64' else 's'
                r = self.reg('rd')
                self.emit(f'{wide[x.op]}.{kind}64 {r}, {a}, {b};')
                return r, at
            key = ('binary', x.op, self.expr_key(x.left), self.expr_key(x.right))
            cached = self.expr_cache.get(key)
            if cached is not None:
                return cached
            r = self.reg('r')
            self.emit(f'{ops[x.op]} {r}, {a}, {b};')
            self.expr_cache[key] = (r, 'i32')
            return r, 'i32'
        if kind == 'TernaryExpr':
            p, pt = self.expr(x.cond)
            a, at = self.expr(x.if_true)
            b, bt = self.expr(x.if_false)
            if pt != 'pred':
                raise Unsupported('Ternary condition must be a predicate')
            if at != bt:
                target = 'f64' if 'f64' in (at, bt) else 'f32'
                a = convert_register(self, a, at, target)
                b = convert_register(self, b, bt, target)
                at = bt = target
            r = self.reg(reg_suffix(at))
            self.emit(f'selp.{select_type(at)} {r}, {a}, {b}, {p};')
            return r, at
        raise Unsupported(f'Unsupported expression {kind}')

    def value(self, name):
        if name not in self.values:
            raise Unsupported(f'Unbound SSA value {name!r}')
        return self.values[name][0]

    def address(self, name, index, element_offset=0):
        if name not in self.buffers:
            raise Unsupported(f'Unknown buffer {name}')
        shape, dtype = self.buffers[name]
        dtype = dtype_name(dtype)
        nbytes = dtype_info(dtype)[0]
        # Address expressions repeat heavily in scheduled IR: a matmul loads
        # the same shared cell for several output fragments, and copy loops
        # rebuild the same affine index each trip. Reuse only pure index
        # expressions, keyed by the current binding version of each variable.
        parts = tuple(self.expr_key(coord) for coord in index)
        key = (name, parts, element_offset) if all(p is not None for p in parts) else None
        if key is not None:
            cached = self.address_cache.get(key)
            if cached is not None:
                return cached
        if len(index) != len(shape):
            # Scalar buffers may have [] IR indices and shape [1] metadata.
            if index or math.prod(shape) != 1:
                raise Unsupported(f'Index rank mismatch for {name}: {len(index)} vs {shape}')
        offset = '0'
        for axis, coord in enumerate(index):
            value, dt = self.expr(coord)
            if dt != 'i32':
                raise Unsupported('Only signed32 indices are supported')
            stride = math.prod(shape[axis+1:])
            tmp = self.reg('r')
            self.emit(f'mad.lo.s32 {tmp}, {value}, {stride}, {offset};')
            offset = tmp
        if element_offset:
            tmp = self.reg('r')
            self.emit(f'add.s32 {tmp}, {offset}, {element_offset};')
            offset = tmp
        byte = self.reg('rd')
        self.emit(f'mul.wide.s32 {byte}, {offset}, {nbytes};')
        addr = self.reg('rd')
        if name in self.shared:
            if self.dynamic_shared:
                self.emit(f'mov.u64 {addr}, __emmy_smem;')
                self.emit(f'add.u64 {addr}, {addr}, {self.shared_offsets[name]};')
            else:
                self.emit(f'mov.u64 {addr}, {ident(name)};')
            base = addr
        else:
            if name not in self.ptrs:
                raise Unsupported(f'Buffer {name} absent from metadata ABI')
            base = self.ptrs[name]
        self.emit(f'add.u64 {addr}, {base}, {byte};')
        result = addr, 'shared' if name in self.shared else 'global'
        if key is not None:
            self.address_cache[key] = result
        return result

    def emit_cp_async(self, source, destination, nbytes, active=None, base_name=None):
        """Issue one global-to-shared cp.async with a shared-window address.

        ``address`` already contains ``__emmy_smem`` or the static slab symbol,
        so recomputing its base here would either double-count the offset or
        produce an unsupported generic-to-shared conversion. PTX's shared
        window is a 32-bit offset from that base.
        """
        # ``address`` builds a generic address. cp.async takes a full 32-bit
        # shared-window address, so retain the slab's window position as well as
        # the destination's offset within it. ptxas materializes a nonzero
        # window position for dynamic shared memory on sm_120; on sm_89 it folds
        # to zero.
        if base_name is None:
            raise Unsupported('cp.async requires a shared base name')
        shared32 = self.reg('r')
        slab = self.cp_async_base if self.dynamic_shared else ident(base_name)
        base = self.reg('rd')
        self.emit(f'mov.u64 {base}, {slab};')
        offset = self.reg('rd')
        self.emit(f'sub.u64 {offset}, {destination}, {base};')
        slab32 = self.reg('r')
        self.emit(f'mov.u32 {slab32}, {slab};')
        self.emit(f'cvt.u32.u64 {shared32}, {offset};')
        self.emit(f'add.u32 {shared32}, {slab32}, {shared32};')
        qualifier = 'cg' if nbytes == 16 else 'ca'
        pred = f'@{active} ' if active is not None else ''
        self.emit(f'{pred}cp.async.{qualifier}.shared.global '
                  f'[{shared32}], [{source}], {nbytes};')

    def assign(self, name, op, args, dtype=None):
        op = reduce_canon(getattr(op, 'name', None))
        operands = [self.value(a) for a in args]
        operand_types = [self.values[a][1] for a in args]
        literal_args = [self.scalar_constants.get(a, self.value(a) if a in self.values else None)
                        for a in args]
        if dtype is not None:
            result = dtype_name(dtype)
        else:
            if not operand_types:
                raise Unsupported(f'Elementwise operation {op} has no operands')
            result = operand_types[0]
        result = dtype_name(result)
        # A bitcast is a same-width reinterpretation, never a numerical
        # conversion. Handle it before any operand promotion.
        if op == 'bitcast':
            if len(operands) != 1:
                raise Unsupported('bitcast requires exactly one operand')
            source_width = dtype_info(operand_types[0])[0]
            target_width = dtype_info(result)[0]
            if source_width != target_width:
                raise Unsupported(f'bitcast width mismatch {operand_types[0]} -> {result}')
            reg = self.reg(reg_suffix(result))
            self.emit(f'mov.b{target_width * 8} {reg}, {operands[0]};')
            self.values[name] = (reg, result)
            return
        # Integer arithmetic, bit manipulation, and comparisons retain their
        # typed registers. Only the explicitly listed operations are legal;
        # an unknown integer op must fail rather than silently truncate.
        integer_result = result in INT_DTYPES
        if integer_result and not (op in INTEGER_ELEMENTWISE or op in COMPARISONS
                                   or op in ('copy', 'negative', 'abs')):
            raise Unsupported(f'Non-floating elementwise arithmetic unsupported: {result}')
        # where carries a predicate alongside two values. A predicate is not a
        # numeric operand and must never be fed through convert_register.
        value_start = 1 if op == 'where' else 0
        if op not in ('copy', 'bitcast') and any(t != result for t in operand_types[value_start:]):
            # Scalar arithmetic is evaluated in the operation's stamped
            # result dtype. Insert explicit conversions rather than relying on
            # PTX's (nonexistent) implicit operand promotion.
            converted = [convert_register(self, value, dtype, result)
                         for value, dtype in zip(operands[value_start:],
                                                 operand_types[value_start:], strict=True)]
            operands = operands[:value_start] + converted
            operand_types = operand_types[:value_start] + [result] * len(converted)
        if op == 'copy' and len(operands) == 1:
            operands[0] = convert_register(self, operands[0], operand_types[0], result)
            operand_types = [result]
        if op in COMPARISONS and len(operands) == 2:
            compare_type = ptx_compare_type(result)
            if result == 'f16':
                # PTX setp has no b16/half carrier form. Compare the promoted
                # f32 views; the produced mask stays in the stamped result
                # dtype so later ``where``/mask consumers see one wide value.
                operands = [convert_register(self, value, 'f16', 'f32') for value in operands]
                compare_type = 'f32'
            p = self.reg('p')
            self.emit(f'setp.{COMPARISONS[op]}.{compare_type} '
                      f'{p}, {operands[0]}, {operands[1]};')
            reg = self.reg(reg_suffix(result))
            if result in FLOAT_DTYPES:
                self.emit(f'selp.{select_type(result)} {reg}, {literal(1, result)}, {literal(0, result)}, {p};')
            else:
                self.emit(f'selp.{compare_type} {reg}, 1, 0, {p};')
            self.values[name] = (reg, result)
            return
        if op == 'where' and len(operands) == 3:
            condition = as_predicate(self, operands[0], operand_types[0])
            reg = self.reg(reg_suffix(result))
            self.emit(f'selp.{select_type(result)} {reg}, {operands[1]}, {operands[2]}, {condition};')
            self.values[name] = (reg, result)
            return
        if integer_result and op in INTEGER_ELEMENTWISE and len(operands) == 2:
            reg = emit_integer_binary(self, op, result, operands[0], operands[1])
            self.values[name] = (reg, result)
            return
        if integer_result and op in ('negative', 'abs') and len(operands) == 1:
            reg = self.reg(reg_suffix(result))
            # PTX neg/abs apply only to signed numeric types. The absolute
            # value of an unsigned integer is its identity, and its negation
            # is the same-width two's-complement subtraction from zero.
            if result in ('u32', 'u64'):
                if op == 'abs':
                    self.emit(f'mov.{dtype_info(result)[1]} {reg}, {operands[0]};')
                else:
                    self.emit(f'sub.{dtype_info(result)[1]} {reg}, 0, {operands[0]};')
            else:
                self.emit(f'{op if op == "abs" else "neg"}.{dtype_info(result)[1]} '
                          f'{reg}, {operands[0]};')
            self.values[name] = (reg, result)
            return
        reg = self.reg(reg_suffix(result))
        binary = {'add':'add', 'subtract':'sub', 'multiply':'mul', 'divide':'div.rn', 'maximum':'max', 'minimum':'min'}
        unary = {'negative':'neg', 'abs':'abs', 'sqrt':'sqrt.rn', 'reciprocal':'rcp.rn'}
        if op in binary and len(operands) == 2:
            if result == 'f16':
                # PTX f16 has add/mul/min/max but no div; use the f32 form
                # for divide, then round once back to the stamped dtype.
                if op == 'divide':
                    left = convert_register(self, operands[0], result, 'f32')
                    right = convert_register(self, operands[1], result, 'f32')
                    wide = self.reg('f')
                    self.emit(f'div.rn.f32 {wide}, {left}, {right};')
                    self.emit(f'cvt.rn.f16.f32 {reg}, {wide};')
                else:
                    instruction = binary[op] + ('.rn' if op in ('add', 'subtract', 'multiply') else '')
                    self.emit(f'{instruction}.f16 {reg}, {operands[0]}, {operands[1]};')
            else:
                self.emit(f'{binary[op]}.{result} {reg}, {operands[0]}, {operands[1]};')
        elif op in unary and len(operands) == 1:
            if result == 'f16':
                if op in ('sqrt', 'reciprocal'):
                    wide = convert_register(self, operands[0], 'f16', 'f32')
                    out = self.reg('f')
                    self.emit(f'{unary[op]}.f32 {out}, {wide};')
                    self.emit(f'cvt.rn.f16.f32 {reg}, {out};')
                else:
                    self.emit(f'{unary[op]}.f16 {reg}, {operands[0]};')
            else:
                self.emit(f'{unary[op]}.{result} {reg}, {operands[0]};')
        elif op == 'tanh' and len(operands) == 1:
            if result != 'f32':
                raise Unsupported('tanh is only supported for f32')
            self.emit(f'tanh.approx.f32 {reg}, {operands[0]};')
        elif op == 'log' and len(operands) == 1:
            if result != 'f32':
                raise Unsupported('log is only supported for f32')
            # PTX has no natural-log instruction. lg2 is base-2, so scale by
            # ln(2) to preserve the IR's natural-log semantics.
            self.emit(f'lg2.approx.f32 {reg}, {operands[0]};')
            self.emit(f'mul.f32 {reg}, {reg}, {f32(math.log(2.0))};')
        elif op == 'rsqrt' and len(operands) == 1:
            if result != 'f32':
                raise Unsupported('rsqrt is only supported for f32')
            self.emit(f'rsqrt.approx.f32 {reg}, {operands[0]};')
        elif op == 'copy' and len(operands) == 1:
            # PTX register moves name signed/unsigned integers explicitly;
            # ``i32``/``u32`` are IR dtype names, not legal ``mov`` modifiers.
            self.emit(f'mov.{dtype_info(result)[1]} {reg}, {operands[0]};')
        elif op in ('exp', 'exp_fast') and len(operands) == 1:
            if result != 'f32':
                raise Unsupported(f'{op} is only supported for f32')
            scaled = self.reg('f')
            self.emit(f'mul.f32 {scaled}, {operands[0]}, {f32(math.log2(math.e))};')
            # .ftz is a single MUFU.EX2 (CUDA's __expf); without it ptxas
            # wraps each exponential in denormal range fix-ups. It only
            # flushes results below ~1.2e-38 to zero.
            self.emit(f'ex2.approx.ftz.f32 {reg}, {scaled};')
        elif op == 'log2' and len(operands) == 1:
            if result != 'f32':
                raise Unsupported('log2 is only supported for f32')
            self.emit(f'lg2.approx.f32 {reg}, {operands[0]};')
        elif op == 'log10' and len(operands) == 1:
            if result != 'f32':
                raise Unsupported('log10 is only supported for f32')
            self.emit(f'lg2.approx.f32 {reg}, {operands[0]};')
            self.emit(f'mul.f32 {reg}, {reg}, {f32(math.log10(2.0))};')
        elif op == 'sigmoid' and len(operands) == 1:
            if result != 'f32':
                raise Unsupported('sigmoid is only supported for f32')
            scaled = self.reg('f')
            exponent = self.reg('f')
            self.emit(f'mul.f32 {scaled}, {operands[0]}, {f32(-math.log2(math.e))};')
            self.emit(f'ex2.approx.f32 {exponent}, {scaled};')
            self.emit(f'add.f32 {exponent}, {exponent}, {f32(1.0)};')
            self.emit(f'rcp.approx.f32 {reg}, {exponent};')
        elif op == 'silu' and len(operands) == 1:
            if result != 'f32':
                raise Unsupported('silu is only supported for f32')
            scaled = self.reg('f')
            exponent = self.reg('f')
            denominator = self.reg('f')
            self.emit(f'mul.f32 {scaled}, {operands[0]}, {f32(-math.log2(math.e))};')
            self.emit(f'ex2.approx.f32 {exponent}, {scaled};')
            self.emit(f'add.f32 {denominator}, {exponent}, {f32(1.0)};')
            self.emit(f'div.rn.f32 {reg}, {operands[0]}, {denominator};')
        elif op == 'softplus' and len(operands) == 1:
            if result != 'f32':
                raise Unsupported('softplus is only supported for f32')
            absolute = self.reg('f')
            exponent = self.reg('f')
            shifted = self.reg('f')
            logarithm = self.reg('f')
            maximum = self.reg('f')
            self.emit(f'abs.f32 {absolute}, {operands[0]};')
            self.emit(f'mul.f32 {absolute}, {absolute}, {f32(-math.log2(math.e))};')
            self.emit(f'ex2.approx.f32 {exponent}, {absolute};')
            self.emit(f'add.f32 {shifted}, {exponent}, {f32(1.0)};')
            self.emit(f'lg2.approx.f32 {logarithm}, {shifted};')
            self.emit(f'mul.f32 {logarithm}, {logarithm}, {f32(math.log(2.0))};')
            self.emit(f'max.f32 {maximum}, {operands[0]}, {f32(0.0)};')
            self.emit(f'add.f32 {reg}, {maximum}, {logarithm};')
        elif op == 'square' and len(operands) == 1:
            suffix = '.rn' if result == 'f16' else ''
            self.emit(f'mul{suffix}.{result} {reg}, {operands[0]}, {operands[0]};')
        elif op in ('floor', 'ceil', 'round', 'trunc') and len(operands) == 1:
            if result not in FLOAT_DTYPES:
                raise Unsupported(f'{op} requires a floating dtype')
            mode = 'rzi' if op == 'trunc' else {'floor': 'rmi', 'ceil': 'rpi', 'round': 'rni'}[op]
            self.emit(f'cvt.{mode}.{result}.{result} {reg}, {operands[0]};')
        elif op == 'relu' and len(operands) == 1:
            self.emit(f'max.{result} {reg}, {operands[0]}, {literal(0, result)};')
        elif op in ('sin', 'cos') and len(operands) == 1:
            if result != 'f32':
                raise Unsupported(f'{op} is only supported for f32')
            self.emit(f'{op}.approx.f32 {reg}, {operands[0]};')
        elif op == 'tan' and len(operands) == 1:
            if result != 'f32':
                raise Unsupported('tan is only supported for f32')
            # PTX has no tan instruction; the quotient preserves the IR's
            # sin/cos semantics and shares the same SFU approximations.
            sine, cosine = self.reg('f'), self.reg('f')
            self.emit(f'sin.approx.f32 {sine}, {operands[0]};')
            self.emit(f'cos.approx.f32 {cosine}, {operands[0]};')
            self.emit(f'div.rn.f32 {reg}, {sine}, {cosine};')
        elif op == 'pow' and len(operands) == 2:
            if result != 'f32':
                raise Unsupported('pow is only supported for f32')
            # PTX has no general pow instruction. x**y = exp2(y*log2(x)) is
            # the normal form, but log2 is undefined for negative x even when
            # y is integral (the common square/cube spelling). Fold integral
            # constant exponents into repeated multiplication so the sign is
            # preserved; everything else stays on the log/exp path.
            exponent = operands[1]
            constant = literal_args[1] if len(literal_args) == 2 else None
            if isinstance(constant, (int, float)) and float(constant).is_integer() and 0 <= constant <= 8:
                count = int(constant)
                if count == 0:
                    self.emit(f'mov.f32 {reg}, {f32(1.0)};')
                else:
                    self.emit(f'mov.f32 {reg}, {operands[0]};')
                    for _ in range(count - 1):
                        self.emit(f'mul.rn.f32 {reg}, {reg}, {operands[0]};')
            else:
                logarithm, product = self.reg('f'), self.reg('f')
                self.emit(f'lg2.approx.f32 {logarithm}, {operands[0]};')
                self.emit(f'mul.rn.f32 {product}, {logarithm}, {exponent};')
                self.emit(f'ex2.approx.f32 {reg}, {product};')
        elif op == 'sign' and len(operands) == 1:
            if result != 'f32':
                raise Unsupported('sign is only supported for f32')
            positive, negative = self.reg('p'), self.reg('p')
            pos_value, neg_value = self.reg('f'), self.reg('f')
            self.emit(f'setp.gt.f32 {positive}, {operands[0]}, {f32(0.0)};')
            self.emit(f'setp.lt.f32 {negative}, {operands[0]}, {f32(0.0)};')
            self.emit(f'selp.f32 {pos_value}, {f32(1.0)}, {f32(0.0)}, {positive};')
            self.emit(f'selp.f32 {neg_value}, {f32(1.0)}, {f32(0.0)}, {negative};')
            self.emit(f'sub.rn.f32 {reg}, {pos_value}, {neg_value};')
        elif op == 'erf' and len(operands) == 1:
            if result != 'f32':
                raise Unsupported('erf is only supported for f32')
            self.emit(f'mov.f32 {reg}, {self.emit_erf(operands[0])};')
        elif op in ('gelu', 'gelu_tanh') and len(operands) == 1:
            if result != 'f32':
                raise Unsupported(f'{op} is only supported for f32')
            if op == 'gelu_tanh':
                cube, inner, argument = self.reg('f'), self.reg('f'), self.reg('f')
                self.emit(f'mul.rn.f32 {cube}, {operands[0]}, {operands[0]};')
                self.emit(f'mul.rn.f32 {cube}, {cube}, {operands[0]};')
                self.emit(f'mul.rn.f32 {inner}, {cube}, {f32(0.044715)};')
                self.emit(f'add.rn.f32 {inner}, {inner}, {operands[0]};')
                self.emit(f'mul.rn.f32 {argument}, {inner}, {f32(math.sqrt(2.0 / math.pi))};')
                tanh_value = self.reg('f')
                self.emit(f'tanh.approx.f32 {tanh_value}, {argument};')
                self.emit(f'add.rn.f32 {tanh_value}, {tanh_value}, {f32(1.0)};')
                self.emit(f'mul.rn.f32 {tanh_value}, {tanh_value}, {f32(0.5)};')
                self.emit(f'mul.rn.f32 {reg}, {tanh_value}, {operands[0]};')
            else:
                scaled = self.reg('f')
                self.emit(f'mul.rn.f32 {scaled}, {operands[0]}, {f32(1.0 / math.sqrt(2.0))};')
                erf_value = self.emit_erf(scaled)
                shifted = self.reg('f')
                self.emit(f'add.rn.f32 {shifted}, {erf_value}, {f32(1.0)};')
                half = self.reg('f')
                self.emit(f'mul.rn.f32 {half}, {operands[0]}, {f32(0.5)};')
                self.emit(f'mul.rn.f32 {reg}, {half}, {shifted};')
        else:
            raise Unsupported(f'Unsupported elementwise operation {op}/{len(args)} under {result}')
        if name in self.mutable:
            stable = self.mutable[name]
            self.emit(f'mov.{dtype_info(result)[1]} {stable}, {reg};')
            self.values[name] = (stable, result)
        else:
            self.values[name] = (reg, result)

    def emit_erf(self, value):
        """Return an f32 register holding erf(value).

        Abramowitz-Stegun 7.1.26: t = 1/(1+p|x|) and
        erf(|x|) = 1 - (a1 t + a2 t^2 + ... + a5 t^5) exp(-x^2)
        with maximum absolute error about 1.5e-7. Horner starts at a5 and
        multiplies by t once after the final coefficient. The sign is restored
        with an explicit predicate/select, not copysign.
        """
        absolute = self.reg('f')
        self.emit(f'abs.f32 {absolute}, {value};')
        t = self.reg('f')
        self.emit(f'mul.rn.f32 {t}, {absolute}, {f32(0.3275911)};')
        self.emit(f'add.rn.f32 {t}, {t}, {f32(1.0)};')
        reciprocal = self.reg('f')
        self.emit(f'rcp.approx.f32 {reciprocal}, {t};')
        polynomial = self.reg('f')
        self.emit(f'mov.f32 {polynomial}, {f32(1.061405429)};')
        for coefficient in (-1.453152027, 1.421413741, -0.284496736, 0.254829592):
            self.emit(f'mul.rn.f32 {polynomial}, {polynomial}, {reciprocal};')
            self.emit(f'add.rn.f32 {polynomial}, {polynomial}, {f32(coefficient)};')
        self.emit(f'mul.rn.f32 {polynomial}, {polynomial}, {reciprocal};')
        square = self.reg('f')
        self.emit(f'mul.rn.f32 {square}, {absolute}, {absolute};')
        self.emit(f'mul.rn.f32 {square}, {square}, {f32(-math.log2(math.e))};')
        exponential = self.reg('f')
        self.emit(f'ex2.approx.f32 {exponential}, {square};')
        magnitude = self.reg('f')
        self.emit(f'mul.rn.f32 {magnitude}, {polynomial}, {exponential};')
        self.emit(f'sub.rn.f32 {magnitude}, {f32(1.0)}, {magnitude};')
        negative = self.reg('p')
        self.emit(f'setp.lt.f32 {negative}, {value}, {f32(0.0)};')
        negated = self.reg('f')
        self.emit(f'neg.f32 {negated}, {magnitude};')
        signed = self.reg('f')
        self.emit(f'selp.f32 {signed}, {negated}, {magnitude}, {negative};')
        return signed

    def init_accums(self, statements):
        for s in statements:
            if type(s).__name__ == 'Accum':
                op = reduce_canon(s.op.name)
                identities = {'add': 0.0, 'multiply': 1.0, 'maximum': -1e30, 'minimum': 1e30}
                if op not in identities:
                    raise Unsupported(f'Unsupported reduction {op}')
                dtype = dtype_name(s.dtype) if s.dtype is not None else 'f32'
                if dtype not in FLOAT_DTYPES:
                    raise Unsupported(f'Only floating reductions supported: {dtype}')
                reg = self.reg(reg_suffix(dtype))
                self.emit(f'mov.{dtype} {reg}, {literal(identities[op], dtype)};')
                self.values[s.name] = (reg, dtype)
                self.mutable[s.name] = reg
                self.accum_ops[s.name] = op
            elif type(s).__name__ == 'Cond':
                self.init_accums(s.body)
                self.init_accums(s.else_body)

    def select(self, statement):
        """Lower Emmy's ordered coordinate-predicated value binding.

        Branch predicates are tested in source order. Later branches are the
        catch-all in Emmy's ``select_to_ternary``, so build the PTX chain from
        the last branch backwards. The upstream renderer declares this result
        as C ``float``; convert every branch value to f32 for the same contract.
        """
        selected = None
        for branch in reversed(statement.branches):
            if branch.value not in self.values:
                raise Unsupported(f'Unknown Select branch value {branch.value}')
            value, value_type = self.values[branch.value]
            value = convert_register(self, value, value_type, 'f32')
            if selected is None:
                selected = value
                continue
            condition, condition_type = self.expr(branch.select)
            if condition_type != 'pred':
                condition = as_predicate(self, condition, condition_type)
            reg = self.reg('f')
            self.emit(f'selp.f32 {reg}, {value}, {selected}, {condition};')
            selected = reg
        self.values[statement.name] = (selected, 'f32')
        self.rebind(statement.name)

    def preinitialize_trees(self, statements):
        """Seed TreeHalve slabs with their combine identities.

        A partial CTA can contain a warp whose every lane is inactive. Such a
        warp publishes no partial, but the shared tree still reads its slot.
        Without a neutral seed that slot is undefined and a maximum/minimum
        reduction can be corrupted even though active folds were masked.
        """
        trees = []
        def walk(items):
            for statement in items:
                if type(statement).__name__ == 'TreeHalve':
                    trees.append(statement)
                for field in ('body', 'else_body'):
                    nested = getattr(statement, field, ())
                    if nested:
                        walk(nested)
        walk(statements)
        seeded = {}
        for tree in trees:
            if tree.length <= 0 or tree.length & (tree.length - 1):
                raise Unsupported('Unsupported shared tree configuration')
            if len(tree.bufs) != len(tree.combine_states):
                raise Unsupported('Tree state and combine arity differ')
            inner_scale = tree.inner[1] if tree.inner is not None else 1
            if inner_scale <= 0:
                raise Unsupported('Shared tree inner extent must be positive')
            total_slots = tree.length * inner_scale
            launched = math.prod(self.meta['block'])
            for buf, combine in zip(tree.bufs, tree.combine_states, strict=True):
                op = combine.op.name
                op = reduce_canon(op)
                identities = {'add': 0.0, 'multiply': 1.0, 'maximum': -1e30, 'minimum': 1e30}
                if op not in identities:
                    raise Unsupported(f'Unsupported tree reduction {op}')
                identity = identities[op]
                if buf in seeded:
                    if seeded[buf] != identity:
                        raise Unsupported('Tree buffer has conflicting identities')
                    continue
                seeded[buf] = identity
                dtype = dtype_name(self.buffers[buf][1])
                literal_value = literal(identity, dtype)
                # The tree layout may contain several inner slots per segment.
                # Seed every address the halving loop can subsequently read,
                # striding across the launched CTA rather than assuming one
                # thread per tree position.
                from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
                slot = self.reg('r')
                self.emit(f'mov.u32 {slot}, {self.value("tid")};')
                self.values['_tree_slot'] = (slot, 'i32')
                begin, done = self.label('tree_seed'), self.label('tree_seed_done')
                self.mark(begin)
                active = self.reg('p')
                self.emit(f'setp.ge.u32 {active}, {slot}, {total_slots};')
                self.emit(f'@{active} bra {done};')
                addr, space = self.address(buf, (Var('_tree_slot'),))
                if space != 'shared':
                    raise Unsupported('Tree buffer is not shared')
                self.emit(f'st.shared.{dtype_info(dtype)[1]} [{addr}], {literal_value};')
                self.emit(f'add.u32 {slot}, {slot}, {launched};')
                self.emit(f'bra {begin};')
                self.mark(done)
                self.values.pop('_tree_slot', None)

    def strided_vector_plan(self, s, stride, bound):
        """Loads a thread-strided reduction may fetch four elements at a time.

        The loop folds f(x[..., a]) for a = t, t + stride, ... where t is a
        Tile axis of extent ``stride``: together the threads cover every
        element once. Reassigning elements so thread t folds 4t..4t+3,
        4t + 4*stride.. keeps that exact cover, and is legal when every fold
        is add/max/min of a value computed from the current element only
        (not from carried state): the per-thread partial is then an
        ordinary associative reduction, merged as before. Sums change only
        by floating-point reassociation, as in any parallel reduction.
        Requires 4*stride | bound and the axis only as the contiguous last
        coordinate of f32 global loads whose other strides are multiples
        of 4. Returns those loads, or None.
        """
        start = s.start
        if (type(start).__name__ != 'Var' or self.tile_extents.get(start.name) != stride
                or bound % (4 * stride) or bound + 4 * stride >= (1 << 31) or self.inactive):
            return None
        axis = s.axis.name
        carried = {x.name for x in s.body if type(x).__name__ == 'Accum'}
        tainted, loads = set(), []
        for x in s.body:
            kind = type(x).__name__
            if kind == 'Load':
                names = [expr_names(c) for c in x.index]
                if any(n is None for n in names):
                    return None
                if axis in set().union(*names):
                    shape, dtype = self.buffers[x.input]
                    last = x.index[-1]
                    if (len(x.names) != 1 or dtype_name(dtype) != 'f32' or x.input not in self.ptrs
                            or x.input in self.shared
                            or not (type(last).__name__ == 'Var' and last.name == axis)
                            or any(axis in n for n in names[:-1])
                            or any(math.prod(shape[i + 1:]) % 4 for i in range(len(shape) - 1))):
                        return None
                    loads.append(x)
            elif kind in ('Assign', 'Let'):
                deps = set(x.deps())
                if axis in deps:
                    return None
                if deps & (carried | tainted):
                    tainted.add(x.name)
            elif kind == 'Accum':
                if (reduce_canon(x.op.name) not in ('add', 'maximum', 'minimum')
                        or x.base not in (None, x.name) or x.value in carried | tainted):
                    return None
            else:
                return None
        return loads or None

    def strided_vector_path(self, s, r, stride, bound, end, loads):
        """Fold the whole strided loop with 128-bit loads, then leave the
        induction register at the bound so the ordered loops below run no
        trips. A misaligned base pointer skips straight to those loops."""
        skip = self.label('vec_skip')
        misaligned = None
        for load in dict.fromkeys(x.input for x in loads):
            low, p = self.reg('rd'), self.reg('p')
            self.emit(f'and.b64 {low}, {self.ptrs[load]}, 15;')
            self.emit(f'setp.ne.u64 {p}, {low}, 0;')
            if misaligned is not None:
                self.emit(f'or.pred {p}, {p}, {misaligned};')
            misaligned = p
        self.emit(f'@{misaligned} bra {skip};')
        va = self.reg('r')
        self.emit(f'shl.b32 {va}, {r}, 2;')
        vstep = 4 * stride
        for group in STRIDED_VECTOR_BATCH:
            begin, done = self.label('vec'), self.label('vec_done')
            self.mark(begin)
            last, p = self.reg('r'), self.reg('p')
            self.emit(f'add.s32 {last}, {va}, {(group - 1) * vstep};')
            self.emit(f'setp.ge.s32 {p}, {last}, {bound};')
            self.emit(f'@{p} bra {done};')
            fetched = []
            for k in range(group):
                ak = self.reg('r')
                self.emit(f'add.s32 {ak}, {va}, {k * vstep};')
                self.values[s.axis.name] = (ak, 'i32')
                self.rebind(s.axis.name)
                lanes = {}
                for load in loads:
                    addr, space = self.address(load.input, load.index)
                    regs = [self.reg('f') for _ in range(4)]
                    self.emit(f'ld.{space}.v4.f32 {{{", ".join(regs)}}}, [{addr}];')
                    lanes[id(load)] = regs
                fetched.append(lanes)
            for lanes in fetched:
                for j in range(4):
                    for x in s.body:
                        if id(x) in lanes:
                            self.values[x.names[0]] = (lanes[id(x)][j], 'f32')
                            self.rebind(x.names[0])
                        else:
                            self.statements([x])
            self.emit(f'add.s32 {va}, {va}, {group * vstep};')
            self.emit(f'bra {begin};')
            self.mark(done)
        self.emit(f'mov.s32 {r}, {end};')
        self.mark(skip)
        self.values[s.axis.name] = (r, 'i32')
        self.rebind(s.axis.name)

    def strided_main_loop(self, s, r, step, end):
        """Run whole groups of iterations of a thread-strided loop back to
        back, largest STRIDED_UNROLL group first; the caller's loop then
        finishes the remainder one iteration at a time.

        A runtime start (usually the thread index) keeps ptxas from unrolling,
        leaving one load in flight per thread. The iterations still execute in
        their original order, so every fold sees the same operands in the same
        sequence and results are bit-identical; the loads merely become
        independent instructions ptxas can issue ahead of the dependent folds.
        """
        stride = static_or_none(s.step)
        bound = static_or_none(s.end if s.end is not None else s.axis.extent)
        if not straight_line(s.body) or stride is None or stride <= 0 or bound is None:
            return
        vector = self.strided_vector_plan(s, stride, bound)
        if vector is not None:
            self.strided_vector_path(s, r, stride, bound, end, vector)
        for group in STRIDED_UNROLL:
            if bound + (group - 1) * stride >= (1 << 31):
                continue
            last = self.reg('r')
            p = self.reg('p')
            begin, done = self.label('loop_unrolled'), self.label('loop_unrolled_done')
            self.mark(begin)
            self.emit(f'add.s32 {last}, {r}, {(group - 1) * stride};')
            self.emit(f'setp.ge.s32 {p}, {last}, {end};')
            self.emit(f'@{p} bra {done};')
            for _ in range(group):
                self.statements(s.body)
                self.emit(f'add.s32 {r}, {r}, {step};')
                self.rebind(s.axis.name)
            self.emit(f'bra {begin};')
            self.mark(done)

    def vector_load_plan(self, loop):
        """Plan 128/64-bit shared loads for a loop unrolled VECTOR_UNROLL times.

        Copy k binds the axis to (trip base + k), and the trip base is a
        multiple of VECTOR_UNROLL. Scalar shared loads (in any copy) whose
        flat offsets share every non-constant term form runs of consecutive
        elements; an aligned run of 4 or 2 becomes one vector load. A run is
        aligned when its first constant and every term coefficient are
        multiples of the width (the axis term contributes a multiple of
        VECTOR_UNROLL) and the buffer's shared offset is width-aligned.
        Returns {(copy, id(load)): (leader, registers slot, lane)} or None.
        """
        extent = static_or_none(loop.axis.extent)
        if (extent is None or extent % VECTOR_UNROLL or self.inactive
                or not straight_line(loop.body)
                or any(id(s) in self.hoisted or id(s) in self.vn_skip for s in loop.body)):
            return None
        axis = ('var', loop.axis.name)
        # Base addresses are computed once before the loop, so an index may
        # read only the axis and names bound outside the body.
        varying = defined_names(loop.body) - {loop.axis.name}
        entries, bases = {}, {}
        for position, s in enumerate(loop.body):
            if type(s).__name__ != 'Load' or len(s.names) != 1 or s.input not in self.shared:
                continue
            names = [expr_names(c) for c in s.index]
            if any(n is None or n & varying for n in names):
                continue
            shape, dtype = self.buffers[s.input]
            if dtype_name(dtype) != 'f32':
                continue
            form = flat_affine(shape, s.index)
            if form is None:
                continue
            constant, terms = form
            step = terms.get(axis, 0)
            others = {k: c for k, c in terms.items() if k != axis}
            if self.dynamic_shared:
                aligned_base = self.shared_offsets[s.input] % 16 == 0
            else:
                aligned_base = smem_alignment(self.shared_statements[s.input]) >= 16
            key = (s.input, frozenset(terms.items()))
            bases.setdefault(key, (s, constant, step))
            for copy in range(VECTOR_UNROLL):
                entries.setdefault(key, []).append(
                    (constant + copy * step, copy, position, s, others, aligned_base))
        plan, slot = {}, 0
        for key, members in entries.items():
            origin = bases[key][1]
            by_const = {}
            for member in members:
                by_const.setdefault(member[0], member)
            consts = sorted(by_const)
            used = set()
            for c in consts:
                if c in used:
                    continue
                for width in (4, 2, 1):
                    run = [by_const.get(c + i) for i in range(width)]
                    if (None in run or any(m[0] in used for m in run) or c % width
                            or not run[0][5]
                            or any(coef % width for coef in run[0][4].values())):
                        continue
                    # The lowest element must be emitted first: it carries
                    # the aligned address the vector load uses.
                    order = sorted(run, key=lambda m: (m[1], m[2]))
                    if order[0] is not run[0]:
                        continue
                    for lane, m in enumerate(run):
                        plan[(m[1], id(m[3]))] = (m is run[0], slot, lane, width, key, c - origin)
                        used.add(m[0])
                    slot += 1
                    break
        if not any(entry[3] > 1 for entry in plan.values()):
            return None
        return plan, bases

    def vector_unrolled_body(self, loop, r, plan):
        """Emit VECTOR_UNROLL consecutive iterations with combined loads.

        Each iteration's statements run in their original order (folds are
        unchanged); a combined load is issued at its lowest element and later
        elements only bind the already-loaded lane.
        """
        plan, bases = plan
        slots, pointers = {}, {}
        for key, (base, space, step) in bases.items():
            if step:
                offset, pointer = self.reg('rd'), self.reg('rd')
                self.emit(f'mul.wide.s32 {offset}, {r}, {4 * step};')
                self.emit(f'add.u64 {pointer}, {base}, {offset};')
            else:
                pointer = base
            pointers[key] = (pointer, space)
        for copy in range(VECTOR_UNROLL):
            if copy:
                rk = self.reg('r')
                self.emit(f'add.s32 {rk}, {r}, {copy};')
            else:
                rk = r
            self.values[loop.axis.name] = (rk, 'i32')
            self.rebind(loop.axis.name)
            for s in loop.body:
                entry = plan.get((copy, id(s)))
                if entry is None:
                    self.statements([s])
                    continue
                leader, slot, lane, width, key, element = entry
                if leader:
                    pointer, space = pointers[key]
                    regs = [self.reg('f') for _ in range(width)]
                    vector = f'.v{width}' if width > 1 else ''
                    dest = f'{{{", ".join(regs)}}}' if width > 1 else regs[0]
                    self.emit(f'ld.{space}{vector}.f32 {dest}, [{pointer}+{4 * element}];')
                    slots[slot] = regs
                self.values[s.names[0]] = (slots[slot][lane], 'f32')
                self.rebind(s.names[0])

    def two_pass_softmax(self, loop, online):
        """Seed online-softmax maxima with their final values.

        A pre-pass over the loop folds each carrier's ``x`` with maximum from
        the same identity, in the same order as the online loop would, so
        the seed equals the online loop's final maximum. The main loop then
        reads that constant: the carrier's fold is dropped, ``gn`` is the
        maximum, ``dg`` is 0 and ``alpha`` is 1, leaving one exponential
        and no rescaling chain per step.
        """
        seeds = {}
        for fold, x, gn, dg, alpha, _ in online:
            m = self.reg('f')
            self.emit(f'mov.f32 {m}, {literal(-1e30, "f32")};')
            seeds[fold.name] = m
        needed = {id(s) for *_, statements in online for s in statements}
        r = self.reg('r')
        self.emit(f'mov.s32 {r}, 0;')
        self.values[loop.axis.name] = (r, 'i32')
        self.rebind(loop.axis.name)
        begin, done = self.label('maxpass'), self.label('maxpass_done')
        self.mark(begin)
        p = self.reg('p')
        self.emit(f'setp.ge.s32 {p}, {r}, {static(loop.axis.extent)};')
        self.emit(f'@{p} bra {done};')
        self.statements([s for s in loop.body if id(s) in needed])
        for fold, x, *_ in online:
            self.emit(f'max.f32 {seeds[fold.name]}, {seeds[fold.name]}, {self.value(x)};')
        self.emit(f'add.s32 {r}, {r}, 1;')
        self.rebind(loop.axis.name)
        self.emit(f'bra {begin};')
        self.mark(done)
        zero, one = self.reg('f'), self.reg('f')
        self.emit(f'mov.f32 {zero}, {literal(0.0, "f32")};')
        self.emit(f'mov.f32 {one}, {literal(1.0, "f32")};')
        for fold, x, gn, dg, alpha, _ in online:
            m = seeds[fold.name]
            self.values[fold.name] = (m, 'f32')
            self.mutable[fold.name] = m
            self.rebind(fold.name)
            self.subst[id(fold)] = []
            self.subst[id(gn)] = [(gn.name, m)]
            self.subst[id(dg)] = [(dg.name, zero)]
            if alpha is not None:
                self.subst[id(alpha)] = [(alpha.name, one)]

    def plan_hoists(self, tile, extents, launched):
        """Find reductions that CTA threads recompute redundantly.

        In a serial loop ``for o``, a nested reduction loop that reads only
        Tile axes, ``o`` and unwritten global inputs yields the same value in
        every thread that agrees on the axes it reads. Attention's QK^T dot
        product is the motivating case: each output column thread recomputes
        its row's scores. Such a reduction can run once per distinct
        (o, used axes) before the serial loop, split across the CTA, with the
        serial loop reading the results back from shared memory. Each value is
        produced by the identical statement sequence, so results are
        bit-identical.
        """
        cut = next((i for i in range(len(extents) + 1)
                    if math.prod(extents[i:]) == launched), None)
        if cut is None:
            return
        axis_extent = {axis.name: extent for axis, extent in zip(tile.axes, extents)}
        cta_axes = [axis.name for axis in tile.axes[cut:]]
        outputs = {arg['name'] for arg in self.meta['args'] if arg.get('output')}
        written = written_buffers(tile.body) | outputs
        budget = HOIST_SHARED_LIMIT
        for outer in tile.body:
            if type(outer).__name__ != 'Loop':
                continue
            extent = static_or_none(outer.axis.extent)
            if extent is None:
                continue
            elsewhere = touched_names([s for s in tile.body if s is not outer])
            carried = any(type(x).__name__ == 'Accum' for x in outer.body)

            def unwritten_input(load):
                return (load.input not in written and load.input not in self.shared
                        and load.input in self.ptrs and self.buffers[load.input][1] == 'f32')

            def contiguous_on_serial(load):
                part = expr_names(load.index[-1])
                return part is not None and outer.axis.name in part

            candidates = []
            for inner in outer.body:
                inner_kind = type(inner).__name__
                if inner_kind == 'Loop':
                    # A reduction recomputed by several threads.
                    if not inner.seed or not straight_line(inner.body):
                        continue
                    free = touched_names(inner.body) - defined_names(inner.body) - {inner.axis.name}
                    loads = [x for x in inner.body if type(x).__name__ == 'Load']
                    accums = [x for x in inner.body if type(x).__name__ == 'Accum']
                    if not accums or any(x.dtype is not None and dtype_name(x.dtype) != 'f32'
                                         for x in accums):
                        continue
                    exports = list(dict.fromkeys(x.name for x in accums))
                elif inner_kind == 'Load' and carried:
                    # A global load on a loop-carried chain: staging it lets the
                    # CTA fetch every iteration's operand with full parallelism.
                    free = set(inner.deps())
                    loads = [inner]
                    exports = list(inner.names)
                else:
                    continue
                if not free <= set(axis_extent) | {outer.axis.name}:
                    continue
                if not all(unwritten_input(x) for x in loads):
                    continue
                siblings = [x for x in outer.body if x is not inner]
                if set(exports) & (defined_names(siblings) | elsewhere):
                    continue
                used = [name for name in cta_axes if name in free]
                # Only values that several threads would recompute or reload
                # are worth a shared round trip.
                if math.prod(axis_extent[name] for name in cta_axes if name not in free) < 2:
                    continue
                configs = math.prod(axis_extent[name] for name in used)
                plan = HoistedValue(inner, exports, [(name, axis_extent[name]) for name in used],
                                    configs, extent, f'__hoist{len(self.declarations)}',
                                    any(contiguous_on_serial(x) for x in loads))
                if plan.nbytes > budget or plan.nbytes + self.shared_bytes > 48 * 1024:
                    continue
                budget -= plan.nbytes
                self.hoist_bytes += plan.nbytes
                self.declarations.append(f'.shared .align 16 .b8 {plan.symbol}[{plan.nbytes}];')
                candidates.append(plan)
            if candidates:
                self.hoist_plans[id(outer)] = (outer, candidates, launched)

    def decode_config(self, index, used):
        """Row-major axis values of a hoist configuration index."""
        values = {}
        for position, (name, extent) in enumerate(used):
            stride = math.prod(e for _, e in used[position + 1:])
            r = self.reg('r')
            self.emit(f'div.u32 {r}, {index}, {stride};')
            if position:
                self.emit(f'rem.u32 {r}, {r}, {extent};')
            values[name] = r
        return values

    def emit_hoisted(self, outer, plans, launched):
        """Evaluate each hoisted reduction across the CTA, then barrier."""
        extent = static(outer.axis.extent)
        names = {outer.axis.name} | {name for plan in plans for name, _ in plan.used}
        saved = {name: self.values.get(name) for name in names}
        tid = self.value('tid')
        groups = {}
        for plan in plans:
            if type(plan.statement).__name__ == 'Loop':
                # Reductions over the same coordinates run fused in one item:
                # independent chains overlap and shared operands load once.
                key = ('Loop', tuple(plan.used), plan.serial_fastest,
                       static(plan.statement.axis.extent))
            else:
                key = ('Load', id(plan))
            groups.setdefault(key, []).append(plan)
        for key, members in groups.items():
            lead = members[0]
            if key[0] == 'Loop':
                shape = self.matmul_members(members, outer)
                if shape is not None:
                    self.emit_hoisted_matmul(shape, members, outer, launched)
                    continue
            work = lead.configs * extent
            w = self.reg('r')
            self.emit(f'mov.u32 {w}, {tid};')
            # Batches keep many independent loads in flight: one per staged
            # load item, a whole fused loop's worth per reduction item.
            batches = STRIDED_UNROLL if key[0] == 'Load' else (HOIST_REDUCTION_BATCH,)
            for batch in (*batches, 1):
                if work + (batch - 1) * launched >= (1 << 31):
                    continue
                begin, done = self.label('hoist'), self.label('hoist_done')
                self.mark(begin)
                last, p = self.reg('r'), self.reg('p')
                self.emit(f'add.u32 {last}, {w}, {(batch - 1) * launched};')
                self.emit(f'setp.ge.u32 {p}, {last}, {work};')
                self.emit(f'@{p} bra {done};')
                coords = []
                for k in range(batch):
                    item = self.reg('r')
                    self.emit(f'add.u32 {item}, {w}, {k * launched};')
                    coords.append(self.hoist_item_coords(lead, outer, item))
                items = self.evaluate_hoist_items(members, coords)
                # Every item's operands are requested before any is stored.
                for config, serial, results in items:
                    for plan, values in results:
                        for k, value in enumerate(values):
                            word = self.reg('r')
                            self.emit(f'mad.lo.u32 {word}, {config}, {plan.row_words}, {serial};')
                            self.emit(f'add.u32 {word}, {word}, {k * plan.configs * plan.row_words};')
                            addr = self.reg('rd')
                            self.emit(f'mul.wide.u32 {addr}, {word}, 4;')
                            base = self.reg('rd')
                            self.emit(f'mov.u64 {base}, {plan.symbol};')
                            self.emit(f'add.u64 {addr}, {base}, {addr};')
                            self.emit(f'st.shared.f32 [{addr}], {value};')
                self.emit(f'add.u32 {w}, {w}, {batch * launched};')
                self.emit(f'bra {begin};')
                self.mark(done)
        for name, value in saved.items():
            if value is None:
                self.values.pop(name, None)
            else:
                self.values[name] = value
            self.rebind(name)
        self.emit('bar.sync 0;')
        for plan in plans:
            # This thread's own configuration index, from its own axis values.
            config = self.reg('r')
            self.emit(f'mov.u32 {config}, 0;')
            for name, extent_ in plan.used:
                self.emit(f'mad.lo.u32 {config}, {config}, {extent_}, {self.value(name)};')
            rows = []
            for k in range(len(plan.exports)):
                word = self.reg('r')
                self.emit(f'mad.lo.u32 {word}, {config}, {plan.row_words}, '
                          f'{k * plan.configs * plan.row_words};')
                row = self.reg('rd')
                self.emit(f'mul.wide.u32 {row}, {word}, 4;')
                base = self.reg('rd')
                self.emit(f'mov.u64 {base}, {plan.symbol};')
                self.emit(f'add.u64 {row}, {base}, {row};')
                rows.append(row)
            self.hoisted[id(plan.statement)] = (plan, outer.axis.name, rows)

    def matmul_members(self, members, outer):
        """Recognize hoisted reductions of the form acc += A[c, k] * B[k, s].

        Every member must be a Load/Load/multiply/add-fold loop whose ``B``
        operand (indexed by the serial axis, not by thread axes) is one
        shared expression. Returns (b_load, [(a_load, a_first)]) or None.
        """
        serial = outer.axis.name
        used = {name for name, _ in members[0].used}
        b_key, operands = None, []
        for plan in members:
            body = list(plan.statement.body)
            kinds = sorted(type(s).__name__ for s in body)
            if kinds != ['Accum', 'Assign', 'Load', 'Load']:
                return None
            loads = [s for s in body if type(s).__name__ == 'Load']
            product = next(s for s in body if type(s).__name__ == 'Assign')
            fold = next(s for s in body if type(s).__name__ == 'Accum')
            if (any(len(s.names) != 1 or self.buffers[s.input][1] != 'f32' for s in loads)
                    or product.op.name != 'multiply' or reduce_canon(fold.op.name) != 'add'
                    or fold.value != product.name or fold.base not in (None, fold.name)
                    or any(x.dtype is not None and dtype_name(x.dtype) != 'f32'
                           for x in (product, fold))
                    or sorted(product.args) != sorted(s.names[0] for s in loads)
                    or body.index(product) < max(body.index(s) for s in loads)
                    or body.index(fold) < body.index(product)):
                return None
            free = [set().union(*(expr_names(c) or {None} for c in s.index)) for s in loads]
            b_side = [serial in f and not f & used and None not in f for f in free]
            a_side = [serial not in f and None not in f for f in free]
            if sorted(b_side) != [False, True] or not a_side[b_side.index(False)]:
                return None
            b, a = loads[b_side.index(True)], loads[b_side.index(False)]
            key = (b.input, tuple(structural_key(c) for c in b.index))
            if None in key[1] or (b_key is not None and key != b_key):
                return None
            b_key = key
            operands.append((a, product.args[0] == a.names[0]))
        return b, operands

    def emit_hoisted_matmul(self, shape, members, outer, launched):
        """Register-tiled evaluation of hoisted A·B reductions.

        Each work item is one thread-axis configuration ``c`` and ``tn``
        consecutive serial columns; the thread keeps every member's
        accumulators for those columns, so each B load feeds all members and
        each A load feeds ``tn`` columns. Every output still folds
        ``mul(a, b)`` over k in order from the same identity, so results
        match the unhoisted loop exactly. With a contiguous, 16-byte aligned
        B row a single 128-bit load fetches the ``tn`` = 4 columns.
        """
        b, operands = shape
        lead = members[0]
        extent = lead.extent
        k_loops = [plan.statement for plan in members]
        k_extent = static(k_loops[0].axis.extent)
        staged = 4 * (len(members) * lead.configs * k_extent + k_extent * extent)
        # Staging pays off only when each B element feeds several output rows.
        if (extent % 4 == 0 and k_extent % 4 == 0 and lead.configs * len(members) >= 2
                and self.shared_bytes + self.hoist_bytes + staged <= 48 * 1024):
            self.hoist_bytes += staged
            self.emit_staged_matmul(shape, members, outer, launched)
            return
        last = b.index[-1]
        shape_b = self.buffers[b.input][0]
        vector = (extent % 4 == 0 and type(last).__name__ == 'Var' and last.name == outer.axis.name
                  and all(math.prod(shape_b[i + 1:]) % 4 == 0 for i in range(len(shape_b) - 1)))
        saved = {name: self.values.get(name) for name in
                 {outer.axis.name, *(name for name, _ in lead.used), *(l.axis.name for l in k_loops)}}
        done = self.label('mm_done')
        if vector:
            scalar = self.label('mm_scalar')
            low, p = self.reg('rd'), self.reg('p')
            self.emit(f'and.b64 {low}, {self.ptrs[b.input]}, 15;')
            self.emit(f'setp.ne.u64 {p}, {low}, 0;')
            self.emit(f'@{p} bra {scalar};')
            self.matmul_work_loop(shape, members, outer, launched, 4)
            self.emit(f'bra {done};')
            self.mark(scalar)
        self.matmul_work_loop(shape, members, outer, launched, 1)
        self.mark(done)
        for name, value in saved.items():
            if value is None:
                self.values.pop(name, None)
            else:
                self.values[name] = value
            self.rebind(name)

    def stage_operands(self, copies, symbol, launched):
        """Copy several loads' CTA footprints to shared memory.

        When every footprint divides evenly among the threads and the total
        per thread is modest, all loads are issued straight-line before any
        store, so the whole copy costs about one memory latency.
        """
        trips = [math.prod(e for _, e in dims) / launched for _, dims, _ in copies]
        if all(x == int(x) for x in trips) and sum(trips) <= STAGE_STRAIGHT_LINE:
            pending = []
            for (load, dims, offset), count in zip(copies, trips):
                for k in range(int(count)):
                    item = self.reg('r')
                    self.emit(f'add.u32 {item}, {self.value("tid")}, {k * launched};')
                    self.bind_coords(self.decode_config(item, dims))
                    addr, space = self.address(load.input, load.index)
                    value = self.reg('f')
                    self.emit(f'ld.{space}.f32 {value}, [{addr}];')
                    pending.append((item, value, offset))
            base = self.reg('rd')
            self.emit(f'mov.u64 {base}, {symbol};')
            for item, value, offset in pending:
                sa = self.reg('rd')
                self.emit(f'mul.wide.u32 {sa}, {item}, 4;')
                self.emit(f'add.u64 {sa}, {base}, {sa};')
                self.emit(f'st.shared.f32 [{sa}+{offset}], {value};')
            return
        for load, dims, offset in copies:
            self.stage_operand(load, dims, symbol, offset, launched)

    def stage_operand(self, load, dims, symbol, offset, launched):
        """Copy one load's CTA footprint to shared memory, row-major over
        ``dims`` ((axis name, extent), last fastest), many loads in flight."""
        work = math.prod(extent for _, extent in dims)
        w = self.reg('r')
        self.emit(f'mov.u32 {w}, {self.value("tid")};')
        for batch in (*STRIDED_UNROLL, 1):
            if work + (batch - 1) * launched >= (1 << 31):
                continue
            begin, done = self.label('stage'), self.label('stage_done')
            self.mark(begin)
            last, p = self.reg('r'), self.reg('p')
            self.emit(f'add.u32 {last}, {w}, {(batch - 1) * launched};')
            self.emit(f'setp.ge.u32 {p}, {last}, {work};')
            self.emit(f'@{p} bra {done};')
            pending = []
            for k in range(batch):
                item = self.reg('r')
                self.emit(f'add.u32 {item}, {w}, {k * launched};')
                self.bind_coords(self.decode_config(item, dims))
                addr, space = self.address(load.input, load.index)
                value = self.reg('f')
                self.emit(f'ld.{space}.f32 {value}, [{addr}];')
                pending.append((item, value))
            for item, value in pending:
                sa = self.reg('rd')
                self.emit(f'mul.wide.u32 {sa}, {item}, 4;')
                base = self.reg('rd')
                self.emit(f'mov.u64 {base}, {symbol};')
                self.emit(f'add.u64 {sa}, {base}, {sa};')
                self.emit(f'st.shared.f32 [{sa}+{offset}], {value};')
            self.emit(f'add.u32 {w}, {w}, {batch * launched};')
            self.emit(f'bra {begin};')
            self.mark(done)

    def emit_staged_matmul(self, shape, members, outer, launched):
        """Shared-memory tiled evaluation of hoisted A·B reductions.

        The A rows of every member and the B tile are first copied to shared
        memory with batched loads (the global latency is paid a few times,
        not once per k step). Each work item then owns one configuration and
        four consecutive columns and walks k four at a time with 128-bit
        shared loads: per four k steps, one A vector per member and four B
        vectors feed 16 x members FMAs. Every output still folds mul(a, b)
        over k in order from the same identity, so results are unchanged.
        """
        b, operands = shape
        lead = members[0]
        extent = lead.extent
        k_axis = members[0].statement.axis.name
        k_extent = static(members[0].statement.axis.extent)
        a_words = lead.configs * k_extent
        symbol = f'__stage{len(self.declarations)}'
        nbytes = 4 * (len(members) * a_words + k_extent * extent)
        self.declarations.append(f'.shared .align 16 .b8 {symbol}[{nbytes}];')
        b_offset = 4 * len(members) * a_words
        dims_a = [*lead.used, (k_axis, k_extent)]
        copies = [(a, dims_a, 4 * m * a_words) for m, (a, _) in enumerate(operands)]
        copies.append((b, [(k_axis, k_extent), (outer.axis.name, extent)], b_offset))
        self.stage_operands(copies, symbol, launched)
        self.emit('bar.sync 0;')
        blocks = extent // 4
        work = lead.configs * blocks
        w = self.reg('r')
        self.emit(f'mov.u32 {w}, {self.value("tid")};')
        begin, done = self.label('mm'), self.label('mm_items_done')
        self.mark(begin)
        p = self.reg('p')
        self.emit(f'setp.ge.u32 {p}, {w}, {work};')
        self.emit(f'@{p} bra {done};')
        block, config, serial = self.reg('r'), self.reg('r'), self.reg('r')
        self.emit(f'rem.u32 {block}, {w}, {blocks};')
        self.emit(f'div.u32 {config}, {w}, {blocks};')
        self.emit(f'shl.b32 {serial}, {block}, 2;')
        base = self.reg('rd')
        self.emit(f'mov.u64 {base}, {symbol};')
        a_rows = []
        for m in range(len(members)):
            word = self.reg('r')
            self.emit(f'mad.lo.u32 {word}, {config}, {k_extent}, {m * a_words};')
            row = self.reg('rd')
            self.emit(f'mul.wide.u32 {row}, {word}, 4;')
            self.emit(f'add.u64 {row}, {base}, {row};')
            a_rows.append(row)
        b_col = self.reg('rd')
        self.emit(f'mul.wide.u32 {b_col}, {serial}, 4;')
        self.emit(f'add.u64 {b_col}, {base}, {b_col};')
        self.emit(f'add.u64 {b_col}, {b_col}, {b_offset};')
        accs = [[self.reg('f') for _ in range(4)] for _ in members]
        for row in accs:
            for acc in row:
                self.emit(f'mov.f32 {acc}, {literal(0.0, "f32")};')
        k = self.reg('r')
        self.emit(f'mov.u32 {k}, 0;')
        k_begin, k_done = self.label('mm_k'), self.label('mm_k_done')
        self.mark(k_begin)
        q = self.reg('p')
        self.emit(f'setp.ge.u32 {q}, {k}, {k_extent};')
        self.emit(f'@{q} bra {k_done};')
        a_off = self.reg('rd')
        self.emit(f'mul.wide.u32 {a_off}, {k}, 4;')
        b_off = self.reg('rd')
        self.emit(f'mul.wide.u32 {b_off}, {k}, {4 * extent};')
        b_ptr = self.reg('rd')
        self.emit(f'add.u64 {b_ptr}, {b_col}, {b_off};')
        bvecs = []
        for step in range(4):
            vec = [self.reg('f') for _ in range(4)]
            self.emit(f'ld.shared.v4.f32 {{{", ".join(vec)}}}, [{b_ptr}+{4 * step * extent}];')
            bvecs.append(vec)
        for (a, a_first), row, a_row in zip(operands, accs, a_rows):
            a_ptr = self.reg('rd')
            self.emit(f'add.u64 {a_ptr}, {a_row}, {a_off};')
            avec = [self.reg('f') for _ in range(4)]
            self.emit(f'ld.shared.v4.f32 {{{", ".join(avec)}}}, [{a_ptr}];')
            for step in range(4):
                for bv, acc in zip(bvecs[step], row):
                    product = self.reg('f')
                    x, y = (avec[step], bv) if a_first else (bv, avec[step])
                    self.emit(f'mul.f32 {product}, {x}, {y};')
                    self.emit(f'add.f32 {acc}, {acc}, {product};')
        self.emit(f'add.u32 {k}, {k}, 4;')
        self.emit(f'bra {k_begin};')
        self.mark(k_done)
        for plan, row in zip(members, accs):
            word = self.reg('r')
            self.emit(f'mad.lo.u32 {word}, {config}, {plan.row_words}, {serial};')
            sa = self.reg('rd')
            self.emit(f'mul.wide.u32 {sa}, {word}, 4;')
            sbase = self.reg('rd')
            self.emit(f'mov.u64 {sbase}, {plan.symbol};')
            self.emit(f'add.u64 {sa}, {sbase}, {sa};')
            for j, acc in enumerate(row):
                self.emit(f'st.shared.f32 [{sa}+{4 * j}], {acc};')
        self.emit(f'add.u32 {w}, {w}, {launched};')
        self.emit(f'bra {begin};')
        self.mark(done)

    def matmul_work_loop(self, shape, members, outer, launched, tn):
        b, operands = shape
        lead = members[0]
        blocks = lead.extent // tn
        work = lead.configs * blocks
        w = self.reg('r')
        self.emit(f'mov.u32 {w}, {self.value("tid")};')
        begin, done = self.label('mm'), self.label('mm_items_done')
        self.mark(begin)
        p = self.reg('p')
        self.emit(f'setp.ge.u32 {p}, {w}, {work};')
        self.emit(f'@{p} bra {done};')
        # Column blocks vary fastest across threads so B rows coalesce.
        block, config, serial = self.reg('r'), self.reg('r'), self.reg('r')
        self.emit(f'rem.u32 {block}, {w}, {blocks};')
        self.emit(f'div.u32 {config}, {w}, {blocks};')
        self.emit(f'mul.lo.u32 {serial}, {block}, {tn};')
        self.bind_coords({outer.axis.name: serial, **self.decode_config(config, lead.used)})
        accs = []
        for _ in members:
            row = []
            for _ in range(tn):
                acc = self.reg('f')
                self.emit(f'mov.f32 {acc}, {literal(0.0, "f32")};')
                row.append(acc)
            accs.append(row)
        k = self.reg('r')
        self.emit(f'mov.s32 {k}, 0;')
        self.bind_coords({plan.statement.axis.name: k for plan in members})
        self.loop_invariants.append({})
        self.loop_expressions.append({})
        k_begin, k_done = self.label('mm_k'), self.label('mm_k_done')
        self.mark(k_begin)
        q = self.reg('p')
        self.emit(f'setp.ge.s32 {q}, {k}, {static(members[0].statement.axis.extent)};')
        self.emit(f'@{q} bra {k_done};')
        addr, space = self.address(b.input, b.index)
        bs = [self.reg('f') for _ in range(tn)]
        if tn == 4:
            self.emit(f'ld.{space}.v4.f32 {{{", ".join(bs)}}}, [{addr}];')
        else:
            self.emit(f'ld.{space}.f32 {bs[0]}, [{addr}];')
        for (a, a_first), row in zip(operands, accs):
            a_addr, a_space = self.address(a.input, a.index)
            av = self.reg('f')
            self.emit(f'ld.{a_space}.f32 {av}, [{a_addr}];')
            for bv, acc in zip(bs, row):
                product = self.reg('f')
                x, y = (av, bv) if a_first else (bv, av)
                self.emit(f'mul.f32 {product}, {x}, {y};')
                self.emit(f'add.f32 {acc}, {acc}, {product};')
        self.emit(f'add.s32 {k}, {k}, 1;')
        self.bind_coords({plan.statement.axis.name: k for plan in members})
        self.emit(f'bra {k_begin};')
        self.mark(k_done)
        self.loop_invariants.pop()
        self.loop_expressions.pop()
        for plan, row in zip(members, accs):
            for j, acc in enumerate(row):
                word = self.reg('r')
                self.emit(f'mad.lo.u32 {word}, {config}, {plan.row_words}, {serial};')
                if j:
                    self.emit(f'add.u32 {word}, {word}, {j};')
                sa = self.reg('rd')
                self.emit(f'mul.wide.u32 {sa}, {word}, 4;')
                base = self.reg('rd')
                self.emit(f'mov.u64 {base}, {plan.symbol};')
                self.emit(f'add.u64 {sa}, {base}, {sa};')
                self.emit(f'st.shared.f32 [{sa}], {acc};')
        self.emit(f'add.u32 {w}, {w}, {launched};')
        self.emit(f'bra {begin};')
        self.mark(done)

    def hoist_item_coords(self, lead, outer, item):
        """Decode a work item into (config, serial, axis bindings)."""
        serial, config = self.reg('r'), self.reg('r')
        if lead.serial_fastest:
            self.emit(f'rem.u32 {serial}, {item}, {lead.extent};')
            self.emit(f'div.u32 {config}, {item}, {lead.extent};')
        else:
            self.emit(f'rem.u32 {config}, {item}, {lead.configs};')
            self.emit(f'div.u32 {serial}, {item}, {lead.configs};')
        return config, serial, {outer.axis.name: serial, **self.decode_config(config, lead.used)}

    def bind_coords(self, bindings):
        for name, r in bindings.items():
            self.values[name] = (r, 'i32')
            self.rebind(name)

    def evaluate_hoist_items(self, members, coords):
        """Evaluate a batch of work items; returns per item its exports."""
        if type(members[0].statement).__name__ == 'Loop':
            exports = self.fused_loop([plan.statement for plan in members],
                                      [bindings for _, _, bindings in coords])
            return [(config, serial, [(plan, [values[name] for name in plan.exports])
                                      for plan in members])
                    for (config, serial, _), values in zip(coords, exports)]
        results = []
        for config, serial, bindings in coords:
            self.bind_coords(bindings)
            self.statements([members[0].statement])
            results.append((config, serial, [(plan, [self.value(name) for name in plan.exports])
                                             for plan in members]))
        return results

    def fused_loop(self, loops, items):
        """Independent straight-line Loops of one extent sharing one induction,
        evaluated for several work items (coordinate bindings) per trip.

        Each item keeps its own fold registers, and each body still runs its
        iterations in order, so every fold is unchanged; bodies and items
        only interleave, which hoisting has already proved independent (each
        reads Tile axes and its own bindings). Returns each item's final fold
        registers by name.
        """
        accums = [s.name for loop in loops for s in loop.body if type(s).__name__ == 'Accum']
        states = []
        for bindings in items:
            self.bind_coords(bindings)
            for loop in loops:
                if loop.seed:
                    self.init_accums(loop.body)
            states.append({name: (self.values[name], self.mutable.get(name)) for name in accums})
        axes = list(dict.fromkeys(loop.axis.name for loop in loops))
        r = self.reg('r')
        self.emit(f'mov.s32 {r}, 0;')
        for axis in axes:
            self.values[axis] = (r, 'i32')
            self.rebind(axis)
        if len(items) == 1:
            body = [statement for loop in loops for statement in loop.body]
            self.hoist_invariants(body, defined_names(body) | set(axes))
        else:
            # Preheader addresses would be valid for one item's bindings only.
            self.loop_invariants.append({})
            self.loop_expressions.append({})
        begin, done = self.label('loop'), self.label('loop_done')
        self.mark(begin)
        p = self.reg('p')
        self.emit(f'setp.ge.s32 {p}, {r}, {static(loops[0].axis.extent)};')
        self.emit(f'@{p} bra {done};')
        for bindings, state in zip(items, states):
            if len(items) > 1:
                self.bind_coords(bindings)
            for name, (value, mutable) in state.items():
                self.values[name] = value
                self.mutable[name] = mutable
                self.rebind(name)
            for loop in loops:
                self.statements(loop.body)
            for name in accums:
                state[name] = (self.values[name], self.mutable.get(name))
        self.emit(f'add.s32 {r}, {r}, 1;')
        for axis in axes:
            self.rebind(axis)
        self.emit(f'bra {begin};')
        self.mark(done)
        self.loop_invariants.pop()
        self.loop_expressions.pop()
        return [{name: value[0] for name, (value, _) in state.items()} for state in states]

    def load_hoisted(self, statement):
        plan, serial, rows = self.hoisted[id(statement)]
        offset = self.reg('rd')
        self.emit(f'mul.wide.u32 {offset}, {self.value(serial)}, 4;')
        for name, row in zip(plan.exports, rows):
            addr, r = self.reg('rd'), self.reg('f')
            self.emit(f'add.u64 {addr}, {row}, {offset};')
            self.emit(f'ld.shared.f32 {r}, [{addr}];')
            self.values[name] = (r, 'f32')
            self.rebind(name)

    def statements(self, statements):
        pending = []
        for s in statements:
            for name, key in pending:
                self.vn_registers[key] = self.values[name]
            pending = self.vn_record.get(id(s), [])
            if id(s) in self.hoist_plans:
                self.emit_hoisted(*self.hoist_plans.pop(id(s)))
            if id(s) in self.hoisted:
                self.load_hoisted(s)
                continue
            if id(s) in self.subst:
                # Rewritten by the two-pass softmax: bind the known value.
                for name, reg in self.subst[id(s)]:
                    self.values[name] = (reg, 'f32')
                    self.rebind(name)
                continue
            if id(s) in self.vn_skip:
                # Value-numbered redundant: reuse the earlier register.
                for name, key in self.vn_skip[id(s)]:
                    self.values[name] = self.vn_registers[key]
                    self.rebind(name)
                continue
            kind = type(s).__name__
            if kind == 'Tile':
                if s.aux_threads or s.raster_orient not in ('m', 'n'):
                    raise Unsupported('Auxiliary tile scheduling unsupported')
                # Raster metadata only permutes whole CTAs. A row-major CTA
                # ordering still visits the same coordinate product exactly once.
                if s.raster_group is not None and (not isinstance(s.raster_group, int) or s.raster_group <= 0):
                    raise Unsupported('Invalid raster group')
                # Window metadata records the parent slice an axis was carved
                # from.  The axis still walks its own local [0, extent) range;
                # base/bound/partition do not change the runtime cell decode.
                extents = [static(a.extent) for a in s.axes]
                launched = math.prod(self.meta['block'])
                cells = s.block_threads if s.block_threads is not None else launched
                if cells != launched:
                    raise Unsupported('Tile block_threads differs from prescribed launch')
                if s.raster_axes:
                    # Prove raster coordinates are block-uniform: all within-CTA
                    # coordinate factors form a suffix of exactly block_threads.
                    cuts = [i for i in range(len(extents) + 1)
                            if math.prod(extents[i:]) == launched]
                    if not cuts or any(name not in [axis.name for axis in s.axes[:max(cuts)]]
                                       for name in s.raster_axes):
                        raise Unsupported('Raster axes are not proven block-uniform')
                n = math.prod(extents)
                if n <= 0 or n >= (1 << 31):
                    raise Unsupported('Tile size outside supported signed32 range')
                blocks = math.ceil(n / launched)
                if math.prod(self.meta['grid']) < blocks:
                    raise Unsupported('Prescribed grid does not cover the Tile')
                # A partial final CTA must keep every launched thread alive through
                # its barriers. Threads outside the Tile run the body under a
                # predicate and contribute the reduction identity instead of
                # exiting early (which would deadlock a CTA-wide barrier).
                collective = has_collective(s.body)
                # The prescribed grid can contain a partial final CTA or extra
                # whole CTAs. Both are outside the Tile and must be excluded,
                # even when the collective body needs every launched thread
                # alive to reach its barriers.
                inactive = launched * math.prod(self.meta['grid']) > n
                p = self.reg('p')
                self.emit(f'setp.ge.u32 {p}, {self.value("_gid")}, {n};')
                if not collective:
                    self.emit(f'@{p} bra kernel_done;')
                for i, axis in enumerate(s.axes):
                    stride = math.prod(extents[i+1:])
                    r = self.reg('r')
                    if stride & (stride - 1) == 0:
                        self.emit(f'shr.u32 {r}, {self.value("_gid")}, {stride.bit_length() - 1};')
                    else:
                        self.emit(f'div.u32 {r}, {self.value("_gid")}, {stride};')
                    extent = extents[i]
                    if i and extent & (extent - 1) == 0:
                        self.emit(f'and.b32 {r}, {r}, {extent - 1};')
                    elif i:
                        self.emit(f'rem.u32 {r}, {r}, {extent};')
                    self.values[axis.name] = (r, 'i32')
                if inactive and collective:
                    # Inactive threads still execute the body to reach barriers.
                    # Their addresses are masked at every memory access; the
                    # axis registers themselves are consumed only within it.
                    self.preinitialize_trees(s.body)
                    active = self.reg('p')
                    self.emit(f'setp.lt.u32 {active}, {self.value("_gid")}, {n};')
                    self.inactive.append(active)
                self.tile_extents = {axis.name: extent for axis, extent in zip(s.axes, extents)}
                if not inactive and not self.inactive and not self.dynamic_shared:
                    # Hoisting adds a CTA barrier, so every launched thread
                    # must be a live Tile cell.
                    self.plan_hoists(s, extents, launched)
                self.statements(s.body)
                if inactive and collective:
                    self.inactive.pop()
            elif kind == 'Load':
                dtype = dtype_name(s.dtype) if s.dtype is not None else dtype_name(self.buffers[s.input][1])
                phys = dtype_info(dtype)[1]
                for i, name in enumerate(s.names):
                    addr, space = self.address(s.input, s.index, i)
                    reg = self.reg(reg_suffix(dtype))
                    if self.inactive:
                        # Inactive lanes must not issue the memory operation, but
                        # the register still needs a deterministic definition.
                        self.emit(f'mov.{dtype} {reg}, {literal(0, dtype)};')
                        self.emit(f'@{self.inactive[-1]} ld.{space}.{phys} {reg}, [{addr}];')
                    else:
                        self.emit(f'ld.{space}.{phys} {reg}, [{addr}];')
                    self.values[name] = (reg, dtype)
            elif kind == 'Write':
                if s.swizzle not in ('NONE', None):
                    raise Unsupported('Swizzled writes unsupported')
                output_dtype = dtype_name(self.buffers[s.output][1])
                value_dtype = dtype_name(s.value_dtype) if s.value_dtype is not None else output_dtype
                phys = dtype_info(output_dtype)[1]
                for i, name in enumerate(s.values):
                    addr, space = self.address(s.output, s.index, i)
                    value = self.value(name)
                    value_type = self.values[name][1]
                    # IR stores convert at the boundary. Selecting the store
                    # width from the output (not the SSA value) preserves both
                    # the stride and the destination representation.
                    value = convert_register(self, value, value_type, output_dtype)
                    active = self.inactive[-1] if self.inactive else None
                    guarded = False
                    if active is not None:
                        # A shared partial for an inactive cell does not exist,
                        # so inactive threads must not publish one. The final
                        # global result is usually written by a single active
                        # thread and is likewise suppressed for inactive lanes.
                        guarded = True
                    if s.atomic:
                        if space != 'global':
                            raise Unsupported('Shared atomic writes unsupported')
                        if output_dtype not in ('f32', 'f64'):
                            raise Unsupported(f'Only floating atomics supported: {output_dtype}')
                        pred = f'@{active} ' if guarded else ''
                        self.emit(f'{pred}red.global.add.{phys} [{addr}], {value};')
                    else:
                        pred = f'@{active} ' if guarded else ''
                        self.emit(f'{pred}st.{space}.{phys} [{addr}], {value};')
            elif kind == 'Assign':
                self.address_cache.clear()
                self.expr_cache.clear()
                start = len(self.lines)
                self.assign(s.name, s.op, s.args, s.dtype)
                if self.inactive and s.name in self.mutable:
                    for i in range(start, len(self.lines)):
                        instruction = self.lines[i].strip()
                        self.lines[i] = f'    @{self.inactive[-1]} {instruction}'
                self.rebind(s.name)
            elif kind == 'Let':
                self.values[s.name] = self.expr(s.value)
                # A repeated Let name is a rebind, not an immutable alias.
                # Clear versioned address/CSE entries before later expressions
                # can observe the new binding.
                self.rebind(s.name)
            elif kind == 'Select':
                self.address_cache.clear()
                self.expr_cache.clear()
                self.select(s)
            elif kind == 'Init':
                self.address_cache.clear()
                dtype = dtype_name(s.dtype)
                if dtype not in FLOAT_DTYPES:
                    raise Unsupported(f'Only floating carried state supported: {dtype}')
                r = self.reg(reg_suffix(dtype))
                self.emit(f'mov.{dtype} {r}, {literal(s.identity, dtype)};')
                self.values[s.name] = (r, dtype)
                self.mutable[s.name] = r
                self.rebind(s.name)
            elif kind == 'Accum':
                self.address_cache.clear()
                old = self.value(s.name)
                left = s.base if s.base is not None else s.name
                if self.inactive:
                    # A masked lane keeps its seeded identity state. Predicating
                    # the whole fold is op-agnostic: it is correct for maximum /
                    # minimum / product as well as sum, and it also preserves
                    # uncovered online-state updates that read the accumulator.
                    start = len(self.lines)
                    self.assign(s.name, s.op, (left, s.value), s.dtype)
                    for i in range(start, len(self.lines)):
                        instruction = self.lines[i].strip()
                        self.lines[i] = f'    @{self.inactive[-1]} {instruction}'
                else:
                    self.assign(s.name, s.op, (left, s.value), s.dtype)
                if self.value(s.name) != old:
                    self.emit(f'mov.{self.values[s.name][1]} {old}, {self.value(s.name)};')
                self.values[s.name] = (old, self.values[s.name][1])
                self.mutable[s.name] = old
                self.rebind(s.name)
            elif kind in ('StridedLoop', 'Loop'):
                # Seeds apply to accumulators owned immediately by this loop.
                # Nested reductions seed themselves when their loop is entered.
                if s.seed:
                    self.init_accums(s.body)
                    body = list(s.body)
                    fixed, removed = {}, set()
                    online = (online_max_carriers(body, set(self.hoisted))
                              if kind == 'Loop' and not self.inactive else [])
                    for fold, x, gn, dg, alpha, _ in online:
                        removed.add(id(fold))
                        fixed[id(gn)] = lambda vn, gn=gn, x=x: [(gn.name, ('maxof', vn(x)))]
                        fixed[id(dg)] = lambda vn, dg=dg: [(dg.name, ('const', 0.0))]
                        if alpha is not None:
                            fixed[id(alpha)] = lambda vn, alpha=alpha: [(alpha.name, ('const', 1.0))]
                    skip, record, carried = loop_value_numbering(
                        [x for x in body if id(x) not in removed], set(self.hoisted), fixed)
                    self.vn_skip.update(skip)
                    self.vn_record.update(record)
                    for name, rep in carried.items():
                        self.values[name] = self.values[rep]
                        self.mutable[name] = self.mutable[rep]
                    if online:
                        self.two_pass_softmax(s, online)
                r = self.reg('r')
                start, step, end = ('0', '1', str(static(s.axis.extent)))
                if kind == 'StridedLoop':
                    start, _ = self.expr(s.start)
                    step, _ = self.expr(s.step)
                    if s.end is not None:
                        end, _ = self.expr(s.end)
                self.emit(f'mov.s32 {r}, {start};')
                self.values[s.axis.name] = (r, 'i32')
                self.rebind(s.axis.name)
                vector = self.vector_load_plan(s) if kind == 'Loop' else None
                if vector is not None:
                    self.vector_loops += 1
                    # Strength reduction: each group's address at axis = 0
                    # (the induction register's value here) is loop-invariant.
                    # Computed before hoisting so the groups share invariant
                    # subexpressions through the expression cache.
                    bases = {}
                    for key, (stmt, _, step) in vector[1].items():
                        addr, space = self.address(stmt.input, stmt.index)
                        base = self.reg('rd')
                        self.emit(f'mov.u64 {base}, {addr};')
                        bases[key] = (base, space, step)
                    vector = (vector[0], bases)
                self.hoist_invariants(s.body, defined_names(s.body) | {s.axis.name})
                begin, done = self.label('loop'), self.label('loop_done')
                if kind == 'StridedLoop':
                    self.strided_main_loop(s, r, step, end)
                self.mark(begin)
                p = self.reg('p')
                self.emit(f'setp.ge.s32 {p}, {r}, {end};')
                self.emit(f'@{p} bra {done};')
                if vector is None:
                    self.statements(s.body)
                    self.emit(f'add.s32 {r}, {r}, {step};')
                else:
                    self.vector_unrolled_body(s, r, vector)
                    self.emit(f'add.s32 {r}, {r}, {VECTOR_UNROLL};')
                self.rebind(s.axis.name)
                self.emit(f'bra {begin};')
                self.mark(done)
                self.loop_invariants.pop()
                self.loop_expressions.pop()
            elif kind == 'WarpShuffle':
                if s.length > 32 or s.length <= 0 or s.length & (s.length-1):
                    raise Unsupported('Shuffle width must be a power of two <=32')
                dtype = dtype_name(s.dtype) if s.dtype is not None else 'f32'
                if dtype not in ('f32', 'i32', 'u32'):
                    raise Unsupported(f'Shuffle dtype {dtype} needs a 64-bit shuffle')
                mask = self.reg('r')
                self.emit(f'activemask.b32 {mask};')
                offset = s.length // 2
                while offset:
                    prior = {name: self.values[name] for name in s.state}
                    for a, b in zip(s.state, s.state_b, strict=True):
                        r = self.reg(reg_suffix(dtype))
                        self.emit(f'shfl.sync.bfly.b32 {r}, {self.value(a)}, {offset}, 31, {mask};')
                        self.values[b] = (r, dtype)
                    # The shuffle source is undefined when this lane's XOR
                    # partner is absent from activemask. Compute the merge
                    # unconditionally, then select the lane's prior state back
                    # when the peer is absent. The per-lane test is exact for a
                    # partial warp and for sparse masks, unlike assuming that
                    # active lanes always form a prefix. Selecting after the
                    # merge ensures the carried register is defined on both
                    # paths for the next shuffle step.
                    partner = self.reg('r')
                    bit = self.reg('r')
                    peer_active = self.reg('p')
                    self.emit(f'xor.b32 {partner}, {self.value("lane")}, {offset};')
                    self.emit(f'shr.u32 {bit}, {mask}, {partner};')
                    self.emit(f'and.b32 {bit}, {bit}, 1;')
                    self.emit(f'setp.ne.u32 {peer_active}, {bit}, 0;')
                    self.statements(s.combine_states)
                    for name in s.state:
                        merged, merged_type = self.values[name]
                        before, before_type = prior[name]
                        if merged_type != before_type:
                            raise Unsupported('Shuffle combine changed the state dtype')
                        kept = self.reg(reg_suffix(merged_type))
                        self.emit(f'selp.{select_type(merged_type)} {kept}, {merged}, '
                                  f'{before}, {peer_active};')
                        self.values[name] = (kept, merged_type)
                    offset //= 2
            elif kind == 'Smem':
                # Layout, buffer registration, and the PTX declaration are
                # emitted once by the initialization scan. Declaration order
                # does not depend on runtime statement order.
                pass
            elif kind == 'CpAsyncCopy':
                if s.swizzle not in ('NONE', None) or s.nbytes not in (4, 8, 16):
                    raise Unsupported('Only nonswizzled 4/8/16-byte copy groups supported')
                dtype = dtype_name(self.buffers[s.smem][1])
                nbytes = dtype_info(dtype)[0]
                if s.nbytes % nbytes:
                    raise Unsupported(f'Copy width {s.nbytes} is not a whole number of {dtype} elements')
                from emmy.compiler.ir.expr import BinaryExpr
                destination = s.smem_index
                if s.lane_index is not None:
                    destination = tuple(BinaryExpr('+', a, b) for a, b in zip(destination, s.lane_index, strict=True))
                width = s.nbytes // nbytes
                src, source_space = self.address(s.src, s.src_index)
                dst, destination_space = self.address(s.smem, destination)
                if source_space != 'global' or destination_space != 'shared':
                    raise Unsupported('Copy must transport global to shared')
                active = self.inactive[-1] if self.inactive else None
                destination_residue = flat_residue(self.buffers[s.smem][0], destination, width, self.var_residue)
                source_residue = flat_residue(self.buffers[s.src][0], s.src_index, width, self.var_residue)
                statically_aligned = (width == 4 and dtype == 'f32'
                                      and self.shared_statements[s.smem].align >= 16
                                      and self.shared_offsets[s.smem] % 16 == 0
                                      and destination_residue == 0 and source_residue == 0)
                if statically_aligned:
                    # The CTA launch can carry extra threads and the hidden
                    # allocator is not required to align a buffer more than its
                    # element. Keep one runtime check with a scalar fallback.
                    src_ok = self.reg('p')
                    dst_ok = self.reg('p')
                    aligned = self.reg('p')
                    src_low = self.reg('rd')
                    dst_low = self.reg('rd')
                    self.emit(f'and.b64 {src_low}, {src}, 15;')
                    self.emit(f'and.b64 {dst_low}, {dst}, 15;')
                    self.emit(f'setp.eq.u64 {src_ok}, {src_low}, 0;')
                    self.emit(f'setp.eq.u64 {dst_ok}, {dst_low}, 0;')
                    self.emit(f'and.pred {aligned}, {src_ok}, {dst_ok};')
                    if active is not None:
                        self.emit(f'and.pred {aligned}, {aligned}, {active};')
                    scalar = self.label('copy_scalar')
                    done = self.label('copy_done')
                    self.emit(f'@!{aligned} bra {scalar};')
                    self.emit_cp_async(src, dst, 16, base_name=s.smem)
                    self.emit(f'bra {done};')
                    self.mark(scalar)
                    for offset in range(width):
                        src_scalar, _ = self.address(s.src, s.src_index, offset)
                        dst_scalar, _ = self.address(s.smem, destination, offset)
                        self.emit_cp_async(src_scalar, dst_scalar, nbytes, active,
                                           base_name=s.smem)
                    self.mark(done)
                    continue
                for offset in range(width):
                    src, _ = self.address(s.src, s.src_index, offset)
                    dst, _ = self.address(s.smem, destination, offset)
                    self.emit_cp_async(src, dst, nbytes, active, base_name=s.smem)
            elif kind in ('CpAsyncCommit', 'CpAsyncWait'):
                if not self.has_cp_async:
                    continue
                if kind == 'CpAsyncCommit':
                    self.emit('cp.async.commit_group;')
                else:
                    if type(s.group) is not int or s.group < 0:
                        raise Unsupported('Invalid cp.async wait group')
                    self.emit(f'cp.async.wait_group {s.group};')
            elif kind == 'Cond':
                p, dtype = self.expr(s.cond)
                if dtype != 'pred':
                    raise Unsupported('Expected predicate for conditional')
                # Cond body SSA bindings are branch-local. Accumulators are
                # the documented exception: a fold crosses the boundary with
                # its finalized mutable register. Restore every other outer
                # binding after each arm so one path cannot leak a register
                # into the join that the other path never defined.
                outer_values = dict(self.values)
                outer_mutable = dict(self.mutable)
                otherwise, done = self.label('else'), self.label('endif')
                self.emit(f'@!{p} bra {otherwise};')
                self.statements(s.body)
                self.values = dict(outer_values)
                self.mutable = dict(outer_mutable)
                self.emit(f'bra {done};')
                self.mark(otherwise)
                self.statements(s.else_body)
                self.mark(done)
                self.values = dict(outer_values)
                self.mutable = dict(outer_mutable)
                for name, register in outer_mutable.items():
                    if name in self.values:
                        self.values[name] = (register, self.values[name][1])
            elif kind == 'Sync':
                if s.warp:
                    if s.barrier_id != 0:
                        raise Unsupported('Warp sync cannot carry a named barrier')
                    self.emit('bar.warp.sync 0xffffffff;')
                elif s.count is not None:
                    self.emit(f'bar.sync {s.barrier_id}, {s.count};')
                else:
                    self.emit(f'bar.sync {s.barrier_id};')
            elif kind == 'TreeHalve':
                if s.length <= 0 or s.length & (s.length-1):
                    raise Unsupported('Unsupported shared tree configuration')
                if len(s.bufs) != len(s.state) or len(s.state) != len(s.state_b) or not s.combine_states:
                    raise Unsupported('Tree state, buffer, and combine arity differ')
                if s.tid_var != 'warp':
                    raise Unsupported('Only warp-indexed TreeHalve is supported')
                if s.barrier_id != 0 and s.barrier_count is None:
                    raise Unsupported('Named TreeHalve barrier requires a thread count')
                if s.inner is not None:
                    inner_var, inner_scale = s.inner
                    if inner_scale <= 0 or inner_var not in self.values:
                        raise Unsupported('Invalid TreeHalve inner segment')
                    from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
                    inner_expr = Var(inner_var)
                    def tree_index(offset):
                        segment = Var(s.tid_var) if offset == 0 else BinaryExpr('+', Var(s.tid_var), Literal(offset, 'int'))
                        return BinaryExpr('+', BinaryExpr('*', segment, Literal(inner_scale, 'int')),
                                          inner_expr)
                    root_index = inner_expr
                else:
                    from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
                    def tree_index(offset):
                        return Var(s.tid_var) if offset == 0 else BinaryExpr('+', Var(s.tid_var), Literal(offset, 'int'))
                    root_index = Literal(0, 'int')
                # Emmy stores from every participating tree thread. A lane-0
                # guard here would publish only one slot per warp and silently
                # drop the other warp partials for a non-unit tree.
                size = s.length // 2
                while size:
                    p = self.reg('p')
                    self.emit(f'setp.lt.s32 {p}, {self.value(s.tid_var)}, {size};')
                    skip = self.label('tree_skip')
                    self.emit(f'@!{p} bra {skip};')
                    for buf, a, b in zip(s.bufs, s.state, s.state_b, strict=True):
                        from emmy.compiler.ir.expr import Var, BinaryExpr, Literal
                        addr, _ = self.address(buf, (tree_index(0),))
                        dtype = dtype_name(self.buffers[buf][1])
                        ar = self.reg(reg_suffix(dtype))
                        self.emit(f'ld.shared.{dtype_info(dtype)[1]} {ar}, [{addr}];')
                        self.values[a] = (ar, dtype)
                        addr2, _ = self.address(buf, (tree_index(size),))
                        br = self.reg(reg_suffix(dtype))
                        self.emit(f'ld.shared.{dtype_info(dtype)[1]} {br}, [{addr2}];')
                        self.values[b] = (br, dtype)
                    self.statements(s.combine_states)
                    for buf, a in zip(s.bufs, s.state, strict=True):
                        addr, _ = self.address(buf, (tree_index(0),))
                        dtype = dtype_name(self.buffers[buf][1])
                        self.emit(f'st.shared.{dtype_info(dtype)[1]} [{addr}], {self.value(a)};')
                    self.mark(skip)
                    if s.barrier_id != 0:
                        self.emit(f'bar.sync {s.barrier_id}, {s.barrier_count};')
                    else:
                        self.emit('bar.sync 0;')
                    size //= 2
                for buf, a in zip(s.bufs, s.state, strict=True):
                    from emmy.compiler.ir.expr import Literal
                    addr, _ = self.address(buf, (root_index,))
                    dtype = dtype_name(self.buffers[buf][1])
                    r = self.reg(reg_suffix(dtype))
                    self.emit(f'ld.shared.{dtype_info(dtype)[1]} {r}, [{addr}];')
                    self.values[a] = (r, dtype)
            else:
                raise Unsupported(f'Unsupported statement {kind}')
        for name, key in pending:
            self.vn_registers[key] = self.values[name]


    def render(self):
        self.statements(self.node.op.body)
        by_name = {arg['name']: arg for arg in self.meta['args']}
        params = []
        for i, name in enumerate(self.meta['arg_order']):
            arg = by_name.get(name)
            if arg is None:
                raise Unsupported(f'Argument {name!r} absent from metadata args')
            phys = 'u64' if not arg.get('scalar') else dtype_info(dtype_name(arg['dtype']))[1]
            params.append(f'    .param .{phys} param_{i}')
        decl = [f'    .reg .{suffix} %{kind}<{self.counts[kind]+1}>;' for kind,suffix in [('r','b32'),('rh','b16'),('rd','b64'),('f','f32'),('fd','f64'),('p','pred')]]
        decl += ['    '+x for x in self.declarations]
        # The metadata block shape is the launch contract. Declaring it lets
        # ptxas budget registers for the real thread count instead of the
        # 1024-thread worst case (a 64-register cap).
        bounds = '.maxntid ' + ', '.join(str(int(x)) for x in self.meta['block'])
        return '\n'.join([*PTX_HEADER, '', *self.externals, f'.visible .entry {ident(self.node.op.name)}(', ',\n'.join(params), ')', bounds, '{', *decl, *self.lines, 'kernel_done:', '    ret;', '}', ''])


def compile_path(path):
    path = Path(path)
    graph = Graph.from_dict(json.loads(path.read_text()))
    kernels = [(name,n) for name,n in graph.nodes.items() if type(n.op).__name__ == 'KernelOp']
    metadata = json.loads(path.with_name('meta.json').read_text())
    if len(kernels) == 1:
        name,node = kernels[0]
    else:
        # Pipeline IR dumps may retain the whole graph. The filename names the
        # requested kernel; node and entry metadata jointly validate selection.
        requested = path.name[:-len('.ir.json')] if path.name.endswith('.ir.json') else path.stem
        selected = [(name, node) for name, node in kernels if node.op.name == requested
                    and any(k['entry'] == requested and k['node'] == name for k in metadata['kernels'])]
        if len(selected) != 1:
            raise Unsupported('Expected a unique filename-selected KernelOp in pipeline graph')
        name,node = selected[0]
    matches = [k for k in metadata['kernels'] if k['entry'] == node.op.name and k['node'] == name]
    if len(matches) != 1:
        raise Unsupported('Expected exactly one metadata kernel matching node and entry')
    if len(set(matches[0]['arg_order'])) != len(matches[0]['arg_order']):
        raise Unsupported('Duplicate argument names in metadata')
    # A SIMT reduction loop whose shared operands lower to strength-reduced
    # vector loads computes in exact fp32 and, on the RTX 5090, beat the
    # three-term TF32 path on m01 (17.0 vs 21.1 us); prefer it outright.
    simt = Emitter(graph, node, matches[0])
    simt_ptx = simt.render()
    if simt.vector_loops:
        return simt_ptx
    # The tensor-core path is fail-closed: only an independently validated
    # one-atom plan bypasses the existing SIMT emitter.  Every other typed
    # region takes the untouched path below, preserving baseline PTX bytes.
    try:
        import tensor_core
        from tf32_emitter import production_ptx
        decision = tensor_core.classify(graph, node, matches[0])
        if (decision.verdict == tensor_core.PLAN and decision.plan is not None
                and decision.plan.staging in ('global', 'shared', 'pipeline')):
            try:
                return production_ptx(decision.plan, matches[0], node=node, graph=graph)
            except Exception as exc:
                # A plan is an optimization, never a correctness dependency.
                # Any emitter rejection falls back to the untouched SIMT path.
                print(f'compile.py: tensor-core emitter fallback: '
                      f'{type(exc).__name__}: {exc}', file=sys.stderr)
                return simt_ptx
    except (ImportError, ValueError, KeyError, TypeError, AttributeError,
            IndexError, ArithmeticError) as exc:
        print(f'compile.py: tensor-core fallback: {type(exc).__name__}: {exc}',
              file=sys.stderr)
    return simt_ptx


def main():
    try:
        if len(sys.argv) != 2:
            raise Unsupported('Usage: python compile.py kernel.ir.json')
        output = compile_path(sys.argv[1])
        output.encode('ascii')
        sys.stdout.write(output)
    except Exception as exc:
        print(f'compile.py: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
