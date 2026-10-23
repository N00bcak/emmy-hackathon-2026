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



I32_MIN = -(1 << 31)
I32_MAX = (1 << 31) - 1

def checked_i32_interval(lower, upper):
    if (isinstance(lower, int) and isinstance(upper, int)
            and I32_MIN <= lower <= upper <= I32_MAX):
        return lower, upper
    return None

def i32_binary_interval(op, left, right):
    """Conservative mathematical bounds; never infer through signed overflow."""
    if left is None or right is None:
        return None
    a, z = left
    b, y = right
    if checked_i32_interval(a, z) is None or checked_i32_interval(b, y) is None:
        return None
    if op == '+':
        return checked_i32_interval(a + b, z + y)
    if op == '-':
        return checked_i32_interval(a - y, z - b)
    if op == '*':
        products = (a * b, a * y, z * b, z * y)
        return checked_i32_interval(min(products), max(products))
    if op in ('/', '//') and b > 0:
        quotients = (a // b, a // y, z // b, z // y)
        return checked_i32_interval(min(quotients), max(quotients))
    if op == '%' and b > 0:
        return checked_i32_interval(0, min(z, y - 1) if a >= 0 else y - 1)
    if op in ('>>', '<<') and b == y and 0 <= b <= 31:
        if op == '>>':
            return checked_i32_interval(a >> b, z >> b)
        return checked_i32_interval(a << b, z << b)
    if op == '&':
        if b == y and b >= 0:
            return checked_i32_interval(0, min(z, b) if a >= 0 else b)
        if a == z and a >= 0:
            return checked_i32_interval(0, min(y, a) if b >= 0 else a)
        if a >= 0 and b >= 0:
            return checked_i32_interval(0, min(z, y))
    if op == '|' and a >= 0 and b >= 0:
        return checked_i32_interval(0, (1 << max(z, y).bit_length()) - 1)
    return None

_INTEGER_ELEMENTWISE_OPS = frozenset(['add', 'bitwise_and', 'bitwise_count', 'bitwise_or', 'bitwise_xor', 'floor_divide', 'left_shift', 'mod', 'multiply', 'remainder', 'right_shift', 'subtract'])
_INTEGER_DTYPES = frozenset(['i16', 'i32', 'i64', 'u16', 'u32', 'u64', 'u8'])
def infer_assign_dtype(op_name: str, arg_dtypes: list[str]) -> str:
    """Promote an elementwise op's arg dtypes to a single result dtype.

    Mirrors the inline rule in ``Assign.render``: the result is the
    common dtype iff every arg agrees; any disagreement promotes to f32.
    Floating point keeps the historical "all f16 → f16, otherwise f32"
    rule. Integer arithmetic/bit manipulation must instead retain a common
    integer dtype: promoting packed words to f32 loses bits before the CUDA
    renderer sees the operation. Comparisons and logical masks deliberately
    stay on the floating-point rule.

    Lifted out of ``Assign.render`` so the ``030_stamp_types`` pass can
    reuse the same rule when stamping ``Assign.dtype`` on the IR.
    """
    if arg_dtypes and op_name in _INTEGER_ELEMENTWISE_OPS and all(d in _INTEGER_DTYPES for d in arg_dtypes):
        # Keep packed arithmetic in the integer family.  NumPy promotes u64+i32 to f64 because
        # neither integer type contains the other's full range; that is disastrous for bit
        # extraction.  Use the usual C-style rank/sign rule instead: the widest unsigned type
        # wins a tie, while a strictly wider signed type can represent the narrower unsigned
        # range.  This gives u16+u32 -> u32, u64+i32 -> u64, and i64+u32 -> i64.
        widths = {"u8": 8, "i16": 16, "u16": 16, "i32": 32, "u32": 32, "i64": 64, "u64": 64}
        signed_width = max((widths[d] for d in arg_dtypes if d.startswith("i")), default=0)
        unsigned_width = max((widths[d] for d in arg_dtypes if d.startswith("u")), default=0)
        if unsigned_width >= signed_width:
            return f"u{unsigned_width}"
        return f"i{signed_width}"
    if arg_dtypes and all(d == "f16" for d in arg_dtypes):
        return "f16"
    return "f32"


from emmy.compiler.ir import expr as _typed_expr
from emmy.compiler.graph import _stmt_eval_scope
_stmt_eval_scope().setdefault("FlatIndex", _typed_expr.FlatIndex)


def _constant_literal_operand(expr):
    if type(expr).__name__ != 'Literal':
        return None
    try:
        declared = getattr(expr, 'dtype', 'float')
        if declared == 'float' or (isinstance(expr.value, float) and declared in ('int','bool')):
            dtype = 'f32'
        else:
            dtype = {'double':'f64','int':'i32','bool':'i32'}.get(declared,declared)
            if dtype not in DTYPES:
                dtype = 'i32' if isinstance(expr.value,(bool,int)) else 'f32'
        if dtype in FLOAT_DTYPES:
            code = {'f16':'e','f32':'f','f64':'d'}[dtype]
            value = struct.unpack('<'+code,struct.pack('<'+code,float(expr.value)))[0]
        else:
            if not isinstance(expr.value,(bool,int)):
                return None
            value = int(expr.value)
            width = dtype_info(dtype)[0]*8
            lower, upper = (0,(1<<width)-1) if dtype.startswith('u') else (-(1<<(width-1)),(1<<(width-1))-1)
            if not lower <= value <= upper:
                return None
        return value,dtype
    except (OverflowError,ValueError,TypeError,struct.error):
        return None

def constant_predicate(expr):
    """Prove only typed literal comparisons and compositions of such predicates."""
    if type(expr).__name__ != 'BinaryExpr':
        return None
    if expr.op in ('&&','||'):
        left,right=constant_predicate(expr.left),constant_predicate(expr.right)
        if left is None or right is None:
            return None
        return (left and right) if expr.op=='&&' else (left or right)
    if expr.op not in ('==','!=','<','<=','>','>='):
        return None
    left,right=_constant_literal_operand(expr.left),_constant_literal_operand(expr.right)
    if left is None or right is None:
        return None
    a,at=left;b,bt=right
    if at != bt:
        if at in INT_DTYPES and bt in INT_DTYPES:
            target=infer_assign_dtype('add',[at,bt]);width=dtype_info(target)[0]*8
            def integer_convert(value):
                value%=1<<width
                return value-(1<<width) if target.startswith('i') and value>=(1<<(width-1)) else value
            a,b=integer_convert(a),integer_convert(b)
        else:
            # Exact F32 conversion of arbitrary I64 values needs a separate proof;
            # retain the ordinary conservative path for that mixed literal domain.
            if at in ('i64','u64') or bt in ('i64','u64'):
                return None
            target='f64' if 'f64'in(at,bt) else 'f32';code='d' if target=='f64'else'f'
            try:a,b=(struct.unpack('<'+code,struct.pack('<'+code,float(v)))[0]for v in(a,b))
            except (OverflowError,ValueError,TypeError,struct.error):return None
    return {'==':a==b,'!=':a!=b,'<':a<b,'<=':a<=b,'>':a>b,'>=':a>=b}[expr.op]

def selected_constant_branches(statements):
    """Project proven selected arms for participation/seeding analysis only.

    Actual Cond nodes and their lexical restoration remain in executable IR.
    """
    result=[]
    for statement in statements:
        if type(statement).__name__=='Cond':
            decision=constant_predicate(statement.cond)
            if decision is not None:
                selected=statement.body if decision else statement.else_body
                result.extend(selected_constant_branches(selected))
                continue
        result.append(statement)
    return tuple(result)

def validate_partial_tile_collectives(statements, emitter):
    """Prove direct full-physical-CTA collective participation."""
    tree_buffer_requirements = {}
    def visit(items, direct):
        for index, statement in enumerate(items):
            kind = type(statement).__name__
            if kind == 'Tile':
                raise Unsupported('Partial collective Tile contains a nested Tile exit')
            if kind == 'WarpShuffle':
                if not direct:
                    raise Unsupported('Partial collective Tile has a nested/divergent shuffle')
                continue
            if kind == 'TreeHalve':
                if not direct:
                    raise Unsupported('Partial collective Tile has a nested/divergent tree')
                emitter.validate_partial_tree(statement)
                for buffer, combine in zip(statement.bufs, statement.combine_states, strict=True):
                    requirement = (statement.length, statement.inner,
                                   emitter.buffers[buffer][1], reduce_canon(combine.op.name))
                    previous_requirement = tree_buffer_requirements.setdefault(buffer, requirement)
                    if previous_requirement != requirement:
                        raise Unsupported('Partial tree slab reuse has incompatible seed requirements')
                # A visible shared publication barrier is required before the tree.
                previous = items[index - 1] if index else None
                if (type(previous).__name__ != 'Sync' or previous.warp
                        or previous.count is not None or previous.barrier_id != 0):
                    raise Unsupported('Partial CTA tree needs a direct publication barrier')
                continue
            if kind == 'Sync':
                if not direct or statement.warp or statement.count is not None:
                    raise Unsupported('Partial collective Tile has an unproven barrier')
                continue
            for body in statement_bodies(statement):
                visit(body, False)
    visit(statements, True)

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


def ssa_dtype_name(dtype):
    """Normalize value annotations separately from buffer storage types."""
    name = getattr(dtype, 'name', dtype)
    return 'pred' if name in ('bool', 'pred') else dtype_name(name)


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
    if source == 'pred' or target == 'pred':
        return emitter.snapshot_operand(value, source, target)[0]
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
        mode = '.rn' if dtype_info(target)[0] < dtype_info(source)[0] else ''
        emitter.emit(f'cvt{mode}.{target}.{source} {reg}, {value};')
        return reg
    if source in INT_DTYPES and target in INT_DTYPES:
        kind = 's' if source in ('i32', 'i64') else 'u'
        reg = emitter.reg(reg_suffix(target))
        emitter.emit(f'cvt.{kind}{dtype_info(target)[0] * 8}.{kind}{dtype_info(source)[0] * 8} {reg}, {value};')
        return reg
    if source in INT_DTYPES and target in FLOAT_DTYPES:
        kind = 's' if source in ('i32', 'i64') else 'u'
        reg = emitter.reg(reg_suffix(target))
        emitter.emit(f'cvt.rn.{target}.{kind}{dtype_info(source)[0] * 8} {reg}, {value};')
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
        emitter.emit(f'setp.neu.{dtype} {p}, {value}, {literal(0, dtype)};')
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
        if width == 64 and op in ('left_shift', 'right_shift'):
            # PTX shift counts are always32-bit, including64-bit values.
            b = convert_register(emitter, b, result, 'u32')
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
        elif kind == 'Tile':
            found.update(axis.name for axis in statement.axes)
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
        elif kind == 'Tile':
            # Entering and leaving a lexical Tile both replace its axis bindings.
            for axis in statement.axes:
                versions[axis.name] = versions.get(axis.name, 0) + 2
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
        for key in ('block', 'grid'):
            dimensions = meta.get(key)
            if (not isinstance(dimensions, (list, tuple)) or len(dimensions) != 3
                    or any(type(d) is not int or not 0 < d <= I32_MAX for d in dimensions)):
                raise Unsupported('Launch geometry requires three positive signed32 dimensions')
        self.block_threads = math.prod(meta['block'])
        self.grid_blocks = math.prod(meta['grid'])
        if self.block_threads > 1024 or self.block_threads * self.grid_blocks > I32_MAX:
            raise Unsupported('Linear launch geometry exceeds supported signed32 bounds')
        self.geometry_3d = any(d != 1 for d in (*meta['block'][1:], *meta['grid'][1:]))
        self.physical_builtins = {}
        self.lines, self.declarations = [], []
        self.externals = []
        self.counts = {'r': 0, 'rh': 0, 'rd': 0, 'f': 0, 'fd': 0, 'p': 0}
        self.values, self.buffers, self.shared = {}, {}, {}
        self.int_ranges = {}
        self.partial_tile_shuffle = False
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
        self._dynamic_shared_base = None
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
        for producer in graph.nodes.values():
            for buffer_name, tensor in zip(producer.buffer_names(), producer.outputs, strict=True):
                self.buffers[buffer_name] = (tuple(static(d) for d in tensor.shape), tensor.dtype.name)
        arguments = {arg['name']: arg for arg in meta['args']}
        if (len(arguments) != len(meta['args']) or len(meta['arg_order']) != len(meta['args'])
                or set(arguments) != set(meta['arg_order'])):
            raise Unsupported('Metadata arguments must match positional arg_order exactly')
        self.scalar_parameters = {}
        self.scalar_args = self.scalar_parameters
        self.scalar_shapes = {}
        # Runtime parameter values are never compiler constants.
        self.scalar_constants = {}
        for argument in meta['args']:
            name = argument['name']
            dtype = dtype_name(argument['dtype'])
            if argument.get('scalar'):
                if dtype not in ('f16', 'f32', 'f64', 'i32', 'i64'):
                    raise Unsupported('Unsupported scalar parameter ABI dtype')
                shape, graph_dtype = self.buffers.get(name, (None, None))
                declared_shape = argument.get('shape', shape if shape is not None else ())
                if (not isinstance(declared_shape, (list, tuple)) or len(declared_shape) > 1
                        or any(type(d) is not int or d <= 0 for d in declared_shape)
                        or math.prod(declared_shape) != 1
                        or (shape is not None and (len(shape) > 1 or math.prod(shape) != 1
                                                   or graph_dtype != dtype))
                        or argument.get('output') or name in meta.get('zero_outputs', ())):
                    raise Unsupported('Scalar parameter requires consistent scalar metadata and graph storage')
                self.scalar_shapes[name] = shape if shape is not None else tuple(declared_shape)
                self.buffers.pop(name, None)
            else:
                if 'shape' not in argument:
                    raise Unsupported(f'Buffer {name!r} lacks metadata shape')
                self.buffers[name] = (tuple(argument['shape']), dtype)
        if node.op.serial:
            raise Unsupported('Serial runtime axes are not supported')
        self.ptrs = {}
        self.parameter_types = []
        for index, name in enumerate(meta['arg_order']):
            ident(name)
            argument = arguments[name]
            dtype = dtype_name(argument['dtype'])
            if argument.get('scalar'):
                suffix = dtype_info(dtype)[1]
                register = self.reg(reg_suffix(dtype))
                self.emit(f'ld.param.{suffix} {register}, [param_{index}];')
                self.scalar_parameters[name] = (register, dtype)
                self.parameter_types.append(suffix)
            else:
                pointer = self.reg('rd')
                self.emit(f'ld.param.u64 {pointer}, [param_{index}];')
                self.emit(f'cvta.to.global.u64 {pointer}, {pointer};')
                self.ptrs[name] = pointer
                self.parameter_types.append('u64')
        for name, op, suffix in [('tid', '%tid.x', 'u32'), ('bid', '%ctaid.x', 'u32'), ('ntid', '%ntid.x', 'u32')]:
            reg = self.reg('r')
            self.emit(f'mov.{suffix} {reg}, {op};')
            self.values[name] = (reg, 'i32')
            physical_name = {'tid': 'thread_idx.x', 'bid': 'block_idx.x', 'ntid': 'block_dim.x'}[name]
            self.physical_builtins[physical_name] = reg
            size = meta['grid'][0] if name == 'bid' else meta['block'][0]
            self.set_integer_range(reg, checked_i32_interval(size, size) if name == 'ntid'
                                   else checked_i32_interval(0, size - 1))
        if self.geometry_3d:
            for logical, family, dimensions in (
                    ('tid', 'thread_idx', meta['block']),
                    ('bid', 'block_idx', meta['grid'])):
                x, y, z = (self.physical_builtin(f'{family}.{axis}')[0] for axis in 'xyz')
                yz, linear = self.reg('r'), self.reg('r')
                self.emit(f'mad.lo.u32 {yz}, {z}, {dimensions[1]}, {y};')
                self.emit(f'mad.lo.u32 {linear}, {yz}, {dimensions[0]}, {x};')
                total = math.prod(dimensions)
                self.set_integer_range(yz, (0, dimensions[1] * dimensions[2] - 1))
                self.set_integer_range(linear, (0, total - 1))
                self.values[logical] = (linear, 'i32')
            total_threads = self.reg('r')
            self.emit(f'mov.u32 {total_threads}, {self.block_threads};')
            self.set_integer_range(total_threads, (self.block_threads, self.block_threads))
            self.values['ntid'] = (total_threads, 'i32')
        tid, bid, ntid = (self.values[k][0] for k in ('tid', 'bid', 'ntid'))
        self.hardware_tid, self.hardware_bid, self.hardware_ntid = tid, bid, ntid
        gid = self.reg('r')
        self.emit(f'mad.lo.u32 {gid}, {bid}, {ntid}, {tid};')
        self.values['_gid'] = (gid, 'i32')
        self.hardware_gid = gid
        product = i32_binary_interval('*', self.integer_range(bid), self.integer_range(ntid))
        self.set_integer_range(gid, i32_binary_interval('+', product, self.integer_range(tid)))
        if self.has_cp_async and self.dynamic_shared:
            base = self.reg('rd')
            self.emit(f'mov.u64 {base}, {self.cp_async_base};')
            self._dynamic_shared_base = base
        for name, instruction, rhs in [('lane', 'and.b32', 31), ('warp', 'shr.u32', 5)]:
            reg = self.reg('r')
            self.emit(f'{instruction} {reg}, {tid}, {rhs};')
            self.values[name] = (reg, 'i32')
            if name == 'lane':
                self.hardware_lane = reg
                self.set_integer_range(reg, checked_i32_interval(0, min(31, self.block_threads - 1)))
            else:
                self.hardware_warp = reg
                self.set_integer_range(reg, checked_i32_interval(0, (self.block_threads - 1) // 32))
        # Keep upstream logical aliases for physical y/z coordinates.
        for axis, position in (('y', 1), ('z', 2)):
            for logical, family, dimensions in (
                    ('tid', 'thread_idx', meta['block']),
                    ('bid', 'block_idx', meta['grid'])):
                value = (self.physical_builtin(f'{family}.{axis}')[0]
                         if dimensions[position] > 1 else '0')
                self.values[f'{logical}_{axis}'] = (value, 'i32')
        # Logical IR names may be rebound or used as carried state. Keep their
        # initial values distinct from immutable physical scheduling operands.
        for name, (operand, dtype) in tuple(self.values.items()):
            self.values[name] = self.convert_operand(operand, dtype, dtype, copy_same_type=True)

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
            left = self.expr_key_at(x.left, versions)
            right = self.expr_key_at(x.right, versions)
            return None if left is None or right is None else ('binary', x.op, left, right)
        if kind == 'TernaryExpr':
            parts = (self.expr_key_at(x.cond, versions),
                     self.expr_key_at(x.if_true, versions), self.expr_key_at(x.if_false, versions))
            return None if any(part is None for part in parts) else ('ternary', *parts)
        return None

    def expr(self, x):
        kind = type(x).__name__
        key = self.expr_key(x)
        if key is not None and key in self.expr_cache:
            return self.expr_cache[key]
        if kind == 'Var':
            return self.binding(x.name)
        if kind == 'Builtin':
            return self.physical_builtin(x.name)
        if kind == 'Literal':
            declared = getattr(x, 'dtype', 'float')
            if declared == 'float' or (isinstance(x.value, float) and declared in ('int', 'bool')):
                return f32(float(x.value)), 'f32'
            aliases = {'double': 'f64', 'int': 'i32', 'bool': 'i32'}
            dtype = aliases.get(declared, declared)
            if dtype in FLOAT_DTYPES:
                return literal(x.value, dtype), dtype
            if isinstance(x.value, (bool, int)):
                dtype = dtype if dtype in INT_DTYPES else 'i32'
                bits = dtype_info(dtype)[0] * 8
                lower, upper = (0, (1 << bits) - 1) if dtype.startswith('u') else (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
                if not lower <= int(x.value) <= upper:
                    raise Unsupported(f'Integer literal exceeds {dtype} bounds')
                return str(int(x.value)), dtype
            return f32(x.value), 'f32'
        if kind == 'BinaryExpr':
            a, at = self.expr(x.left)
            b, bt = self.expr(x.right)
            if x.op in ('<<', '>>') and at in INT_DTYPES and bt in INT_DTYPES:
                count = b if dtype_info(bt)[0] == 4 else convert_register(self, b, bt, 'u32')
                width = dtype_info(at)[0] * 8
                kind = 'u' if at.startswith('u') else 's'
                r = self.reg(reg_suffix(at))
                instruction = 'shl.b' + str(width) if x.op == '<<' else 'shr.' + kind + str(width)
                self.emit(f'{instruction} {r}, {a}, {count};')
                if at == 'i32':
                    self.set_integer_range(r, i32_binary_interval(x.op, self.integer_range(a), self.integer_range(b)))
                return r, at
            if x.op in ('&&', '||'):
                if at != 'pred' or bt != 'pred':
                    raise Unsupported('Logical expressions require predicates')
                r = self.reg('p')
                self.emit(f'{"and.pred" if x.op == "&&" else "or.pred"} {r}, {a}, {b};')
                return r, 'pred'
            if at == bt == 'pred' and x.op in ('==', '!='):
                r = self.reg('p')
                self.emit(f'xor.pred {r}, {a}, {b};')
                if x.op == '==':
                    self.emit(f'not.pred {r}, {r};')
                return r, 'pred'
            comparisons = {'==':'eq', '!=':'ne', '<':'lt', '<=':'le', '>':'gt', '>=':'ge'}
            if x.op in comparisons:
                if at != bt:
                    target = infer_assign_dtype('add', [at, bt]) if at in INT_DTYPES and bt in INT_DTYPES else ('f64' if 'f64' in (at, bt) else 'f32')
                    a = convert_register(self, a, at, target)
                    b = convert_register(self, b, bt, target)
                    at = bt = target
                if at == 'f16':
                    a = convert_register(self, a, at, 'f32')
                    b = convert_register(self, b, bt, 'f32')
                    at = bt = 'f32'
                p = self.reg('p')
                comparison = 'neu' if at in FLOAT_DTYPES and x.op == '!=' else comparisons[x.op]
                self.emit(f'setp.{comparison}.{ptx_compare_type(at)} {p}, {a}, {b};')
                return p, 'pred'
            if at != bt:
                target = infer_assign_dtype('add', [at, bt]) if at in INT_DTYPES and bt in INT_DTYPES else ('f64' if 'f64' in (at, bt) else 'f32')
                a = convert_register(self, a, at, target)
                b = convert_register(self, b, bt, target)
                at = bt = target
            if at == 'f32':
                instructions = {'+':'add.rn.f32', '-':'sub.rn.f32', '*':'mul.rn.f32', '/':'div.rn.f32'}
                if x.op not in instructions:
                    raise Unsupported(f'Unsupported floating expression {x.op}')
                r = self.reg('f')
                self.emit(f'{instructions[x.op]} {r}, {a}, {b};')
                return r, 'f32'
            ops = {'+':'add.s32', '-':'sub.s32', '*':'mul.lo.s32', '&':'and.b32', '|':'or.b32', '>>':'shr.s32', '<<':'shl.b32'}
            if x.op in ('&&', '||'):
                if at != 'pred' or bt != 'pred':
                    raise Unsupported(f'Logical {x.op} requires predicates')
                p = self.reg('p')
                self.emit(f'{"and.pred" if x.op == "&&" else "or.pred"} {p}, {a}, {b};')
                return p, 'pred'
            if x.op in ('/', '//', '%'):
                if at == 'i32':
                    return self.integer_floor_divmod(x.op, a, b)
                if re.fullmatch(r'-?[0-9]+', b) and int(b) == 0:
                    raise Unsupported('Integer division by known zero')
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
                instruction_type = 'b64' if x.op in ('&', '|', '<<') else kind + '64'
                self.emit(f'{wide[x.op]}.{instruction_type} {r}, {a}, {b};')
                return r, at
            left_key, right_key = self.expr_key(x.left), self.expr_key(x.right)
            key = ('binary', x.op, left_key, right_key) if left_key is not None and right_key is not None else None
            cached = self.expr_cache.get(key) if key is not None else None
            if cached is not None:
                return cached
            r = self.reg('r')
            instruction = 'shr.u32' if at == 'u32' and x.op == '>>' else ops[x.op]
            self.emit(f'{instruction} {r}, {a}, {b};')
            if at == 'i32':
                self.set_integer_range(r, i32_binary_interval(x.op, self.integer_range(a), self.integer_range(b)))
            if key is not None:
                self.expr_cache[key] = (r, at)
            return r, at
        if kind == 'TernaryExpr':
            p, pt = self.expr(x.cond)
            a, at = self.expr(x.if_true)
            b, bt = self.expr(x.if_false)
            if pt != 'pred':
                raise Unsupported('Ternary condition must be a predicate')
            if at != bt:
                target = infer_assign_dtype('add', [at, bt]) if at in INT_DTYPES and bt in INT_DTYPES else ('f64' if 'f64' in (at, bt) else 'f32')
                a = convert_register(self, a, at, target)
                b = convert_register(self, b, bt, target)
                at = bt = target
            if at == 'pred':
                yes, no, inverted, r = (self.reg('p') for _ in range(4))
                self.emit(f'and.pred {yes}, {p}, {a};')
                self.emit(f'not.pred {inverted}, {p};')
                self.emit(f'and.pred {no}, {inverted}, {b};')
                self.emit(f'or.pred {r}, {yes}, {no};')
                return r, 'pred'
            r = self.reg(reg_suffix(at))
            self.emit(f'selp.{select_type(at)} {r}, {a}, {b}, {p};')
            return r, at
        if kind == 'CastExpr':
            operand, source = self.expr(x.expr)
            aliases = {'float':'f32','int':'i32','bool':'pred','double':'f64'}
            target = aliases.get(x.dtype, x.dtype)
            if target not in DTYPES and target != 'pred':
                raise Unsupported(f'Unsupported expression cast {target}')
            return self.convert_operand(operand, source, target, copy_same_type=False)
        if kind == 'FuncCallExpr':
            return self.funcall_operation(x.name, [self.expr(a) for a in x.args])
        if kind == 'FlatIndex':
            return self.flatten_offset(x.buffer, x.index, record_ranges=True), 'i32'
        raise Unsupported(f'Unsupported expression {kind}')

    def value(self, name):
        return self.binding(name)[0]

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
        # PTX's shared window is a 32-bit offset from the shared base. Derive
        # that offset by subtracting the slab base from the generic address;
        # a 64-bit cvta.to.shared is not a valid operand form here.
        if base_name is None:
            raise Unsupported('cp.async requires a shared base name')
        shared32 = self.reg('r')
        if self.dynamic_shared:
            base = self._dynamic_shared_base
            offset = self.reg('rd')
            self.emit(f'sub.u64 {offset}, {destination}, {base};')
            self.emit(f'cvt.u32.u64 {shared32}, {offset};')
        else:
            # A static slab symbol is already a shared-window address. Add the
            # destination's offset within that slab to it; using only the
            # offset would drop the slab's position for the second buffer.
            base = self.reg('rd')
            self.emit(f'mov.u64 {base}, {ident(base_name)};')
            offset = self.reg('rd')
            self.emit(f'sub.u64 {offset}, {destination}, {base};')
            slab32 = self.reg('r')
            self.emit(f'mov.u32 {slab32}, {ident(base_name)};')
            self.emit(f'cvt.u32.u64 {shared32}, {offset};')
            self.emit(f'add.u32 {shared32}, {slab32}, {shared32};')
        qualifier = 'cg' if nbytes == 16 else 'ca'
        pred = f'@{active} ' if active is not None else ''
        self.emit(f'{pred}cp.async.{qualifier}.shared.global '
                  f'[{shared32}], [{source}], {nbytes};')

    def assign(self, name, op, args, dtype=None):
        typed = [self.binding(a) for a in args]
        operation = reduce_canon(getattr(op, 'name', op))
        argument_types = [t for _, t in typed]
        result = dtype_name(dtype) if dtype is not None else infer_assign_dtype(operation, argument_types)
        native_integer = (INTEGER_ELEMENTWISE - {'divide'}) | set(COMPARISONS) | {'copy', 'negative', 'abs', 'maximum', 'minimum', 'where'}
        native_half = {'add','subtract','multiply','divide','maximum','minimum','negative','abs','sqrt','reciprocal','copy','relu','where'} | set(COMPARISONS)
        native_double = {'add','subtract','multiply','divide','maximum','minimum','negative','abs','sqrt','reciprocal','copy','relu','where','square','floor','ceil','round','trunc'} | set(COMPARISONS)
        value_types = argument_types[1:] if operation == 'where' else argument_types
        all_at_result = bool(value_types) and all(t == result for t in value_types)
        integer_promotion = result in INT_DTYPES and bool(value_types) and all(t in INT_DTYPES for t in value_types)
        native_op = (result == 'f32' or (result in INT_DTYPES and operation in native_integer)
                     or (result == 'f16' and operation in native_half)
                     or (result == 'f64' and operation in native_double))
        native_computation = True
        if operation == 'bitcast':
            if dtype is None:
                raise Unsupported('Bitcast requires explicit destination dtype')
            register, result = self.emit_operation(op, typed, result)
        elif operation == 'copy':
            # Copy is a numeric conversion of one value, with no arithmetic rounding step.
            # This deliberately preserves typed native integer/wide copies.
            register, result = self.emit_operation(op, typed, result)
        elif result == 'f32' or (native_op and (all_at_result or integer_promotion)):
            register, result = self.emit_operation(op, typed, result)
        else:
            if any(t in ('f64','i64','u64') for t in argument_types):
                raise Unsupported('Mixed/non-native wide stamped arithmetic has no established computation law')
            # Mixed operands no wider than F32 compute in F32, then demote exactly once.
            # In particular F32(.6)+F32(.6) stampedI32 is1, not0+0.
            register, actual = self.emit_operation(op, typed, 'f32')
            register, result = self.convert_operand(register, actual, result, copy_same_type=False)
            native_computation = False
        if name in self.mutable:
            stable = self.mutable[name]
            if self.values[name][1] != result:
                raise Unsupported('Mutable assignment changes carrier dtype')
            self.emit(f'mov.{select_type(result)} {stable}, {register};')
            register = stable
        self.values[name] = (register, result)
        if result == 'i32':
            interval_op = {'add':'+','subtract':'-','multiply':'*'}.get(operation)
            interval = (i32_binary_interval(interval_op, *(self.integer_range(v) for v, _ in typed))
                        if native_computation and interval_op and len(typed) == 2
                        else self.integer_range(typed[0][0]) if operation == 'copy' and len(typed) == 1 and typed[0][1] == 'i32'
                        else None)
            self.set_integer_range(register, interval)

    def emit_operation(self, op, args, dtype=None, literal_args=None):
        op = reduce_canon(getattr(op, 'name', op))
        operands = [v for v, _ in args]
        operand_types = [t for _, t in args]
        if not operands:
            raise Unsupported('Elementwise operation needs operands')
        result = dtype_name(dtype) if dtype is not None else infer_assign_dtype(op, operand_types)
        literal_args = [None] * len(args) if literal_args is None else literal_args
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
            return reg, result
        # Integer arithmetic, bit manipulation, and comparisons retain their
        # typed registers. Only the explicitly listed operations are legal;
        # an unknown integer op must fail rather than silently truncate.
        integer_result = result in INT_DTYPES
        if integer_result and not (op in INTEGER_ELEMENTWISE or op in COMPARISONS
                                   or op in ('copy', 'negative', 'abs', 'maximum', 'minimum')):
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
            comparison = 'neu' if compare_type in FLOAT_DTYPES and COMPARISONS[op] in ('ne', 'neu') else COMPARISONS[op]
            self.emit(f'setp.{comparison}.{compare_type} '
                      f'{p}, {operands[0]}, {operands[1]};')
            reg = self.reg(reg_suffix(result))
            if result in FLOAT_DTYPES:
                self.emit(f'selp.{select_type(result)} {reg}, {literal(1, result)}, {literal(0, result)}, {p};')
            else:
                self.emit(f'selp.{compare_type} {reg}, 1, 0, {p};')
            return reg, result
        if op == 'where' and len(operands) == 3:
            condition = as_predicate(self, operands[0], operand_types[0])
            reg = self.reg(reg_suffix(result))
            self.emit(f'selp.{select_type(result)} {reg}, {operands[1]}, {operands[2]}, {condition};')
            return reg, result
        if integer_result and op in ('maximum', 'minimum') and len(operands) == 2:
            reg = self.reg(reg_suffix(result))
            instruction = 'max' if op == 'maximum' else 'min'
            self.emit(f'{instruction}.{dtype_info(result)[1]} {reg}, {operands[0]}, {operands[1]};')
            return reg, result
        if integer_result and op in INTEGER_ELEMENTWISE and len(operands) == 2:
            reg = emit_integer_binary(self, op, result, operands[0], operands[1])
            return reg, result
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
            return reg, result
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
                    instruction = binary[op] + ('.NaN' if op in ('maximum', 'minimum') else '') + ('.rn' if op in ('add', 'subtract', 'multiply') else '')
                    self.emit(f'{instruction}.f16 {reg}, {operands[0]}, {operands[1]};')
            else:
                instruction = binary[op] + ('.NaN' if result == 'f32' and op in ('maximum', 'minimum') else '')
                self.emit(f'{instruction}.{result} {reg}, {operands[0]}, {operands[1]};')
                if result == 'f64' and op in ('maximum', 'minimum'):
                    nan = self.reg('p')
                    self.emit(f'setp.nan.f64 {nan}, {operands[0]}, {operands[1]};')
                    self.emit(f'@{nan} mov.f64 {reg}, {f64(float("nan"))};')
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
        elif op == 'exp' and len(operands) == 1:
            if result != 'f32':
                raise Unsupported('exp is only supported for f32')
            scaled = self.reg('f')
            self.emit(f'mul.f32 {scaled}, {operands[0]}, {f32(math.log2(math.e))};')
            self.emit(f'ex2.approx.f32 {reg}, {scaled};')
        elif op == 'exp_fast' and len(operands) == 1:
            if result != 'f32':
                raise Unsupported('exp_fast is only supported for f32')
            scaled = self.reg('f')
            self.emit(f'mul.f32 {scaled}, {operands[0]}, {f32(math.log2(math.e))};')
            self.emit(f'ex2.approx.f32 {reg}, {scaled};')
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
            modifier = '.NaN' if result in ('f16', 'f32') else ''
            self.emit(f'max{modifier}.{result} {reg}, {operands[0]}, {literal(0, result)};')
            if result == 'f64':
                nan = self.reg('p')
                self.emit(f'setp.nan.f64 {nan}, {operands[0]}, {operands[0]};')
                self.emit(f'@{nan} mov.f64 {reg}, {f64(float("nan"))};')
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
            raise Unsupported('General pow accuracy/domain is not established; no metadata-value specialization')
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
            unordered = self.reg('p')
            self.emit(f'setp.nan.f32 {unordered}, {operands[0]}, {operands[0]};')
            self.emit(f'@{unordered} mov.f32 {reg}, {f32(float("nan"))};')
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
        return reg, result

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
                self.emit(f'mov.{select_type(dtype)} {reg}, {literal(identities[op], dtype)};')
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
            value, value_type = self.binding(branch.value)
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
                identities = {'add': 0.0, 'multiply': 1.0, 'maximum': float('-inf'), 'minimum': float('inf')}
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
                self.emit(f'mov.u32 {slot}, {self.hardware_tid};')
                previous_slot = self.values.get('_tree_slot')
                self.rebind('_tree_slot')
                try:
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
                finally:
                    if previous_slot is None:
                        self.values.pop('_tree_slot', None)
                    else:
                        self.values['_tree_slot'] = previous_slot
                    self.rebind('_tree_slot')

    def integer_range(self, operand):
            if re.fullmatch(r'-?[0-9]+', operand):
                value = int(operand)
                return checked_i32_interval(value, value)
            return self.int_ranges.get(operand)

    def set_integer_range(self, operand, interval):
            if interval is None:
                self.int_ranges.pop(operand, None)
            else:
                self.int_ranges[operand] = interval

    def physical_builtin(self, name):
            families = {'thread_idx': ('tid', 'block', True),
                        'block_idx': ('ctaid', 'grid', True),
                        'block_dim': ('ntid', 'block', False),
                        'grid_dim': ('nctaid', 'grid', False)}
            known = {f'{family}.{axis}': (special, dimensions, index, position, axis)
                     for family, (special, dimensions, index) in families.items()
                     for position, axis in enumerate('xyz')}
            if name not in known:
                raise Unsupported(f'Unsupported physical Builtin {name}')
            if name not in self.physical_builtins:
                special, dimensions, index, position, axis = known[name]
                reg = self.reg('r')
                self.emit(f'mov.u32 {reg}, %{special}.{axis};')
                size = self.meta[dimensions][position]
                self.set_integer_range(reg, (0, size - 1) if index else (size, size))
                self.physical_builtins[name] = reg
            return self.physical_builtins[name], 'i32'

    def integer_floor_divmod(self, op, a, b):
            # Literal operands arrive as decimal immediates; register operands are
            # dynamic and retain the IR's nonzero/representable signed32 precondition.
            divisor = int(b) if re.fullmatch(r'-?[0-9]+', b) else None
            if divisor == 0:
                raise Unsupported('Integer division by a known zero divisor')
            result = self.reg('r')
            left_range, right_range = self.integer_range(a), self.integer_range(b)
            self.set_integer_range(result, i32_binary_interval(op, left_range, right_range))
            if op == '%' and divisor in (-1, 1):
                self.set_integer_range(result, (0, 0))
                self.emit(f'mov.b32 {result}, 0;')
                return result, 'i32'
            if divisor == -1 and a == str(-(1 << 31)):
                raise Unsupported('Integer quotient exceeds signed32 bounds')
            if divisor is not None and divisor > 0 and not divisor & (divisor - 1):
                if op == '%':
                    self.emit(f'and.b32 {result}, {a}, {divisor - 1};')
                else:
                    self.emit(f'shr.s32 {result}, {a}, {divisor.bit_length() - 1};')
                return result, 'i32'

            if (left_range is not None and right_range is not None
                    and left_range[0] >= 0 and right_range[0] > 0):
                instruction = 'rem.u32' if op == '%' else 'div.u32'
                self.emit(f'{instruction} {result}, {a}, {b};')
                return result, 'i32'

            modulo_done = None
            if op == '%' and divisor is None:
                # INTMIN % -1 is zero even though its intermediate quotient cannot
                # be represented. Bypass division for every dynamic divisor -1.
                ordinary = self.label('modulo_divide')
                modulo_done = self.label('modulo_done')
                negative_one = self.reg('p')
                self.emit(f'setp.eq.s32 {negative_one}, {b}, -1;')
                self.emit(f'@!{negative_one} bra {ordinary};')
                self.emit(f'mov.b32 {result}, 0;')
                self.emit(f'bra {modulo_done};')
                self.mark(ordinary)

            quotient, product, remainder = (self.reg('r') for _ in range(3))
            self.emit(f'div.s32 {quotient}, {a}, {b};')
            self.emit(f'mul.lo.s32 {product}, {quotient}, {b};')
            self.emit(f'sub.s32 {remainder}, {a}, {product};')
            nonzero, opposite, adjust = (self.reg('p') for _ in range(3))
            signs = self.reg('r')
            self.emit(f'setp.ne.s32 {nonzero}, {remainder}, 0;')
            self.emit(f'xor.b32 {signs}, {remainder}, {b};')
            self.emit(f'setp.lt.s32 {opposite}, {signs}, 0;')
            self.emit(f'and.pred {adjust}, {nonzero}, {opposite};')
            corrected = self.reg('r')
            if op == '%':
                self.emit(f'add.s32 {corrected}, {remainder}, {b};')
                self.emit(f'selp.b32 {result}, {corrected}, {remainder}, {adjust};')
            else:
                self.emit(f'sub.s32 {corrected}, {quotient}, 1;')
                self.emit(f'selp.b32 {result}, {corrected}, {quotient}, {adjust};')
            if modulo_done is not None:
                self.mark(modulo_done)
            return result, 'i32'

    def flatten_offset(self, name, index, element_offset=0, *, record_ranges=False):
            if name not in self.buffers:
                raise Unsupported(f'Unknown buffer {name}')
            shape, _ = self.buffers[name]
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
                if record_ranges:
                    product = i32_binary_interval('*', self.integer_range(value), (stride, stride))
                    self.set_integer_range(tmp, i32_binary_interval('+', product, self.integer_range(offset)))
                offset = tmp
            if element_offset:
                tmp = self.reg('r')
                self.emit(f'add.s32 {tmp}, {offset}, {element_offset};')
                offset = tmp
            return offset

    def scalar_load(self, statement):
        name = statement.input
        shape = self.scalar_shapes[name]
        if (len(statement.names) != 1
                or (len(statement.index) != len(shape) and statement.index)):
            raise Unsupported('Scalar parameter Load must address exactly one scalar cell')
        for coordinate in statement.index:
            operand, dtype = self.expr(coordinate)
            if dtype != 'i32' or self.integer_range(operand) != (0, 0):
                raise Unsupported('Scalar parameter Load index must be proven zero')
        operand, source_dtype = self.scalar_parameters[name]
        target_dtype = source_dtype if statement.dtype is None else dtype_name(statement.dtype)
        register, target_dtype = self.convert_operand(operand, source_dtype, target_dtype,
                                                     copy_same_type=True)
        self.values[statement.names[0]] = (register, target_dtype)
        self.mutable.pop(statement.names[0], None)
        self.rebind(statement.names[0])

    def snapshot_operand(self, operand, source_dtype, dtype):
        """Capture a typed value without aliasing a mutable physical carrier."""
        dtype = getattr(dtype, 'name', dtype)
        if dtype == 'bool':
            dtype = 'pred'
        if dtype != 'pred':
            dtype = dtype_name(dtype)
        if dtype == 'pred':
            register = self.reg('p')
            if source_dtype == 'pred':
                self.emit(f'mov.pred {register}, {operand};')
            else:
                compare_dtype = source_dtype
                if compare_dtype == 'f16':
                    operand = convert_register(self, operand, 'f16', 'f32')
                    compare_dtype = 'f32'
                if compare_dtype in FLOAT_DTYPES:
                    self.emit(f'setp.neu.{compare_dtype} {register}, {operand}, {literal(0, compare_dtype)};')
                elif compare_dtype in INT_DTYPES:
                    self.emit(f'setp.ne.{ptx_compare_type(compare_dtype)} {register}, {operand}, 0;')
                else:
                    raise Unsupported(f'Unsupported Let source dtype {source_dtype}')
            return register, dtype
        if source_dtype == 'pred':
            register = self.reg(reg_suffix(dtype))
            self.emit(f'selp.{select_type(dtype)} {register}, {literal(1, dtype)}, {literal(0, dtype)}, {operand};')
            return register, dtype
        if source_dtype == dtype:
            register = self.reg(reg_suffix(dtype))
            self.emit(f'mov.{select_type(dtype)} {register}, {operand};')
            return register, dtype
        return convert_register(self, operand, source_dtype, dtype), dtype


    def find_key_batches(self, statements):
        """Batch independent readonly scores, keeping every F32 operation ordered."""
        if self.geometry_3d or len(statements) != 1 or type(statements[0]).__name__ != 'Tile':
            return {}
        tile = statements[0]
        names = [a.name for a in tile.axes]
        extents = [static(a.extent) for a in tile.axes]
        if not names or len(set(names)) != len(names):
            return {}
        width = extents[-1]
        if (width < 2 or width > 32 or width & (width - 1)
                or tile.aux_threads or tile.block_threads != self.block_threads
                or self.block_threads % 32
                or math.prod(extents) != self.block_threads * self.grid_blocks):
            return {}
        if tile.raster_group and tile.raster_axes:
            pair = tile.raster_axes
            if (len(pair) != 2 or any(a not in names[:-1] for a in pair)
                    or names.index(pair[1]) != names.index(pair[0]) + 1):
                return {}
        def flat(body):
            for statement in body:
                yield statement
                for nested in statement_bodies(statement):
                    yield from flat(nested)
        whole = list(flat(tile.body))
        allowed = {'Load', 'Assign', 'Let', 'Init', 'Loop', 'Accum', 'Write'}
        if any(type(statement).__name__ not in allowed for statement in whole):
            return {}
        written = {statement.output for statement in whole
                   if type(statement).__name__ == 'Write'}
        def used(body):
            result = set()
            for statement in flat(body):
                result.update(statement.deps())
                result.update(getattr(statement, 'carried_names', lambda: ())())
                for expression in statement.exprs():
                    result.update(expression.free_vars())
            return result
        def index_ok(expression, dot_axis, key_axis, uniform):
            kind = type(expression).__name__
            if kind == 'Literal':
                return (isinstance(expression.value, (bool, int))
                        and expression.dtype in ('int', 'bool', 'i32')
                        and I32_MIN <= int(expression.value) <= I32_MAX)
            if kind == 'Var':
                return expression.name in uniform | {dot_axis, key_axis}
            return (kind == 'BinaryExpr'
                    and expression.op in ('+', '-', '*', '/', '//', '%', '&', '|', '<<', '>>')
                    and index_ok(expression.left, dot_axis, key_axis, uniform)
                    and index_ok(expression.right, dot_axis, key_axis, uniform))
        plans = {}
        def scan(body, uniform, external):
            uniform = set(uniform)
            body = list(body)
            for position, loop in enumerate(body):
                if type(loop).__name__ != 'Loop':
                    uniform -= defined_names((loop,))
                    continue
                key = loop.axis.name
                extent = static(loop.axis.extent)
                definitions = defined_names(loop.body)
                stable = uniform - definitions
                outside = external | used(body[:position]) | used(body[position + 1:])
                parts, dots = list(loop.body), []
                for part in parts:
                    if type(part).__name__ != 'Loop':
                        break
                    dots.append(part)
                tail = parts[len(dots):]
                valid = (0 < extent <= I32_MAX - (width - 1)
                         and key not in definitions and dots and tail
                         and all(type(part).__name__ in ('Load', 'Assign', 'Let', 'Accum')
                                 for part in tail))
                all_names, score_names, temps, address_names, dot_axes = set(), set(), set(), set(), set()
                for dot in dots:
                    axis = dot.axis.name
                    dot_axes.add(axis)
                    if (not dot.seed or not 0 < static(dot.axis.extent) <= I32_MAX
                            or axis in defined_names(dot.body) or axis == key
                            or [type(part).__name__ for part in dot.body]
                               != ['Load', 'Load', 'Assign', 'Accum']):
                        valid = False
                        break
                    left, right, product, accumulator = dot.body
                    local_temps = (*left.names, *right.names, product.name)
                    local_names = (*local_temps, accumulator.name)
                    load_ok = all(
                        len(load.names) == 1 and load.input not in written
                        and load.input in self.buffers and load.input not in self.scalar_parameters
                        and load.input not in self.shared
                        and dtype_name(self.buffers[load.input][1]) == 'f32'
                        and (load.dtype is None or dtype_name(load.dtype) == 'f32')
                        and all(index_ok(index, axis, key, stable) for index in load.index)
                        for load in (left, right))
                    if (not load_ok or len(set(local_names)) != 4
                            or set(local_names) & (all_names | stable | {axis, key})
                            or product.op.name != 'multiply'
                            or sorted(product.args) != sorted((left.names[0], right.names[0]))
                            or (product.dtype is not None and dtype_name(product.dtype) != 'f32')
                            or accumulator.value != product.name or accumulator.op.name != 'add'
                            or accumulator.base is not None or tuple(accumulator.axes) != (axis,)
                            or (accumulator.dtype is not None and dtype_name(accumulator.dtype) != 'f32')):
                        valid = False
                        break
                    all_names.update(local_names)
                    score_names.add(accumulator.name)
                    temps.update(local_temps)
                    address_names.update(set().union(*(set(index.free_vars())
                        for load in (left, right) for index in load.index)) - {axis, key})
                tail_defs = defined_names(tail)
                if (valid and not temps & (outside | used(tail))
                        and not score_names & tail_defs
                        and not tail_defs & (stable | dot_axes | {key})
                        and not all_names & dot_axes
                        and all(type(part).__name__ != 'Load' or
                                part.input not in written for part in tail)):
                    plans[id(loop)] = (width, tuple(dots), tuple(tail),
                                       frozenset(address_names), frozenset(temps))
                if key not in definitions and 0 < extent <= I32_MAX:
                    scan(loop.body, stable | {key}, outside)
                old_axis_uniform = key in uniform
                uniform -= definitions
                if old_axis_uniform:
                    uniform.add(key)
        scan(tile.body, set(names[:-1]), set())
        return plans

    def ordered_key_batch(self, loop):
        plan = self.key_batches.get(id(loop))
        if plan is None or self.inactive:
            return False
        width, dots, tail, uniform, temps = plan
        if (any(name not in self.values or self.values[name][1] != 'i32'
                for name in uniform)
                or any(name in self.mutable for name in temps)):
            return False
        extent = static(loop.axis.extent)
        base, lane, segment = self.reg('r'), self.reg('r'), self.reg('r')
        self.emit(f'mov.s32 {base}, 0;')
        self.emit(f'and.b32 {lane}, {self.hardware_lane}, {width - 1};')
        self.emit(f'and.b32 {segment}, {self.hardware_lane}, {32 - width};')
        private = [self.reg('f') for _ in dots]
        begin, done = self.label('key_batch'), self.label('key_batch_done')
        self.mark(begin)
        stop = self.reg('p')
        self.emit(f'setp.ge.s32 {stop}, {base}, {extent};')
        self.emit(f'@{stop} bra {done};')
        for saved in private:
            self.emit(f'mov.f32 {saved}, 0f00000000;')
        key = self.reg('r')
        self.emit(f'add.s32 {key}, {base}, {lane};')
        self.values[loop.axis.name] = (key, 'i32')
        self.set_integer_range(key, checked_i32_interval(0, extent + width - 2))
        self.rebind(loop.axis.name)
        absent, joined = self.reg('p'), self.label('key_scores_ready')
        self.emit(f'setp.ge.s32 {absent}, {key}, {extent};')
        self.emit(f'@{absent} bra {joined};')
        # These are the original serial-D loops. Snapshot explicitly: replay
        # must not overwrite the per-lane private scores needed by later slots.
        for dot, saved in zip(dots, private, strict=True):
            self.statements((dot,))
            value, dtype = self.values[dot.body[-1].name]
            if dtype != 'f32':
                raise Unsupported('Key batching requires an F32 score')
            self.emit(f'mov.f32 {saved}, {value};')
        self.mark(joined)
        # Every physical lane reconverges before any full-mask indexed shuffle.
        slot = self.reg('r')
        self.emit(f'mov.s32 {slot}, 0;')
        replay, replay_done = self.label('key_replay'), self.label('key_replay_done')
        self.mark(replay)
        self.emit(f'setp.ge.s32 {stop}, {slot}, {width};')
        self.emit(f'@{stop} bra {replay_done};')
        replay_key = self.reg('r')
        self.emit(f'add.s32 {replay_key}, {base}, {slot};')
        self.emit(f'setp.ge.s32 {stop}, {replay_key}, {extent};')
        self.emit(f'@{stop} bra {replay_done};')
        self.values[loop.axis.name] = (replay_key, 'i32')
        self.set_integer_range(replay_key, checked_i32_interval(0, extent - 1))
        self.rebind(loop.axis.name)
        source = self.reg('r')
        self.emit(f'add.s32 {source}, {segment}, {slot};')
        for dot, saved in zip(dots, private, strict=True):
            name = dot.body[-1].name
            public = self.reg('f')
            self.emit(f'shfl.sync.idx.b32 {public}, {saved}, {source}, 31, 0xffffffff;')
            self.values[name] = (public, 'f32')
            self.mutable[name] = public
            self.rebind(name)
        self.statements(tail)
        self.emit(f'add.s32 {slot}, {slot}, 1;')
        self.emit(f'bra {replay};')
        self.mark(replay_done)
        self.emit(f'add.s32 {base}, {base}, {width};')
        self.rebind(loop.axis.name)
        self.emit(f'bra {begin};')
        self.mark(done)
        return True

    def shared_compute_body(self, loop):
        """Prove a flat, active, shared-F32 compute loop without emitting code."""
        if self.inactive or loop.axis.name in defined_names(loop.body):
            return None
        types = {name: dtype for name, (_, dtype) in self.scalar_parameters.items()}
        types.update({name: dtype for name, (_, dtype) in self.values.items()})
        def i32_index(expr):
            kind = type(expr).__name__
            if kind == 'Literal':
                return (isinstance(expr.value, (bool, int))
                        and expr.dtype in ('int', 'bool', 'i32')
                        and I32_MIN <= int(expr.value) <= I32_MAX)
            if kind == 'Var':
                return expr.name == loop.axis.name or types.get(expr.name) == 'i32'
            return (kind == 'BinaryExpr'
                    and expr.op in ('+', '-', '*', '/', '//', '%', '&', '|', '<<', '>>')
                    and i32_index(expr.left) and i32_index(expr.right))
        loads = 0
        for statement in loop.body:
            kind = type(statement).__name__
            if kind == 'Load':
                if (statement.input not in self.shared
                        or dtype_name(self.buffers[statement.input][1]) != 'f32'
                        or (statement.dtype is not None and dtype_name(statement.dtype) != 'f32')
                        or not all(i32_index(index) for index in statement.index)):
                    return None
                for name in statement.names:
                    types[name] = 'f32'
                loads += 1
            elif kind in ('Assign', 'Accum'):
                op = reduce_canon(getattr(statement.op, 'name', statement.op))
                if op not in ('add', 'subtract', 'multiply', 'divide', 'maximum', 'minimum', 'copy'):
                    return None
                args = (statement.args if kind == 'Assign' else
                        (statement.base if statement.base is not None else statement.name, statement.value))
                if (not args or any(types.get(name) != 'f32' for name in args)
                        or (statement.dtype is not None and dtype_name(statement.dtype) != 'f32')):
                    return None
                types[statement.name] = 'f32'
            else:
                return None
        return loads

    def shared_vector_plan(self, loop):
        """Prove four existing scalar shared reads form one aligned vector.

        Forms use actual i32 operands rather than SSA spelling. Every expression
        subtree and flattening intermediate is bounded in signed32 arithmetic.
        This analysis emits no instructions and never changes range/value state.
        """
        body, axis_name = loop.body, loop.axis.name
        if any(type(s).__name__ not in ('Load', 'Assign', 'Accum') for s in body):
            return {}
        defined = set()
        for s in body:
            defined.update(s.names if type(s).__name__ == 'Load' else (s.name,))
        if axis_name in defined:
            return {}
        extent = static(loop.axis.extent)
        axis_key = ('axis',)
        atoms = {axis_key: checked_i32_interval(0, extent - 1)}

        def finish(c, terms):
            terms = {key: value for key, value in terms.items() if value}
            low = high = c
            for key, value in terms.items():
                bounds = atoms[key]
                if bounds is None:
                    raise ValueError('Unknown atom range')
                a, b = (value * bounds[0], value * bounds[1])
                low += min(a, b)
                high += max(a, b)
            bounds = checked_i32_interval(low, high)
            if bounds is None:
                raise ValueError('Unbounded signed32 intermediate')
            if low == high:
                return low, {}, bounds
            return c, terms, bounds

        def scale(form, coefficient):
            return finish(form[0] * coefficient,
                          {k: v * coefficient for k, v in form[1].items()})

        def add(left, right, sign=1):
            terms = dict(left[1])
            for key, value in right[1].items():
                terms[key] = terms.get(key, 0) + sign * value
            return finish(left[0] + sign * right[0], terms)

        def key(form):
            return form[0], tuple(sorted(form[1].items(), key=repr))

        def parse(expr):
            kind = type(expr).__name__
            # Mirror expr(): Python floats and default/float-tagged literals
            # lower to f32, even when their Python/JSON value is an integer.
            if (kind == 'Literal' and isinstance(expr.value, (bool, int))
                    and expr.dtype in ('int', 'bool', 'i32')):
                return finish(int(expr.value), {})
            if kind == 'Var':
                if expr.name == axis_name:
                    return finish(0, {axis_key: 1})
                if expr.name in defined or expr.name not in self.values:
                    raise ValueError('Modified or unknown address dependency')
                operand, dtype = self.values[expr.name]
                if dtype != 'i32':
                    raise ValueError('Noninteger address dependency')
                bounds = self.integer_range(operand)
                if bounds is None:
                    raise ValueError('Unknown address range')
                if bounds[0] == bounds[1]:
                    return finish(bounds[0], {})
                atom = ('operand', operand)
                atoms[atom] = bounds
                return finish(0, {atom: 1})
            if kind != 'BinaryExpr':
                raise ValueError('Unsupported expression proof')
            left, right = parse(expr.left), parse(expr.right)
            if expr.op == '+':
                return add(left, right)
            if expr.op == '-':
                return add(left, right, -1)
            if expr.op == '*' and not left[1]:
                return scale(right, left[0])
            if expr.op == '*' and not right[1]:
                return scale(left, right[0])
            if axis_key in left[1] or axis_key in right[1]:
                raise ValueError('Nonaffine axis dependency')
            bounds = i32_binary_interval(expr.op, left[2], right[2])
            if bounds is None:
                raise ValueError('Unsupported opaque integer expression')
            if bounds[0] == bounds[1]:
                return finish(bounds[0], {})
            atom = ('binary', expr.op, key(left), key(right))
            atoms[atom] = bounds
            return finish(0, {atom: 1})

        result = {}
        for position, statement in enumerate(body):
            if type(statement).__name__ != 'Load' or len(statement.names) != 1:
                continue
            if statement.input not in self.shared:
                continue
            if statement.dtype is not None and dtype_name(statement.dtype) != 'f32':
                continue
            declaration = self.shared[statement.input]
            if self.dynamic_shared:
                aligned = self.shared_offsets[statement.input] % 16 == 0
            else:
                aligned = smem_alignment(declaration) % 16 == 0
            if not aligned:
                continue
            shape, dtype = self.buffers[statement.input]
            if (dtype_name(dtype) != 'f32' or len(statement.index) != len(shape)
                    or math.prod(shape) > I32_MAX):
                continue
            try:
                flat = finish(0, {})
                for coordinate, dimension in zip(statement.index, shape, strict=True):
                    form = parse(coordinate)
                    if not 0 <= form[2][0] <= form[2][1] < static(dimension):
                        raise ValueError('Coordinate outside shared declaration')
                    flat = add(scale(flat, static(dimension)), form)
                if flat[1].get(axis_key) != 1 or flat[0] % 4:
                    continue
                if any(value % 4 for atom, value in flat[1].items()
                       if atom != axis_key):
                    continue
                result[position] = statement
            except ValueError:
                continue
        return result

    def shared_vector_prefetch(self, plan):
        result = {}
        for position, statement in plan.items():
            address, space = self.address(statement.input, statement.index)
            assert space == 'shared'
            registers = tuple(self.reg('f') for _ in range(4))
            self.emit('ld.shared.v4.f32 {' + ', '.join(registers)
                      + f'}}, [{address}];')
            result[position] = registers
        return result


    def init_scope_frame(self, statements):
        # Existing Init names are lexical shadows. Fresh names retain the
        # enclosing construct's existing export/join policy.
        return {s.name: (self.values[s.name], self.mutable.get(s.name))
                for s in statements
                if type(s).__name__ == 'Init' and s.name in self.values}

    def statements_with_init_scope(self, statements, frame, shared_prefetch=None):
        try:
            self.statements(statements, shared_prefetch)
        finally:
            for name, (value, mutable) in frame.items():
                self.values[name] = value
                if mutable is None:
                    self.mutable.pop(name, None)
                else:
                    self.mutable[name] = mutable
                self.rebind(name)


    def binding(self, name):
        if name in self.values:
            return self.values[name]
        if name in self.scalar_parameters:
            return self.scalar_parameters[name]
        raise Unsupported(f'Unbound SSA value {name!r}')

    def live_loop_step(self, expression, header_values, axis_name, axis_register):
        """Evaluate a backedge step in its header scope using live carrier storage."""
        saved_values = self.values
        saved_expressions, saved_addresses = self.expr_cache, self.address_cache
        self.values = dict(header_values)
        self.values[axis_name] = (axis_register, 'i32')
        # Body locals and preheader CSE entries cannot redirect or stale the update.
        self.expr_cache, self.address_cache = {}, {}
        try:
            return self.expr(expression)
        finally:
            self.values = saved_values
            self.expr_cache, self.address_cache = saved_expressions, saved_addresses

    def convert_operand(self, operand, source_dtype, dtype, *, copy_same_type=True):
        dtype = getattr(dtype, 'name', dtype)
        if dtype == 'bool':
            dtype = 'pred'
        if source_dtype == dtype and not copy_same_type:
            return operand, dtype
        value, dtype = self.snapshot_operand(operand, source_dtype, dtype)
        if source_dtype == dtype == 'i32':
            self.set_integer_range(value, self.integer_range(operand))
        elif source_dtype == 'pred' and dtype == 'i32':
            self.set_integer_range(value, (0, 1))
        return value, dtype

    def scalar_operation(self, op, args, dtype=None):
        return self.emit_operation(op, args, dtype)
    def funcall_operation(self, op, args):
            # Direct registry calls retain their operand dtype; Assign has its own
            # deliberate promotion law and continues to use scalar_operation.
            if args and all(dtype == 'f32' for _, dtype in args):
                return self.scalar_operation(op, args)
            if not args or any(dtype != 'i32' for _, dtype in args):
                raise Unsupported('FuncCall requires all f32 or supported all i32 operands')
            if op in ('add', 'subtract', 'multiply'):
                return self.scalar_operation(op, args, 'i32')
            native = {'maximum': ('max.s32', 2), 'minimum': ('min.s32', 2),
                      'copy': ('mov.b32', 1), 'abs': ('abs.s32', 1),
                      'negative': ('neg.s32', 1)}
            if op not in native or len(args) != native[op][1]:
                raise Unsupported(f'Unsupported i32 FuncCall {op}/{len(args)}')
            instruction, _ = native[op]
            register = self.reg('r')
            operands = ', '.join(operand for operand, _ in args)
            self.emit(f'{instruction} {register}, {operands};')
            if op == 'copy':
                self.set_integer_range(register, self.integer_range(args[0][0]))
            return register, 'i32'

    def pad_partial_shuffle(self, statement, active):
        """Pad only independently proved monoid components, before shuffling."""
        if self.block_threads % statement.length:
            raise Unsupported('Partial shuffle requires complete physical groups')
        states, others = tuple(statement.state), tuple(statement.state_b)
        if (len(states) != len(others) or len(statement.combine_states) != len(states)
                or len(set(states)) != len(states) or len(set(others)) != len(others)
                or set(states) & set(others)):
            raise Unsupported('Partial shuffle requires independent state components')
        updates = {}
        for combine in statement.combine_states:
            if type(combine).__name__ != 'Assign' or combine.name in updates:
                raise Unsupported('Partial shuffle requires one pure Assign per component')
            updates[combine.name] = combine
        plans = []
        for state, other in zip(states, others, strict=True):
            combine = updates.get(state)
            if combine is None or tuple(combine.args) not in ((state, other), (other, state)):
                raise Unsupported('Partial shuffle combine crosses state components')
            operation = reduce_canon(combine.op.name)
            operand, dtype = self.values[state]
            if dtype != (dtype_name(statement.dtype) if statement.dtype is not None else 'f32'):
                raise Unsupported('Partial shuffle state dtype differs from its declaration')
            if combine.dtype is not None and dtype_name(combine.dtype) != dtype:
                raise Unsupported('Partial shuffle combine changes component dtype')
            if dtype == 'f32':
                identities = {'add': 0.0, 'multiply': 1.0,
                              'maximum': float('-inf'), 'minimum': float('inf')}
            elif dtype in ('i32', 'u32'):
                identities = {'add': 0, 'multiply': 1}
            else:
                raise Unsupported('Partial shuffle identity is unproved for this dtype')
            if operation not in identities:
                raise Unsupported('Partial shuffle combine identity is unproved')
            plans.append((state, operand, dtype, identities[operation]))
        for state, operand, dtype, identity in plans:
            target = self.mutable.get(state)
            if target is None:
                target = self.reg(reg_suffix(dtype))
            self.emit(f'selp.{select_type(dtype)} {target}, {operand}, '
                      f'{literal(identity, dtype)}, {active};')
            self.values[state] = (target, dtype)

    def validate_partial_tree(self, tree):
        """Prove the simple unsegmented partial-CTA tree padding law."""
        if (tree.tid_var != 'warp' or tree.inner is not None
                or tree.barrier_id != 0 or tree.barrier_count is not None
                or self.block_threads % 32 or tree.length <= 0
                or tree.length & (tree.length - 1)
                or tree.length > self.block_threads // 32):
            raise Unsupported('Partial CTA TreeHalve participation is unproved')
        states, others = tuple(tree.state), tuple(tree.state_b)
        if (not states or len(tree.bufs) != len(states) or len(set(tree.bufs)) != len(tree.bufs)
                or len(others) != len(states)
                or len(tree.combine_states) != len(states)
                or len(set(states)) != len(states) or len(set(others)) != len(others)
                or set(states) & set(others)):
            raise Unsupported('Partial CTA tree requires independent state components')
        for buf, state, other, combine in zip(tree.bufs, states, others,
                                               tree.combine_states, strict=True):
            if (type(combine).__name__ != 'Assign' or combine.name != state
                    or tuple(combine.args) not in ((state, other), (other, state))
                    or reduce_canon(combine.op.name) not in
                       ('add', 'multiply', 'maximum', 'minimum')):
                raise Unsupported('Partial CTA tree combine identity is unproved')
            if buf not in self.shared or buf not in self.buffers:
                raise Unsupported('Partial CTA tree requires shared slabs')
            shape, dtype = self.buffers[buf]
            if (len(shape) != 1 or shape[0] < tree.length or dtype != 'f32'
                    or (dtype_name(tree.dtype) if tree.dtype is not None else 'f32') != 'f32'
                    or (combine.dtype is not None and dtype_name(combine.dtype) != 'f32')):
                raise Unsupported('Partial CTA tree dtype/layout identity is unproved')

    def statements(self, statements, shared_prefetch=None):
        for position, s in enumerate(statements):
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
                self.emit(f'setp.ge.u32 {p}, {self.hardware_gid}, {n};')
                tile_done = None
                if not collective:
                    # A Tile guard scopes this body; later sibling statements
                    # still run for threads outside this Tile's coordinate set.
                    tile_done = self.label('tile_done')
                    self.emit(f'@{p} bra {tile_done};')
                previous_axes = {axis.name: self.values.get(axis.name) for axis in s.axes}
                try:
                    for i, axis in enumerate(s.axes):
                        stride = math.prod(extents[i+1:])
                        r = self.reg('r')
                        self.emit(f'div.u32 {r}, {self.hardware_gid}, {stride};')
                        extent = extents[i]
                        if i:
                            self.emit(f'rem.u32 {r}, {r}, {extent};')
                        self.rebind(axis.name)
                        self.values[axis.name] = (r, 'i32')
                        # Partial collective lanes also execute this body. Do not
                        # assign their out-of-domain axes active-lane facts.
                        if all(extent > 0 for extent in extents) and not (inactive and collective):
                            self.set_integer_range(r, checked_i32_interval(0, extents[i] - 1))
                    if inactive and collective:
                        # Inactive threads still execute the body to reach barriers.
                        # Their addresses are masked at every memory access; the
                        # axis registers themselves are consumed only within it.
                        if len(self.node.op.body) != 1 or self.node.op.body[0] is not s:
                            raise Unsupported('Partial collective Tile requires proven converged CTA entry')
                        analysis_body = selected_constant_branches(s.body)
                        validate_partial_tile_collectives(analysis_body, self)
                        self.preinitialize_trees(analysis_body)
                        self.emit('bar.sync 0;')
                        active = self.reg('p')
                        self.emit(f'setp.lt.u32 {active}, {self.hardware_gid}, {n};')
                        self.inactive.append(active)
                    try:
                        self.statements_with_init_scope(s.body, self.init_scope_frame(s.body))
                    finally:
                        if inactive and collective:
                            self.inactive.pop()
                finally:
                    # Axis registers are lexical bindings. Do not roll back physical
                    # integer facts: the body may have changed a carried register.
                    for name, previous in previous_axes.items():
                        if previous is None:
                            self.values.pop(name, None)
                        else:
                            self.values[name] = previous
                        self.rebind(name)
                if tile_done is not None:
                    self.mark(tile_done)
            elif kind == 'Load':
                if shared_prefetch is not None and position in shared_prefetch:
                    self.values[s.names[0]] = (shared_prefetch[position], 'f32')
                    self.mutable.pop(s.names[0], None)
                    self.rebind(s.names[0])
                    continue
                if s.input in self.scalar_parameters:
                    self.scalar_load(s)
                    continue
                source_dtype = dtype_name(self.buffers[s.input][1])
                dtype = dtype_name(s.dtype) if s.dtype is not None else source_dtype
                if len(s.names) > 1 and dtype != source_dtype:
                    # Emmy's vector renderer reinterprets a typed vector pack;
                    # numerical conversion is only established for scalar Load.
                    raise Unsupported('Vector Load stamp must match buffer storage dtype')
                phys = dtype_info(source_dtype)[1]
                loaded_bindings = []
                for i, name in enumerate(s.names):
                    addr, space = self.address(s.input, s.index, i)
                    raw = self.reg(reg_suffix(source_dtype))
                    if self.inactive:
                        self.emit(f'mov.{select_type(source_dtype)} {raw}, {literal(0, source_dtype)};')
                        self.emit(f'@{self.inactive[-1]} ld.{space}.{phys} {raw}, [{addr}];')
                    else:
                        self.emit(f'ld.{space}.{phys} {raw}, [{addr}];')
                    register, result_dtype = self.convert_operand(raw, source_dtype, dtype,
                                                                 copy_same_type=False)
                    loaded_bindings.append((name, register, result_dtype))
                # Grouped loads evaluate every address before publishing any
                # destination binding, including names that shadow an index.
                for name, register, result_dtype in loaded_bindings:
                    self.values[name] = (register, result_dtype)
                    self.mutable.pop(name, None)
                    self.rebind(name)
            elif kind == 'Write':
                if s.swizzle not in ('NONE', None):
                    raise Unsupported('Swizzled writes unsupported')
                output_dtype = dtype_name(self.buffers[s.output][1])
                value_dtype = ssa_dtype_name(s.value_dtype) if s.value_dtype is not None else output_dtype
                phys = dtype_info(output_dtype)[1]
                for i, name in enumerate(s.values):
                    addr, space = self.address(s.output, s.index, i)
                    value = self.value(name)
                    value_type = self.binding(name)[1]
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
                self.mutable.pop(s.name, None)
                start = len(self.lines)
                self.assign(s.name, s.op, s.args, s.dtype)
                if self.inactive and s.name in self.mutable:
                    for i in range(start, len(self.lines)):
                        instruction = self.lines[i].strip()
                        self.lines[i] = f'    @{self.inactive[-1]} {instruction}'
                self.rebind(s.name)
            elif kind == 'Let':
                operand, source_dtype = self.expr(s.value)
                target_dtype = source_dtype if s.dtype is None else s.dtype
                self.values[s.name] = self.snapshot_operand(operand, source_dtype, target_dtype)
                self.mutable.pop(s.name, None)
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
                self.emit(f'mov.{select_type(dtype)} {r}, {literal(s.identity, dtype)};')
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
                previous_axis = self.values.get(s.axis.name)
                previous_invariant_depth = len(self.loop_invariants)
                previous_expression_depth = len(self.loop_expressions)
                try:
                    # Seeds apply to accumulators owned immediately by this loop.
                    # Nested reductions seed themselves when their loop is entered.
                    if s.seed:
                        self.init_accums(s.body)
                    loop_init_frame = self.init_scope_frame(s.body)
                    extent = static(s.axis.extent)
                    if kind == 'Loop' and self.ordered_key_batch(s):
                        continue
                    if kind == 'Loop' and 8 <= extent <= I32_MAX and self.shared_compute_body(s):
                        factor = 8
                        vector_plan = self.shared_vector_plan(s)
                        limit = extent - extent % factor
                        r = self.reg('r')
                        self.emit(f'mov.s32 {r}, 0;')
                        self.set_integer_range(r, checked_i32_interval(0, limit - factor))
                        self.values[s.axis.name] = (r, 'i32')
                        self.rebind(s.axis.name)
                        # Preserve the existing pure-index hoist proof and its
                        # scoped frames. No body-defined name or axis is invariant.
                        self.hoist_invariants(s.body, defined_names(s.body) | {s.axis.name})
                        begin, done = self.label('unroll_loop'), self.label('unroll_done')
                        self.mark(begin)
                        p = self.reg('p')
                        self.emit(f'setp.ge.s32 {p}, {r}, {limit};')
                        self.emit(f'@{p} bra {done};')
                        for offset in range(factor):
                            axis = r
                            if offset:
                                axis = self.reg('r')
                                self.emit(f'add.s32 {axis}, {r}, {offset};')
                                self.set_integer_range(axis, checked_i32_interval(offset, limit - factor + offset))
                            self.values[s.axis.name] = (axis, 'i32')
                            self.rebind(s.axis.name)
                            if offset % 4 == 0:
                                prefetched = self.shared_vector_prefetch(vector_plan)
                            bindings = {position: registers[offset % 4]
                                        for position, registers in prefetched.items()}
                            self.statements_with_init_scope(s.body, loop_init_frame, bindings)
                        self.emit(f'add.s32 {r}, {r}, {factor};')
                        self.rebind(s.axis.name)
                        self.emit(f'bra {begin};')
                        self.mark(done)
                        self.set_integer_range(r, checked_i32_interval(limit, limit))
                        for coordinate in range(limit, extent):
                            self.values[s.axis.name] = (str(coordinate), 'i32')
                            self.rebind(s.axis.name)
                            self.statements_with_init_scope(s.body, loop_init_frame)
                        self.emit(f'mov.s32 {r}, {extent};')
                        self.set_integer_range(r, checked_i32_interval(extent, extent))
                        continue
                    r = self.reg('r')
                    start, step, end = ('0', '1', str(static(s.axis.extent)))
                    live_step, step_header_values = False, None
                    if kind == 'StridedLoop':
                        end_names = set(s.end.free_vars()) if s.end is not None else set()
                        if s.axis.name in end_names:
                            raise Unsupported('StridedLoop end header scope for its induction axis is unsupported')
                        step_names = set(s.step.free_vars())
                        live_step = bool(step_names & (defined_names(s.body) | {s.axis.name}))
                        start, _ = self.expr(s.start)
                        if not live_step:
                            step, _ = self.expr(s.step)
                        if s.end is not None:
                            end, end_dtype = self.expr(s.end)
                            if end_names & (defined_names(s.body) | {s.axis.name}):
                                # The header captures its bound once, even when a
                                # collective later overwrites the source carrier.
                                end, _ = self.convert_operand(end, end_dtype, 'i32')
                        if live_step:
                            # Capture lexical bindings, not runtime values. Outer carried
                            # registers remain live across body updates and skipped paths.
                            step_header_values = dict(self.values)
                    self.emit(f'mov.s32 {r}, {start};')
                    # Private plain induction cannot wrap before exit. A body
                    # that writes its axis (including a collective carrier) is
                    # conservatively left without a persistent induction fact.
                    if kind == 'Loop' and s.axis.name not in defined_names(s.body) and 0 < extent <= I32_MAX:
                        self.set_integer_range(r, checked_i32_interval(0, extent - 1))
                    elif kind == 'StridedLoop' and s.axis.name not in defined_names(s.body):
                        # Only immutable typed-I32 constants establish this fact.
                        # The final add also stays signed32: the loop cannot wrap
                        # back into its body after the first exit comparison.
                        def i32_constant(expression):
                            if (type(expression).__name__ == 'Literal'
                                    and isinstance(expression.value, (bool, int))
                                    and expression.dtype in ('int', 'bool', 'i32')
                                    and I32_MIN <= int(expression.value) <= I32_MAX):
                                return int(expression.value)
                            return None
                        lo, stride = i32_constant(s.start), i32_constant(s.step)
                        hi = extent if s.end is None else i32_constant(s.end)
                        if (lo is not None and stride is not None and hi is not None
                                and 0 <= lo < hi <= I32_MAX and stride > 0
                                and hi - 1 + stride <= I32_MAX):
                            self.set_integer_range(r, checked_i32_interval(lo, hi - 1))
                    self.values[s.axis.name] = (r, 'i32')
                    self.rebind(s.axis.name)
                    self.hoist_invariants(s.body, defined_names(s.body) | {s.axis.name})
                    begin, done = self.label('loop'), self.label('loop_done')
                    self.mark(begin)
                    p = self.reg('p')
                    self.emit(f'setp.ge.s32 {p}, {r}, {end};')
                    self.emit(f'@{p} bra {done};')
                    self.statements_with_init_scope(s.body, loop_init_frame)
                    if live_step:
                        step, _ = self.live_loop_step(s.step, step_header_values, s.axis.name, r)
                    self.emit(f'add.s32 {r}, {r}, {step};')
                    self.rebind(s.axis.name)
                    self.emit(f'bra {begin};')
                    self.mark(done)
                    self.loop_invariants.pop()
                    self.loop_expressions.pop()
                finally:
                    # Loop induction names are lexical, including empty loops and shadowed Tile axes.
                    # Preserve data/mutable facts computed by the body; only the axis binding is restored.
                    del self.loop_invariants[previous_invariant_depth:]
                    del self.loop_expressions[previous_expression_depth:]
                    if previous_axis is None:
                        self.values.pop(s.axis.name, None)
                    else:
                        self.values[s.axis.name] = previous_axis
                    self.rebind(s.axis.name)
            elif kind == 'WarpShuffle':
                if s.length > 32 or s.length <= 0 or s.length & (s.length-1):
                    raise Unsupported('Shuffle width must be a power of two <=32')
                dtype = dtype_name(s.dtype) if s.dtype is not None else 'f32'
                if dtype not in ('f32', 'i32', 'u32'):
                    raise Unsupported(f'Shuffle dtype {dtype} needs a 64-bit shuffle')
                if self.block_threads % s.length:
                    raise Unsupported('Shuffle requires complete physical subgroups')
                # Collective state is updated in place, including across skipped
                # enclosing Cond/Loop paths. Capture before padding can allocate.
                carried_values = {name: self.values[name] for name in s.state}
                saved_inactive = self.inactive
                if saved_inactive:
                    self.pad_partial_shuffle(s, saved_inactive[-1])
                    for name, (stable, original_type) in carried_values.items():
                        padded, padded_type = self.values[name]
                        if padded_type != original_type:
                            raise Unsupported('Shuffle padding changed the state dtype')
                        if padded != stable:
                            self.emit(f'mov.{select_type(original_type)} {stable}, {padded};')
                        self.values[name] = (stable, original_type)
                        if name in self.mutable:
                            self.mutable[name] = stable
                        if original_type == 'i32':
                            self.set_integer_range(stable, None)
                        self.rebind(name)
                    self.inactive = []
                try:
                    mask = self.reg('r')
                    self.emit(f'activemask.b32 {mask};')
                    offset = s.length // 2
                    while offset:
                        # A mutable Assign in the combine may overwrite its
                        # carrier. Preserve the absent-peer fallback beforehand.
                        prior = {name: self.convert_operand(*self.values[name],
                                                          self.values[name][1])
                                 for name in s.state}
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
                        self.emit(f'xor.b32 {partner}, {self.hardware_lane}, {offset};')
                        self.emit(f'shr.u32 {bit}, {mask}, {partner};')
                        self.emit(f'and.b32 {bit}, {bit}, 1;')
                        self.emit(f'setp.ne.u32 {peer_active}, {bit}, 0;')
                        self.statements(s.combine_states)
                        for name in s.state:
                            merged, merged_type = self.values[name]
                            before, before_type = prior[name]
                            if merged_type != before_type:
                                raise Unsupported('Shuffle combine changed the state dtype')
                            stable, original_type = carried_values[name]
                            if merged_type != original_type:
                                raise Unsupported('Shuffle changed the carrier dtype')
                            self.emit(f'selp.{select_type(merged_type)} {stable}, {merged}, '
                                      f'{before}, {peer_active};')
                            self.values[name] = (stable, merged_type)
                            if name in self.mutable:
                                self.mutable[name] = stable
                            if merged_type == 'i32':
                                self.set_integer_range(stable, None)
                            self.rebind(name)
                        offset //= 2
                finally:
                    self.inactive = saved_inactive
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
                self.statements_with_init_scope(s.body, self.init_scope_frame(s.body))
                self.values = dict(outer_values)
                self.mutable = dict(outer_mutable)
                self.address_cache.clear()
                self.expr_cache.clear()
                self.emit(f'bra {done};')
                self.mark(otherwise)
                self.statements_with_init_scope(s.else_body, self.init_scope_frame(s.else_body))
                self.mark(done)
                self.values = dict(outer_values)
                self.mutable = dict(outer_mutable)
                self.address_cache.clear()
                self.expr_cache.clear()
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
                saved_tree_inactive = self.inactive
                if saved_tree_inactive:
                    self.validate_partial_tree(s)
                    self.inactive = []
                try:
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
                    # An enclosing mutable carrier must survive skipped control flow.
                    carried_values = {a: self.values.get(a) for a in s.state}
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
                            predication = ''
                            if saved_tree_inactive and s.inner is None:
                                writer = self.reg('p')
                                self.emit(f'setp.eq.s32 {writer}, {self.hardware_lane}, 0;')
                                predication = f'@{writer} '
                            self.emit(f'{predication}st.shared.{dtype_info(dtype)[1]} [{addr}], {self.value(a)};')
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
                        carried = carried_values[a]
                        if carried is None:
                            r = self.reg(reg_suffix(dtype))
                        else:
                            r, carried_dtype = carried
                            if carried_dtype != dtype:
                                raise Unsupported('Tree carrier and shared buffer dtypes differ')
                        self.emit(f'ld.shared.{dtype_info(dtype)[1]} {r}, [{addr}];')
                        if dtype == 'i32':
                            self.set_integer_range(r, None)
                        self.values[a] = (r, dtype)
                        if a in self.mutable:
                            self.mutable[a] = r
                        self.rebind(a)
                finally:
                    self.inactive = saved_tree_inactive
            else:
                raise Unsupported(f'Unsupported statement {kind}')

    def render(self):
        self.key_batches = self.find_key_batches(self.node.op.body)
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
        return '\n'.join(['.version 8.0', '.target sm_89', '.address_size 64', '', *self.externals, f'.visible .entry {ident(self.node.op.name)}(', ',\n'.join(params), ')', '{', *decl, *self.lines, 'kernel_done:', '    ret;', '}', ''])


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
    return Emitter(graph, node, matches[0]).render()


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
