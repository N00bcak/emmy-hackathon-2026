#!/usr/bin/env python3
"""Small direct typed Kernel IR -> PTX compiler, intentionally conservative.

No emitter, renderer, or external compiler performs the IR lowering. ptxas may
assemble the resulting PTX separately. Unsupported constructs fail explicitly.
"""
from __future__ import annotations
import dataclasses
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
    uniform_cache = {}
    def uniform_loop(statement):
        # This predicate depends only on the immutable typed statement tree.
        # Cache within this validation call; participation ancestry is separate.
        key = id(statement)
        if key not in uniform_cache:
            uniform_cache[key] = prove_uniform_loop(statement)
        return uniform_cache[key]
    def prove_uniform_loop(statement):
        kind = type(statement).__name__
        if kind not in ('Loop', 'StridedLoop'):
            return False
        axis = statement.axis.name
        known = frozenset(('Load', 'Write', 'Assign', 'Accum', 'Init', 'Let',
                           'Select', 'Smem', 'CpAsyncCopy', 'CpAsyncCommit',
                           'CpAsyncWait', 'Sync', 'Loop', 'StridedLoop',
                           'Cond', 'WarpShuffle', 'TreeHalve'))
        def preserves_axis(items):
            for child in items:
                child_kind = type(child).__name__
                if child_kind not in known:
                    return False
                # Every descendant must terminate for padding physical lanes,
                # even if its own body contains no barrier.
                if child_kind in ('Loop', 'StridedLoop') and not uniform_loop(child):
                    return False
                if (getattr(child, 'name', None) == axis
                        or axis in getattr(child, 'names', ())
                        or axis in getattr(child, 'state', ())
                        or axis in getattr(child, 'state_b', ())
                        or getattr(getattr(child, 'axis', None), 'name', None) == axis):
                    return False
                if not all(preserves_axis(body) for body in statement_bodies(child)):
                    return False
            return True
        if not preserves_axis(statement.body):
            return False
        extent = static(statement.axis.extent)
        if not 0 < extent <= I32_MAX:
            return False
        if kind == 'Loop':
            return True
        def integer_literal(expr):
            if (type(expr).__name__ == 'Literal' and type(expr.value) is int
                    and expr.dtype in ('int', 'i32')
                    and I32_MIN <= expr.value <= I32_MAX):
                return expr.value
            return None
        start, step = integer_literal(statement.start), integer_literal(statement.step)
        end = extent if statement.end is None else integer_literal(statement.end)
        return (start is not None and step is not None and end is not None
                and 0 <= start < end <= extent and step > 0
                and end - 1 + step <= I32_MAX)
    def visit(items, direct, uniform):
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
                if not uniform or statement.warp or statement.count is not None:
                    raise Unsupported('Partial collective Tile has an unproven barrier')
                continue
            for body in statement_bodies(statement):
                visit(body, False, uniform and uniform_loop(statement))
    visit(statements, True, True)

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


def cta_cooperative(statements):
    """True when threads of a CTA interact: barriers, shuffles, tree
    reductions, shared memory or asynchronous copies anywhere in the tree."""
    for statement in statements:
        if type(statement).__name__ in ('WarpShuffle', 'TreeHalve', 'Sync', 'Smem', 'CpAsyncCopy',
                                        'CpAsyncCommit', 'CpAsyncWait'):
            return True
        if any(cta_cooperative(nested) for nested in statement_bodies(statement)):
            return True
    return False


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


def substitute_axis(value, mapping):
    """Rebuild a statement tree with mapped ``Var``s substituted. A mapped
    name used as a bare string is not understood, so it raises."""
    if isinstance(value, _typed_expr._ExprOps):
        return value.substitute(mapping)
    if isinstance(value, str):
        if value in mapping:
            raise Unsupported(f'Axis {value} referenced outside an expression')
        return value
    if isinstance(value, tuple):
        rebuilt = [substitute_axis(item, mapping) for item in value]
        return type(value)(rebuilt) if type(value) is not tuple else tuple(rebuilt)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        if type(value).__name__ in ('Axis', 'Dim', 'DataType', 'Window', 'ElementwiseImpl'):
            if type(value).__name__ == 'Axis' and value.name in mapping:
                raise Unsupported(f'Axis {value.name} rebound inside the Tile')
            return value
        changed = {}
        for f in dataclasses.fields(value):
            if not f.init:
                continue
            old = getattr(value, f.name)
            new = substitute_axis(old, mapping)
            if new is not old:
                changed[f.name] = new
        return dataclasses.replace(value, **changed) if changed else value
    return value


def unique_retile_writes(tile, meta, names, extents):
    """Prove that changing lane ownership cannot reorder colliding writes.

    This intentionally accepts only direct scalar writes with affine Tile-axis
    coordinates. Exhaustive static enumeration is bounded; large/opaque Tiles
    keep the original lowering.
    """
    writes = [item for item in tile.body if type(item).__name__ == 'Write']
    if (not writes or any(type(item).__name__ != 'Write'
                          and written_buffers((item,)) for item in tile.body)
            or math.prod(extents) * len(writes) > 262144):
        return False
    outputs = {arg['name']: tuple(arg['shape']) for arg in meta['args']
               if arg.get('output') and all(isinstance(d, int) and d > 0
                                            for d in arg['shape'])}
    definitions = []
    axis_terms = {('var', name) for name in names}
    for write in writes:
        shape = outputs.get(write.output)
        if (shape is None or write.atomic or len(write.values) != 1
                or len(write.index) != len(shape)):
            return False
        coords = []
        for part in write.index:
            form = affine_form(part)
            if form is None or not set(form[1]) <= axis_terms:
                return False
            coords.append(form)
        definitions.append((write.output, shape, coords))
    seen = {name: set() for name in outputs}
    # Decode each Tile point in the same row-major order as the generic Tile.
    for flat in range(math.prod(extents)):
        values, remainder = {}, flat
        for name, extent in reversed(list(zip(names, extents))):
            values[name], remainder = remainder % extent, remainder // extent
        for output, shape, coords in definitions:
            linear = 0
            for index, (constant, terms) in enumerate(coords):
                coordinate = constant + sum(coef * values[term[1]]
                                            for term, coef in terms.items())
                if coordinate < 0 or coordinate >= shape[index]:
                    return False
                linear += coordinate * math.prod(shape[index + 1:])
            if linear in seen[output]:
                return False
            seen[output].add(linear)
    return True


def retile_redundant_hoists(node, meta):
    """Reorder a Tile's axes so each CTA owns whole rows of a hoisted reduction
    (e.g. attention's QK^T scores) instead of recomputing rows another CTA also
    needs; one axis may be split as hi * f + lo. Only for a non-cooperative Tile
    that exactly fills the grid, where any cell-to-thread order is equivalent."""
    body = node.op.body
    if len(body) != 1 or type(body[0]).__name__ != 'Tile':
        return node
    tile = body[0]
    # Without raster_group, raster_axes is only a tag; the decode is row-major.
    if tile.aux_threads or tile.raster_group is not None or cta_cooperative(tile.body):
        return node
    # Reordering independent cells is legal only if global writes cannot
    # affect any load, and the Tile has no atomic side effect.
    def accesses(items):
        for item in items:
            yield item
            for nested in statement_bodies(item):
                yield from accesses(nested)
    items = list(accesses(tile.body))
    writes = [item for item in items if type(item).__name__ == 'Write']
    if (not writes or any(item.atomic for item in writes)
            or any(type(item).__name__ == 'Load' and item.input in written_buffers(tile.body)
                   for item in items)):
        return node
    names = [axis.name for axis in tile.axes]
    extents = [static_or_none(axis.extent) for axis in tile.axes]
    launched, grid = math.prod(meta['block']), math.prod(meta['grid'])
    if (None in extents or len(set(names)) != len(names) or any(e <= 0 for e in extents)
            or any(d != 1 for d in (*meta['block'][1:], *meta['grid'][1:]))
            or tile.block_threads not in (None, launched)
            or math.prod(extents) != launched * grid
            or not unique_retile_writes(tile, meta, names, extents)):
        return node
    axis_extent = dict(zip(names, extents))
    dependent = set()
    for outer in tile.body:
        if type(outer).__name__ != 'Loop':
            continue
        for inner in outer.body:
            if (type(inner).__name__ != 'Loop' or not inner.seed or not straight_line(inner.body)
                    or not any(type(x).__name__ == 'Accum' for x in inner.body)):
                continue
            free = touched_names(inner.body) - defined_names(inner.body) - {inner.axis.name}
            if free <= set(names) | {outer.axis.name}:
                dependent |= free & set(names)
    independent = [n for n in names if n not in dependent]
    if not dependent or not independent:
        return node
    cuts = [i for i in range(len(extents) + 1) if math.prod(extents[i:]) == launched]
    if cuts and all(n in names[cuts[-1]:] for n in independent):
        return node  # already no cross-CTA recompute
    inner_size = math.prod(axis_extent[n] for n in independent)
    if launched % inner_size:
        return node
    want = launched // inner_size
    order = [n for n in names if n in dependent]
    lo, size, split = [], 1, None
    for name in reversed(order):
        if size == want:
            break
        extent = axis_extent[name]
        if size * extent <= want and want % (size * extent) == 0:
            lo.insert(0, name)
            size *= extent
            continue
        factor = want // size
        if want % size or extent % factor:
            return node
        split = (name, extent // factor, factor)
        size = want
        break
    if size != want:
        return node
    by_name = {axis.name: axis for axis in tile.axes}
    hi = [n for n in order if n not in lo and (split is None or n != split[0])]
    axes, mapping = [by_name[n] for n in hi], {}
    if split is not None:
        name, upper, factor = split
        taken = touched_names(tile.body) | set(names)
        high, low = f'{name}__hi', f'{name}__lo'
        if high in taken or low in taken:
            return node
        Dim = type(by_name[name].extent)
        axes.append(dataclasses.replace(by_name[name], name=high, extent=Dim(upper), window=None))
        axes.append(dataclasses.replace(by_name[name], name=low, extent=Dim(factor), window=None))
        mapping[name] = _typed_expr.Var(high) * _typed_expr.Literal(factor, 'int') + _typed_expr.Var(low)
    axes += [by_name[n] for n in lo] + [by_name[n] for n in independent]
    try:
        new_body = substitute_axis(tile.body, mapping) if mapping else tile.body
    except Unsupported:
        return node
    retiled = dataclasses.replace(tile, axes=tuple(axes), body=new_body, raster_axes=None)
    op = dataclasses.replace(node.op, body=type(body)([retiled]))
    return dataclasses.replace(node, op=op)


class AccessLane:
    """One element of a proven contiguous f32 access group."""
    __slots__ = ('position', 'statement', 'element', 'lane', 'name')

    def __init__(self, position, statement, element, lane, name):
        self.position = position
        self.statement = statement
        self.element = element
        self.lane = lane
        self.name = name


class AccessDescriptor:
    """A typed 128-bit access with affine contiguous source offsets.

    ``terms`` are the flattened index's non-constant affine terms and
    ``constant`` is its constant offset. ``reads`` and ``writes`` count the
    accesses to each lane in source order. Shared bases have a static
    alignment; global bases use a runtime predicate and retain the scalar
    fallback.
    """
    __slots__ = ('kind', 'buffer', 'space', 'dtype', 'terms', 'constant',
                 'alignment', 'static_aligned', 'reads', 'writes', 'lanes')

    def __init__(self, kind, buffer, space, dtype, terms, constant, alignment,
                 static_aligned, reads, writes, lanes):
        self.kind = kind
        self.buffer = buffer
        self.space = space
        self.dtype = dtype
        self.terms = terms
        self.constant = constant
        self.alignment = alignment
        self.static_aligned = static_aligned
        self.reads = reads
        self.writes = writes
        self.lanes = lanes

    @property
    def width(self):
        return len(self.lanes)


class AccessPlan:
    """Block-local descriptors and loaded vector registers."""

    def __init__(self):
        self.loads = {}
        self.stores = {}
        self.load_lanes = {}
        self.store_lanes = {}
        self.registers = {}


def _terms_key(terms):
    return tuple(sorted(terms.items(), key=lambda item: repr(item[0])))


def access_descriptors(statements, buffers, shared, shared_alignment=None, blocked=()):
    """Prove f32 four-element contiguous groups in one straight-line block.

    The proof is intentionally local to a statement list. Elements must be
    adjacent uses in source order and have the same affine non-constant terms;
    only that constant offset may differ, by exactly the lane number. A
    destination may name an index operand only if every lane address is
    evaluated before any destination is published, so the lowering keeps that
    ordering. No statement is moved.
    """
    plan = AccessPlan()
    written = written_buffers(statements)
    bindings = {}
    for statement in statements:
        kind = type(statement).__name__
        if kind == 'Load':
            source_dtype = dtype_name(buffers[statement.input][1]) if statement.input in buffers else 'f32'
            dtype = dtype_name(statement.dtype) if statement.dtype is not None else source_dtype
            for name in statement.names:
                bindings[name] = dtype
        elif kind in ('Assign', 'Accum'):
            bindings[statement.name] = ssa_dtype_name(statement.dtype) if statement.dtype is not None else 'f32'
        elif kind == 'Let':
            bindings[statement.name] = ssa_dtype_name(statement.dtype) if statement.dtype is not None else 'f32'
    runs = {}
    blocked = set(blocked)
    for position, statement in enumerate(statements):
        kind = type(statement).__name__
        if kind not in ('Load', 'Write'):
            continue
        if id(statement) in blocked:
            continue
        if kind == 'Load':
            buffer, index, names = statement.input, statement.index, statement.names
            if len(names) not in (1, 4):
                continue
            elements = [(position, statement, names[element], element)
                        for element in range(len(names))]
        else:
            if statement.atomic or statement.swizzle not in ('NONE', None):
                continue
            buffer, index, names = statement.output, statement.index, statement.values
            if len(names) != 4:
                continue
            elements = [(position, statement, names[element], element)
                        for element in range(len(names))]
        if buffer not in buffers:
            continue
        shape, storage_dtype = buffers[buffer]
        storage_dtype = dtype_name(storage_dtype)
        if storage_dtype != 'f32':
            continue
        if kind == 'Load':
            if statement.dtype is not None and dtype_name(statement.dtype) != 'f32':
                continue
        else:
            value_dtype = (ssa_dtype_name(statement.value_dtype)
                           if statement.value_dtype is not None else storage_dtype)
            if value_dtype != 'f32':
                continue
            if any(bindings.get(name) not in ('f32', 'i32', 'u32', 'i64', 'u64') for name in names):
                continue
        if kind == 'Load' and buffer in written:
            continue
        if len(set(names)) != len(names):
            continue
        form = flat_affine(shape, index)
        if form is None:
            continue
        constant, terms = form
        if any(coefficient % 4 for coefficient in terms.values()):
            continue
        if any(part is None for part in (expr_names(coord) for coord in index)):
            continue
        key = (kind, buffer, _terms_key(terms))
        group = runs.setdefault(key, [])
        group.append((constant, index, elements, statement))

    for (kind, buffer, terms_key), members in runs.items():
        entries = {}
        for constant, index, elements, statement in members:
            for position, lane_statement, name, element in elements:
                entries.setdefault(constant + element,
                                   (position, lane_statement, constant, element, name, index))
        used = set()
        ordered = sorted(entries)
        for offset in ordered:
            if offset in used:
                continue
            run = [entries.get(offset + lane) for lane in range(4)]
            if any(member is None for member in run) or any(offset + lane in used for lane in range(4)):
                continue
            if offset % 4:
                continue
            # A load group may span pure statements, but no index operand or
            # lane destination may be rebound while later lanes are moved to
            # the leader. A destination is also live from its original load
            # onward, so an intervening read of it cannot be moved past that
            # definition. (Reads before the destination's own load are fine.)
            first, last = run[0][0], run[-1][0]
            intervening = statements[first + 1:last]
            if any(type(statement).__name__ not in ('Load', 'Let', 'Assign', 'Accum')
                   for statement in intervening):
                continue
            # Moving a later load to its run leader is only effect-free when
            # nothing between them can write a buffer that might alias it.
            # Tuple stores are single statements and do not cross anything.
            if kind == 'Load' and written_buffers(intervening):
                continue
            index_names = set()
            for member in run:
                for coord in member[5]:
                    index_names |= expr_names(coord)
            lane_ids = {id(member[1]) for member in run}
            intervening_names = defined_names(
                statement for statement in statements[first + 1:last]
                if id(statement) not in lane_ids)
            names = tuple(member[4] for member in run)
            if (index_names & (intervening_names | set(names))
                    or set(names) & intervening_names):
                continue
            if kind == 'Load':
                # Lanes bind at the leader. An intervening read of a name whose
                # lane definition comes later would observe the pre-group
                # binding instead, so reject that interleave. Reads after the
                # name's own lane are unchanged by moving later lanes up.
                lane_positions = {member[4]: member[0] for member in run}
                if any(lane_positions[name] > position
                       for position in range(first + 1, last)
                       for name in statements[position].deps()
                       if name in lane_positions):
                    continue
            if len(set(names)) != 4:
                continue
            lanes = []
            prior_position = -1
            for lane_number, member in enumerate(run):
                position, lane_statement, constant, element, name, _ = member
                lane_constant = constant + element
                if lane_constant != offset + lane_number:
                    break
                if position < prior_position or (lane_number and position == prior_position
                                                 and lane_statement is not run[lane_number - 1][1]):
                    break
                prior_position = position
                lanes.append(AccessLane(position, lane_statement, element,
                                        lane_number, name))
            if len(lanes) != 4:
                continue
            leader = lanes[0].statement
            statements_used = {id(lane.statement) for lane in lanes}
            if statements_used & blocked:
                continue
            if kind == 'Write' and any(lane.statement is not leader for lane in lanes[1:]):
                continue
            space = 'shared' if buffer in shared else 'global'
            static_aligned = False
            if space == 'shared':
                base_offset = shared_alignment.get(buffer, 0) if shared_alignment else 0
                if (base_offset + 4 * run[0][2]) % 16 != 0:
                    continue
                static_aligned = True
            descriptor = AccessDescriptor(
                kind, buffer, space, 'f32', terms_key, run[0][2], 16,
                static_aligned,
                tuple(1 if kind == 'Load' else 0 for _ in lanes),
                tuple(1 if kind == 'Write' else 0 for _ in lanes), lanes)
            leader_id = id(leader)
            if kind == 'Load':
                plan.loads[leader_id] = descriptor
                for lane in lanes[1:]:
                    plan.load_lanes[id(lane.statement)] = leader_id
            else:
                plan.stores[leader_id] = descriptor
            used.update(offset + lane for lane in range(4))
    return plan


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


VN_KINDS = frozenset(('Load', 'Let', 'Assign', 'Accum'))


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
    # These rewrites require unique scalar bindings. General IR permits
    # lexical shadowing and mutation, which stay on the scoped fallback.
    if not pure:
        return {}, {}, {}
    bindings = []
    for statement in body:
        if id(statement) in opaque:
            # A hoisted loop exports its accumulators, not its private axis
            # and load temporaries. Repeated private axis names are lexical.
            if type(statement).__name__ == 'Loop':
                bindings.extend(dict.fromkeys(x.name for x in statement.body
                                              if type(x).__name__ == 'Accum'))
            elif type(statement).__name__ == 'Load':
                bindings.extend(statement.names)
            else:
                return {}, {}, {}
        elif type(statement).__name__ == 'Load':
            bindings.extend(statement.names)
        else:
            bindings.append(statement.name)
    if len(bindings) != len(set(bindings)):
        return {}, {}, {}
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
                base = ('load', s.input, str(s.dtype), index) if pure else ('opaque', id(s))
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


SIBLING_WORK_BATCH_MAX_VALUES = 64

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
        self.hoist_plans = {}
        self.hoisted = {}
        self.hoist_bytes = 0
        self.vn_skip, self.vn_record, self.vn_registers = {}, {}, {}
        self.vector_loops = 0
        self.tile_extents = {}
        self.subst = {}
        self.access_plan = AccessPlan()
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
                if at == bt == 'pred' and x.op in ('<', '<=', '>', '>='):
                    # Ordered BOOL comparisons use numeric zero/one operands;
                    # PTX setp has no predicate source-operand form.
                    a = convert_register(self, a, 'pred', 'i32')
                    b = convert_register(self, b, 'pred', 'i32')
                    at = bt = 'i32'
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
        typed = [self.binding(a) for a in args]
        operation = reduce_canon(getattr(op, 'name', op))
        argument_types = [t for _, t in typed]
        result = ('pred' if getattr(dtype, 'name', dtype) == 'bool' else dtype_name(dtype)) if dtype is not None else infer_assign_dtype(operation, argument_types)
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
        if result == 'pred':
            # Public BOOL scalar results promote their BOOL inputs to F32,
            # evaluate the ordinary op, then convert back to truth once.
            # Numeric-to-BOOL folds have a different upstream conversion law.
            if not typed or any(t != 'pred' for t in argument_types):
                raise Unsupported('BOOL assignments require all-BOOL operands')
            if operation == 'bitcast':
                raise Unsupported('Predicate bitcasts unsupported')
            if operation == 'copy' and len(typed) == 1:
                register, result = self.snapshot_operand(*typed[0], 'pred')
            else:
                register, actual = self.emit_operation(op, typed, 'f32')
                register, result = self.snapshot_operand(register, actual, 'pred')
            native_computation = False
        elif operation == 'bitcast':
            if dtype is None:
                raise Unsupported('Bitcast requires explicit destination dtype')
            register, result = self.emit_operation(op, typed, result)
        elif operation == 'copy':
            # Copy is a numeric conversion of one value, with no arithmetic rounding step.
            # This deliberately preserves typed native integer/wide copies.
            register, result = self.emit_operation(op, typed, result)
        elif result == 'f32' or (native_op and (all_at_result or integer_promotion)):
            if (operation == 'reciprocal' and result == 'f32' and len(args) == 1
                    and args[0] == getattr(self, 'softmax_pair_sum_name', None)
                    and getattr(self, 'allow_softmax_approx_rcp', False)):
                denominator = typed[0][0]
                normal, finite, fast = self.reg('p'), self.reg('p'), self.reg('p')
                fallback, joined = self.label('softmax_rcp_exact'), self.label('softmax_rcp_join')
                register = self.reg('f')
                self.emit(f'setp.ge.f32 {normal}, {denominator}, {f32(2.0**-126)};')
                self.emit(f'setp.lt.f32 {finite}, {denominator}, {f32(float("inf"))};')
                self.emit(f'and.pred {fast}, {normal}, {finite};')
                self.emit(f'@!{fast} bra {fallback};')
                self.emit(f'rcp.approx.f32 {register}, {denominator};')
                self.emit(f'bra {joined};')
                self.mark(fallback)
                self.emit(f'rcp.rn.f32 {register}, {denominator};')
                self.mark(joined)
            else:
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
            move_type = 'pred' if result == 'pred' else select_type(result)
            self.emit(f'mov.{move_type} {stable}, {register};')
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
        elif op in ('exp', 'exp_fast') and len(operands) == 1:
            if result != 'f32':
                raise Unsupported(f'{op} is only supported for f32')
            scaled = self.reg('f')
            self.emit(f'mul.f32 {scaled}, {operands[0]}, {f32(math.log2(math.e))};')
            # Keep exp2's operand/result normal, then restore tiny results.
            # The offset is exact in the only range with nonzero subnormals;
            # the final multiply deliberately does not use .ftz.
            tiny = self.reg('p')
            self.emit(f'setp.lt.f32 {tiny}, {scaled}, {f32(-125.0)};')
            self.emit(f'@{tiny} add.f32 {scaled}, {scaled}, {f32(126.0)};')
            self.emit(f'ex2.approx.ftz.f32 {reg}, {scaled};')
            self.emit(f'@{tiny} mul.rn.f32 {reg}, {reg}, {f32(2.0 ** -126)};')
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
            # log(1+y) must not first round tiny y away in the addition to1.
            # For0<=y<=1/8 the alternating degree6 series has relative
            # truncation error <=(1+y)*y**6/7, below6.14e-7. The threshold
            # is an analytic approximation interval, independent of inputs.
            small = self.reg('p')
            large, joined = self.label('softplus_log'), self.label('softplus_ready')
            self.emit(f'setp.le.f32 {small}, {exponent}, {f32(0.125)};')
            self.emit(f'@!{small} bra {large};')
            polynomial = self.reg('f')
            self.emit(f'fma.rn.f32 {polynomial}, {exponent}, {f32(-1.0 / 6.0)}, {f32(1.0 / 5.0)};')
            for coefficient in (-1.0 / 4.0, 1.0 / 3.0, -1.0 / 2.0, 1.0):
                self.emit(f'fma.rn.f32 {polynomial}, {exponent}, {polynomial}, {f32(coefficient)};')
            self.emit(f'mul.rn.f32 {logarithm}, {exponent}, {polynomial};')
            self.emit(f'bra {joined};')
            self.mark(large)
            self.emit(f'add.f32 {shifted}, {exponent}, {f32(1.0)};')
            self.emit(f'lg2.approx.f32 {logarithm}, {shifted};')
            self.emit(f'mul.f32 {logarithm}, {logarithm}, {f32(math.log(2.0))};')
            self.mark(joined)
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
        signed = self.reg('f')
        small = self.reg('p')
        large, joined = self.label('erf_large'), self.label('erf_ready')
        self.emit(f'setp.le.f32 {small}, {absolute}, {f32(0.5)};')
        self.emit(f'@!{small} bra {large};')
        # The odd series retains the linear term near zero. For z=x*x<=1/4,
        # the next alternating term gives relative truncation below2.85e-8.
        square_small, polynomial_small, linear = self.reg('f'), self.reg('f'), self.reg('f')
        self.emit(f'mul.rn.f32 {square_small}, {value}, {value};')
        self.emit(f'mov.f32 {polynomial_small}, {f32(-1.0 / 1320.0)};')
        for coefficient in (1.0 / 216.0, -1.0 / 42.0, 1.0 / 10.0, -1.0 / 3.0, 1.0):
            self.emit(f'fma.rn.f32 {polynomial_small}, {square_small}, {polynomial_small}, {f32(coefficient)};')
        self.emit(f'mul.rn.f32 {linear}, {value}, {f32(2.0 / math.sqrt(math.pi))};')
        self.emit(f'mul.rn.f32 {signed}, {linear}, {polynomial_small};')
        self.emit(f'bra {joined};')
        self.mark(large)
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
        self.emit(f'selp.f32 {signed}, {negated}, {magnitude}, {negative};')
        self.mark(joined)
        return signed

    @staticmethod
    def reduction_identity_literal(op, dtype):
        identities = {'add': 0.0, 'multiply': 1.0, 'maximum': -1e30, 'minimum': 1e30}
        if op not in identities:
            raise Unsupported(f'Unsupported reduction {op}')
        return literal(identities[op], dtype)

    def init_accums(self, statements):
        for s in statements:
            if type(s).__name__ == 'Accum':
                op = reduce_canon(s.op.name)
                dtype = dtype_name(s.dtype) if s.dtype is not None else 'f32'
                if dtype not in FLOAT_DTYPES:
                    raise Unsupported(f'Only floating reductions supported: {dtype}')
                reg = self.reg(reg_suffix(dtype))
                self.emit(f'mov.{select_type(dtype)} {reg}, {self.reduction_identity_literal(op, dtype)};')
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
        self.values[statement.name] = self.snapshot_operand(selected, 'f32', 'f32')
        self.mutable.pop(statement.name, None)
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
        # Cooperative precomputation has already removed duplicated score work.
        # Replaying those shared scalars through shuffles adds only overhead.
        if all(id(dot) in self.hoisted for dot in dots):
            return False
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

    def find_serial_read_plans(self, statements):
        """Recognize a closed readonly, uniform-trip scalar reduction stream."""
        if self.geometry_3d or len(statements) != 1 or type(statements[0]).__name__ != 'Tile':
            return {}
        tile = statements[0]
        axes = {axis.name: static(axis.extent) for axis in tile.axes}
        if (len(axes) != len(tile.axes) or tile.aux_threads
                or tile.block_threads != self.block_threads
                or math.prod(axes.values()) != self.block_threads * self.grid_blocks
                or set(axes) & defined_names(tile.body)):
            return {}
        known = {'Tile', 'StridedLoop', 'Loop', 'Load', 'Accum', 'Assign', 'Let',
                 'Init', 'WarpShuffle', 'TreeHalve', 'Cond', 'Sync', 'Smem', 'Write'}
        def walk(body):
            for statement in body:
                yield statement
                for nested in statement_bodies(statement):
                    yield from walk(nested)
        whole = list(walk(statements))
        if any(type(statement).__name__ not in known for statement in whole):
            return {}
        written = {statement.output for statement in whole if type(statement).__name__ == 'Write'}
        plans = {}
        for loop in tile.body:
            if (type(loop).__name__ != 'StridedLoop' or loop.unroll or loop.end is not None
                    or len(loop.body) != 2 or type(loop.start).__name__ != 'Var'
                    or type(loop.step).__name__ != 'Literal'
                    or type(loop.step.value) is not int or loop.step.dtype not in ('int', 'i32')):
                continue
            step, end = loop.step.value, static(loop.axis.extent)
            if (step <= 0 or axes.get(loop.start.name) != step or end <= 0
                    or end > I32_MAX or end % step or end + step - 1 > I32_MAX):
                continue
            trips = end // step
            if trips < 8 or loop.axis.name in axes:
                continue
            load, accum = loop.body
            if (type(load).__name__ != 'Load' or type(accum).__name__ != 'Accum'
                    or len(load.names) != 1 or load.dtype is None
                    or getattr(load.dtype, 'name', load.dtype) != 'f32'
                    or len(load.index) != 1 or type(load.index[0]).__name__ != 'Var'
                    or load.index[0].name != loop.axis.name
                    or load.input in self.shared or load.input in self.scalar_parameters
                    or load.input not in self.buffers or load.input in written):
                continue
            shape, dtype = self.buffers[load.input]
            if len(shape) != 1 or shape[0] < end or dtype_name(dtype) != 'f32':
                continue
            if (accum.dtype is None or getattr(accum.dtype, 'name', accum.dtype) != 'f32'
                    or accum.op.name != 'add' or accum.base is not None
                    or tuple(accum.axes) != (loop.axis.name,)
                    or accum.value != load.names[0]
                    or len({loop.axis.name, loop.start.name, load.names[0], accum.name}) != 4):
                continue
            # Conservative lexical inventory: any use or same-name declaration
            # outside this direct loop declines; no public load value escapes.
            outside = repr(tuple(statement for statement in tile.body if statement is not loop))
            if re.search(r'(?<![A-Za-z0-9_])' + re.escape(load.names[0]) + r'(?![A-Za-z0-9_])', outside):
                continue
            plans[id(loop)] = (step, trips, end, load, accum)
        return plans


    def find_rms_input_replay(self, tile, extents, launched):
        """Prove a complete eight-value F32 row reduction and later row map.

        Repartition each 1024-element row into two four-wide groups per CTA
        lane. Every output element keeps its original input values, while the
        reduction's summation order may differ at the unchanged final tolerance.
        """
        if (self.inactive or self.geometry_3d or tile.aux_threads
                or len(tile.axes) != 2 or len(extents) != 2
                or extents[1] != launched or tile.block_threads != launched
                or math.prod(extents) != launched * math.prod(self.meta['grid'])):
            return None
        loops = [item for item in tile.body if type(item).__name__ == 'StridedLoop']
        if len(loops) != 2 or tile.body[-1] is not loops[1]:
            return None
        first, second = loops
        if (first is second or not first.seed or not second.seed
                or first.unroll or second.unroll or first.end is not None or second.end is not None
                or first.axis.name == second.axis.name
                or first.axis.name in (tile.axes[0].name, tile.axes[1].name)
                or second.axis.name in (tile.axes[0].name, tile.axes[1].name)
                or static(first.axis.extent) != static(second.axis.extent)
                or static(first.axis.extent) != 8 * launched):
            return None
        for item in loops:
            if (type(item.start).__name__ != 'Var'
                    or item.start.name != tile.axes[1].name
                    or type(item.step).__name__ != 'Literal'
                    or type(item.step.value) is not int
                    or item.step.dtype not in ('int', 'i32')
                    or item.step.value != launched):
                return None
        one, two = list(first.body), list(second.body)
        if ([type(item).__name__ for item in one] != ['Load', 'Assign', 'Accum']
                or [type(item).__name__ for item in two]
                != ['Load', 'Assign', 'Load', 'Assign', 'Write']):
            return None
        x1, square, fold = one
        x2, scale, weight, product, write = two
        if (len(x1.names) != 1 or len(x2.names) != 1 or len(weight.names) != 1
                or x1.input != x2.input or x1.input not in self.ptrs
                or x1.input in self.shared or x1.input in self.scalar_parameters
                or x1.input in written_buffers(tile.body)
                or weight.input in written_buffers(tile.body)
                or weight.input not in self.ptrs or weight.input in self.shared
                or x1.input not in self.buffers or write.output not in self.ptrs
                or dtype_name(self.buffers[x1.input][1]) != 'f32'
                or dtype_name(self.buffers[weight.input][1]) != 'f32'
                or dtype_name(self.buffers[write.output][1]) != 'f32'
                or any(item.dtype is not None and dtype_name(item.dtype) != 'f32'
                       for item in (x1, x2, square, fold, scale, weight, product))
                or square.op.name != 'multiply'
                or tuple(square.args) != (x1.names[0], x1.names[0])
                or reduce_canon(fold.op.name) != 'add'
                or fold.value != square.name or fold.base is not None
                or tuple(fold.axes) != (first.axis.name,)
                or scale.op.name != 'multiply' or len(scale.args) != 2
                or x2.names[0] not in scale.args
                or product.op.name != 'multiply' or len(product.args) != 2
                or set(product.args) != {weight.names[0], scale.name}
                or write.atomic or write.swizzle not in ('NONE', None)
                or len(write.values) != 1 or write.values[0] != product.name
                or write.output in (x1.input, weight.input)
                or write.output not in {arg['name'] for arg in self.meta['args']
                                        if arg.get('output')}
                or (write.value_dtype is not None
                    and dtype_name(write.value_dtype) != 'f32')):
            return None
        shape = self.buffers[x1.input][0]
        wshape = self.buffers[weight.input][0]
        oshape = self.buffers[write.output][0]
        if (len(shape) != 2 or shape[0] < extents[0] or shape[1] != 8 * launched
                or len(wshape) != 1 or wshape[0] < 8 * launched
                or len(oshape) != 2 or oshape[0] < extents[0]
                or oshape[1] != 8 * launched):
            return None
        def row_column(index, axis):
            return (len(index) == 2
                    and all(type(part).__name__ == 'Var' for part in index)
                    and tuple(part.name for part in index)
                    == (tile.axes[0].name, axis))
        if (not row_column(x1.index, first.axis.name)
                or not row_column(x2.index, second.axis.name)
                or not row_column(write.index, second.axis.name)
                or len(weight.index) != 1
                or type(weight.index[0]).__name__ != 'Var'
                or weight.index[0].name != second.axis.name):
            return None
        # The second Load has exactly one consumer. The first Load's square
        # and fold remain in their original order; all Tile statements between
        # the loops, including WarpShuffle, shared tree and barriers, stay put.
        if (sum(x2.names[0] in item.deps() for item in two[1:]) != 1
                or any(x2.names[0] in item.deps() for item in two[2:])):
            return None
        return {'first': first, 'second': second, 'load1': x1, 'load2': x2,
                'scale': scale, 'weight': weight, 'product': product, 'write': write,
                'bank': None, 'step': launched, 'align': {}}

    def rms_replay_alignment(self, name, plan):
        predicate = plan['align'].get(name)
        if predicate is None:
            low, predicate = self.reg('rd'), self.reg('p')
            self.emit(f'and.b64 {low}, {self.ptrs[name]}, 15;')
            self.emit(f'setp.eq.u64 {predicate}, {low}, 0;')
            plan['align'][name] = predicate
        return predicate

    def rms_replay_vector_load(self, load, axis, base, plan):
        self.values[axis] = (base, 'i32')
        self.rebind(axis)
        address, space = self.address(load.input, load.index)
        if space != 'global':
            raise Unsupported('RMS row replay requires global inputs')
        registers = [self.reg('f') for _ in range(4)]
        predicate = self.rms_replay_alignment(load.input, plan)
        self.emit(f'@{predicate} ld.global.v4.f32 '
                  f'{{{", ".join(registers)}}}, [{address}];')
        for offset, register in enumerate(registers):
            self.emit(f'@!{predicate} ld.global.f32 {register}, '
                      f'[{address}+{4 * offset}];')
        return registers

    def rms_replay_vector_store(self, write, axis, base, values, plan):
        self.values[axis] = (base, 'i32')
        self.rebind(axis)
        address, space = self.address(write.output, write.index)
        if space != 'global':
            raise Unsupported('RMS row replay requires global output')
        predicate = self.rms_replay_alignment(write.output, plan)
        self.emit(f'@{predicate} st.global.v4.f32 [{address}], '
                  f'{{{", ".join(values)}}};')
        for offset, value in enumerate(values):
            self.emit(f'@!{predicate} st.global.f32 '
                      f'[{address}+{4 * offset}], {value};')

    def rms_input_replay_loop(self, loop, frame):
        plan = getattr(self, 'rms_input_replay', None)
        if plan is None or (loop is not plan['first'] and loop is not plan['second']):
            return False
        first = loop is plan['first']
        if not first and (plan['bank'] is None or len(plan['bank']) != 2):
            raise Unsupported('RMS row replay must follow its certified producer')
        start, dtype = self.expr(loop.start)
        if dtype != 'i32':
            raise Unsupported('RMS row replay requires an I32 tile coordinate')
        step = plan['step']
        groups = []
        for group in range(2):
            base = self.reg('r')
            self.emit(f'mad.lo.u32 {base}, {start}, 4, {group * 4 * step};')
            self.set_integer_range(base, checked_i32_interval(group * 4 * step,
                                                               (group + 1) * 4 * step - 1))
            if first:
                values = self.rms_replay_vector_load(
                    plan['load1'], loop.axis.name, base, plan)
            else:
                values = plan['bank'][group][1]
            groups.append((base, values))
        if first:
            # Both global requests precede the first dependent multiply.
            plan['bank'] = groups
        for group, (base, xvalues) in enumerate(groups):
            wvalues = None
            if not first:
                wvalues = self.rms_replay_vector_load(
                    plan['weight'], loop.axis.name, base, plan)
            output = []
            for offset, xvalue in enumerate(xvalues):
                axis = base
                if offset:
                    axis = self.reg('r')
                    self.emit(f'add.s32 {axis}, {base}, {offset};')
                self.set_integer_range(axis, checked_i32_interval(
                    group * 4 * step + offset,
                    (group + 1) * 4 * step - 4 + offset))
                self.values[loop.axis.name] = (axis, 'i32')
                self.rebind(loop.axis.name)
                load_name = plan['load1'].names[0] if first else plan['load2'].names[0]
                self.values[load_name] = (xvalue, 'f32')
                self.mutable.pop(load_name, None)
                self.rebind(load_name)
                if first:
                    self.statements_with_init_scope(loop.body[1:], frame)
                else:
                    weight_name = plan['weight'].names[0]
                    self.values[weight_name] = (wvalues[offset], 'f32')
                    self.mutable.pop(weight_name, None)
                    self.rebind(weight_name)
                    self.statements_with_init_scope(
                        (plan['scale'], plan['product']), frame)
                    output.append(self.value(plan['product'].name))
            if not first:
                self.rms_replay_vector_store(
                    plan['write'], loop.axis.name, base, output, plan)
        return True

    def serial_read_pipeline(self, loop):
        plan = self.serial_read_plans.get(id(loop))
        if plan is None or self.inactive:
            return False
        step, trips, end, load, accum = plan
        if (loop.start.name not in self.values or self.values[loop.start.name][1] != 'i32'
                or accum.name not in self.values or self.values[accum.name][1] != 'f32'
                or load.names[0] in self.mutable or load.input not in self.ptrs):
            return False
        start, dtype = self.expr(loop.start)
        if dtype != 'i32':
            return False
        current_axis = self.reg('r')
        self.emit(f'mov.s32 {current_axis}, {start};')
        self.set_integer_range(current_axis, checked_i32_interval(0, end - 1))
        # The typed serial plan has a single readonly 1D F32 global input,
        # complete Tile participation and an in-bounds uniform trip count.
        # Keep each original load and fold in order; only advance its address.
        pointer_bank = trips >= 16 and 0 < step and 7 * step * 4 < 4096
        if pointer_bank:
            start_byte = self.reg('rd')
            self.emit(f'mul.wide.s32 {start_byte}, {start}, 4;')
            stream_pointer = self.reg('rd')
            self.emit(f'add.u64 {stream_pointer}, {self.ptrs[load.input]}, {start_byte};')

        def load_at(offset):
            point = current_axis
            if offset:
                point = self.reg('r')
                self.emit(f'add.s32 {point}, {current_axis}, {offset};')
            self.set_integer_range(point, checked_i32_interval(0, end - 1))
            self.values[loop.axis.name] = (point, 'i32')
            self.rebind(loop.axis.name)
            if pointer_bank:
                value = self.reg('f')
                self.emit(f'ld.global.f32 {value}, [{stream_pointer}+{offset * 4}];')
                self.values[load.names[0]] = (value, 'f32')
                self.mutable.pop(load.names[0], None)
                self.rebind(load.names[0])
            else:
                self.statements((load,))
            value, dtype = self.values[load.names[0]]
            if dtype != 'f32':
                raise Unsupported('Serial prefetch requires an F32 load')
            return value

        def consume(bank):
            # Accum has no expression/address operand; only its original loaded
            # value changes. The named carrier performs the same four updates.
            self.values[loop.axis.name] = (current_axis, 'i32')
            self.rebind(loop.axis.name)
            for value in bank:
                self.values[load.names[0]] = (value, 'f32')
                self.mutable.pop(load.names[0], None)
                self.rebind(load.names[0])
                self.statements((accum,))

        # Every current-bank load is issued before its first use. Next-bank
        # registers are distinct and survive until the backedge move.
        current = [load_at(j * step) for j in range(4)]
        group = self.reg('r')
        self.emit(f'mov.s32 {group}, 0;')
        begin, drain = self.label('serial_prefetch'), self.label('serial_prefetch_drain')
        self.mark(begin)
        stop = self.reg('p')
        groups, remainder = divmod(trips, 4)
        self.emit(f'setp.ge.s32 {stop}, {group}, {groups - 1};')
        self.emit(f'@{stop} bra {drain};')
        future = [load_at((j + 4) * step) for j in range(4)]
        consume(current)
        for destination, value in zip(current, future, strict=True):
            self.emit(f'mov.f32 {destination}, {value};')
        self.emit(f'add.s32 {current_axis}, {current_axis}, {4 * step};')
        if pointer_bank:
            self.emit(f'add.u64 {stream_pointer}, {stream_pointer}, {4 * step * 4};')
        self.emit(f'add.s32 {group}, {group}, 1;')
        self.rebind(loop.axis.name)
        self.emit(f'bra {begin};')
        self.mark(drain)
        consume(current)
        for j in range(remainder):
            value = load_at((4 + j) * step)
            consume((value,))
        self.emit(f'add.s32 {current_axis}, {current_axis}, {(4 + remainder) * step};')
        self.values[loop.axis.name] = (current_axis, 'i32')
        self.set_integer_range(current_axis, checked_i32_interval(end, end + step - 1))
        self.rebind(loop.axis.name)
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

    def global_dot_vector_plan(self, loop):
        """Prove four existing readonly global dot reads form one vector group.

        Forms use actual i32 operands rather than SSA spelling. Every expression
        subtree and flattening intermediate is bounded in signed32 arithmetic.
        This analysis emits no instructions and never changes range/value state.
        """
        body, axis_name = loop.body, loop.axis.name
        if (id(loop) not in self.global_dot_candidates or self.inactive or not loop.seed
                or not 4 <= static(loop.axis.extent) <= I32_MAX
                or [type(s).__name__ for s in body] != ['Load', 'Load', 'Assign', 'Accum']):
            return {}
        first, second, product, update = body
        if (len(first.names) != 1 or len(second.names) != 1
                or product.op.name != 'multiply' or update.op.name != 'add'
                or update.value != product.name or update.base is not None
                or tuple(update.axes) != (axis_name,)
                or any(s.dtype is None or ssa_dtype_name(s.dtype) != 'f32'
                       for s in (product, update))):
            return {}
        loaded = (first.names[0], second.names[0])
        if any(name in self.mutable for name in (*loaded, product.name)):
            return {}
        if (len(set((*loaded, product.name, update.name, axis_name))) != 5
                or len(product.args) != 2 or set(product.args) != set(loaded)):
            return {}
        for load in (first, second):
            if (load.input in self.scalar_parameters or load.input not in self.buffers
                    or load.input in self.shared or load.input not in self.ptrs
                    or dtype_name(self.buffers[load.input][1]) != 'f32'
                    or (load.dtype is not None and ssa_dtype_name(load.dtype) != 'f32')):
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
            shape, dtype = self.buffers[statement.input]
            if (not shape or shape[-1] % 4 or not statement.index
                    or type(statement.index[-1]).__name__ != 'Var'
                    or statement.index[-1].name != axis_name
                    or any(axis_name in coordinate.free_vars()
                           for coordinate in statement.index[:-1])):
                continue
            if (dtype_name(dtype) != 'f32' or len(statement.index) != len(shape)
                    or math.prod(shape) > I32_MAX):
                continue
            try:
                flat = finish(0, {})
                for coordinate, dimension in zip(statement.index, shape, strict=True):
                    form = parse(coordinate)
                    if not 0 <= form[2][0] <= form[2][1] < static(dimension):
                        raise ValueError('Coordinate outside global declaration')
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


    def global_dot_alignment(self, name):
        # Parameter low bits are immutable. Insert the test before all body
        # control so a skipped first use can never leave a cached predicate unset.
        if name not in self.global_dot_alignments:
            low, pred = self.reg('rd'), self.reg('p')
            lines = [f'    and.b64 {low}, {self.ptrs[name]}, 15;',
                     f'    setp.eq.u64 {pred}, {low}, 0;']
            anchor = self.global_dot_prologue_end
            self.lines[anchor:anchor] = lines
            self.global_dot_prologue_end += len(lines)
            self.global_dot_alignments[name] = pred
        return self.global_dot_alignments[name]

    def global_dot_prefetch(self, plan):
        result = {}
        for position, statement in plan.items():
            address, space = self.address(statement.input, statement.index)
            assert space == 'global'
            registers = tuple(self.reg('f') for _ in range(4))
            self.emit('ld.global.v4.f32 {' + ', '.join(registers)
                      + f'}}, [{address}];')
            result[position] = registers
        return result

    def global_dot_vector_loop(self, loop, frame):
        plan = self.global_dot_vector_plan(loop)
        if not plan:
            return False
        extent = static(loop.axis.extent)
        limit = extent - extent % 4
        # KEY admission proves these dot temporaries are dead outside the dot.
        # Both routes mutate the SAME already-seeded public accumulator register.
        name = loop.body[-1].name
        carrier = self.values[name]
        if carrier[1] != 'f32' or self.mutable.get(name) != carrier[0]:
            return False
        saved_values = dict(self.values)
        saved_mutable = dict(self.mutable)
        saved_ranges = dict(self.int_ranges)
        saved_ops = dict(self.accum_ops)
        invariant_depth, expression_depth = len(self.loop_invariants), len(self.loop_expressions)

        def restore_context():
            self.values = dict(saved_values)
            self.mutable = dict(saved_mutable)
            self.int_ranges = dict(saved_ranges)
            self.accum_ops = dict(saved_ops)
            self.address_cache.clear()
            self.expr_cache.clear()
            del self.loop_invariants[invariant_depth:]
            del self.loop_expressions[expression_depth:]

        predicates = [self.global_dot_alignment(statement.input)
                      for statement in plan.values()]
        aligned = predicates[0]
        for pred in predicates[1:]:
            both = self.reg('p')
            self.emit(f'and.pred {both}, {aligned}, {pred};')
            aligned = both
        scalar, joined = self.label('dot_scalar'), self.label('dot_complete')
        self.emit(f'@!{aligned} bra {scalar};')
        r = self.reg('r')
        self.emit(f'mov.s32 {r}, 0;')
        self.set_integer_range(r, checked_i32_interval(0, limit - 4))
        self.values[loop.axis.name] = (r, 'i32')
        self.rebind(loop.axis.name)
        self.hoist_invariants(loop.body, defined_names(loop.body) | {loop.axis.name})
        begin, tail = self.label('dot_vector4'), self.label('dot_vector4_tail')
        self.mark(begin)
        stop = self.reg('p')
        self.emit(f'setp.ge.s32 {stop}, {r}, {limit};')
        self.emit(f'@{stop} bra {tail};')
        prefetched = self.global_dot_prefetch(plan)
        for offset in range(4):
            axis = r
            if offset:
                axis = self.reg('r')
                self.emit(f'add.s32 {axis}, {r}, {offset};')
                self.set_integer_range(axis, checked_i32_interval(offset, limit - 4 + offset))
            self.values[loop.axis.name] = (axis, 'i32')
            self.rebind(loop.axis.name)
            bindings = {position: registers[offset]
                        for position, registers in prefetched.items()}
            self.statements_with_init_scope(loop.body, frame, bindings)
        self.emit(f'add.s32 {r}, {r}, 4;')
        self.rebind(loop.axis.name)
        self.emit(f'bra {begin};')
        self.mark(tail)
        self.set_integer_range(r, checked_i32_interval(limit, limit))
        for coordinate in range(limit, extent):
            self.values[loop.axis.name] = (str(coordinate), 'i32')
            self.rebind(loop.axis.name)
            self.statements_with_init_scope(loop.body, frame)
        assert self.values[name] == carrier and self.mutable[name] == carrier[0]
        self.emit(f'bra {joined};')

        # Exact inherited scalar iteration, with the same seed register and
        # original per-iteration memory/arithmetic order. No second seed or FMA
        # result capture is introduced.
        restore_context()
        self.mark(scalar)
        scalar_axis = self.reg('r')
        self.emit(f'mov.s32 {scalar_axis}, 0;')
        self.set_integer_range(scalar_axis, checked_i32_interval(0, extent - 1))
        self.values[loop.axis.name] = (scalar_axis, 'i32')
        self.rebind(loop.axis.name)
        self.hoist_invariants(loop.body, defined_names(loop.body) | {loop.axis.name})
        begin, done = self.label('dot_scalar_loop'), self.label('dot_scalar_done')
        self.mark(begin)
        stop = self.reg('p')
        self.emit(f'setp.ge.s32 {stop}, {scalar_axis}, {extent};')
        self.emit(f'@{stop} bra {done};')
        self.statements_with_init_scope(loop.body, frame)
        self.emit(f'add.s32 {scalar_axis}, {scalar_axis}, 1;')
        self.rebind(loop.axis.name)
        self.emit(f'bra {begin};')
        self.mark(done)
        assert self.values[name] == carrier and self.mutable[name] == carrier[0]
        self.mark(joined)
        # All temporary definitions are local under the whole-KEY proof.
        # Preserve only the shared carrier's runtime value and outer contexts.
        restore_context()
        self.rebind(name)
        self.rebind(loop.axis.name)
        return True


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
        # A seeded child fold declares its carrier in THIS body, before its
        # own loop. Its finalized value remains visible inside this body, but
        # an ancestor binding must return when this body closes. seedFalse
        # children declare nothing and keep updating their existing carrier.
        def owned_seeds(items):
            # Mirror init_accums ownership, without crossing a nested loop.
            for statement in items:
                kind = type(statement).__name__
                if kind == 'Accum':
                    yield statement.name
                elif kind == 'Cond':
                    yield from owned_seeds(statement.body)
                    yield from owned_seeds(statement.else_body)

        names = set()
        for statement in statements:
            kind = type(statement).__name__
            if kind == 'Init':
                names.add(statement.name)
            elif kind in ('Loop', 'StridedLoop') and statement.seed:
                names.update(owned_seeds(statement.body))
        # Fresh names retain the enclosing construct's export/join policy.
        return {name: (self.values[name], self.mutable.get(name))
                for name in names if name in self.values}

    def restore_init_scope(self, frame):
        for name, (value, mutable) in frame.items():
            self.values[name] = value
            if mutable is None:
                self.mutable.pop(name, None)
            else:
                self.mutable[name] = mutable
            self.rebind(name)

    def merge_scope_frame(self, statement):
        # Typed collective merges export only their carried state. Right-hand
        # shuffle/tree operands and every other Assign target are block locals.
        states = set(statement.state)
        names = (set(statement.state_b) |
                 {s.name for s in statement.combine_states
                  if type(s).__name__ == 'Assign'}) - states
        return {name: (self.values.get(name), self.mutable.get(name))
                for name in names}

    def restore_merge_scope(self, frame):
        for name, (value, mutable) in frame.items():
            if value is None:
                self.values.pop(name, None)
            else:
                self.values[name] = value
            if mutable is None:
                self.mutable.pop(name, None)
            else:
                self.mutable[name] = mutable
            # Restored outer registers retain their range facts. Every index
            # expression using this lexical name must bind to the outer value.
            self.rebind(name)

    def statements_with_init_scope(self, statements, frame, shared_prefetch=None):
        try:
            self.statements(statements, shared_prefetch)
        finally:
            self.restore_init_scope(frame)


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


    def small_strided_softmax_pair(self, loop):
        """Collapse a complete two-trip F32 online local max/sum state.

        The typed body and bounds prove two unconditional read-only scores per
        lane. Only its final max/sum carriers may escape to later collectives;
        no prefix or temporary is observable. The collective code is untouched.
        """
        if (self.inactive or not loop.seed or loop.end is not None
                or len(self.node.op.body) != 1
                or type(self.node.op.body[0]).__name__ != 'Tile'):
            return False
        tile = self.node.op.body[0]
        if loop not in tile.body:
            return False
        stride = static_or_none(loop.step)
        extent = static_or_none(loop.axis.extent)
        if (stride is None or extent is None or stride <= 0
                or extent != 2 * stride or extent > I32_MAX):
            return False
        body = tuple(loop.body)
        if [type(x).__name__ for x in body] != [
                'Let', 'Load', 'Assign', 'Assign', 'Assign', 'Assign',
                'Assign', 'Assign', 'Assign', 'Accum', 'Accum']:
            return False
        one, load, maximum, old_delta, alpha, new_delta, beta, old_scale, new_scale, sum_fold, max_fold = body
        if (type(one.value).__name__ != 'Literal' or one.value.value != 1.0
                or len(load.names) != 1 or load.input not in self.ptrs
                or load.input in self.shared or load.input in self.scalar_parameters
                or load.input in written_buffers(tile.body)
                or load.input not in self.buffers
                or dtype_name(self.buffers[load.input][1]) != 'f32'
                or any(x.dtype is not None and dtype_name(x.dtype) != 'f32'
                       for x in body)):
            return False
        score = load.names[0]
        max_name, sum_name = max_fold.name, sum_fold.name
        def assign(statement, op, args):
            return statement.op.name == op and tuple(statement.args) == tuple(args)
        if (not assign(maximum, 'maximum', (max_name, score))
                or not assign(old_delta, 'subtract', (max_name, maximum.name))
                or not assign(alpha, 'exp', (old_delta.name,))
                or not assign(new_delta, 'subtract', (score, maximum.name))
                or not assign(beta, 'exp', (new_delta.name,))
                or not assign(old_scale, 'multiply', (sum_name, alpha.name))
                or tuple(new_scale.args) not in ((beta.name, one.name), (one.name, beta.name))
                or new_scale.op.name != 'multiply'
                or reduce_canon(sum_fold.op.name) != 'add'
                or sum_fold.base != old_scale.name or sum_fold.value != new_scale.name
                or reduce_canon(max_fold.op.name) != 'maximum'
                or max_fold.base is not None or max_fold.value != score
                or tuple(sum_fold.axes) != (loop.axis.name,)
                or tuple(max_fold.axes) != (loop.axis.name,)):
            return False
        defined = defined_names(body)
        if (len(defined) != 11 or loop.axis.name in defined
                or set(loop.start.free_vars()) & defined
                or any(set(index.free_vars()) & defined for index in load.index)):
            return False
        outside = touched_names([item for item in tile.body if item is not loop])
        if (outside & (defined - {max_name, sum_name})):
            return False
        start, dtype = self.expr(loop.start)
        bounds = self.integer_range(start) if dtype == 'i32' else None
        if bounds is None or not (0 <= bounds[0] <= bounds[1] < stride):
            return False
        if (max_name not in self.values or sum_name not in self.values
                or self.values[max_name][1] != 'f32'
                or self.values[sum_name][1] != 'f32'):
            return False
        self.values[loop.axis.name] = (start, 'i32')
        self.rebind(loop.axis.name)
        self.statements((load,))
        first = self.value(score)
        second_index = self.reg('r')
        self.emit(f'add.s32 {second_index}, {start}, {stride};')
        self.set_integer_range(second_index,
                               checked_i32_interval(bounds[0] + stride, bounds[1] + stride))
        self.values[loop.axis.name] = (second_index, 'i32')
        self.rebind(loop.axis.name)
        self.statements((load,))
        second = self.value(score)
        local_first = self.reg('f')
        local_max = self.reg('f')
        self.emit(f'max.f32 {local_first}, {self.value(max_name)}, {first};')
        self.emit(f'max.f32 {local_max}, {local_first}, {second};')
        delta_first, _ = self.emit_operation('subtract', [(first, 'f32'), (local_max, 'f32')], 'f32')
        delta_second, _ = self.emit_operation('subtract', [(second, 'f32'), (local_max, 'f32')], 'f32')
        exp_first, _ = self.emit_operation('exp', [(delta_first, 'f32')], 'f32')
        exp_second, _ = self.emit_operation('exp', [(delta_second, 'f32')], 'f32')
        total, _ = self.emit_operation('add', [(exp_first, 'f32'), (exp_second, 'f32')], 'f32')
        old_max, old_sum = self.value(max_name), self.value(sum_name)
        self.emit(f'mov.f32 {old_max}, {local_max};')
        self.emit(f'mov.f32 {old_sum}, {total};')
        self.values[max_name] = (old_max, 'f32')
        self.values[sum_name] = (old_sum, 'f32')
        self.softmax_pair_sum_name = sum_name
        self.mutable[max_name] = old_max
        self.mutable[sum_name] = old_sum
        self.rebind(max_name)
        self.rebind(sum_name)
        return True

    def exact_short_strided_loop(self, loop, init_frame):
        """Emit each proven trip once when a Tile lane owns a short stripe.

        The start is the Tile axis itself, ranging from zero to step-1.  A
        static extent divisible by step gives every physical lane exactly the
        same number of trips.  Emit the original body and fold updates in
        textual order; only induction and control instructions disappear.
        """
        if (self.inactive or self.geometry_3d or len(self.node.op.body) != 1
                or type(self.node.op.body[0]).__name__ != 'Tile'):
            return False
        tile = self.node.op.body[0]
        if (not tile.body or tile.body[0] is not loop
                or tile.aux_threads or tile.raster_axes
                or tile.block_threads != self.block_threads
                or math.prod(static(axis.extent) for axis in tile.axes)
                   != self.block_threads * self.grid_blocks
                or type(loop.start).__name__ != 'Var'
                or type(loop.step).__name__ != 'Literal'
                or type(loop.step.value) is not int
                or loop.step.dtype not in ('int', 'i32')
                or loop.end is not None or loop.unroll):
            return False
        step, extent = loop.step.value, static(loop.axis.extent)
        if (step <= 0 or extent <= 0 or extent > I32_MAX - step
                or extent % step or not 2 <= extent // step <= 8
                or self.tile_extents.get(loop.start.name) != step
                or loop.axis.name == loop.start.name
                or loop.axis.name in defined_names(loop.body)
                or not straight_line(loop.body)
                or any(id(x) in self.hoisted or id(x) in self.hoist_plans
                       for x in loop.body)):
            return False
        start, dtype = self.binding(loop.start.name)
        if dtype != 'i32':
            return False
        for trip in range(extent // step):
            index = self.reg('r')
            if trip:
                self.emit(f'add.s32 {index}, {start}, {trip * step};')
            else:
                self.emit(f'mov.s32 {index}, {start};')
            self.set_integer_range(index, checked_i32_interval(trip * step,
                                                               (trip + 1) * step - 1))
            self.values[loop.axis.name] = (index, 'i32')
            self.rebind(loop.axis.name)
            self.statements_with_init_scope(loop.body, init_frame)
        return True

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
        if (not straight_line(s.body) or stride is None or stride <= 0 or bound is None
                or s.axis.name in defined_names(s.body)):
            return
        for group in STRIDED_UNROLL:
            if bound + (group - 1) * stride >= (1 << 31):
                continue
            last = self.reg('r')
            p = self.reg('p')
            begin, done = self.label('loop_unrolled'), self.label('loop_unrolled_done')
            self.mark(begin)
            self.emit(f'setp.ge.s32 {p}, {r}, {bound};')
            self.emit(f'@{p} bra {done};')
            self.emit(f'add.s32 {last}, {r}, {(group - 1) * stride};')
            self.emit(f'setp.ge.s32 {p}, {last}, {end};')
            self.emit(f'@{p} bra {done};')
            for _ in range(group):
                self.statements(s.body)
                self.emit(f'add.s32 {r}, {r}, {step};')
                self.rebind(s.axis.name)
            self.emit(f'bra {begin};')
            self.mark(done)

    def vector_group_base(self, descriptor):
        """Address of a descriptor's first element, checked for alignment."""
        first = descriptor.lanes[0]
        address, space = self.address(descriptor.buffer, first.statement.index, first.element)
        if descriptor.static_aligned:
            return address, space, None
        low, predicate = self.reg('rd'), self.reg('p')
        self.emit(f'and.b64 {low}, {address}, 15;')
        self.emit(f'setp.eq.u64 {predicate}, {low}, 0;')
        return address, space, predicate


    def emit_vector_load(self, address, space, predicate):
        """128-bit f32 load with an exact scalar fallback for a misaligned base."""
        registers = [self.reg('f') for _ in range(4)]
        operands = ', '.join(registers)
        if predicate is not None:
            scalar = self.label('access_scalar')
            vector_done = self.label('access_vector_done')
            self.emit(f'@!{predicate} bra {scalar};')
            self.emit(f'ld.{space}.v4.f32 {{{operands}}}, [{address}];')
            self.emit(f'bra {vector_done};')
            self.mark(scalar)
            for lane, register in enumerate(registers):
                self.emit(f'ld.{space}.f32 {register}, [{address}+{4 * lane}];')
            self.mark(vector_done)
        else:
            self.emit(f'ld.{space}.v4.f32 {{{operands}}}, [{address}];')
        return registers


    def vector_load_group(self, descriptor):
        """Load four proven-contiguous f32 lanes at the leader statement."""
        address, space, predicate = self.vector_group_base(descriptor)
        if self.inactive:
            # Preserve the scalar path's zero for masked destinations, keep the
            # barrier/participation predicate an outer conjunction, and keep
            # an active misaligned lane on the scalar fallback instead of
            # letting the inactive-lane zero path erase its value.
            active = self.inactive[-1]
            registers = [self.reg('f') for _ in range(4)]
            operands = ', '.join(registers)
            scalar = self.label('access_masked_scalar') if predicate is not None else None
            zero = self.label('access_masked_zero')
            done = self.label('access_masked_done')
            self.emit(f'@!{active} bra {zero};')
            if predicate is not None:
                self.emit(f'@!{predicate} bra {scalar};')
            self.emit(f'ld.{space}.v4.f32 {{{operands}}}, [{address}];')
            self.emit(f'bra {done};')
            if scalar is not None:
                self.mark(scalar)
                for lane, register in enumerate(registers):
                    self.emit(f'ld.{space}.f32 {register}, [{address}+{4 * lane}];')
                self.emit(f'bra {done};')
            self.mark(zero)
            for register in registers:
                self.emit(f'mov.f32 {register}, {literal(0, "f32")};')
            self.mark(done)
        else:
            registers = self.emit_vector_load(address, space, predicate)
        for lane, register in zip(descriptor.lanes, registers, strict=True):
            self.values[lane.name] = (register, 'f32')
            self.mutable.pop(lane.name, None)
            self.rebind(lane.name)
        leader_id = id(descriptor.lanes[0].statement)
        self.access_plan.registers[leader_id] = registers


    def vector_store_group(self, descriptor):
        """Store four proven-contiguous f32 lanes as one aligned tuple store."""
        address, space, predicate = self.vector_group_base(descriptor)
        values = [convert_register(self, self.value(lane.name),
                                   self.binding(lane.name)[1], 'f32')
                  for lane in descriptor.lanes]
        operands = ', '.join(values)
        active = self.inactive[-1] if self.inactive else None
        scalar = self.label('access_store_scalar') if predicate is not None else None
        done = self.label('access_store_done') if predicate is not None else None
        if predicate is not None:
            self.emit(f'@!{predicate} bra {scalar};')
        prefix = f'@{active} ' if active is not None else ''
        self.emit(f'{prefix}st.{space}.v4.f32 [{address}], {{{operands}}};')
        if predicate is not None:
            self.emit(f'bra {done};')
            self.mark(scalar)
            for lane, value in enumerate(values):
                pred = f'@{active} ' if active is not None else ''
                self.emit(f'{pred}st.{space}.f32 [{address}+{4 * lane}], {value};')
            self.mark(done)


    def access_plan_for(self, statements, prefetched=()):
        """Only unchanged straight-line memory accesses are grouped."""
        special = (self.hoisted, self.subst, self.vn_skip,
                   self.vn_record, self.hoist_plans,
                   self.shared_probability_skip, self.shared_probability_betas,
                   self.prescaled_assignments)
        blocked = {id(s) for s in statements
                   if any(id(s) in table for table in special)}
        blocked.update(id(statements[position]) for position in prefetched
                       if 0 <= position < len(statements))
        return access_descriptors(statements, self.buffers, self.shared,
                                  {name: self.shared_offsets[name] for name in self.shared_offsets},
                                  blocked)

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
                or loop.axis.name in defined_names(loop.body)
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
            if dtype_name(dtype) != 'f32' or (s.dtype is not None and dtype_name(s.dtype) != 'f32'):
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



    def find_shared_probability_plan(self, loop, online, planned=None):
        """Certify dead scaled scores and identical row maxima before reuse."""
        if not getattr(self, 'shared_probability_enabled', False) or self.inactive:
            return None
        if len(self.node.op.body) != 1:
            return None
        tile = self.node.op.body[0]
        if (type(tile).__name__ != 'Tile' or len(tile.axes) != 4
                or loop not in tile.body or not loop.seed or tile.aux_threads
                or self.geometry_3d or tile.block_threads != self.block_threads):
            return None
        extents = [static(a.extent) for a in tile.axes]
        width = extents[-1]
        if (not 2 <= width <= 32 or width & (width - 1)
                or extents[-2] * width != self.block_threads
                or math.prod(extents) != self.grid_blocks * self.block_threads
                or not 0 < static(loop.axis.extent) <= I32_MAX - width):
            return None
        def flat(body):
            for x in body:
                yield x
                for child in statement_bodies(x):
                    yield from flat(child)
        whole = list(flat(tile.body))
        if any(type(x).__name__ not in {'Loop','Load','Assign','Accum','Let','Write'}
               for x in whole):
            return None
        # Unique leaf bindings exclude mutable snapshots, builtin/axis shadows,
        # and the demonstrated dead Assign/Accum probe (which retains parent).
        axes = {a.name for a in tile.axes} | {x.axis.name for x in whole
                if type(x).__name__ == 'Loop'} | {'lane','warp','tid','bid','ntid'}
        definitions = {}
        leaves = [x for x in whole if not statement_bodies(x)]
        for x in leaves:
            for name in defined_names((x,)):
                if name in axes or name in definitions:
                    return None
                definitions[name] = x
        def uses(x):
            names = set(x.deps())
            for expr in x.exprs():
                names.update(expr.free_vars())
            return names
        def readers(name):
            return {id(x) for x in leaves if name in uses(x)}
        body = list(loop.body)
        dots = []
        for x in body:
            if type(x).__name__ != 'Loop':
                break
            dots.append(x)
        if not dots or any(type(x).__name__ == 'Loop' for x in body[len(dots):]):
            return None
        if ({id(x) for x in whole if type(x).__name__ == 'Loop'}
                != {id(loop), *(id(x) for x in dots)}):
            return None
        after = tile.body[list(tile.body).index(loop) + 1:]
        if any(type(x).__name__ not in ('Assign', 'Write') for x in after):
            return None
        outputs = set(self.node.buffer_names())
        if (any(x.atomic or x.output not in outputs or x.swizzle not in ('NONE', None)
                for x in whole if type(x).__name__ == 'Write')
                or any(x.input in outputs for x in whole if type(x).__name__ == 'Load')
                or tile.axes[-1].name not in self.values
                or self.values[tile.axes[-1].name][1] != 'i32'):
            return None
        expected_used = ((tile.axes[-2].name, extents[-2]),)
        descriptors = ({identity: item[0] for identity, item in self.hoisted.items()}
                       if planned is None else planned)
        groups, skip, betas = {}, set(), {}
        for fold, score, maximum, delta, alpha, needed in online:
            score_assign = definitions.get(score)
            if (score_assign is None or id(score_assign) not in self.prescaled_assignments
                    or score_assign not in loop.body or alpha is None):
                return None
            raw = self.prescaled_assignments[id(score_assign)]
            choices = [dot for dot in dots if id(dot) in self.prescaled_hoists
                       and id(dot) in descriptors
                       and tuple(descriptors[id(dot)].exports) == (raw,)]
            if len(choices) != 1:
                return None
            dot = choices[0]
            plan = descriptors[id(dot)]
            serial, rows = ((loop.axis.name, ()) if planned is not None
                            else self.hoisted[id(dot)][1:])
            if (tuple(plan.used) != expected_used or plan.configs != extents[-2]
                    or plan.extent != static(loop.axis.extent)
                    or serial != loop.axis.name or (planned is None and len(rows) != 1)
                    or len(plan.exports) != 1):
                return None
            differences = [x for x in loop.body if type(x).__name__ == 'Assign'
                           and x.op.name == 'subtract'
                           and tuple(x.args) == (score, maximum.name)]
            if len(differences) != 1:
                return None
            difference = differences[0]
            exponentials = [x for x in loop.body if type(x).__name__ == 'Assign'
                            and x.op.name == 'exp'
                            and tuple(x.args) == (difference.name,)]
            if len(exponentials) != 1:
                return None
            beta = exponentials[0]
            if (readers(score) != {id(fold), id(maximum), id(difference)}
                    or readers(difference.name) != {id(beta)}
                    or any(x.dtype is not None and dtype_name(x.dtype) != 'f32'
                           for x in (score_assign, maximum, difference, beta, fold))):
                return None
            row = rows[0] if planned is None else None
            groups.setdefault(id(dot), (row, fold.name))
            betas[id(beta)] = (row, loop.axis.name)
            skip.update((id(dot), id(score_assign), id(difference)))
        # Every raw score and every score-scale consumer must be accounted for.
        # A single unknown observer causes the whole new phase to decline.
        if set(groups) != {id(x) for x in dots}:
            return None
        scale_consumers = {id(x) for x in body if id(x) in self.prescaled_assignments}
        if scale_consumers != {id(definitions[p[1]]) for p in online}:
            return None
        return width, groups, skip, betas, tile.axes[-1].name

    def install_shared_probabilities(self, loop, online, seeds):
        if id(loop) in self.producer_probability_ready:
            _, _, skip, betas, _ = self.require_producer_probability_binding(loop, online)
            self.shared_probability_frames[id(loop)] = (
                set(self.shared_probability_skip), dict(self.shared_probability_betas),
                dict(self.vn_registers))
            self.shared_probability_skip.update(skip)
            self.shared_probability_betas.update(betas)
            return
        plan = self.find_shared_probability_plan(loop, online)
        if plan is None:
            return
        width, groups, skip, betas, column = plan
        # Finish ALL original raw-score observers before any thread overwrites.
        self.emit('bar.sync 0;')
        key = self.reg('r')
        self.emit(f'mov.u32 {key}, {self.value(column)};')
        begin, done = self.label('shared_probability'), self.label('shared_probability_done')
        self.mark(begin)
        stop = self.reg('p')
        self.emit(f'setp.ge.u32 {stop}, {key}, {static(loop.axis.extent)};')
        self.emit(f'@{stop} bra {done};')
        offset = self.reg('rd')
        self.emit(f'mul.wide.u32 {offset}, {key}, 4;')
        for row, maximum in groups.values():
            address, score = self.reg('rd'), self.reg('f')
            self.emit(f'add.u64 {address}, {row}, {offset};')
            self.emit(f'ld.shared.f32 {score}, [{address}];')
            difference, dtype = self.emit_operation(
                'subtract', [(score, 'f32'), (seeds[maximum], 'f32')], 'f32')
            probability, dtype = self.emit_operation('exp', [(difference, dtype)], 'f32')
            self.emit(f'st.shared.f32 [{address}], {probability};')
        self.emit(f'add.u32 {key}, {key}, {width};')
        self.emit(f'bra {begin};')
        self.mark(done)
        # Tail lanes still join: no barrier was placed inside the producer loop.
        self.emit('bar.sync 0;')
        self.shared_probability_frames[id(loop)] = (
            set(self.shared_probability_skip), dict(self.shared_probability_betas),
            dict(self.vn_registers))
        self.shared_probability_skip.update(skip)
        self.shared_probability_betas.update(betas)

    def scalar_shared_address_packet_loop(self, loop, init_frame):
        """Keep scalar shared reads, sharing only each packet's address base."""
        extent = static(loop.axis.extent)
        if (id(loop) not in self.shared_probability_frames or self.inactive
                or not 4 <= extent <= I32_MAX - 3
                or loop.axis.name in defined_names(loop.body)):
            return False
        # The installed shared-P phase already proved the full closed Tile,
        # readonly allocations, unique leaf definitions and no axis shadows.
        # Recheck the executable body: only certified eliminated prefixes,
        # synthetic P reads and already allocated hoisted F32 reads may load.
        known_rows = set()
        for plan, serial, rows in self.hoisted.values():
            if (serial == loop.axis.name and plan.extent == extent
                    and plan.row_words >= extent
                    and 0 < plan.nbytes < (1 << 32)
                    and len(rows) == len(plan.exports)):
                known_rows.update(rows)
        streams = set()
        for statement in loop.body:
            identity = id(statement)
            if identity in self.shared_probability_skip:
                continue
            if identity in self.shared_probability_betas:
                row, serial = self.shared_probability_betas[identity]
                if serial != loop.axis.name or row not in known_rows:
                    return False
                streams.add((row, serial))
            elif identity in self.hoisted:
                plan, serial, rows = self.hoisted[identity]
                if (serial != loop.axis.name or plan.extent != extent
                        or len(rows) != len(plan.exports)
                        or any(row not in known_rows for row in rows)):
                    return False
                streams.update((row, serial) for row in rows)
            elif (type(statement).__name__ not in ('Let', 'Assign', 'Accum')
                    or statement_bodies(statement)):
                return False
        if not streams:
            return False
        # Every allocation/row is immutable after the dominating producer
        # barriers. Bounds are inherited from HoistedValue's complete typed
        # row construction; scalar loads require only its original 4B alignment.
        streams = sorted(streams)
        limit = extent - extent % 4
        induction = self.reg('r')
        self.emit(f'mov.s32 {induction}, 0;')
        self.set_integer_range(induction, checked_i32_interval(0, limit - 4))
        self.values[loop.axis.name] = (induction, 'i32')
        self.rebind(loop.axis.name)
        self.hoist_invariants(loop.body, defined_names(loop.body) | {loop.axis.name})
        begin, done = self.label('shared_address_packet'), self.label('shared_address_packet_done')
        self.mark(begin)
        stop = self.reg('p')
        self.emit(f'setp.ge.s32 {stop}, {induction}, {limit};')
        self.emit(f'@{stop} bra {done};')
        offset = self.reg('rd')
        self.emit(f'mul.wide.u32 {offset}, {induction}, 4;')
        pointers = {}
        for stream in streams:
            pointer = self.reg('rd')
            self.emit(f'add.u64 {pointer}, {stream[0]}, {offset};')
            pointers[stream] = pointer
        previous = getattr(self, 'shared_scalar_packet', None)
        try:
            for copy in range(4):
                axis = induction
                if copy:
                    axis = self.reg('r')
                    self.emit(f'add.s32 {axis}, {induction}, {copy};')
                    self.set_integer_range(axis, checked_i32_interval(copy, limit - 4 + copy))
                self.values[loop.axis.name] = (axis, 'i32')
                self.rebind(loop.axis.name)
                self.shared_scalar_packet = {
                    stream: (pointer, 4 * copy) for stream, pointer in pointers.items()}
                self.statements_with_init_scope(loop.body, init_frame)
        finally:
            self.shared_scalar_packet = previous
        self.emit(f'add.s32 {induction}, {induction}, 4;')
        self.rebind(loop.axis.name)
        self.emit(f'bra {begin};')
        self.mark(done)
        for coordinate in range(limit, extent):
            self.values[loop.axis.name] = (str(coordinate), 'i32')
            self.rebind(loop.axis.name)
            self.statements_with_init_scope(loop.body, init_frame)
        self.emit(f'mov.s32 {induction}, {extent};')
        self.set_integer_range(induction, checked_i32_interval(extent, extent))
        return True


    def load_shared_probability(self, statement):
        row, serial = self.shared_probability_betas[id(statement)]
        packet = (getattr(self, 'shared_scalar_packet', None) or {}).get((row, serial))
        if packet is None:
            offset, address = self.reg('rd'), self.reg('rd')
            self.emit(f'mul.wide.u32 {offset}, {self.value(serial)}, 4;')
            self.emit(f'add.u64 {address}, {row}, {offset};')
            operand = address
        else:
            address, immediate = packet
            operand = f'{address}+{immediate}'
        probability, snapshot = self.reg('f'), self.reg('f')
        self.emit(f'ld.shared.f32 {probability}, [{operand}];')
        self.emit(f'mov.f32 {snapshot}, {probability};')
        self.values[statement.name] = (snapshot, 'f32')
        self.mutable.pop(statement.name, None)
        self.rebind(statement.name)

    def restore_shared_probability_phase(self, loop):
        frame = self.shared_probability_frames.pop(id(loop), None)
        if frame is not None:
            self.shared_probability_skip, self.shared_probability_betas, self.vn_registers = frame


    def find_distributed_maximum_plan(self, loop, online, seed_names, planned=None):
        """Exact finite-seeded max over complete physical column subgroups."""
        plan = self.find_shared_probability_plan(loop, online, planned)
        if plan is None:
            return None
        width, groups, _, _, column = plan
        tile = self.node.op.body[0]
        extents = [static(axis.extent) for axis in tile.axes]
        if (width not in (16, 32) or self.block_threads % 32
                or extents[-2] * width != self.block_threads
                or math.prod(extents) != self.grid_blocks * self.block_threads
                or self.geometry_3d or self.inactive):
            return None
        # Tile decodes the physical hardware_gid row-major. The last axis is
        # gid%W=physical_tid%W because the complete CTA size is divisible by W;
        # the preceding row axis is floor(tid/W), constant in each aligned
        # subgroup. Do not use any rebindable logical tid/lane name.
        def leaves(body):
            for statement in body:
                children = statement_bodies(statement)
                if not children:
                    yield statement
                for child in children:
                    yield from leaves(child)
        whole = list(leaves(tile.body))
        def reads(statement, name):
            return (name in statement.deps()
                    or any(name in expr.free_vars() for expr in statement.exprs()))
        for fold, _, maximum, delta, _, _ in online:
            if {id(x) for x in whole if x is not fold and reads(x, fold.name)} != {
                    id(maximum), id(delta)}:
                return None
        # Reconstruct EVERY online fold's canonical typed dot/row membership.
        # find_shared_probability_plan intentionally retains only the first fold
        # per dot; dropping its cardinality guard alone would lose duplicate seeds.
        descriptors = ({identity: item[0] for identity, item in self.hoisted.items()}
                       if planned is None else planned)
        members = {identity: [] for identity in groups}
        signatures = {}
        seen = set()
        for fold, score, maximum, delta, alpha, needed in online:
            assignments = [x for x in loop.body if type(x).__name__ == 'Assign'
                           and x.name == score and id(x) in self.prescaled_assignments]
            if len(assignments) != 1 or fold.name in seen or fold.name not in seed_names:
                return None
            statement = assignments[0]
            raw = self.prescaled_assignments[id(statement)]
            choices = [x for x in loop.body if type(x).__name__ == 'Loop'
                       and id(x) in groups and id(x) in self.prescaled_hoists
                       and id(x) in descriptors
                       and tuple(descriptors[id(x)].exports) == (raw,)]
            if len(choices) != 1:
                return None
            identity = id(choices[0])
            signature = (raw, tuple(statement.args))
            if identity in signatures and signatures[identity] != signature:
                return None
            signatures[identity] = signature
            members[identity].append(fold.name)
            seen.add(fold.name)
        if (seen != set(seed_names) or len(seen) != len(online)
                or any(not names or groups[identity][1] not in names
                       for identity, names in members.items())):
            return None
        return width, groups, members

    def distributed_shared_maximum(self, loop, online, seeds):
        """Exact finite-seeded max over complete physical column subgroups."""
        if id(loop) in self.producer_probability_ready:
            self.require_producer_probability_binding(loop, online)
        plan = self.find_distributed_maximum_plan(loop, online, seeds)
        if plan is None:
            return False
        width, groups, members = plan
        cached = self.producer_maxima.get(id(loop))
        signature = (width, tuple((identity, tuple(names))
                                  for identity, names in members.items()))
        if (cached is not None and cached[0] == signature
                and set(cached[1]) == set(groups)
                and all(identity in self.hoisted
                        and self.hoisted[identity][0] is item
                        for identity, item in cached[2].items())):
            for identity, (_, maximum) in groups.items():
                self.emit(f'mov.f32 {seeds[maximum]}, {cached[1][identity]};')
        else:
            # This phase uses only fresh private registers and immutable score
            # row pointers. It publishes no partial bindings or VN facts.
            key = self.reg('r')
            self.emit(f'and.b32 {key}, {self.hardware_tid}, {width - 1};')
            begin, done = self.label('distributed_max'), self.label('distributed_max_done')
            self.mark(begin)
            stop = self.reg('p')
            self.emit(f'setp.ge.u32 {stop}, {key}, {static(loop.axis.extent)};')
            self.emit(f'@{stop} bra {done};')
            offset = self.reg('rd')
            self.emit(f'mul.wide.u32 {offset}, {key}, 4;')
            for row, maximum in groups.values():
                address, score = self.reg('rd'), self.reg('f')
                self.emit(f'add.u64 {address}, {row}, {offset};')
                self.emit(f'ld.shared.f32 {score}, [{address}];')
                self.emit(f'max.f32 {seeds[maximum]}, {seeds[maximum]}, {score};')
            self.emit(f'add.u32 {key}, {key}, {width};')
            self.emit(f'bra {begin};')
            self.mark(done)
            # Every full-warp thread reaches every shuffle. The segment mask keeps
            # W16 half-warps in separate rows; XOR offsets never leave the segment.
            clamp = ((32 - width) << 8) | (width - 1)
            offset = width // 2
            while offset:
                for _, maximum in groups.values():
                    peer = self.reg('f')
                    self.emit(f'shfl.sync.bfly.b32 {peer}, {seeds[maximum]}, {offset}, {clamp}, 0xffffffff;')
                    self.emit(f'max.f32 {seeds[maximum]}, {seeds[maximum]}, {peer};')
                offset //= 2
        # Preserve distinct mutable carrier publication even when MAX values
        # coincide. Copy the canonical completed value into a fresh register for
        # EVERY duplicate member, including the canonical member itself. The
        # single-member path emits exactly the parent's original instructions.
        for identity, names in members.items():
            if len(names) == 1:
                continue
            canonical = seeds[groups[identity][1]]
            for name in names:
                snapshot = self.reg('f')
                self.emit(f'mov.f32 {snapshot}, {canonical};')
                seeds[name] = snapshot
        # Original fold/gn/delta publication follows in two_pass_softmax;
        # shared-P's original pre-overwrite CTA barrier remains after that.
        return True

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
            self.emit(f'mov.f32 {m}, {self.reduction_identity_literal(reduce_canon(fold.op.name), "f32")};')
            seeds[fold.name] = m
        if not self.distributed_shared_maximum(loop, online, seeds):
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
        self.install_shared_probabilities(loop, online, seeds)


    def normalized_softmax_plans(self, loop, hoisted_ids=None):
        """Require the complete rescaled numerator/denominator recurrence.

        Matching a running maximum alone is insufficient: prefix maxima and
        raw rescaled sums are observable. Only ratios of equally rescaled
        states may leave this loop; intermediate max/exp values may not escape.
        """
        plans = online_max_carriers(loop.body, set(self.hoisted)
                                    if hoisted_ids is None else set(hoisted_ids))
        if not plans:
            return []
        body = list(loop.body)
        definitions = {s.name: s for s in body if type(s).__name__ in ('Assign', 'Let', 'Accum', 'Init')}
        descendants = set()
        outside = []
        def collect(statements, inside=False):
            for statement in statements:
                nested_inside = inside or statement is loop
                if nested_inside:
                    descendants.add(id(statement))
                else:
                    outside.append(statement)
                for nested in statement_bodies(statement):
                    collect(nested, nested_inside)
        collect(self.node.op.body)
        def readers(name, statements):
            return [s for s in statements if name in set(s.deps())]
        def assignment(s, op, args=None):
            return (type(s).__name__ == 'Assign' and s.op.name == op
                    and (s.dtype is None or dtype_name(s.dtype) == 'f32')
                    and (args is None or tuple(s.args) == tuple(args)))
        accepted = []
        for plan in plans:
            fold, score, maximum, delta, alpha, _ = plan
            if alpha is None or (fold.dtype is not None and dtype_name(fold.dtype) != 'f32'):
                continue
            differences = [s for s in body if assignment(s, 'subtract', (score, maximum.name))]
            if len(differences) != 1:
                continue
            difference = differences[0]
            exponentials = readers(difference.name, body)
            if len(exponentials) != 1 or not assignment(exponentials[0], 'exp', (difference.name,)):
                continue
            beta = exponentials[0]
            if {id(s) for s in readers(maximum.name, body)} != {id(delta), id(difference)}:
                continue
            all_folds = {s.name for s in body if type(s).__name__ == 'Accum'}
            value_definitions = dict(definitions)
            for statement in body:
                if type(statement).__name__ == 'Load':
                    value_definitions.update((name, statement) for name in statement.names)
            def independent(name, seen=None):
                seen = set() if seen is None else seen
                if name in all_folds:
                    return False
                if name in seen or name not in value_definitions:
                    return True
                statement = value_definitions[name]
                if type(statement).__name__ not in ('Load', 'Let', 'Assign'):
                    return False
                return all(independent(dep, seen | {name}) for dep in statement.deps())
            states, old_scales, new_scales, denominators = {}, [], [], set()
            valid = True
            for old in readers(alpha.name, body):
                if not assignment(old, 'multiply') or len(old.args) != 2:
                    valid = False; break
                other = [a for a in old.args if a != alpha.name]
                if len(other) != 1:
                    valid = False; break
                name = other[0]
                update = definitions.get(name)
                if (type(update).__name__ != 'Accum' or reduce_canon(update.op.name) != 'add'
                        or update.base != old.name
                        or (update.dtype is not None and dtype_name(update.dtype) != 'f32')):
                    valid = False; break
                new = definitions.get(update.value)
                if not assignment(new, 'multiply') or len(new.args) != 2 or beta.name not in new.args:
                    valid = False; break
                coefficient = next((a for a in new.args if a != beta.name), None)
                if coefficient is None or not independent(coefficient):
                    valid = False; break
                if ([id(s) for s in readers(name, body)] != [id(old)]
                        or [id(s) for s in readers(old.name, body)] != [id(update)]
                        or [id(s) for s in readers(new.name, body)] != [id(update)]):
                    valid = False; break
                constant = definitions.get(coefficient)
                if (type(constant).__name__ == 'Let' and type(constant.value).__name__ == 'Literal'
                        and constant.value.value == 1.0):
                    denominators.add(name)
                states[name] = update
                old_scales.append(old); new_scales.append(new)
            if (not valid or not denominators or not states
                    or {id(s) for s in readers(beta.name, body)} != {id(s) for s in new_scales}):
                continue
            temporaries = [maximum, delta, alpha, difference, beta, *old_scales, *new_scales]
            if any(readers(s.name, outside) for s in temporaries):
                continue
            reciprocal_denominators = {}
            for denominator in denominators:
                for user in readers(denominator, outside):
                    if assignment(user, 'reciprocal', (denominator,)):
                        reciprocal_denominators[user.name] = denominator
            def normalized(user, numerator):
                if assignment(user, 'divide') and tuple(user.args) in [(numerator, d) for d in denominators]:
                    return True
                return (assignment(user, 'multiply') and len(user.args) == 2
                        and any(sorted(user.args) == sorted((numerator, reciprocal))
                                for reciprocal in reciprocal_denominators))
            for name in states:
                for user in readers(name, outside):
                    if name in denominators and assignment(user, 'reciprocal', (name,)):
                        continue
                    if not normalized(user, name):
                        valid = False
            for reciprocal in reciprocal_denominators:
                if any(not any(normalized(user, name) for name in states)
                       for user in readers(reciprocal, outside)):
                    valid = False
            if valid:
                accepted.append(plan)
        return accepted

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
                        and load.input in self.ptrs and self.buffers[load.input][1] == 'f32'
                        and (load.dtype is None or dtype_name(load.dtype) == 'f32'))

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


    def plan_hoisted_score_scales(self, outer, plans):
        """Accept only private scores whose sole consumers share one uniform scale."""
        if len(self.node.op.body) != 1:
            return
        tile = self.node.op.body[0]
        if type(tile).__name__ != 'Tile' or outer not in tile.body:
            return
        def flat(body):
            for statement in body:
                yield statement
                for nested in statement_bodies(statement):
                    yield from flat(nested)
        whole = list(flat(tile.body))
        allowed = {'Tile', 'Loop', 'Load', 'Write', 'Assign', 'Let', 'Init', 'Accum'}
        if any(type(x).__name__ not in allowed for x in whole):
            return
        written = {x.output for x in whole if type(x).__name__ == 'Write'}
        uniform = set()
        prefix = list(tile.body[:list(tile.body).index(outer)])
        # A uniform literal-index load is invariant even while producer coordinates
        # are rebound. Requiring unique global definitions closes all shadow paths.
        definitions = {}
        for x in whole:
            for name in defined_names((x,)) if not statement_bodies(x) else ():
                definitions.setdefault(name, []).append(x)
        for x in prefix:
            kind = type(x).__name__
            if kind == 'Load':
                if (x.input in written or x.input not in self.buffers
                        or dtype_name(self.buffers[x.input][1]) != 'f32'
                        or x.input in self.shared or len(x.names) != 1
                        or (x.dtype is not None and dtype_name(x.dtype) != 'f32')
                        or any(type(e).__name__ != 'Literal' or type(e.value) is not int
                               for e in x.index)):
                    return
                uniform.update(x.names)
            elif kind == 'Assign':
                if (x.op.name != 'reciprocal' or len(x.args) != 1
                        or x.args[0] not in uniform
                        or (x.dtype is not None and dtype_name(x.dtype) != 'f32')):
                    return
                uniform.add(x.name)
            else:
                return
        axes = {a.name for a in tile.axes} | {x.axis.name for x in whole
                if type(x).__name__ in ('Loop', 'StridedLoop')}
        axes.update({'tid', 'bid', 'ntid', 'lane', 'warp'})
        uniform = {name for name in uniform if name not in axes
                   and len(definitions.get(name, ())) == 1
                   and name in self.values and self.values[name][1] == 'f32'
                   and name not in self.mutable}
        for plan in plans:
            if type(plan.statement).__name__ != 'Loop' or len(plan.exports) != 1:
                continue
            score = plan.exports[0]
            if score in axes:
                continue
            body = list(plan.statement.body)
            if ([type(x).__name__ for x in body] != ['Load', 'Load', 'Assign', 'Accum']
                    or body[-1].name != score or body[-1].base is not None
                    or body[-1].op.name != 'add'
                    or (body[-1].dtype is not None and dtype_name(body[-1].dtype) != 'f32')):
                continue
            consumers = []
            for x in whole:
                if statement_bodies(x) or x is body[-1]:
                    continue
                uses = set(x.deps())
                for expr in x.exprs():
                    uses.update(expr.free_vars())
                if score in uses:
                    consumers.append(x)
            if not consumers:
                continue
            scales = []
            valid = True
            for x in consumers:
                if (type(x).__name__ != 'Assign' or x not in outer.body
                        or x.op.name != 'multiply' or len(x.args) != 2
                        or x.args.count(score) != 1
                        or (x.dtype is not None and dtype_name(x.dtype) != 'f32')):
                    valid = False
                    break
                other = x.args[1] if x.args[0] == score else x.args[0]
                scales.append((other, x.args[0] == score))
            if (not valid or len(set(scales)) != 1 or scales[0][0] not in uniform
                    or len(definitions.get(score, ())) != 1):
                continue
            scale, score_first = scales[0]
            self.prescaled_hoists[id(plan.statement)] = (self.values[scale][0], score_first)
            for x in consumers:
                self.prescaled_assignments[id(x)] = score

    def prescale_hoisted_score(self, plan, value):
        scale = self.prescaled_hoists.get(id(plan.statement))
        if scale is None:
            return value
        register, score_first = scale
        x, y = (value, register) if score_first else (register, value)
        result = self.reg('f')
        self.emit(f'mul.rn.f32 {result}, {x}, {y};')
        return result

    def certify_producer_maxima(self, outer, plans, groups, launched):
        """Pure typed proof of staged one-item/thread and identical row groups."""
        if launched != self.block_threads or self.inactive or self.geometry_3d:
            return None
        planned = {id(plan.statement): plan for plan in plans}
        online = self.normalized_softmax_plans(outer, planned)
        if not online:
            return None
        certified = self.find_distributed_maximum_plan(
            outer, online, {item[0].name for item in online}, planned)
        if certified is None:
            return None
        width, rows, members = certified
        loops = [items for key, items in groups.items() if key[0] == 'Loop']
        if len(loops) != 1:
            return None
        items = loops[0]
        if {id(item.statement) for item in items} != set(rows):
            return None
        lead = items[0]
        extent = static(outer.axis.extent)
        shape = self.matmul_members(items, outer)
        if shape is None or extent % 4 or extent // 4 != width:
            return None
        k_extent = static(lead.statement.axis.extent)
        staged = 4 * (len(items) * lead.configs * k_extent + k_extent * extent)
        if (k_extent % 4 or lead.configs * len(items) < 2
                or self.shared_bytes + self.hoist_bytes + staged > 48 * 1024
                or lead.configs * width != launched
                or any(item.configs != lead.configs
                       or tuple(item.used) != tuple(lead.used)
                       or item.extent != extent or len(item.exports) != 1
                       or id(item.statement) not in self.prescaled_hoists
                       for item in items)):
            return None
        # Shared-P proves the sole used axis is the physical preceding row axis.
        # width==extent/4 makes producer config=floor(physical_tid/width);
        # work==T gives exactly one item to every physical thread. The full
        # row-major Tile proof in the shared matcher applies to every CTA.
        signature = (width, tuple((identity, tuple(names))
                                  for identity, names in members.items()))
        return signature, set(rows)

    @staticmethod
    def producer_online_signature(online):
        return tuple((id(fold), score, id(maximum), id(delta), id(alpha),
                      tuple(id(item) for item in needed))
                     for fold, score, maximum, delta, alpha, needed in online)

    def certify_producer_probabilities(self, outer, plans, maximum_certificate):
        """Pure whole-phase admission before raw score stores can be removed."""
        if maximum_certificate is None:
            return None
        tile = self.node.op.body[0]
        axis_names = [axis.name for axis in tile.axes]
        if (len(axis_names) != len(set(axis_names))
                or outer.axis.name in axis_names
                or id(outer) in self.vn_skip or id(outer) in self.subst
                or id(outer) in self.shared_probability_skip
                or id(outer) in self.shared_probability_betas):
            return None
        planned = {id(item.statement): item for item in plans}
        online = self.normalized_softmax_plans(outer, planned)
        shared = self.find_shared_probability_plan(outer, online, planned)
        if shared is None:
            return None
        width, groups, skip, betas, column = shared
        if (set(groups) != maximum_certificate[1]
                or width != maximum_certificate[0][0]
                or any(item.extent != static(outer.axis.extent)
                       or len(item.exports) != 1
                       for identity, item in planned.items() if identity in groups)):
            return None
        # All conditions below are immutable typed descriptors or saved physical
        # axis bindings. emit_hoisted restores these exact bindings after its
        # producer work, registers these same descriptors, and executes the final
        # P/V publication barrier before the consumer can be entered.
        return {
            'outer': outer, 'planned': planned,
            'online': self.producer_online_signature(online),
            'maximum': maximum_certificate,
            'groups': tuple((identity, fold) for identity, (_, fold) in groups.items()),
            'skip': set(skip), 'betas': set(betas), 'column': column,
            'column_binding': self.values[column],
            'scales': {identity: self.prescaled_hoists[identity] for identity in groups},
            'seeds': tuple((fold.name, self.reduction_identity_literal(
                reduce_canon(fold.op.name), 'f32')) for fold, *_ in online)
        }

    def require_producer_probability_binding(self, outer, online):
        """Commit the preproved phase; never reinterpret P cells as raw scores."""
        certificate = self.producer_probability_ready[id(outer)]
        valid = (certificate['outer'] is outer
                 and certificate['online'] == self.producer_online_signature(online)
                 and self.values.get(certificate['column']) == certificate['column_binding']
                 and all(identity in self.hoisted
                         and self.hoisted[identity][0] is item
                         and self.hoisted[identity][1] == outer.axis.name
                         for identity, item in certificate['planned'].items())
                 and all(self.prescaled_hoists.get(identity) == scale
                         for identity, scale in certificate['scales'].items())
                 and certificate['seeds'] == tuple(
                     (fold.name, self.reduction_identity_literal(
                         reduce_canon(fold.op.name), 'f32')) for fold, *_ in online))
        plan = self.find_shared_probability_plan(outer, online) if valid else None
        distributed = (self.find_distributed_maximum_plan(
            outer, online, {fold.name for fold, *_ in online}) if valid else None)
        if plan is not None and distributed is not None:
            width, groups, skip, betas, column = plan
            members = distributed[2]
            valid = (width == certificate['maximum'][0][0]
                     and tuple((identity, fold) for identity, (_, fold) in groups.items())
                         == certificate['groups']
                     and set(skip) == certificate['skip']
                     and set(betas) == certificate['betas']
                     and column == certificate['column']
                     and (width, tuple((identity, tuple(names))
                                       for identity, names in members.items()))
                         == certificate['maximum'][0])
        else:
            valid = False
        cached = self.producer_maxima.get(id(outer))
        if (not valid or cached is None
                or cached[0] != certificate['maximum'][0]
                or set(cached[1]) != certificate['maximum'][1]
                or any(cached[2].get(identity) is not item
                       for identity, item in certificate['planned'].items()
                       if identity in certificate['maximum'][1])):
            # This is an internal construction invariant, not a fallback for
            # unknown IR: all unknown conditions declined before producer emit.
            # Returning PTX with the raw-score path would be incorrect here.
            raise Unsupported('Internal direct probability phase binding mismatch')
        return plan

    def emit_hoisted(self, outer, plans, launched):
        """Evaluate each hoisted reduction across the CTA, then barrier."""
        self.plan_hoisted_score_scales(outer, plans)
        extent = static(outer.axis.extent)
        names = {outer.axis.name} | {name for plan in plans for name, _ in plan.used}
        saved = {name: self.values.get(name) for name in names}
        tid = self.hardware_tid
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
        self.producer_maximum_certificates[id(outer)] = self.certify_producer_maxima(
            outer, plans, groups, launched)
        self.producer_probability_certificates[id(outer)] = self.certify_producer_probabilities(
            outer, plans, self.producer_maximum_certificates[id(outer)])
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
                            value = self.prescale_hoisted_score(plan, value)
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
                    self.emit(f'add.u32 {item}, {self.hardware_tid}, {k * launched};')
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
        self.emit(f'mov.u32 {w}, {self.hardware_tid};')
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
        certificate = self.producer_maximum_certificates.get(id(outer))
        direct = self.producer_probability_certificates.get(id(outer))
        if direct is not None and (
                certificate != direct['maximum']
                or {id(item.statement) for item in members} != direct['maximum'][1]
                or any(direct['planned'].get(id(item.statement)) is not item
                       for item in members)):
            raise Unsupported('Internal direct probability producer route mismatch')
        maxima = {}
        if (certificate is not None
                and {id(item.statement) for item in members} == certificate[1]):
            for item in members:
                register = self.reg('f')
                fold = next(x for x in outer.body if type(x).__name__ == 'Accum'
                            and x.name in dict(certificate[0][1])[id(item.statement)])
                seed = self.reduction_identity_literal(reduce_canon(fold.op.name), 'f32')
                self.emit(f'mov.f32 {register}, {seed};')
                maxima[id(item.statement)] = register
        blocks = extent // 4
        work = lead.configs * blocks
        w = self.reg('r')
        self.emit(f'mov.u32 {w}, {self.hardware_tid};')
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
        scores = []
        for plan, row in zip(members, accs):
            if direct is None:
                word = self.reg('r')
                self.emit(f'mad.lo.u32 {word}, {config}, {plan.row_words}, {serial};')
                sa = self.reg('rd')
                self.emit(f'mul.wide.u32 {sa}, {word}, 4;')
                sbase = self.reg('rd')
                self.emit(f'mov.u64 {sbase}, {plan.symbol};')
                self.emit(f'add.u64 {sa}, {sbase}, {sa};')
            retained = []
            for j, acc in enumerate(row):
                acc = self.prescale_hoisted_score(plan, acc)
                if direct is None:
                    self.emit(f'st.shared.f32 [{sa}+{4 * j}], {acc};')
                else:
                    retained.append(acc)
                if id(plan.statement) in maxima:
                    maximum = maxima[id(plan.statement)]
                    self.emit(f'max.f32 {maximum}, {maximum}, {acc};')
            if direct is not None:
                scores.append((plan, retained))
        self.emit(f'add.u32 {w}, {w}, {launched};')
        self.emit(f'bra {begin};')
        self.mark(done)
        if maxima:
            # work==T proves all physical lanes reach this join together; the
            # XOR tree never leaves its aligned W16/W32 physical row subgroup.
            width = certificate[0][0]
            clamp = ((32 - width) << 8) | (width - 1)
            offset = width // 2
            while offset:
                for maximum in maxima.values():
                    peer = self.reg('f')
                    self.emit(f'shfl.sync.bfly.b32 {peer}, {maximum}, {offset}, {clamp}, 0xffffffff;')
                    self.emit(f'max.f32 {maximum}, {maximum}, {peer};')
                offset //= 2
            self.producer_maxima[id(outer)] = (
                certificate[0], maxima, {id(item.statement): item for item in members})
            if direct is not None:
                # Fullwork==T proves EVERY lane initialized its four scores per
                # member before this join. Each row's full MAX is now complete.
                # Scores expire here, before subsequent V hoist work.
                for item, retained in scores:
                    word = self.reg('r')
                    self.emit(f'mad.lo.u32 {word}, {config}, {item.row_words}, {serial};')
                    address, base = self.reg('rd'), self.reg('rd')
                    self.emit(f'mul.wide.u32 {address}, {word}, 4;')
                    self.emit(f'mov.u64 {base}, {item.symbol};')
                    self.emit(f'add.u64 {address}, {base}, {address};')
                    maximum = maxima[id(item.statement)]
                    for component, score in enumerate(retained):
                        difference, dtype = self.emit_operation(
                            'subtract', [(score, 'f32'), (maximum, 'f32')], 'f32')
                        probability, dtype = self.emit_operation(
                            'exp', [(difference, dtype)], 'f32')
                        self.emit(f'st.shared.f32 [{address}+{4 * component}], {probability};')
                self.producer_probability_ready[id(outer)] = direct

    def matmul_work_loop(self, shape, members, outer, launched, tn):
        b, operands = shape
        lead = members[0]
        blocks = lead.extent // tn
        batch = self.sibling_work_batch(lead, members, tn, launched, blocks)
        if batch > 1:
            self.matmul_work_loop_batch(shape, members, outer, launched, tn, batch)
            return
        work = lead.configs * blocks
        w = self.reg('r')
        self.emit(f'mov.u32 {w}, {self.hardware_tid};')
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
                acc = self.prescale_hoisted_score(plan, acc)
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

    def sibling_work_batch(self, lead, members, tn, launched, blocks):
        """Admit config-only sibling items that share one serial B operand.

        ``matmul_work_loop`` advances a thread by ``launched`` work items.
        When that stride is a multiple of the column-block count, the serial
        coordinate is fixed and only the CTA-configuration coordinate moves.
        ``matmul_members`` has already proved the B expression independent of
        that configuration, so distinct items in the batch are independent
        accumulators reading one identical B value.  Their source folds retain
        their original per-element order.
        """
        if self.inactive or getattr(self, 'sibling_fusion_disabled', False):
            return 1
        if tn <= 0 or blocks <= 0 or launched % blocks:
            return 1
        stride = launched // blocks
        if stride <= 0 or lead.configs % stride:
            return 1
        items = lead.configs // stride
        # A partial batch leaves the same outer work loop around a narrower
        # body: it duplicates loop control and extends the live set without
        # removing the residual trip.  Admit batching only when one batch is
        # the thread's complete configuration run and that full live set fits
        # the explicit value budget.
        live = items * len(members) * tn + tn + 2 * len(members)
        if items > 1 and live <= SIBLING_WORK_BATCH_MAX_VALUES:
            return items
        return 1

    def matmul_work_loop_batch(self, shape, members, outer, launched, tn, batch):
        """Hoisted sibling dots sharing one serial B read per K step."""
        b, operands = shape
        lead = members[0]
        blocks = lead.extent // tn
        work = lead.configs * blocks
        stride = launched // blocks
        w = self.reg('r')
        self.emit(f'mov.u32 {w}, {self.hardware_tid};')
        begin, done = self.label('mm_batch'), self.label('mm_batch_items_done')
        self.mark(begin)
        p = self.reg('p')
        self.emit(f'setp.ge.u32 {p}, {w}, {work};')
        self.emit(f'@{p} bra {done};')
        block, config, serial = self.reg('r'), self.reg('r'), self.reg('r')
        self.emit(f'rem.u32 {block}, {w}, {blocks};')
        self.emit(f'div.u32 {config}, {w}, {blocks};')
        self.emit(f'mul.lo.u32 {serial}, {block}, {tn};')
        configs = []
        bindings = []
        for j in range(batch):
            if j:
                item_config = self.reg('r')
                self.emit(f'add.u32 {item_config}, {config}, {j * stride};')
            else:
                item_config = config
            configs.append(item_config)
            bindings.append({outer.axis.name: serial,
                             **self.decode_config(item_config, lead.used)})
        self.bind_coords(bindings[0])
        accs = []
        for _ in range(batch):
            member_rows = []
            for _ in members:
                row = []
                for _ in range(tn):
                    acc = self.reg('f')
                    self.emit(f'mov.f32 {acc}, {literal(0.0, "f32")};')
                    row.append(acc)
                member_rows.append(row)
            accs.append(member_rows)
        k = self.reg('r')
        self.emit(f'mov.s32 {k}, 0;')
        self.bind_coords({plan.statement.axis.name: k for plan in members})
        self.loop_invariants.append({})
        self.loop_expressions.append({})
        k_begin, k_done = self.label('mm_batch_k'), self.label('mm_batch_k_done')
        self.mark(k_begin)
        q = self.reg('p')
        self.emit(f'setp.ge.s32 {q}, {k}, {static(members[0].statement.axis.extent)};')
        self.emit(f'@{q} bra {k_done};')
        addr, space = self.address(b.input, b.index)
        bs = [self.reg('f') for _ in range(tn)]
        if tn == 4:
            self.emit(f'ld.{space}.v4.f32 {{{', '.join(bs)}}}, [{addr}];')
        else:
            self.emit(f'ld.{space}.f32 {bs[0]}, [{addr}];')
        for j, binding in enumerate(bindings):
            self.bind_coords({**binding,
                              **{plan.statement.axis.name: k for plan in members}})
            for (a, a_first), row in zip(operands, accs[j]):
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
        for item_config, member_rows in zip(configs, accs):
            for plan, row in zip(members, member_rows):
                for j, acc in enumerate(row):
                    acc = self.prescale_hoisted_score(plan, acc)
                    word = self.reg('r')
                    self.emit(f'mad.lo.u32 {word}, {item_config}, {plan.row_words}, {serial};')
                    if j:
                        self.emit(f'add.u32 {word}, {word}, {j};')
                    sa = self.reg('rd')
                    self.emit(f'mul.wide.u32 {sa}, {word}, 4;')
                    base = self.reg('rd')
                    self.emit(f'mov.u64 {base}, {plan.symbol};')
                    self.emit(f'add.u64 {sa}, {base}, {sa};')
                    self.emit(f'st.shared.f32 [{sa}], {acc};')
        self.emit(f'add.u32 {w}, {w}, {batch * launched};')
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
        packet = getattr(self, 'shared_scalar_packet', None) or {}
        if not packet:
            offset = self.reg('rd')
            self.emit(f'mul.wide.u32 {offset}, {self.value(serial)}, 4;')
            for name, row in zip(plan.exports, rows):
                addr, r = self.reg('rd'), self.reg('f')
                self.emit(f'add.u64 {addr}, {row}, {offset};')
                self.emit(f'ld.shared.f32 {r}, [{addr}];')
                self.values[name] = (r, 'f32')
                self.rebind(name)
            return
        offset = None
        for name, row in zip(plan.exports, rows):
            entry = packet.get((row, serial))
            r = self.reg('f')
            if entry is None:
                if offset is None:
                    offset = self.reg('rd')
                    self.emit(f'mul.wide.u32 {offset}, {self.value(serial)}, 4;')
                address = self.reg('rd')
                self.emit(f'add.u64 {address}, {row}, {offset};')
                operand = address
            else:
                address, immediate = entry
                operand = f'{address}+{immediate}'
            self.emit(f'ld.shared.f32 {r}, [{operand}];')
            self.values[name] = (r, 'f32')
            self.rebind(name)

    def statements(self, statements, shared_prefetch=None):
        pending = []
        plan = self.access_plan_for(statements, shared_prefetch or ())
        self.access_plan.loads.update(plan.loads)
        self.access_plan.stores.update(plan.stores)
        self.access_plan.load_lanes.update(plan.load_lanes)
        for position, s in enumerate(statements):
            for name, key in pending:
                self.vn_registers[key] = self.values[name]
            if id(s) in self.shared_probability_skip:
                pending = []
                continue
            pending = self.vn_record.get(id(s), [])
            if id(s) in self.shared_probability_betas:
                self.load_shared_probability(s)
                continue
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
            if id(s) in self.prescaled_assignments:
                score = self.prescaled_assignments[id(s)]
                snapshot = self.reg('f')
                self.emit(f'mov.f32 {snapshot}, {self.value(score)};')
                self.values[s.name] = (snapshot, 'f32')
                self.mutable.pop(s.name, None)
                self.rebind(s.name)
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
                self.emit(f'setp.ge.u32 {p}, {self.hardware_gid}, {n};')
                tile_done = None
                # A top-level Tile starts converged. When its logical extent
                # ends exactly at a physical CTA boundary, surplus CTAs can
                # skip the entire body uniformly, including coupled collectives.
                # Keep a Tile-local continuation so later siblings still run.
                whole_cta_surplus = (collective and inactive and n % launched == 0
                                     and not self.inactive
                                     and any(s is top for top in self.node.op.body))
                if not collective or whole_cta_surplus:
                    # A Tile guard scopes this body; later sibling statements
                    # still run for threads outside this Tile's coordinate set.
                    tile_done = self.label('tile_done')
                    self.emit(f'@{p} bra {tile_done};')
                    if whole_cta_surplus:
                        inactive = False
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
                    self.tile_extents = {axis.name: extent for axis, extent in zip(s.axes, extents)}
                    if not inactive and not self.inactive and not self.dynamic_shared:
                        self.plan_hoists(s, extents, launched)
                    previous_replay = getattr(self, 'rms_input_replay', None)
                    self.rms_input_replay = (
                        self.find_rms_input_replay(s, extents, launched)
                        if not inactive and not self.inactive else None)
                    try:
                        self.statements_with_init_scope(s.body, self.init_scope_frame(s.body))
                    finally:
                        self.rms_input_replay = previous_replay
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
                descriptor = self.access_plan.loads.get(id(s))
                if descriptor is not None:
                    self.vector_load_group(descriptor)
                    continue
                if id(s) in self.access_plan.load_lanes:
                    continue
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
                descriptor = self.access_plan.stores.get(id(s))
                if descriptor is not None and output_dtype == 'f32' and not s.atomic:
                    self.vector_store_group(descriptor)
                    continue
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
                dtype = ssa_dtype_name(s.dtype)
                if dtype == 'pred':
                    # The declaration casts its literal to BOOL; no BOOL storage.
                    r, dtype = self.snapshot_operand('1' if s.identity else '0', 'i32', 'pred')
                else:
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
                if getattr(s.dtype, 'name', s.dtype) == 'bool':
                    if reduce_canon(s.op.name) not in ('add', 'multiply', 'maximum', 'minimum'):
                        raise Unsupported('Unsupported BOOL reduction operation')
                    if any(self.binding(name)[1] != 'pred' for name in (s.name, left, s.value)):
                        raise Unsupported('BOOL reductions require all-BOOL carriers and operands')
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
                    # A seedFalse BOOL carrier may originate from a pure Let.
                    # Keep padding lanes' prior state when no mutable slot exists.
                    pred = (f'@{self.inactive[-1]} '
                            if self.values[s.name][1] == 'pred' and self.inactive else '')
                    self.emit(f'{pred}mov.{self.values[s.name][1]} {old}, {self.value(s.name)};')
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
                        body = list(s.body)
                        fixed, removed = {}, set()
                        online = (self.normalized_softmax_plans(s)
                                  if kind == 'Loop' and not self.inactive else [])
                        if id(s) in self.producer_probability_ready:
                            self.require_producer_probability_binding(s, online)
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
                    loop_init_frame = self.init_scope_frame(s.body)
                    extent = static(s.axis.extent)
                    if kind == 'StridedLoop' and self.rms_input_replay_loop(s, loop_init_frame):
                        continue
                    if kind == 'StridedLoop' and self.small_strided_softmax_pair(s):
                        continue
                    if kind == 'StridedLoop' and self.exact_short_strided_loop(s, loop_init_frame):
                        continue
                    if kind == 'StridedLoop' and self.serial_read_pipeline(s):
                        continue
                    if kind == 'Loop' and self.scalar_shared_address_packet_loop(s, loop_init_frame):
                        continue
                    if kind == 'Loop' and self.ordered_key_batch(s):
                        # KEY replays this loop body without the generic wrapper.
                        # Child dot seeds are body-local even on that route.
                        self.restore_init_scope(loop_init_frame)
                        continue
                    if kind == 'Loop' and self.global_dot_vector_loop(s, loop_init_frame):
                        continue
                    if (kind == 'Loop' and 8 <= extent <= I32_MAX and self.shared_compute_body(s)
                            and self.vector_load_plan(s) is None):
                        self.vector_loops += 1
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
                    if kind == 'StridedLoop' and not live_step:
                        self.strided_main_loop(s, r, step, end)
                    self.mark(begin)
                    p = self.reg('p')
                    self.emit(f'setp.ge.s32 {p}, {r}, {end};')
                    self.emit(f'@{p} bra {done};')
                    if vector is None:
                        self.statements_with_init_scope(s.body, loop_init_frame)
                    else:
                        self.vector_unrolled_body(s, r, vector)
                        step = str(VECTOR_UNROLL)
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
                    self.restore_shared_probability_phase(s)
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
                merge_frame = self.merge_scope_frame(s)
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
                    self.restore_merge_scope(merge_frame)
            elif kind == 'Smem':
                # Layout, buffer registration, and the PTX declaration are
                # emitted once by the initialization scan. Declaration order
                # does not depend on runtime statement order.
                pass
            elif kind == 'CpAsyncCopy':
                if s.swizzle not in ('NONE', None) or s.nbytes not in (4, 8, 16):
                    raise Unsupported('Only nonswizzled 4/8/16-byte copy groups supported')
                dtype = dtype_name(self.buffers[s.smem][1])
                source_dtype = dtype_name(self.buffers[s.src][1])
                nbytes = dtype_info(dtype)[0]
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
                if dtype == 'f16' and dtype_name(self.buffers[s.src][1]) == 'f16':
                    # cp.async has no 2-byte form, and a legal half-cell start
                    # need not be 4-byte aligned. Preserve every payload bit
                    # with scalar 16-bit transport, including half-offset
                    # chunks. Later commit/wait/barriers retain their order.
                    pred = f'@{active} ' if active is not None else ''
                    for offset in range(width):
                        src_half, _ = self.address(s.src, s.src_index, offset)
                        dst_half, _ = self.address(s.smem, destination, offset)
                        bits = self.reg('rh')
                        self.emit(f'{pred}ld.global.b16 {bits}, [{src_half}];')
                        self.emit(f'{pred}st.shared.b16 [{dst_half}], {bits};')
                    continue
                if source_dtype != dtype or nbytes < 4 or s.nbytes % nbytes:
                    # CpAsyncCopy transports contiguous bytes, not converted
                    # cells. Typed start pointers account for element widths
                    # once; subsequent offsets must be bytes. Byte operations
                    # also cover subword alignment and partial destination cells.
                    pred = f'@{active} ' if active is not None else ''
                    for byte_offset in range(s.nbytes):
                        src_byte, dst_byte = src, dst
                        if byte_offset:
                            src_byte, dst_byte = self.reg('rd'), self.reg('rd')
                            self.emit(f'add.u64 {src_byte}, {src}, {byte_offset};')
                            self.emit(f'add.u64 {dst_byte}, {dst}, {byte_offset};')
                        bits = self.reg('r')
                        self.emit(f'{pred}ld.global.u8 {bits}, [{src_byte}];')
                        self.emit(f'{pred}st.shared.u8 [{dst_byte}], {bits};')
                    continue
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
                merge_frame = self.merge_scope_frame(s)
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
                    self.restore_merge_scope(merge_frame)
            else:
                raise Unsupported(f'Unsupported statement {kind}')
        for name, key in pending:
            self.vn_registers[key] = self.values[name]


    def render(self):
        self.shared_probability_skip = set()
        self.shared_probability_betas = {}
        self.shared_probability_frames = {}
        self.producer_probability_certificates = {}
        self.producer_probability_ready = {}
        self.producer_maximum_certificates = {}
        self.producer_maxima = {}
        self.prescaled_hoists = {}
        self.prescaled_assignments = {}
        self.serial_read_plans = self.find_serial_read_plans(self.node.op.body)
        self.global_dot_alignments = {}
        self.global_dot_prologue_end = len(self.lines)
        self.key_batches = self.find_key_batches(self.node.op.body)
        self.global_dot_candidates = {id(dot) for plan in self.key_batches.values()
                                      for dot in plan[1]}
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
        return '\n'.join(['.version 8.0', '.target sm_89', '.address_size 64', '', *self.externals, f'.visible .entry {ident(self.node.op.name)}(', ',\n'.join(params), ')', bounds, '{', *decl, *self.lines, 'kernel_done:', '    ret;', '}', ''])


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
    simt = Emitter(graph, retile_redundant_hoists(node, matches[0]), matches[0])
    # Approximate reciprocal is eligible only for a single final F32
    # normalized output with an explicit, sufficiently broad error budget.
    # The typed two-trip matcher and sum-carrier check gate activation below.
    outputs = list(graph.outputs)
    simt.allow_softmax_approx_rcp = (
        len(kernels) == len(metadata['kernels']) == len(outputs) == 1
        and matches[0].get('final') is True
        and metadata.get('graph_outputs') == outputs
        and isinstance(metadata.get('rtol'), (int, float))
        and isinstance(metadata.get('atol'), (int, float))
        and metadata['rtol'] >= 1e-4 and metadata['atol'] >= 1e-6
        and len(metadata.get('kernels', ())) == 1
        and len(matches[0].get('args', ())) == 2
        and sum(bool(a.get('output')) for a in matches[0]['args']) == 1
        and all(a.get('dtype') == 'f32' for a in matches[0]['args']))
    own = tuple(node.buffer_names())
    simt.shared_probability_enabled = (
        len(kernels) == len(metadata['kernels']) == 1 and len(own) == 1
        and matches[0].get('final') is True
        and list(graph.outputs) == list(own) and metadata.get('graph_outputs') == list(own)
        and all(isinstance(n.inputs, (list, tuple))
                and all(isinstance(x, str) for x in n.inputs) for n in graph.nodes.values())
        and all(own[0] not in n.inputs for n in graph.nodes.values() if n is not node)
        and sum(own[0] in n.buffer_names() for n in graph.nodes.values()) == 1)
    simt_ptx = simt.render()
    if simt.vector_loops:
        return simt_ptx
    # The tensor-core path is fail-closed: only an independently validated
    # one-atom plan bypasses the existing SIMT emitter.  Every other typed
    # region retains the scoped SIMT lowering.
    try:
        import tensor_core
        from tf32_emitter import production_ptx
        if not tensor_core.implicit_f32_mma_numerically_legal():
            return simt_ptx
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
