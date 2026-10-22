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
        if self.has_cp_async and self.dynamic_shared:
            base = self.reg('rd')
            self.emit(f'mov.u64 {base}, {self.cp_async_base};')
            self._dynamic_shared_base = base
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

    def statements(self, statements):
        for s in statements:
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
                self.hoist_invariants(s.body, defined_names(s.body) | {s.axis.name})
                begin, done = self.label('loop'), self.label('loop_done')
                self.mark(begin)
                p = self.reg('p')
                self.emit(f'setp.ge.s32 {p}, {r}, {end};')
                self.emit(f'@{p} bra {done};')
                self.statements(s.body)
                self.emit(f'add.s32 {r}, {r}, {step};')
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
                return Emitter(graph, node, matches[0]).render()
    except (ImportError, ValueError, KeyError, TypeError, AttributeError,
            IndexError, ArithmeticError) as exc:
        print(f'compile.py: tensor-core fallback: {type(exc).__name__}: {exc}',
              file=sys.stderr)
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
