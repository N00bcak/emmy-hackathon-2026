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
    'i32': (4, 's32', 's32'),
    'u32': (4, 'u32', 'u32'),
    'i64': (8, 's64', 's64'),
    'u64': (8, 'u64', 'u64'),
}
FLOAT_DTYPES = frozenset(('f32', 'f64'))
INT_DTYPES = frozenset(('i32', 'u32', 'i64', 'u64'))


def dtype_name(dtype):
    name = getattr(dtype, 'name', dtype)
    if name not in DTYPES:
        raise Unsupported(f'Unsupported buffer dtype {name}')
    return name


def dtype_info(dtype):
    return DTYPES[dtype_name(dtype)]


def literal(value, dtype):
    if dtype == 'f32':
        return f32(float(value))
    if dtype == 'f64':
        return f64(float(value))
    if dtype in INT_DTYPES:
        return str(int(value))
    raise Unsupported(f'Unsupported literal dtype {dtype}')


def reg_suffix(dtype):
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
    if dtype == 'f32':
        return 'f32'
    if dtype == 'f64':
        return 'f64'
    if dtype in ('i64', 'u64'):
        return 'b64'
    return 'b32'


def ensure_same(left, right):
    if left != right:
        raise Unsupported(f'Mixed expression dtypes {left}, {right}')


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


def statement_nbytes(statement):
    """Bytes occupied by a typed smem declaration."""
    dtype = getattr(statement, 'dtype', None)
    if dtype in ('float', 'f32', None):
        return 4 * math.prod(static(v) for v in statement.extents)
    if dtype in ('f64', 'i64', 'u64'):
        return 8 * math.prod(static(v) for v in statement.extents)
    if dtype in ('i32', 'u32'):
        return 4 * math.prod(static(v) for v in statement.extents)
    raise Unsupported(f'Unsupported shared memory dtype {dtype}')


class Emitter:
    def __init__(self, graph, node, meta):
        self.graph, self.node, self.meta = graph, node, meta
        self.lines, self.declarations = [], []
        self.externals = []
        self.counts = {'r': 0, 'rd': 0, 'f': 0, 'fd': 0, 'p': 0}
        self.values, self.buffers, self.shared = {}, {}, {}
        self.label_count = 0
        self.accum_ops = {}
        self.mutable = {}
        self.inactive = []
        self.shared_offsets = {}
        self.shared_statements = {}
        self.shared_bytes = 0
        def shared_layout(statements):
            for statement in statements:
                if type(statement).__name__ == 'Smem':
                    if statement.name in self.shared_offsets:
                        raise Unsupported('Duplicate shared buffer name')
                    align = max(4, statement.align)
                    self.shared_bytes = (self.shared_bytes + align - 1) // align * align
                    self.shared_offsets[statement.name] = self.shared_bytes
                    self.shared_bytes += statement_nbytes(statement)
                    dtype = 'f32' if statement.dtype in ('float', 'f32') else dtype_name(statement.dtype)
                    ident(statement.name)
                    self.buffers[statement.name] = (tuple(static(v) for v in statement.extents), dtype)
                    self.shared[statement.name] = statement
                    self.shared_statements[statement.name] = statement
                for field in ('body', 'else_body', 'combine_states'):
                    nested = getattr(statement, field, ())
                    if nested:
                        shared_layout(nested)
        shared_layout(node.op.body)
        self.dynamic_shared = bool(self.shared_bytes and meta.get('smem_bytes', 0) >= self.shared_bytes)
        if self.shared_bytes > 48 * 1024 and not self.dynamic_shared:
            raise Unsupported('Shared buffers exceed static limit and metadata dynamic allocation')
        if self.dynamic_shared:
            self.externals.append('.extern .shared .align 16 .b8 __emmy_smem[];')
        else:
            for name, statement in self.shared_statements.items():
                self.declarations.append(
                    f'.shared .align {max(4, statement.align)} .b8 {name}[{statement_nbytes(statement)}];')
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
        self.lines.append(label + ':')

    def rebind(self, name):
        """Invalidate address expressions that depend on a rebound value."""
        self.versions[name] = self.versions.get(name, 0) + 1

    def expr_key(self, x):
        """Stable value identity for pure index expressions.

        The textual index repeats across a loop body while its induction
        register is updated at the backedge. Including the register's
        version in this key lets those addresses share one computation
        without ever reusing a value from a previous iteration.
        """
        kind = type(x).__name__
        if kind == 'Var':
            return x.name
        if kind == 'Literal':
            return 'literal', repr(x.value), getattr(x, 'dtype', None)
        if kind == 'BinaryExpr':
            return 'binary', x.op, self.expr_key(x.left), self.expr_key(x.right)
        if kind == 'TernaryExpr':
            return 'ternary', self.expr_key(x.cond), self.expr_key(x.if_true), self.expr_key(x.if_false)
        return None

    def expr(self, x):
        kind = type(x).__name__
        if kind == 'Var':
            if x.name not in self.values:
                raise Unsupported(f'Unknown IR value {x.name}')
            return self.values[x.name]
        if kind == 'Literal':
            if isinstance(x.value, (bool, int)):
                if not -(1 << 31) <= int(x.value) < (1 << 31):
                    raise Unsupported('Integer literal exceeds signed32 bounds')
                return str(int(x.value)), 'i32'
            return f32(x.value), 'f32'
        if kind == 'BinaryExpr':
            a, at = self.expr(x.left)
            b, bt = self.expr(x.right)
            ensure_same(at, bt)
            comparisons = {'==':'eq', '!=':'ne', '<':'lt', '<=':'le', '>':'gt', '>=':'ge'}
            if x.op in comparisons:
                p = self.reg('p')
                self.emit(f'setp.{comparisons[x.op]}.{ptx_compare_type(at)} {p}, {a}, {b};')
                return p, 'pred'
            ops = {'+':'add.s32', '-':'sub.s32', '*':'mul.lo.s32', '//':'div.s32', '/':'div.s32', '%':'rem.s32', '&':'and.b32', '|':'or.b32', '>>':'shr.s32', '<<':'shl.b32'}
            if at != 'i32' or x.op not in ops:
                raise Unsupported(f'Unsupported expression operator {x.op} for {at}')
            r = self.reg('r')
            self.emit(f'{ops[x.op]} {r}, {a}, {b};')
            return r, 'i32'
        if kind == 'TernaryExpr':
            p, pt = self.expr(x.cond)
            a, at = self.expr(x.if_true)
            b, bt = self.expr(x.if_false)
            if pt != 'pred' or at != bt:
                raise Unsupported('Ternary dtype mismatch')
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
        return result

    def assign(self, name, op, args, dtype=None):
        op = getattr(op, 'name', None)
        operands = [self.value(a) for a in args]
        operand_types = [self.values[a][1] for a in args]
        if dtype is not None:
            result = dtype_name(dtype)
        else:
            if not operand_types:
                raise Unsupported(f'Elementwise operation {op} has no operands')
            result = operand_types[0]
        result = dtype_name(result)
        # A scalar copy is a pure register move and is well defined for every
        # typed buffer width. Keep all arithmetic float-only, but let integer
        # loads/address computations round-trip through an IR ``copy``.
        if result not in FLOAT_DTYPES and not (op == 'copy' and len(operands) == 1
                                               and operand_types == [result]):
            raise Unsupported(f'Non-floating elementwise arithmetic unsupported: {result}')
        if any(t != result for t in operand_types):
            raise Unsupported(f'Elementwise operands do not match result {result}: {operand_types}')
        reg = self.reg(reg_suffix(result))
        binary = {'add':'add', 'subtract':'sub', 'multiply':'mul', 'divide':'div.rn', 'maximum':'max', 'minimum':'min'}
        unary = {'negative':'neg', 'abs':'abs', 'sqrt':'sqrt.rn', 'reciprocal':'rcp.rn'}
        if op in binary and len(operands) == 2:
            self.emit(f'{binary[op]}.{result} {reg}, {operands[0]}, {operands[1]};')
        elif op in unary and len(operands) == 1:
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
        elif op == 'sigmoid' and len(operands) == 1:
            if result != 'f32':
                raise Unsupported('sigmoid is only supported for f32')
            scaled = self.reg('f')
            exponent = self.reg('f')
            self.emit(f'mul.f32 {scaled}, {operands[0]}, {f32(-math.log2(math.e))};')
            self.emit(f'ex2.approx.f32 {exponent}, {scaled};')
            self.emit(f'add.f32 {exponent}, {exponent}, {f32(1.0)};')
            self.emit(f'rcp.approx.f32 {reg}, {exponent};')
        elif op == 'relu' and len(operands) == 1:
            self.emit(f'max.{result} {reg}, {operands[0]}, {literal(0, result)};')
        else:
            raise Unsupported(f'Unsupported elementwise operation {op}/{len(args)} under {result}')
        if name in self.mutable:
            stable = self.mutable[name]
            self.emit(f'mov.{result} {stable}, {reg};')
            self.values[name] = (stable, result)
        else:
            self.values[name] = (reg, result)

    def init_accums(self, statements):
        for s in statements:
            if type(s).__name__ == 'Accum':
                op = s.op.name
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
            if tree.inner is not None or tree.length <= 0 or tree.length & (tree.length - 1):
                raise Unsupported('Unsupported shared tree configuration')
            if len(tree.bufs) != len(tree.combine_states):
                raise Unsupported('Tree state and combine arity differ')
            for buf, combine in zip(tree.bufs, tree.combine_states, strict=True):
                op = combine.op.name
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
                for slot in range(tree.length):
                    active = self.reg('p')
                    self.emit(f'setp.eq.u32 {active}, {self.value("tid")}, {slot};')
                    from emmy.compiler.ir.expr import Literal
                    addr, space = self.address(buf, (Literal(slot, 'int'),))
                    if space != 'shared':
                        raise Unsupported('Tree buffer is not shared')
                    self.emit(f'@{active} st.shared.{dtype_info(dtype)[1]} [{addr}], {literal_value};')

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
                for axis in s.axes:
                    window = axis.window
                    if window is not None and (window.base is not None or window.bound is not None or window.partition):
                        raise Unsupported('Nontrivial window bounds unsupported')
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
                partial = n % launched != 0
                active = None
                p = self.reg('p')
                self.emit(f'setp.ge.u32 {p}, {self.value("_gid")}, {n};')
                if partial:
                    active = self.reg('p')
                    self.emit(f'setp.lt.u32 {active}, {self.value("_gid")}, {n};')
                    if not collective:
                        self.emit(f'@!{active} bra kernel_done;')
                elif not collective:
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
                if partial and collective:
                    # Inactive threads still execute the body to reach barriers.
                    # Their addresses are masked at every memory access; the
                    # axis registers themselves are consumed only within it.
                    self.preinitialize_trees(s.body)
                    self.inactive.append(active)
                self.statements(s.body)
                if partial and collective:
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
                dtype = dtype_name(s.value_dtype) if s.value_dtype is not None else dtype_name(self.buffers[s.output][1])
                phys = dtype_info(dtype)[1]
                for i, name in enumerate(s.values):
                    addr, space = self.address(s.output, s.index, i)
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
                        if dtype not in ('f32', 'f64'):
                            raise Unsupported(f'Only floating atomics supported: {dtype}')
                        pred = f'@{active} ' if guarded else ''
                        self.emit(f'{pred}red.global.add.{phys} [{addr}], {self.value(name)};')
                    else:
                        pred = f'@{active} ' if guarded else ''
                        self.emit(f'{pred}st.{space}.{phys} [{addr}], {self.value(name)};')
            elif kind == 'Assign':
                self.assign(s.name, s.op, s.args, s.dtype)
            elif kind == 'Let':
                self.values[s.name] = self.expr(s.value)
            elif kind == 'Init':
                dtype = dtype_name(s.dtype)
                if dtype not in FLOAT_DTYPES:
                    raise Unsupported(f'Only floating carried state supported: {dtype}')
                r = self.reg(reg_suffix(dtype))
                self.emit(f'mov.{dtype} {r}, {literal(s.identity, dtype)};')
                self.values[s.name] = (r, dtype)
                self.mutable[s.name] = r
            elif kind == 'Accum':
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
                begin, done = self.label('loop'), self.label('loop_done')
                self.mark(begin)
                p = self.reg('p')
                self.emit(f'setp.ge.s32 {p}, {r}, {end};')
                self.emit(f'@{p} bra {done};')
                self.statements(s.body)
                self.emit(f'add.s32 {r}, {r}, {step};')
                self.emit(f'bra {begin};')
                self.mark(done)
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
                    for a, b in zip(s.state, s.state_b, strict=True):
                        r = self.reg(reg_suffix(dtype))
                        self.emit(f'shfl.sync.bfly.b32 {r}, {self.value(a)}, {offset}, 31, {mask};')
                        self.values[b] = (r, dtype)
                    self.statements(s.combine_states)
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
                for offset in range(s.nbytes // nbytes):
                    src, source_space = self.address(s.src, s.src_index, offset)
                    dst, destination_space = self.address(s.smem, destination, offset)
                    if source_space != 'global' or destination_space != 'shared':
                        raise Unsupported('Copy must transport global to shared')
                    register = self.reg(reg_suffix(dtype))
                    self.emit(f'ld.global.{dtype_info(dtype)[1]} {register}, [{src}];')
                    self.emit(f'st.shared.{dtype_info(dtype)[1]} [{dst}], {register};')
            elif kind in ('CpAsyncCommit', 'CpAsyncWait'):
                # Copies above are synchronous. Preserve the IR's existing CTA
                # barriers; adding barriers inside cooperative copy loops is unsafe.
                pass
            elif kind == 'Cond':
                p, dtype = self.expr(s.cond)
                if dtype != 'pred':
                    raise Unsupported('Expected predicate for conditional')
                otherwise, done = self.label('else'), self.label('endif')
                self.emit(f'@!{p} bra {otherwise};')
                self.statements(s.body)
                self.emit(f'bra {done};')
                self.mark(otherwise)
                self.statements(s.else_body)
                self.mark(done)
            elif kind == 'Sync':
                if s.warp or s.count is not None:
                    raise Unsupported('Only full CTA barriers supported')
                self.emit(f'bar.sync {s.barrier_id};')
            elif kind == 'TreeHalve':
                if s.tid_var != "warp":
                    raise Unsupported("Only warp-indexed TreeHalve is supported")
                if s.inner is not None or s.barrier_count is not None or s.length <= 0 or s.length & (s.length-1):
                    raise Unsupported('Unsupported shared tree configuration')
                # Initial shared partials have already been written by the IR.
                # One lane per warp performs stores to avoid identical-write races.
                size = s.length // 2
                while size:
                    p = self.reg('p')
                    self.emit(f'setp.lt.s32 {p}, {self.value(s.tid_var)}, {size};')
                    skip = self.label('tree_skip')
                    self.emit(f'@!{p} bra {skip};')
                    for buf, a, b in zip(s.bufs, s.state, s.state_b, strict=True):
                        from emmy.compiler.ir.expr import Var, BinaryExpr, Literal
                        idx = Var(s.tid_var)
                        addr, _ = self.address(buf, (idx,))
                        dtype = dtype_name(self.buffers[buf][1])
                        ar = self.reg(reg_suffix(dtype))
                        self.emit(f'ld.shared.{dtype_info(dtype)[1]} {ar}, [{addr}];')
                        self.values[a] = (ar, dtype)
                        addr2, _ = self.address(buf, (BinaryExpr('+',idx,Literal(size,'int')),))
                        br = self.reg(reg_suffix(dtype))
                        self.emit(f'ld.shared.{dtype_info(dtype)[1]} {br}, [{addr2}];')
                        self.values[b] = (br, dtype)
                    self.statements(s.combine_states)
                    single = self.reg('p')
                    self.emit(f'setp.eq.s32 {single}, {self.value("lane")}, 0;')
                    for buf, a in zip(s.bufs, s.state, strict=True):
                        addr, _ = self.address(buf, (idx,))
                        dtype = dtype_name(self.buffers[buf][1])
                        self.emit(f'@{single} st.shared.{dtype_info(dtype)[1]} [{addr}], {self.value(a)};')
                    self.mark(skip)
                    self.emit(f'bar.sync {s.barrier_id};')
                    size //= 2
                for buf, a in zip(s.bufs, s.state, strict=True):
                    from emmy.compiler.ir.expr import Literal
                    addr, _ = self.address(buf, (Literal(0,'int'),))
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
        decl = [f'    .reg .{suffix} %{kind}<{self.counts[kind]+1}>;' for kind,suffix in [('r','b32'),('rd','b64'),('f','f32'),('fd','f64'),('p','pred')]]
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
