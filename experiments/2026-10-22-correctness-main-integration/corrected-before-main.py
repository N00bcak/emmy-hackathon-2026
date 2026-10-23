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


def partial_tile_has_complete_subgroups(statements, cells, block_threads):
    """Prove a partial CTA contains complete, converged shuffle groups.

    Only direct pure-combine shuffles are accepted. Nested tiles can exit live
    lanes; reject them too. A collective inside a condition or loop
    needs a separate participation proof; CTA barriers/shared trees cannot use
    the per-shuffle active mask to exclude the tile's exited threads.
    """
    for statement in statements:
        kind = type(statement).__name__
        if kind == 'WarpShuffle':
            width = statement.length
            if (not isinstance(width, int) or width <= 0 or width > 32
                    or width & (width - 1)
                    or block_threads % width or cells % width):
                return False
            if any(type(combine).__name__ != 'Assign' for combine in statement.combine_states):
                return False
        elif kind in ('Sync', 'TreeHalve', 'Tile') or has_collective((statement,)):
            return False
        else:
            for field in ('body', 'else_body'):
                nested = getattr(statement, field, ())
                if nested and not partial_tile_has_complete_subgroups(nested, cells, block_threads):
                    return False
    return True


I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1


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


def positive_strided_interval(start, step, end):
    """Bounds hold at every executed body entry, including later iterations."""
    if start is None or step is None or end is None:
        return None
    if any(checked_i32_interval(*bounds) is None for bounds in (start, step, end)):
        return None
    if start[0] < 0 or step[0] <= 0 or start[0] >= end[1]:
        return None
    upper = end[1] - 1
    # The last executed update must remain representable before the next test.
    if upper + step[1] > I32_MAX:
        return None
    return checked_i32_interval(start[0], upper)


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
        if self.block_threads * self.grid_blocks > I32_MAX:
            raise Unsupported('Linear launch geometry exceeds signed32 bounds')
        self.geometry_3d = any(d != 1 for d in (*meta['block'][1:], *meta['grid'][1:]))
        self.physical_builtins = {}
        self.lines, self.declarations = [], []
        self.externals = []
        self.counts = {'r': 0, 'rd': 0, 'f': 0, 'p': 0, 'h': 0, 'd': 0}
        self.values, self.buffers, self.shared = {}, {}, {}
        self.int_ranges = {}
        self.label_count = 0
        self.accum_ops = {}
        self.partial_tile_shuffle = False
        self.mutable = {}
        self.shared_offsets = {}
        shared_bytes = 0
        def shared_layout(statements):
            nonlocal shared_bytes
            for statement in statements:
                if type(statement).__name__ == 'Smem':
                    if statement.name in self.shared_offsets:
                        raise Unsupported('Duplicate shared buffer name')
                    align = max(4, statement.align)
                    shared_bytes = (shared_bytes + align - 1) // align * align
                    self.shared_offsets[statement.name] = shared_bytes
                    shared_bytes += 4 * math.prod(static(v) for v in statement.extents)
                for field in ('body', 'else_body'):
                    nested = getattr(statement, field, ())
                    if nested:
                        shared_layout(nested)
        shared_layout(node.op.body)
        self.dynamic_shared = bool(shared_bytes and meta.get('smem_bytes', 0) >= shared_bytes)
        if shared_bytes > 48 * 1024 and not self.dynamic_shared:
            raise Unsupported('Shared buffers exceed static limit and metadata dynamic allocation')
        if self.dynamic_shared:
            self.externals.append('.extern .shared .align 16 .b8 __emmy_smem[];')
        for n in graph.nodes.values():
            for buffer_name, tensor in zip(n.buffer_names(), n.outputs, strict=True):
                self.buffers[buffer_name] = (tuple(static(d) for d in tensor.shape), tensor.dtype.name)
        arguments = {arg['name']: arg for arg in meta['args']}
        if (len(arguments) != len(meta['args']) or len(meta['arg_order']) != len(meta['args'])
                or set(arguments) != set(meta['arg_order'])):
            raise Unsupported('Metadata arguments must match positional arg_order exactly')
        self.scalar_parameters = {}
        self.scalar_shapes = {}
        for arg in meta['args']:
            name = arg['name']
            if arg.get('scalar'):
                shape, graph_dtype = self.buffers.get(name, (None, None))
                declared_shape = arg.get('shape', shape if shape is not None else ())
                if (declared_shape is None or len(declared_shape) > 1 or math.prod(declared_shape) != 1
                        or (shape is not None and (len(shape) > 1 or math.prod(shape) != 1
                                                   or graph_dtype != arg['dtype']))
                        or arg.get('output') or name in meta.get('zero_outputs', ())):
                    raise Unsupported('Scalar parameter requires consistent scalar metadata and graph storage')
                if arg['dtype'] not in ('f16', 'f32', 'f64', 'i32', 'i64'):
                    raise Unsupported('Unsupported scalar parameter dtype')
                self.scalar_shapes[name] = shape if shape is not None else tuple(declared_shape)
                self.buffers.pop(name, None)
            else:
                self.buffers[name] = (tuple(arg['shape']), arg['dtype'])
        if node.op.serial:
            raise Unsupported('Serial runtime axes are not supported')
        self.ptrs = {}
        self.parameter_types = []
        scalar_types = {'f16': ('h', 'b16'), 'f32': ('f', 'f32'),
                        'f64': ('d', 'f64'), 'i32': ('r', 's32'),
                        'i64': ('rd', 's64')}
        for i, name in enumerate(meta['arg_order']):
            ident(name)
            argument = arguments[name]
            if argument.get('scalar'):
                register_kind, suffix = scalar_types[argument['dtype']]
                operand = self.reg(register_kind)
                self.emit(f'ld.param.{suffix} {operand}, [param_{i}];')
                self.scalar_parameters[name] = (operand, argument['dtype'])
                self.parameter_types.append(suffix)
            else:
                ptr = self.reg('rd')
                self.emit(f'ld.param.u64 {ptr}, [param_{i}];')
                self.emit(f'cvta.to.global.u64 {ptr}, {ptr};')
                self.ptrs[name] = ptr
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
                self.set_integer_range(reg, checked_i32_interval(0, (self.block_threads - 1) // 32))

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

    def expr(self, x):
        kind = type(x).__name__
        if kind == 'Builtin':
            return self.physical_builtin(x.name)
        if kind == 'Var':
            if x.name not in self.values:
                raise Unsupported(f'Unknown IR value {x.name}')
            return self.values[x.name]
        if kind == 'Literal':
            # Match typed expression semantics: default literals are floating,
            # even when their JSON/Python value happens to be an integer.
            if isinstance(x.value, float) or x.dtype == 'float':
                return f32(float(x.value)), 'f32'
            if isinstance(x.value, (bool, int)):
                if not -(1 << 31) <= int(x.value) < (1 << 31):
                    raise Unsupported('Integer literal exceeds signed32 bounds')
                return str(int(x.value)), 'i32'
            return f32(x.value), 'f32'
        if kind == 'BinaryExpr':
            a, at = self.expr(x.left)
            b, bt = self.expr(x.right)
            if at != bt:
                raise Unsupported(f'Mixed expression dtypes {at}, {bt}')
            comparisons = {'==':'eq', '!=':'ne', '<':'lt', '<=':'le', '>':'gt', '>=':'ge'}
            if x.op in comparisons:
                p = self.reg('p')
                comparison = 'neu' if at == 'f32' and x.op == '!=' else comparisons[x.op]
                self.emit(f'setp.{comparison}.{("f32" if at == "f32" else "s32")} {p}, {a}, {b};')
                return p, 'pred'
            if at == 'f32':
                operations = {'+': 'add.rn.f32', '-': 'sub.rn.f32',
                              '*': 'mul.rn.f32', '/': 'div.rn.f32'}
                if x.op not in operations:
                    raise Unsupported(f'Unsupported expression operator {x.op} for {at}')
                r = self.reg('f')
                self.emit(f'{operations[x.op]} {r}, {a}, {b};')
                return r, 'f32'
            ops = {'+':'add.s32', '-':'sub.s32', '*':'mul.lo.s32', '//':'div.s32', '/':'div.s32', '%':'rem.s32', '&':'and.b32', '|':'or.b32', '>>':'shr.s32', '<<':'shl.b32'}
            if at != 'i32' or x.op not in ops:
                raise Unsupported(f'Unsupported expression operator {x.op} for {at}')
            if x.op in ('/', '//', '%'):
                return self.integer_floor_divmod(x.op, a, b)
            r = self.reg('r')
            self.emit(f'{ops[x.op]} {r}, {a}, {b};')
            self.set_integer_range(r, i32_binary_interval(x.op, self.integer_range(a), self.integer_range(b)))
            return r, 'i32'
        if kind == 'TernaryExpr':
            p, pt = self.expr(x.cond)
            a, at = self.expr(x.if_true)
            b, bt = self.expr(x.if_false)
            if pt != 'pred' or at != bt:
                raise Unsupported('Ternary dtype mismatch')
            if at == 'pred':
                # Select predicates without treating predicate registers as b32.
                yes, no, inverted, r = (self.reg('p') for _ in range(4))
                self.emit(f'and.pred {yes}, {p}, {a};')
                self.emit(f'not.pred {inverted}, {p};')
                self.emit(f'and.pred {no}, {inverted}, {b};')
                self.emit(f'or.pred {r}, {yes}, {no};')
                return r, 'pred'
            r = self.reg('f' if at == 'f32' else 'r')
            self.emit(f'selp.{("f32" if at == "f32" else "b32")} {r}, {a}, {b}, {p};')
            if at == 'i32':
                left, right = self.integer_range(a), self.integer_range(b)
                if left is not None and right is not None:
                    self.set_integer_range(r, checked_i32_interval(min(left[0], right[0]), max(left[1], right[1])))
            return r, at
        if kind == 'CastExpr':
            operand, source_dtype = self.expr(x.expr)
            dtype = {'float': 'f32', 'int': 'i32', 'bool': 'pred',
                     'f32': 'f32', 'i32': 'i32'}.get(x.dtype)
            if dtype is None:
                raise Unsupported(f'Unsupported expression cast {x.dtype}')
            return self.convert_operand(operand, source_dtype, dtype, copy_same_type=False)
        if kind == 'FuncCallExpr':
            return self.funcall_operation(x.name, [self.expr(argument) for argument in x.args])
        if kind == 'FlatIndex':
            return self.flatten_offset(x.buffer, x.index, record_ranges=True), 'i32'
        raise Unsupported(f'Unsupported expression {kind}')


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

    def value(self, name):
        if name not in self.values:
            raise Unsupported(f'Unbound SSA value {name!r}')
        return self.values[name][0]

    def convert_operand(self, operand, source_dtype, dtype, *, copy_same_type=True):
        if dtype == source_dtype and not copy_same_type:
            return operand, dtype
        register_kind = {'f32': 'f', 'i32': 'r', 'pred': 'p'}.get(dtype)
        if register_kind is None:
            raise Unsupported(f'Unsupported scalar conversion destination {dtype}')
        register = self.reg(register_kind)
        if dtype == source_dtype:
            suffix = {'f32': 'f32', 'i32': 'b32', 'pred': 'pred'}[dtype]
            self.emit(f'mov.{suffix} {register}, {operand};')
            if dtype == 'i32':
                self.set_integer_range(register, self.integer_range(operand))
        elif (source_dtype, dtype) == ('f32', 'i32'):
            # The valid conversion domain is finite, representable signed32.
            self.emit(f'cvt.rzi.s32.f32 {register}, {operand};')
        elif (source_dtype, dtype) == ('i32', 'f32'):
            self.emit(f'cvt.rn.f32.s32 {register}, {operand};')
        elif dtype == 'pred' and source_dtype in ('f32', 'i32'):
            suffix = 'neu.f32' if source_dtype == 'f32' else 'ne.s32'
            zero = f32(0.0) if source_dtype == 'f32' else '0'
            self.emit(f'setp.{suffix} {register}, {operand}, {zero};')
        elif source_dtype == 'pred' and dtype in ('f32', 'i32'):
            suffix = 'f32' if dtype == 'f32' else 'b32'
            one, zero = (f32(1.0), f32(0.0)) if dtype == 'f32' else ('1', '0')
            self.emit(f'selp.{suffix} {register}, {one}, {zero}, {operand};')
            if dtype == 'i32':
                self.set_integer_range(register, (0, 1))
        else:
            raise Unsupported(f'Unsupported scalar conversion {source_dtype} to {dtype}')
        return register, dtype

    def f32_value(self, name):
        operand, dtype = self.values[name]
        return self.convert_operand(operand, dtype, 'f32', copy_same_type=False)[0]

    def scalar_load(self, statement):
        name = statement.input
        shape = self.scalar_shapes[name]
        if len(statement.names) != 1 or (len(statement.index) != len(shape) and statement.index):
            raise Unsupported('Scalar parameter Load must address exactly one scalar cell')
        for coordinate in statement.index:
            operand, dtype = self.expr(coordinate)
            if dtype != 'i32' or self.integer_range(operand) != (0, 0):
                raise Unsupported('Scalar parameter Load index must be proven zero')
        operand, source_dtype = self.scalar_parameters[name]
        dtype = source_dtype if statement.dtype is None else statement.dtype.name
        register_kind = {'f16': 'h', 'f32': 'f', 'f64': 'd', 'i32': 'r', 'i64': 'rd'}.get(dtype)
        if register_kind is None:
            raise Unsupported('Unsupported scalar Load dtype')
        register = self.reg(register_kind)
        if dtype == source_dtype:
            suffix = {'f16': 'b16', 'f32': 'f32', 'f64': 'f64', 'i32': 'b32', 'i64': 'b64'}[dtype]
            self.emit(f'mov.{suffix} {register}, {operand};')
        elif dtype == 'f32':
            suffix = {'f16': 'f32.f16', 'f64': 'rn.f32.f64',
                      'i32': 'rn.f32.s32', 'i64': 'rn.f32.s64'}[source_dtype]
            self.emit(f'cvt.{suffix} {register}, {operand};')
        elif dtype == 'i32':
            if source_dtype == 'f16':
                widened = self.reg('f')
                self.emit(f'cvt.f32.f16 {widened}, {operand};')
                self.emit(f'cvt.rzi.s32.f32 {register}, {widened};')
            else:
                suffix = {'f32': 'rzi.s32.f32', 'f64': 'rzi.s32.f64', 'i64': 's32.s64'}[source_dtype]
                self.emit(f'cvt.{suffix} {register}, {operand};')
        else:
            raise Unsupported(f'Unsupported scalar Load conversion {source_dtype} to {dtype}')
        self.values[statement.names[0]] = (register, dtype)

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

    def address(self, name, index, element_offset=0, *, expected_dtype='f32'):
        if name not in self.buffers:
            raise Unsupported(f'Unknown buffer {name}')
        _, dtype = self.buffers[name]
        if dtype != expected_dtype or dtype not in ('f32', 'i32'):
            raise Unsupported(f'Address dtype mismatch: expected {expected_dtype}, got {dtype}')
        offset = self.flatten_offset(name, index, element_offset)
        byte = self.reg('rd')
        self.emit(f'mul.wide.s32 {byte}, {offset}, 4;')
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
        return addr, 'shared' if name in self.shared else 'global'

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

    def scalar_operation(self, op, args, dtype=None):
        operands = [operand for operand, _ in args]
        argument_types = [dtype for _, dtype in args]
        integer_operations = {'add': ('add.s32', '+'), 'subtract': ('sub.s32', '-'),
                              'multiply': ('mul.lo.s32', '*')}
        result_dtype = getattr(dtype, 'name', dtype) if dtype is not None else (
            'i32' if op in integer_operations and argument_types
            and all(t == 'i32' for t in argument_types) else 'f32')
        if result_dtype == 'i32':
            if any(t != 'i32' for t in argument_types):
                raise Unsupported('Native i32 Assign requires i32 operands')
            register = self.reg('r')
            if op in integer_operations and len(operands) == 2:
                instruction, interval_op = integer_operations[op]
                self.emit(f'{instruction} {register}, {operands[0]}, {operands[1]};')
                interval = i32_binary_interval(interval_op, *(self.integer_range(a) for a in operands))
            elif op == 'copy' and len(operands) == 1:
                self.emit(f'mov.b32 {register}, {operands[0]};')
                interval = self.integer_range(operands[0])
            else:
                raise Unsupported(f'Unsupported i32 Assign {op}/{len(args)}')
            self.set_integer_range(register, interval)
            return register, 'i32'
        if result_dtype != 'f32':
            raise Unsupported(f'Only f32/i32 Assign supported: {result_dtype}')
        operands = [self.convert_operand(operand, dtype, 'f32', copy_same_type=False)[0]
                    for operand, dtype in args]
        reg = self.reg('f')
        binary = {'add':'add.f32', 'subtract':'sub.f32', 'multiply':'mul.f32', 'divide':'div.rn.f32', 'maximum':'max.NaN.f32', 'minimum':'min.NaN.f32'}
        unary = {'negative':'neg.f32', 'negate':'neg.f32', 'abs':'abs.f32', 'sqrt':'sqrt.rn.f32', 'reciprocal':'rcp.rn.f32', 'rsqrt':'rsqrt.approx.f32'}
        if op in binary and len(operands) == 2:
            self.emit(f'{binary[op]} {reg}, {operands[0]}, {operands[1]};')
        elif op in unary and len(operands) == 1:
            self.emit(f'{unary[op]} {reg}, {operands[0]};')
        elif op == 'copy' and len(operands) == 1:
            self.emit(f'mov.f32 {reg}, {operands[0]};')
        elif op == 'exp' and len(operands) == 1:
            scaled = self.reg('f')
            self.emit(f'mul.f32 {scaled}, {operands[0]}, {f32(math.log2(math.e))};')
            self.emit(f'ex2.approx.f32 {reg}, {scaled};')
        elif op == 'log' and len(operands) == 1:
            logarithm = self.reg('f')
            self.emit(f'lg2.approx.f32 {logarithm}, {operands[0]};')
            self.emit(f'mul.rn.f32 {reg}, {logarithm}, {f32(math.log(2.0))};')
        elif op == 'relu' and len(operands) == 1:
            self.emit(f'max.NaN.f32 {reg}, {operands[0]}, {f32(0)};')
        else:
            raise Unsupported(f'Unsupported elementwise operation {op}/{len(args)}')
        return reg, 'f32'

    def assign(self, name, op, args, dtype=None):
        register, result_dtype = self.scalar_operation(getattr(op, 'name', None),
                                                      [self.values[a] for a in args], dtype)
        if name in self.mutable:
            if result_dtype != 'f32':
                raise Unsupported('Cannot replace an f32 carried state with i32 Assign')
            stable = self.mutable[name]
            self.emit(f'mov.f32 {stable}, {register};')
            self.values[name] = (stable, 'f32')
        else:
            self.values[name] = (register, result_dtype)

    def init_accums(self, statements):
        for s in statements:
            if type(s).__name__ == 'Accum':
                op = s.op.name
                identities = {'add': 0.0, 'multiply': 1.0, 'maximum': -1e30, 'minimum': 1e30}
                if op not in identities:
                    raise Unsupported(f'Unsupported reduction {op}')
                reg = self.reg('f')
                self.emit(f'mov.f32 {reg}, {f32(identities[op])};')
                self.values[s.name] = (reg, 'f32')
                self.mutable[s.name] = reg
                self.accum_ops[s.name] = op
            elif type(s).__name__ == 'Cond':
                self.init_accums(s.body)
                self.init_accums(s.else_body)

    def shared_compute_body(self, statements):
        """Count shared loads in a register/shared-only body, or reject it.

        No instruction is emitted here. Conditions may have scalar bodies, but
        no allowed statement exits a thread or synchronizes another thread.
        """
        loads = 0
        for statement in statements:
            kind = type(statement).__name__
            if kind == 'Load':
                if statement.input not in self.shared:
                    return None
                loads += 1
            elif kind == 'Cond':
                for branch in (statement.body, statement.else_body):
                    count = self.shared_compute_body(branch)
                    if count is None:
                        return None
                    loads += count
            elif kind not in ('Assign', 'Accum', 'Let', 'Init'):
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
                    and expr.dtype != 'float'):
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
            if statement.dtype is not None and statement.dtype.name != 'f32':
                continue
            declaration = self.shared[statement.input]
            if self.dynamic_shared:
                aligned = self.shared_offsets[statement.input] % 16 == 0
            else:
                aligned = max(4, declaration.align) % 16 == 0
            if not aligned:
                continue
            shape, dtype = self.buffers[statement.input]
            if (dtype != 'f32' or len(statement.index) != len(shape)
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
        # Only an existing binding is shadowed; fresh Init exports retain
        # the compiler's established carried-state behavior.
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
                for axis in s.axes:
                    window = axis.window
                    if window is not None and (window.base is not None or window.bound is not None or window.partition):
                        raise Unsupported('Nontrivial window bounds unsupported')
                extents = [static(a.extent) for a in s.axes]
                if s.block_threads is not None and s.block_threads != self.block_threads:
                    raise Unsupported('Tile block_threads differs from prescribed launch')
                if s.raster_axes:
                    # Prove raster coordinates are block-uniform: all within-CTA
                    # coordinate factors form a suffix of exactly block_threads.
                    cuts = [i for i in range(len(extents) + 1)
                            if math.prod(extents[i:]) == self.block_threads]
                    if not cuts or any(name not in [axis.name for axis in s.axes[:max(cuts)]]
                                       for name in s.raster_axes):
                        raise Unsupported('Raster axes are not proven block-uniform')
                n = math.prod(extents)
                if n <= 0 or n >= (1 << 31):
                    raise Unsupported('Tile size outside supported signed32 range')
                # CTA collectives need full participation. Direct shuffle-only tails
                # are safe when the guard removes complete aligned subgroups.
                previous_tile_ranges = dict(self.int_ranges)
                collective = has_collective(s.body)
                if (collective and n % self.block_threads
                        and not (len(self.node.op.body) == 1 and self.node.op.body[0] is s
                                 and partial_tile_has_complete_subgroups(s.body, n, self.block_threads))):
                    raise Unsupported('Collective tile with partial final CTA unsupported')
                p = self.reg('p')
                self.emit(f'setp.ge.u32 {p}, {self.hardware_gid}, {n};')
                self.emit(f'@{p} bra kernel_done;')
                self.set_integer_range(self.hardware_gid, (0, n - 1))
                positive_axes = all(extent > 0 for extent in extents)
                for i, axis in enumerate(s.axes):
                    stride = math.prod(extents[i+1:])
                    r = self.reg('r')
                    self.emit(f'div.u32 {r}, {self.hardware_gid}, {stride};')
                    if i:
                        self.emit(f'rem.u32 {r}, {r}, {extents[i]};')
                    self.values[axis.name] = (r, 'i32')
                    self.set_integer_range(r, checked_i32_interval(0, extents[i] - 1) if positive_axes else None)
                previous_partial = self.partial_tile_shuffle
                self.partial_tile_shuffle = bool(collective and n % self.block_threads)
                self.statements(s.body)
                self.partial_tile_shuffle = previous_partial
                self.int_ranges = previous_tile_ranges
            elif kind == 'Load':
                if s.input in self.scalar_parameters:
                    self.scalar_load(s)
                    continue
                if shared_prefetch is not None and position in shared_prefetch:
                    self.values[s.names[0]] = (shared_prefetch[position], 'f32')
                    continue
                if s.input not in self.buffers:
                    raise Unsupported(f'Unknown buffer {s.input}')
                source_dtype = self.buffers[s.input][1]
                dtype = source_dtype if s.dtype is None else s.dtype.name
                if dtype not in ('f32', 'i32') or dtype != source_dtype:
                    raise Unsupported('Load dtype must match f32/i32 source storage')
                for i, name in enumerate(s.names):
                    addr, space = self.address(s.input, s.index, i, expected_dtype=dtype)
                    reg = self.reg('f' if dtype == 'f32' else 'r')
                    suffix = 'f32' if dtype == 'f32' else 's32'
                    self.emit(f'ld.{space}.{suffix} {reg}, [{addr}];')
                    self.values[name] = (reg, dtype)
            elif kind == 'Write':
                if s.swizzle not in ('NONE', None):
                    raise Unsupported('Swizzled writes unsupported')
                declared_dtype = None if s.value_dtype is None else s.value_dtype.name
                if declared_dtype not in (None, 'f32', 'i32', 'bool'):
                    raise Unsupported('Only f32/i32/bool write values supported')
                for i, name in enumerate(s.values):
                    value = self.value(name)
                    # For these scalar types the f32 destination receives a
                    # numeric conversion of the actual SSA expression.
                    value = self.f32_value(name)
                    addr, space = self.address(s.output, s.index, i)
                    if s.atomic:
                        if space != 'global':
                            raise Unsupported('Shared atomic writes unsupported')
                        self.emit(f'red.global.add.f32 [{addr}], {value};')
                    else:
                        self.emit(f'st.{space}.f32 [{addr}], {value};')
            elif kind == 'Assign':
                self.assign(s.name, s.op, s.args, s.dtype)
            elif kind == 'Let':
                # A Let captures a value, including reads of mutable carried state.
                # Copy every supported operand so aliases cannot observe later writes.
                operand, source_dtype = self.expr(s.value)
                dtype = source_dtype if s.dtype is None else s.dtype.name
                if dtype == 'bool':
                    dtype = 'pred'
                self.values[s.name] = self.convert_operand(operand, source_dtype, dtype)
            elif kind == 'Init':
                if s.dtype.name != 'f32':
                    raise Unsupported('Only f32 state supported')
                r = self.reg('f')
                self.emit(f'mov.f32 {r}, {f32(s.identity)};')
                self.values[s.name] = (r, 'f32')
                self.mutable[s.name] = r
            elif kind == 'Accum':
                old = self.value(s.name)
                left = s.base if s.base is not None else s.name
                self.assign(s.name, s.op, (left, s.value), s.dtype)
                if self.value(s.name) != old:
                    self.emit(f'mov.f32 {old}, {self.value(s.name)};')
                self.values[s.name] = (old, 'f32')
                self.mutable[s.name] = old
            elif kind in ('StridedLoop', 'Loop'):
                previous_axis = self.values.get(s.axis.name)
                previous_loop_ranges = dict(self.int_ranges)
                try:
                    # Seeds apply to accumulators owned immediately by this loop.
                    # Nested reductions seed themselves when their loop is entered.
                    if s.seed:
                        self.init_accums(s.body)
                    loop_init_frame = self.init_scope_frame(s.body)
                    # Unroll only innermost static plain loops. The loop seed above
                    # is still initialized exactly once, and each copy updates the
                    # same carried accumulator in the original sequential order.
                    def contains_loop(body):
                        for statement in body:
                            if type(statement).__name__ in ('Loop', 'StridedLoop'):
                                return True
                            for field in ('body', 'else_body'):
                                if contains_loop(getattr(statement, field, ())):
                                    return True
                        return False
                    extent = static(s.axis.extent)
                    shared_loads = self.shared_compute_body(s.body)
                    if (kind == 'Loop' and extent >= 8 and shared_loads
                            and not contains_loop(s.body)):
                        factor = 8
                        vector_plan = self.shared_vector_plan(s)
                        limit = extent - extent % factor
                        r = self.reg('r')
                        self.emit(f'mov.s32 {r}, 0;')
                        self.set_integer_range(r, checked_i32_interval(0, limit - factor))
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
                            if offset % 4 == 0:
                                prefetched = self.shared_vector_prefetch(vector_plan)
                            bindings = {position: registers[offset % 4]
                                        for position, registers in prefetched.items()}
                            self.statements_with_init_scope(s.body, loop_init_frame, bindings)
                        self.emit(f'add.s32 {r}, {r}, {factor};')
                        self.emit(f'bra {begin};')
                        self.mark(done)
                        self.set_integer_range(r, checked_i32_interval(limit, limit))
                        for coordinate in range(limit, extent):
                            self.values[s.axis.name] = (str(coordinate), 'i32')
                            self.statements_with_init_scope(s.body, loop_init_frame)
                        # Complete the internal induction state before restoring its binding.
                        self.emit(f'mov.s32 {r}, {extent};')
                        self.set_integer_range(r, checked_i32_interval(extent, extent))
                        self.values[s.axis.name] = (r, 'i32')
                        continue
                    r = self.reg('r')
                    start, step, end = ('0', '1', str(static(s.axis.extent)))
                    if kind == 'StridedLoop':
                        start, _ = self.expr(s.start)
                        step, _ = self.expr(s.step)
                        if s.end is not None:
                            end, _ = self.expr(s.end)
                    self.emit(f'mov.s32 {r}, {start};')
                    self.set_integer_range(r, positive_strided_interval(self.integer_range(start),
                                                                   self.integer_range(step), self.integer_range(end)))
                    self.values[s.axis.name] = (r, 'i32')
                    begin, done = self.label('loop'), self.label('loop_done')
                    self.mark(begin)
                    p = self.reg('p')
                    self.emit(f'setp.ge.s32 {p}, {r}, {end};')
                    self.emit(f'@{p} bra {done};')
                    self.statements_with_init_scope(s.body, loop_init_frame)
                    self.emit(f'add.s32 {r}, {r}, {step};')
                    self.emit(f'bra {begin};')
                    self.mark(done)
                finally:
                    self.int_ranges = previous_loop_ranges
                    # Axis binders are lexical; Accum and other data results
                    # retain their existing escape/update semantics.
                    if previous_axis is None:
                        self.values.pop(s.axis.name, None)
                    else:
                        self.values[s.axis.name] = previous_axis
            elif kind == 'WarpShuffle':
                if s.length > 32 or s.length <= 0 or s.length & (s.length-1):
                    raise Unsupported('Shuffle width must be a power of two <=32')
                if self.block_threads % s.length:
                    raise Unsupported('Physical CTA does not contain complete shuffle groups')
                mask = self.reg('r')
                if self.partial_tile_shuffle:
                    # Membership follows the static complete-subgroup proof,
                    # independent of which logical members issue first.
                    if s.length == 32:
                        self.emit(f'mov.b32 {mask}, 0xffffffff;')
                    else:
                        group_lane = self.reg('r')
                        self.emit(f'and.b32 {group_lane}, {self.hardware_lane}, {32 - s.length};')
                        self.emit(f'shl.b32 {mask}, {(1 << s.length) - 1}, {group_lane};')
                else:
                    self.emit(f'activemask.b32 {mask};')
                offset = s.length // 2
                while offset:
                    for a, b in zip(s.state, s.state_b, strict=True):
                        r = self.reg('f')
                        self.emit(f'shfl.sync.bfly.b32 {r}, {self.value(a)}, {offset}, 31, {mask};')
                        self.values[b] = (r, 'f32')
                    self.statements(s.combine_states)
                    offset //= 2
            elif kind == 'Smem':
                if s.dtype not in ('float', 'f32'):
                    raise Unsupported('Only float shared memory supported')
                ident(s.name)
                self.buffers[s.name] = (tuple(s.extents), 'f32')
                self.shared[s.name] = s
                if not self.dynamic_shared:
                    self.declarations.append(f'.shared .align {max(4, s.align)} .b8 {s.name}[{4*math.prod(s.extents)}];')
            elif kind == 'CpAsyncCopy':
                if s.swizzle not in ('NONE', None) or s.nbytes not in (4, 8, 16):
                    raise Unsupported('Only nonswizzled f32 copy groups supported')
                from emmy.compiler.ir.expr import BinaryExpr
                destination = s.smem_index
                if s.lane_index is not None:
                    destination = tuple(BinaryExpr('+', a, b) for a, b in zip(destination, s.lane_index, strict=True))
                # Native typed transaction alignment is a precondition, as for
                # cp.async itself. The vectors preserve bytes without arithmetic.
                src, source_space = self.address(s.src, s.src_index)
                dst, destination_space = self.address(s.smem, destination)
                if source_space != 'global' or destination_space != 'shared':
                    raise Unsupported('Copy must transport global to shared')
                registers = [self.reg('f') for _ in range(s.nbytes // 4)]
                suffix = '' if len(registers) == 1 else f'.v{len(registers)}'
                operands = registers[0] if len(registers) == 1 else '{' + ', '.join(registers) + '}'
                self.emit(f'ld.global{suffix}.f32 {operands}, [{src}];')
                self.emit(f'st.shared{suffix}.f32 [{dst}], {operands};')
            elif kind in ('CpAsyncCommit', 'CpAsyncWait'):
                # Copies above are synchronous. Preserve the IR's existing CTA
                # barriers; adding barriers inside cooperative copy loops is unsafe.
                pass
            elif kind == 'Cond':
                p, dtype = self.expr(s.cond)
                if dtype != 'pred':
                    raise Unsupported('Expected predicate for conditional')
                branch_init_frame = self.init_scope_frame(tuple(s.body) + tuple(s.else_body))
                otherwise, done = self.label('else'), self.label('endif')
                self.emit(f'@!{p} bra {otherwise};')
                previous_branch_ranges = dict(self.int_ranges)
                self.statements_with_init_scope(s.body, branch_init_frame)
                self.int_ranges = dict(previous_branch_ranges)
                self.emit(f'bra {done};')
                self.mark(otherwise)
                self.statements_with_init_scope(s.else_body, branch_init_frame)
                self.int_ranges = previous_branch_ranges
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
                # Broadcast is an in-place update of the enclosing carrier.
                # Keep its initialized register live across skipped control paths.
                carried_registers = {a: self.value(a) for a in s.state}
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
                        ar = self.reg('f')
                        self.emit(f'ld.shared.f32 {ar}, [{addr}];')
                        self.values[a] = (ar, 'f32')
                        addr2, _ = self.address(buf, (BinaryExpr('+',idx,Literal(size,'int')),))
                        br = self.reg('f')
                        self.emit(f'ld.shared.f32 {br}, [{addr2}];')
                        self.values[b] = (br, 'f32')
                    self.statements(s.combine_states)
                    single = self.reg('p')
                    self.emit(f'setp.eq.s32 {single}, {self.hardware_lane}, 0;')
                    for buf, a in zip(s.bufs, s.state, strict=True):
                        addr, _ = self.address(buf, (idx,))
                        self.emit(f'@{single} st.shared.f32 [{addr}], {self.value(a)};')
                    self.mark(skip)
                    self.emit(f'bar.sync {s.barrier_id};')
                    size //= 2
                for buf, a in zip(s.bufs, s.state, strict=True):
                    from emmy.compiler.ir.expr import Literal
                    addr, _ = self.address(buf, (Literal(0,'int'),))
                    r = carried_registers[a]
                    self.emit(f'ld.shared.f32 {r}, [{addr}];')
                    self.values[a] = (r, 'f32')
                    if a in self.mutable:
                        self.mutable[a] = r
            else:
                raise Unsupported(f'Unsupported statement {kind}')

    def render(self):
        self.statements(self.node.op.body)
        params = ',\n'.join(f'    .param .{suffix} param_{i}' for i, suffix in enumerate(self.parameter_types))
        decl = [f'    .reg .{suffix} %{kind}<{self.counts[kind]+1}>;' for kind,suffix in [('r','b32'),('rd','b64'),('f','f32'),('p','pred')]]
        decl += [f'    .reg .{suffix} %{kind}<{self.counts[kind]+1}>;'
                 for kind, suffix in [('h', 'b16'), ('d', 'f64')] if self.counts[kind]]
        decl += ['    '+x for x in self.declarations]
        return '\n'.join(['.version 8.0', '.target sm_89', '.address_size 64', '', *self.externals, f'.visible .entry {ident(self.node.op.name)}(', params, ')', '{', *decl, *self.lines, 'kernel_done:', '    ret;', '}', ''])


def compile_path(path):
    path = Path(path)
    # Emmy 0.3.11 omits this expression from its typed constructor registry.
    from emmy.compiler import graph as graph_module
    from emmy.compiler.ir.expr import FlatIndex
    graph_module._stmt_eval_scope().setdefault('FlatIndex', FlatIndex)
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
