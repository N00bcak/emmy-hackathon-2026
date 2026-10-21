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


class Emitter:
    def __init__(self, graph, node, meta):
        self.graph, self.node, self.meta = graph, node, meta
        self.lines, self.declarations = [], []
        self.counts = {'r': 0, 'rd': 0, 'f': 0, 'p': 0}
        self.values, self.buffers, self.shared = {}, {}, {}
        self.label_count = 0
        self.accum_ops = {}
        for n in graph.nodes.values():
            for tensor in n.outputs:
                self.buffers[tensor.name] = (tuple(static(d) for d in tensor.shape), tensor.dtype.name)
        for arg in meta['args']:
            if arg.get('scalar'):
                raise Unsupported('Scalar-by-value ABI is not supported')
            self.buffers[arg['name']] = (tuple(arg['shape']), arg['dtype'])
        if node.op.serial:
            raise Unsupported('Serial runtime axes are not supported')
        self.ptrs = {}
        for i, name in enumerate(meta['arg_order']):
            ident(name)
            ptr = self.reg('rd')
            self.emit(f'ld.param.u64 {ptr}, [param_{i}];')
            self.emit(f'cvta.to.global.u64 {ptr}, {ptr};')
            self.ptrs[name] = ptr
        for name, op, suffix in [('tid', '%tid.x', 'u32'), ('bid', '%ctaid.x', 'u32'), ('ntid', '%ntid.x', 'u32')]:
            reg = self.reg('r')
            self.emit(f'mov.{suffix} {reg}, {op};')
            self.values[name] = (reg, 'i32')
        tid, bid, ntid = (self.values[k][0] for k in ('tid', 'bid', 'ntid'))
        gid = self.reg('r')
        self.emit(f'mad.lo.u32 {gid}, {bid}, {ntid}, {tid};')
        self.values['_gid'] = (gid, 'i32')
        for name, instruction, rhs in [('lane', 'and.b32', 31), ('warp', 'shr.u32', 5)]:
            reg = self.reg('r')
            self.emit(f'{instruction} {reg}, {tid}, {rhs};')
            self.values[name] = (reg, 'i32')

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
            if at != bt:
                raise Unsupported(f'Mixed expression dtypes {at}, {bt}')
            comparisons = {'==':'eq', '!=':'ne', '<':'lt', '<=':'le', '>':'gt', '>=':'ge'}
            if x.op in comparisons:
                p = self.reg('p')
                self.emit(f'setp.{comparisons[x.op]}.{("f32" if at == "f32" else "s32")} {p}, {a}, {b};')
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
            r = self.reg('f' if at == 'f32' else 'r')
            self.emit(f'selp.{("f32" if at == "f32" else "b32")} {r}, {a}, {b}, {p};')
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
        if dtype != 'f32':
            raise Unsupported(f'Only f32 buffers supported, got {dtype}')
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
        self.emit(f'mul.wide.s32 {byte}, {offset}, 4;')
        addr = self.reg('rd')
        if name in self.shared:
            self.emit(f'mov.u64 {addr}, {ident(name)};')
            base = addr
        else:
            if name not in self.ptrs:
                raise Unsupported(f'Buffer {name} absent from metadata ABI')
            base = self.ptrs[name]
        self.emit(f'add.u64 {addr}, {base}, {byte};')
        return addr, 'shared' if name in self.shared else 'global'

    def assign(self, name, op, args, dtype=None):
        if dtype is not None and getattr(dtype, 'name', dtype) != 'f32':
            raise Unsupported(f'Only f32 arithmetic supported: {dtype}')
        op = getattr(op, 'name', None)
        operands = [self.value(a) for a in args]
        reg = self.reg('f')
        binary = {'add':'add.f32', 'subtract':'sub.f32', 'multiply':'mul.f32', 'divide':'div.rn.f32', 'maximum':'max.f32', 'minimum':'min.f32'}
        unary = {'negative':'neg.f32', 'negate':'neg.f32', 'abs':'abs.f32', 'sqrt':'sqrt.rn.f32', 'reciprocal':'rcp.rn.f32', 'rsqrt':'rsqrt.approx.f32'}
        if op in binary and len(operands) == 2:
            self.emit(f'{binary[op]} {reg}, {operands[0]}, {operands[1]};')
        elif op in unary and len(operands) == 1:
            self.emit(f'{unary[op]} {reg}, {operands[0]};')
        elif op == 'relu' and len(operands) == 1:
            self.emit(f'max.f32 {reg}, {operands[0]}, {f32(0)};')
        else:
            raise Unsupported(f'Unsupported elementwise operation {op}/{len(args)}')
        self.values[name] = (reg, 'f32')

    def init_accums(self, statements):
        for s in statements:
            if type(s).__name__ == 'Accum':
                if s.base is not None:
                    raise Unsupported('Accumulator base is unsupported')
                op = s.op.name
                identities = {'add': 0.0, 'multiply': 1.0, 'maximum': -math.inf, 'minimum': math.inf}
                if op not in identities:
                    raise Unsupported(f'Unsupported reduction {op}')
                reg = self.reg('f')
                self.emit(f'mov.f32 {reg}, {f32(identities[op])};')
                self.values[s.name] = (reg, 'f32')
                self.accum_ops[s.name] = op
            elif type(s).__name__ == 'Cond':
                self.init_accums(s.body)
                self.init_accums(s.else_body)

    def statements(self, statements):
        for s in statements:
            kind = type(s).__name__
            if kind == 'Tile':
                if s.aux_threads or s.raster_axes or s.raster_group or s.raster_orient != 'm':
                    raise Unsupported('Auxiliary/raster tile scheduling unsupported')
                if any(a.window is not None for a in s.axes):
                    raise Unsupported('Windowed tile axes unsupported')
                if self.meta['block'][1:] != [1, 1] or self.meta['grid'][1:] != [1, 1]:
                    raise Unsupported('Only one dimensional launch geometry supported')
                extents = [static(a.extent) for a in s.axes]
                n = math.prod(extents)
                if n <= 0 or n >= (1 << 31):
                    raise Unsupported('Tile size outside supported signed32 range')
                # Partial CTA exits are unsafe for synchronization; reject rather than
                # quietly producing a deadlock or ignoring the bounds guard.
                collective = has_collective(s.body)
                if collective and n % self.meta['block'][0]:
                    raise Unsupported('Collective tile with partial final CTA unsupported')
                p = self.reg('p')
                self.emit(f'setp.ge.u32 {p}, {self.value("_gid")}, {n};')
                self.emit(f'@{p} bra kernel_done;')
                for i, axis in enumerate(s.axes):
                    stride = math.prod(extents[i+1:])
                    r = self.reg('r')
                    self.emit(f'div.u32 {r}, {self.value("_gid")}, {stride};')
                    if i:
                        self.emit(f'rem.u32 {r}, {r}, {extents[i]};')
                    self.values[axis.name] = (r, 'i32')
                self.statements(s.body)
            elif kind == 'Load':
                if s.dtype.name != 'f32':
                    raise Unsupported('Only f32 loads supported')
                for i, name in enumerate(s.names):
                    addr, space = self.address(s.input, s.index, i)
                    reg = self.reg('f')
                    self.emit(f'ld.{space}.f32 {reg}, [{addr}];')
                    self.values[name] = (reg, 'f32')
            elif kind == 'Write':
                if s.swizzle not in ('NONE', None):
                    raise Unsupported('Swizzled writes unsupported')
                if s.value_dtype is not None and s.value_dtype.name != 'f32':
                    raise Unsupported('Only f32 writes supported')
                for i, name in enumerate(s.values):
                    addr, space = self.address(s.output, s.index, i)
                    if s.atomic:
                        if space != 'global':
                            raise Unsupported('Shared atomic writes unsupported')
                        self.emit(f'red.global.add.f32 [{addr}], {self.value(name)};')
                    else:
                        self.emit(f'st.{space}.f32 [{addr}], {self.value(name)};')
            elif kind == 'Assign':
                self.assign(s.name, s.op, s.args, s.dtype)
            elif kind == 'Let':
                self.values[s.name] = self.expr(s.value)
            elif kind == 'Init':
                if s.dtype.name != 'f32':
                    raise Unsupported('Only f32 state supported')
                r = self.reg('f')
                self.emit(f'mov.f32 {r}, {f32(s.identity)};')
                self.values[s.name] = (r, 'f32')
            elif kind == 'Accum':
                old = self.value(s.name)
                self.assign(s.name, s.op, (s.name, s.value), s.dtype)
                self.emit(f'mov.f32 {old}, {self.value(s.name)};')
                self.values[s.name] = (old, 'f32')
            elif kind in ('StridedLoop', 'Loop'):
                if not s.seed:
                    raise Unsupported('Nonseeded loops not yet supported')
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
                offset = s.length // 2
                while offset:
                    for a, b in zip(s.state, s.state_b, strict=True):
                        r = self.reg('f')
                        self.emit(f'shfl.sync.bfly.b32 {r}, {self.value(a)}, {offset}, 31, 0xffffffff;')
                        self.values[b] = (r, 'f32')
                    self.statements(s.combine_states)
                    offset //= 2
            elif kind == 'Smem':
                if s.dtype not in ('float', 'f32'):
                    raise Unsupported('Only float shared memory supported')
                ident(s.name)
                self.buffers[s.name] = (tuple(s.extents), 'f32')
                self.shared[s.name] = s
                self.declarations.append(f'.shared .align {max(4, s.align)} .b8 {s.name}[{4*math.prod(s.extents)}];')
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
                        ar = self.reg('f')
                        self.emit(f'ld.shared.f32 {ar}, [{addr}];')
                        self.values[a] = (ar, 'f32')
                        addr2, _ = self.address(buf, (BinaryExpr('+',idx,Literal(size,'int')),))
                        br = self.reg('f')
                        self.emit(f'ld.shared.f32 {br}, [{addr2}];')
                        self.values[b] = (br, 'f32')
                    self.statements(s.combine_states)
                    single = self.reg('p')
                    self.emit(f'setp.eq.s32 {single}, {self.value("lane")}, 0;')
                    for buf, a in zip(s.bufs, s.state, strict=True):
                        addr, _ = self.address(buf, (idx,))
                        self.emit(f'@{single} st.shared.f32 [{addr}], {self.value(a)};')
                    self.mark(skip)
                    self.emit(f'bar.sync {s.barrier_id};')
                    size //= 2
                for buf, a in zip(s.bufs, s.state, strict=True):
                    from emmy.compiler.ir.expr import Literal
                    addr, _ = self.address(buf, (Literal(0,'int'),))
                    r = self.reg('f')
                    self.emit(f'ld.shared.f32 {r}, [{addr}];')
                    self.values[a] = (r, 'f32')
            else:
                raise Unsupported(f'Unsupported statement {kind}')

    def render(self):
        self.statements(self.node.op.body)
        params = ',\n'.join(f'    .param .u64 param_{i}' for i, _ in enumerate(self.meta['arg_order']))
        decl = [f'    .reg .{suffix} %{kind}<{self.counts[kind]+1}>;' for kind,suffix in [('r','b32'),('rd','b64'),('f','f32'),('p','pred')]]
        decl += ['    '+x for x in self.declarations]
        return '\n'.join(['.version 8.0', '.target sm_89', '.address_size 64', '', f'.visible .entry {ident(self.node.op.name)}(', params, ')', '{', *decl, *self.lines, 'kernel_done:', '    ret;', '}', ''])


def compile_path(path):
    path = Path(path)
    graph = Graph.from_dict(json.loads(path.read_text()))
    kernels = [(name,n) for name,n in graph.nodes.items() if type(n.op).__name__ == 'KernelOp']
    if len(kernels) != 1:
        raise Unsupported(f'Expected one KernelOp, got {len(kernels)}')
    name,node = kernels[0]
    metadata = json.loads(path.with_name('meta.json').read_text())
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
