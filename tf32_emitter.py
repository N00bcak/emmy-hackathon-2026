"""Direct one-atom TF32 ``mma.sync`` PTX emission and CPU lane oracle.

The emitter deliberately has a small interface: one CTA, one warp, one
``m16n8k8`` atom, scalar global loads and stores.  It is a correctness unit,
not a performance path.  Production plans reuse its exact fragment/lane
sequence for each independently proved atom.
"""
from __future__ import annotations

from dataclasses import dataclass
import struct

import numpy as np


MMA = 'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32'


def tf32_rna(value):
    """Bit-exact model of ``cvt.rna.tf32.f32`` (round to nearest, ties away)."""
    bits = struct.unpack('<I', struct.pack('<f', float(np.float32(value))))[0]
    finite = np.isfinite(np.float32(value))
    if not finite:
        return float(np.float32(value))
    rounded = (bits + 0x1000) & 0xFFFFE000
    return float(struct.unpack('<f', struct.pack('<I', rounded))[0])


def lane_map(lane):
    if not 0 <= lane < 32:
        raise ValueError('lane must be in [0, 32)')
    g, t = lane >> 2, lane & 3
    return {
        'a': [(g, t), (g + 8, t), (g, t + 4), (g + 8, t + 4)],
        'b': [(t, g), (t + 4, g)],
        'c': [(g, 2*t), (g, 2*t + 1), (g + 8, 2*t), (g + 8, 2*t + 1)],
    }


def simulate(a, b):
    """Return the exact three-term TF32 result for one 16x8x8 atom."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != (16, 8) or b.shape != (8, 8):
        raise ValueError('one atom expects A=(16,8), B=(8,8)')
    ah = np.vectorize(tf32_rna, otypes=[np.float32])(a)
    bh = np.vectorize(tf32_rna, otypes=[np.float32])(b)
    al = np.vectorize(tf32_rna, otypes=[np.float32])(a - ah)
    bl = np.vectorize(tf32_rna, otypes=[np.float32])(b - bh)
    # Validate the MMA's eight-product dot against the documented SASS
    # ``mma.sync`` unit.  The hardware rounds the exact sum once at the
    # accumulator boundary; carrying f64 into that boundary and converting
    # once models this instruction exactly on both tested architectures.
    hh = np.asarray(ah.astype(np.float64) @ bh.astype(np.float64), dtype=np.float32)
    hl = np.asarray(ah.astype(np.float64) @ bl.astype(np.float64), dtype=np.float32)
    lh = np.asarray(al.astype(np.float64) @ bh.astype(np.float64), dtype=np.float32)
    # Fragment combination is explicitly hh + hl, then + lh, in f32.
    return np.float32(np.float32(hh + hl) + lh), ah, al, bh, bl


def simulate_matmul(a, b, atom_k=8):
    """CPU model of the emitted three-term TF32 contraction.

    Each MMA accumulates one exact 8-product dot into a persistent f32 C
    fragment.  This model rounds each chunk at the accumulator boundary, so it
    is independent of the PTX instruction schedule while modelling the emitted
    arithmetic rather than an exact f64 matmul.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError('simulate_matmul expects compatible rank-two operands')
    if a.shape[1] % atom_k:
        raise ValueError('K must be a multiple of the TF32 atom K extent')
    ah = np.vectorize(tf32_rna, otypes=[np.float32])(a)
    bh = np.vectorize(tf32_rna, otypes=[np.float32])(b)
    al = np.vectorize(tf32_rna, otypes=[np.float32])(a - ah)
    bl = np.vectorize(tf32_rna, otypes=[np.float32])(b - bh)
    c_hh = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)
    c_hl = np.zeros_like(c_hh)
    c_lh = np.zeros_like(c_hh)
    for kk in range(0, a.shape[1], atom_k):
        ahk = ah[:, kk:kk + atom_k].astype(np.float64)
        alk = al[:, kk:kk + atom_k].astype(np.float64)
        bhk = bh[kk:kk + atom_k].astype(np.float64)
        blk = bl[kk:kk + atom_k].astype(np.float64)
        # The MMA instruction rounds each exact 8-product dot once into the
        # accumulator boundary.  Keep the three products in separate C
        # fragments, matching the PTX emitter's independent hh/hl/lh chains.
        c_hh = np.asarray(c_hh.astype(np.float64) + ahk @ bhk, dtype=np.float32)
        c_hl = np.asarray(c_hl.astype(np.float64) + ahk @ blk, dtype=np.float32)
        c_lh = np.asarray(c_lh.astype(np.float64) + alk @ bhk, dtype=np.float32)
    return np.float32(np.float32(c_hh + c_hl) + c_lh)


def lane_oracle(a, b):
    """Return per-lane C fragments from the independent CPU simulation."""
    c, ah, al, bh, bl = simulate(a, b)
    result = {}
    for lane in range(32):
        cells = lane_map(lane)['c']
        result[lane] = {
            'a': [float(ah[r, k]) for r, k in lane_map(lane)['a']],
            'al': [float(al[r, k]) for r, k in lane_map(lane)['a']],
            'b': [float(bh[k, c]) for k, c in lane_map(lane)['b']],
            'bl': [float(bl[k, c]) for k, c in lane_map(lane)['b']],
            'c': [float(c[r, col]) for r, col in cells],
        }
    return result, c


def _f32(value, prefix):
    bits = struct.unpack('<I', struct.pack('<f', float(value)))[0]
    return f'0f{bits:08x}' if prefix == '' else f'{prefix}{bits:08x}'


def synthetic_ptx(a, b, entry='synthetic_mma'):
    """Emit one warp computing a 16x8x8 atom from two packed f32 operands.

    Parameters are A, B, and C; the caller supplies exactly one warp.  A and B
    are row-major ``[16,8]`` and ``[8,8]`` buffers.  The kernel converts each
    lane's six operand values with ``cvt.rna.tf32.f32`` and issues three MMAs
    into separate f32 C fragment sets.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != (16, 8) or b.shape != (8, 8):
        raise ValueError('one atom expects A=(16,8), B=(8,8)')
    lines = [
        '.version 8.0', '.target sm_89', '.address_size 64', '',
        f'.visible .entry {entry}(',
        '    .param .u64 param_A,', '    .param .u64 param_B,',
        '    .param .u64 param_C', ')', '{',
        '    .reg .b32 %r<96>;', '    .reg .b64 %rd<16>;',
        '    .reg .f32 %f<32>;',
        '    .reg .pred %p<2>;',
        '    ld.param.u64 %rd1, [param_A];',
        '    ld.param.u64 %rd2, [param_B];',
        '    ld.param.u64 %rd3, [param_C];',
        '    cvta.to.global.u64 %rd1, %rd1;',
        '    cvta.to.global.u64 %rd2, %rd2;',
        '    cvta.to.global.u64 %rd3, %rd3;',
        '    mov.u32 %r1, %tid.x;',
        '    and.b32 %r2, %r1, 31;',
        '    shr.u32 %r3, %r2, 2;',
        '    and.b32 %r4, %r2, 3;',
        '    mul.lo.u32 %r5, %r3, 8;',
        '    add.u32 %r5, %r5, %r4;',
        '    mul.wide.u32 %rd4, %r5, 4;',
        '    add.u64 %rd4, %rd1, %rd4;',
        '    ld.global.f32 %f1, [%rd4];',
        '    add.u32 %r6, %r5, 64;',
        '    mul.wide.u32 %rd5, %r6, 4;',
        '    add.u64 %rd5, %rd1, %rd5;',
        '    ld.global.f32 %f2, [%rd5];',
        '    add.u32 %r7, %r5, 4;',
        '    mul.wide.u32 %rd6, %r7, 4;',
        '    add.u64 %rd6, %rd1, %rd6;',
        '    ld.global.f32 %f3, [%rd6];',
        '    add.u32 %r8, %r5, 68;',
        '    mul.wide.u32 %rd7, %r8, 4;',
        '    add.u64 %rd7, %rd1, %rd7;',
        '    ld.global.f32 %f4, [%rd7];',
        '    mul.lo.u32 %r9, %r4, 8;',
        '    add.u32 %r9, %r9, %r3;',
        '    mul.wide.u32 %rd8, %r9, 4;',
        '    add.u64 %rd8, %rd2, %rd8;',
        '    ld.global.f32 %f5, [%rd8];',
        '    add.u32 %r10, %r9, 32;',
        '    mul.wide.u32 %rd9, %r10, 4;',
        '    add.u64 %rd9, %rd2, %rd9;',
        '    ld.global.f32 %f6, [%rd9];',
        '    cvt.rna.tf32.f32 %r12, %f1;',
        '    cvt.rna.tf32.f32 %r13, %f2;',
        '    cvt.rna.tf32.f32 %r14, %f3;',
        '    cvt.rna.tf32.f32 %r15, %f4;',
        '    cvt.rna.tf32.f32 %r16, %f5;',
        '    cvt.rna.tf32.f32 %r17, %f6;',
        '    mov.b32 %f19, %r12;', '    sub.f32 %f19, %f1, %f19;',
        '    cvt.rna.tf32.f32 %r18, %f19;',
        '    mov.b32 %f20, %r13;', '    sub.f32 %f20, %f2, %f20;',
        '    cvt.rna.tf32.f32 %r19, %f20;',
        '    mov.b32 %f21, %r14;', '    sub.f32 %f21, %f3, %f21;',
        '    cvt.rna.tf32.f32 %r20, %f21;',
        '    mov.b32 %f22, %r15;', '    sub.f32 %f22, %f4, %f22;',
        '    cvt.rna.tf32.f32 %r21, %f22;',
        '    mov.b32 %f23, %r16;', '    sub.f32 %f23, %f5, %f23;',
        '    cvt.rna.tf32.f32 %r22, %f23;',
        '    mov.b32 %f24, %r17;', '    sub.f32 %f24, %f6, %f24;',
        '    cvt.rna.tf32.f32 %r23, %f24;',
        '    mov.f32 %f7, 0f00000000;', '    mov.f32 %f8, 0f00000000;',
        '    mov.f32 %f9, 0f00000000;', '    mov.f32 %f10, 0f00000000;',
        '    mov.f32 %f11, 0f00000000;', '    mov.f32 %f12, 0f00000000;',
        '    mov.f32 %f13, 0f00000000;', '    mov.f32 %f14, 0f00000000;',
        '    mov.f32 %f15, 0f00000000;', '    mov.f32 %f16, 0f00000000;',
        '    mov.f32 %f17, 0f00000000;', '    mov.f32 %f18, 0f00000000;',
        f'    {MMA} {{%f7, %f8, %f9, %f10}}, {{%r12, %r13, %r14, %r15}}, {{%r16, %r17}}, {{%f7, %f8, %f9, %f10}};',
        f'    {MMA} {{%f11, %f12, %f13, %f14}}, {{%r12, %r13, %r14, %r15}}, {{%r22, %r23}}, {{%f11, %f12, %f13, %f14}};',
        f'    {MMA} {{%f15, %f16, %f17, %f18}}, {{%r18, %r19, %r20, %r21}}, {{%r16, %r17}}, {{%f15, %f16, %f17, %f18}};',
        '    add.f32 %f7, %f7, %f11;', '    add.f32 %f8, %f8, %f12;',
        '    add.f32 %f9, %f9, %f13;', '    add.f32 %f10, %f10, %f14;',
        '    add.f32 %f7, %f7, %f15;', '    add.f32 %f8, %f8, %f16;',
        '    add.f32 %f9, %f9, %f17;', '    add.f32 %f10, %f10, %f18;',
    ]
    # row = g or g+8; col = 2t or 2t+1.  r25 is g*8+2t.
    lines.extend([
        '    mul.lo.u32 %r25, %r3, 8;',
        '    mul.lo.u32 %r26, %r4, 2;',
        '    add.u32 %r25, %r25, %r26;',
    ])
    for i, offset in enumerate((0, 1, 64, 65)):
        lines.extend([
            f'    add.u32 %r{27+i}, %r25, {offset};' if offset else '    mov.u32 %r27, %r25;',
            f'    mul.wide.u32 %rd{10+i}, %r{27+i}, 4;',
            f'    add.u64 %rd{10+i}, %rd3, %rd{10+i};',
            f'    st.global.f32 [%rd{10+i}], %f{7+i};',
        ])
    lines.extend(['    ret;', '}', ''])
    return '\n'.join(lines)


@dataclass(frozen=True)
class AtomResult:
    ptx: str
    oracle: np.ndarray
    lanes: dict


def emit_and_oracle(a, b, entry='synthetic_mma'):
    lanes, c = lane_oracle(a, b)
    return AtomResult(synthetic_ptx(a, b, entry), c, lanes)


def _shared_layout(node):
    """Return the emitter-compatible static shared layout from typed Smem nodes."""
    from compile import smem_alignment, statement_nbytes
    offsets = {}
    total = 0

    def scan(statements):
        nonlocal total
        for statement in statements:
            if type(statement).__name__ == 'Smem':
                if statement.name in offsets:
                    raise ValueError('duplicate shared buffer name')
                align = smem_alignment(statement)
                total = (total + align - 1) // align * align
                offsets[statement.name] = total
                total += statement_nbytes(statement)
            for field in ('body', 'else_body', 'combine_states'):
                nested = getattr(statement, field, ())
                if nested:
                    scan(nested)

    scan(node.op.body)
    return offsets, total


def _static_eval(expr, env):
    """Evaluate a typed integer expression under a fully concrete environment."""
    from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
    if isinstance(expr, Literal):
        return int(expr.value)
    if isinstance(expr, Var):
        return env[expr.name]
    if isinstance(expr, BinaryExpr):
        left, right = _static_eval(expr.left, env), _static_eval(expr.right, env)
        if expr.op == '+': return left + right
        if expr.op == '-': return left - right
        if expr.op == '*': return left * right
        if expr.op in ('/', '//'): return int(left / right)
        if expr.op == '%': return left % right
    if isinstance(expr, TernaryExpr):
        return _static_eval(expr.if_true, env) if _static_eval(expr.cond, env) else _static_eval(expr.if_false, env)
    raise ValueError(f'non-static shared coordinate {type(expr).__name__}')


def _flat_index(index, extents):
    """Return a constant row-major index or raise for a non-static coordinate."""
    value = 0
    if len(index) != len(extents):
        raise ValueError('shared coordinate rank differs from the typed extent rank')
    for coordinate, extent in zip(index, extents):
        value = value * int(extent) + int(coordinate)
    return value


def production_ptx(plan, meta, *, node=None, graph=None):
    """Emit direct PTX for a fully validated one-atom-per-warp plan.

    Simple staged plans with a typed ``cp.async`` prologue are copied into
    shared memory and read back through the proven shared layout.  Direct plans
    and a rotating pipeline plan are read through the proved global coordinates;
    the pipeline's stage ordering is not replayed because the direct read is
    the same tensor the proven copies would have produced.  Every form performs
    the same three-term TF32 contraction and never invokes a renderer or another
    compiler.
    """
    plan.validate()
    args = {arg['name']: arg for arg in meta['args']}
    if plan.a_buffer not in args or plan.b_buffer not in args or plan.output not in args:
        raise ValueError('plan buffer is absent from metadata')
    if any(arg.get('scalar') for arg in meta['args']):
        raise ValueError('scalar arguments are not supported by the MMA plan')
    a_shape = tuple(int(x) for x in args[plan.a_buffer].get('shape', ()))
    b_shape = tuple(int(x) for x in args[plan.b_buffer].get('shape', ()))
    o_shape = tuple(int(x) for x in args[plan.output].get('shape', ()))
    if len(a_shape) != 2 or len(b_shape) != 2 or o_shape != (plan.m, plan.n):
        raise ValueError('plan shapes are not rank-two matmul operands')
    a_k = a_shape[0] if plan.a_transpose else a_shape[1]
    a_m = a_shape[1] if plan.a_transpose else a_shape[0]
    b_k = b_shape[1] if plan.b_transpose else b_shape[0]
    b_n = b_shape[0] if plan.b_transpose else b_shape[1]
    if (a_m, a_k) != (plan.m, plan.k) or (b_k, b_n) != (plan.k, plan.n):
        raise ValueError('plan extents disagree with metadata shapes')

    shared = plan.staging == 'shared'
    shared_tile = None
    shared_walk = None
    if shared:
        if node is None or graph is None:
            raise ValueError('shared plan requires typed IR for its copy map')
        import tensor_core
        shared_tile = next((s for s in node.op.body if type(s).__name__ == 'Tile'), None)
        if shared_tile is None:
            raise ValueError('shared plan has no Tile region')
        shared_walk = tensor_core._walk(shared_tile.body)
        # Re-run the independent proof against this exact emitter input.
        grid_size = plan.grid_m * plan.grid_n
        a_shape = (plan.m, plan.k) if not plan.a_transpose else (plan.k, plan.m)
        b_shape = (plan.n, plan.k) if plan.b_transpose else (plan.k, plan.n)
        tensor_core._staged_copy_proof(
            shared_tile, shared_walk, plan.a_smem,
            shared_walk.copies.get(plan.a_smem, []), a_shape,
            (plan.m, plan.n), grid_size, plan.block_threads)
        tensor_core._staged_copy_proof(
            shared_tile, shared_walk, plan.b_smem,
            shared_walk.copies.get(plan.b_smem, []), b_shape,
            (plan.m, plan.n), grid_size, plan.block_threads)

    class Registers:
        def __init__(self):
            self.count = {'r': 0, 'rd': 0, 'f': 0, 'p': 0}

        def __call__(self, kind):
            self.count[kind] += 1
            return f'%{kind}{self.count[kind]}'

    regs = Registers()
    lines = []
    declarations = []
    externals = []
    def emit(text):
        lines.append('    ' + text)

    labels = 0
    def label(prefix):
        nonlocal labels
        labels += 1
        return f'{prefix}_{labels}'

    params = []
    pointers = {}
    for i, name in enumerate(meta['arg_order']):
        params.append(f'    .param .u64 param_{i}')
        reg = regs('rd')
        emit(f'ld.param.u64 {reg}, [param_{i}];')
        emit(f'cvta.to.global.u64 {reg}, {reg};')
        pointers[name] = reg

    shared_base = None
    if shared:
        shared_base = regs('r')
        shared_base64 = regs('rd')
        emit(f'mov.u64 {shared_base64}, __emmy_smem;')
        emit(f'cvt.u32.u64 {shared_base}, {shared_base64};')
        externals.append('.extern .shared .align 16 .b8 __emmy_smem[];')


    tid = regs('r')
    emit(f'mov.u32 {tid}, %tid.x;')
    lane = regs('r')
    g = regs('r')
    t = regs('r')
    warp = regs('r')
    emit(f'and.b32 {lane}, {tid}, 31;')
    emit(f'shr.u32 {warp}, {tid}, 5;')
    emit(f'shr.u32 {g}, {lane}, 2;')
    emit(f'and.b32 {t}, {lane}, 3;')
    bid = regs('r')
    emit(f'mov.u32 {bid}, %ctaid.x;')
    cta_m = regs('r')
    cta_n = regs('r')
    if plan.grid_n == 1:
        emit(f'mov.u32 {cta_m}, {bid};')
        emit(f'mov.u32 {cta_n}, 0;')
    else:
        emit(f'div.u32 {cta_m}, {bid}, {plan.grid_n};')
        emit(f'rem.u32 {cta_n}, {bid}, {plan.grid_n};')
    warp_m = regs('r')
    warp_n = regs('r')
    if plan.warps_n == 1:
        emit(f'mov.u32 {warp_m}, {warp};')
        emit(f'mov.u32 {warp_n}, 0;')
    else:
        emit(f'div.u32 {warp_m}, {warp}, {plan.warps_n};')
        emit(f'rem.u32 {warp_n}, {warp}, {plan.warps_n};')
    row0 = regs('r')
    col0 = regs('r')
    tmp = regs('r')
    emit(f'mul.lo.u32 {tmp}, {cta_m}, {plan.block_m};')
    emit(f'mul.lo.u32 {row0}, {warp_m}, 16;')
    emit(f'add.u32 {row0}, {tmp}, {row0};')
    emit(f'mul.lo.u32 {tmp}, {cta_n}, {plan.block_n};')
    emit(f'mul.lo.u32 {col0}, {warp_n}, 8;')
    emit(f'add.u32 {col0}, {tmp}, {col0};')

    def static_eval(expr, env):
        return _static_eval(expr, env)

    def emit_u32(expr, env):
        # Emit one typed integer expression as a sequence of u32 instructions.
        from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
        if isinstance(expr, Literal):
            value = int(expr.value)
            if value < 0:
                raise ValueError('negative shared copy coordinate')
            reg = regs('r')
            emit(f'mov.u32 {reg}, {value};')
            return reg
        if isinstance(expr, Var):
            if expr.name == 'tid':
                return tid
            if expr.name == 'bid':
                return bid
            if expr.name in env:
                return env[expr.name]
            raise ValueError(f'unknown copy coordinate variable {expr.name!r}')
        if isinstance(expr, BinaryExpr):
            left = emit_u32(expr.left, env)
            right = emit_u32(expr.right, env)
            reg = regs('r')
            if expr.op == '+':
                emit(f'add.u32 {reg}, {left}, {right};')
            elif expr.op == '-':
                emit(f'sub.u32 {reg}, {left}, {right};')
            elif expr.op == '*':
                emit(f'mul.lo.u32 {reg}, {left}, {right};')
            elif expr.op in ('/', '//'):
                emit(f'div.u32 {reg}, {left}, {right};')
            elif expr.op == '%':
                emit(f'rem.u32 {reg}, {left}, {right};')
            else:
                raise ValueError(f'unsupported shared copy operator {expr.op!r}')
            return reg
        raise ValueError(f'unsupported shared copy expression {type(expr).__name__}')

    if shared:
        # Bind the Tile axes in their launched decode order, then emit the
        # exact typed copy loop.  Coordinates are derived from the same nodes
        # the recognizer proved, not from a table of byte offsets.
        shared_env = {}
        extents = [int(tensor_core._static(a.extent)) for a in shared_tile.axes]
        grid_size = plan.grid_m * plan.grid_n
        for i, axis in enumerate(shared_tile.axes):
            stride = 1
            for later in extents[i+1:]:
                stride *= later
            axis_reg = regs('r')
            if stride == 1:
                emit(f'mov.u32 {axis_reg}, {tid};')
            else:
                emit(f'shr.u32 {axis_reg}, {tid}, {stride.bit_length()-1};')
                if i:
                    extent = extents[i]
                    emit(f'and.b32 {axis_reg}, {axis_reg}, {extent-1};')
            shared_env[axis.name] = axis_reg
        # Override the outer (CTA) axes with the actual launched CTA coordinate.
        # CTA strides span only the block-uniform axes; including the thread
        # suffix here would make every block decode as the first tile.
        _, block_axes, _, _ = tensor_core._block_and_thread_axes(shared_tile)
        cut = len(block_axes)
        block_extents = extents[:cut]
        for i, axis in enumerate(block_axes):
            stride = 1
            for later in block_extents[i+1:]:
                stride *= later
            axis_reg = regs('r')
            if stride == 1:
                emit(f'rem.u32 {axis_reg}, {bid}, {extents[i]};')
            else:
                emit(f'div.u32 {axis_reg}, {bid}, {stride};')
                emit(f'rem.u32 {axis_reg}, {axis_reg}, {extents[i]};')
            shared_env[axis.name] = axis_reg
        for smem_name, src_name, smem_shape in (
                (plan.a_smem, plan.a_buffer, plan.a_smem_shape),
                (plan.b_smem, plan.b_buffer, plan.b_smem_shape)):
            if not smem_shape or len(smem_shape) != 2:
                raise ValueError('missing shared shape')
            copies = shared_walk.copies.get(smem_name, [])
            if len(copies) != 1:
                raise ValueError('shared operand does not have one copy producer')
            copy, chain, _ = copies[0]
            if len(chain) != 1 or type(chain[0]).__name__ != 'StridedLoop':
                raise ValueError('shared copy is not a simple strided loop')
            loop = chain[0]
            loop_reg = regs('r')
            start = tensor_core.eval_expr(loop.start, shared_env)
            emit(f'mov.u32 {loop_reg}, {start};')
            step = int(tensor_core.eval_expr(loop.step, {}))
            end = (int(tensor_core.eval_expr(loop.end, shared_env))
                   if loop.end is not None else int(tensor_core._static(loop.axis.extent)))
            label_begin = label('copy')
            label_done = label('copy_done')
            lines.append(label_begin + ':')
            predc = regs('p')
            emit(f'setp.ge.u32 {predc}, {loop_reg}, {end};')
            emit(f'@{predc} bra {label_done};')
            copy_env = dict(shared_env)
            copy_env[loop.axis.name] = loop_reg
            dest_indices = list(copy.smem_index)
            if copy.lane_index is not None:
                from emmy.compiler.ir.expr import BinaryExpr
                dest_indices = [BinaryExpr('+', a, b) for a, b in zip(dest_indices, copy.lane_index)]
            dest_flat = regs('r')
            if len(dest_indices) != 2:
                raise ValueError('shared destination is not rank two')
            dest0 = emit_u32(dest_indices[0], copy_env)
            dest1 = emit_u32(dest_indices[1], copy_env)
            emit(f'mul.lo.u32 {dest_flat}, {dest0}, {smem_shape[1]};')
            emit(f'add.u32 {dest_flat}, {dest_flat}, {dest1};')
            base_off = (plan.smem_offsets[0] if smem_name == plan.a_smem
                        else plan.smem_offsets[1])
            byte_off = regs('r')
            emit(f'mul.lo.u32 {byte_off}, {dest_flat}, 4;')
            emit(f'add.u32 {byte_off}, {byte_off}, {base_off};')
            dest_addr = regs('r')
            emit(f'add.u32 {dest_addr}, {shared_base}, {byte_off};')
            source_indices = list(copy.src_index)
            if len(source_indices) != 2:
                raise ValueError('shared source is not rank two')
            source0 = emit_u32(source_indices[0], copy_env)
            source1 = emit_u32(source_indices[1], copy_env)
            src_stride = int(args[src_name]['shape'][1])
            src_flat = regs('r')
            emit(f'mul.lo.u32 {src_flat}, {source0}, {src_stride};')
            emit(f'add.u32 {src_flat}, {src_flat}, {source1};')
            src_offset = regs('rd')
            emit(f'mul.wide.u32 {src_offset}, {src_flat}, 4;')
            src_addr = regs('rd')
            emit(f'add.u64 {src_addr}, {pointers[src_name]}, {src_offset};')
            emit(f'cp.async.ca.shared.global [{dest_addr}], [{src_addr}], 4;')
            emit(f'add.u32 {loop_reg}, {loop_reg}, {step};')
            emit(f'bra {label_begin};')
            lines.append(label_done + ':')
        emit('cp.async.commit_group;')
        emit('cp.async.wait_group 0;')
        emit('bar.sync 0;')

    # C fragment registers persist across the complete K reduction.
    c = [[regs('f') for _ in range(4)] for _ in range(3)]
    for frag in c:
        for reg in frag:
            emit(f'mov.f32 {reg}, 0f00000000;')
    k = regs('r')
    emit(f'mov.u32 {k}, 0;')
    loop = label('kloop')
    done = label('kdone')
    pred = regs('p')
    lines.append(loop + ':')
    emit(f'setp.ge.u32 {pred}, {k}, {plan.k};')
    emit(f'@{pred} bra {done};')

    # Index rows/columns exactly as the verified MMA lane map requires.
    r0 = regs('r')
    r1 = regs('r')
    ccol = regs('r')
    kt0 = regs('r')
    kt1 = regs('r')
    emit(f'add.u32 {r0}, {row0}, {g};')
    emit(f'add.u32 {r1}, {r0}, 8;')
    emit(f'add.u32 {ccol}, {col0}, {g};')
    emit(f'add.u32 {kt0}, {k}, {t};')
    emit(f'add.u32 {kt1}, {kt0}, 4;')

    def load_a(row, kk):
        value = regs('f')
        off = regs('r')
        addr = regs('rd')
        if shared:
            # Shared tile coordinates are exactly the typed compute-load
            # indices proved by the recognizer: (warp-local row, k).
            local_row = regs('r')
            emit(f'sub.u32 {local_row}, {row}, {row0};')
            off = regs('r')
            emit(f'mul.lo.u32 {off}, {local_row}, {plan.a_smem_shape[1]};')
            emit(f'add.u32 {off}, {off}, {kk};')
            emit(f'mul.lo.u32 {off}, {off}, 4;')
            emit(f'add.u32 {off}, {off}, {plan.smem_offsets[0]};')
            emit(f'add.u32 {off}, {shared_base}, {off};')
            emit(f'cvt.u64.u32 {addr}, {off};')
            emit(f'ld.shared.f32 {value}, [{addr}];')
            return value
        if plan.a_transpose:
            emit(f'mul.lo.u32 {off}, {kk}, {a_shape[1]};')
            emit(f'add.u32 {off}, {off}, {row};')
        else:
            emit(f'mul.lo.u32 {off}, {row}, {a_shape[1]};')
            emit(f'add.u32 {off}, {off}, {kk};')
        emit(f'mul.wide.u32 {addr}, {off}, 4;')
        emit(f'add.u64 {addr}, {pointers[plan.a_buffer]}, {addr};')
        emit(f'ld.global.f32 {value}, [{addr}];')
        return value

    def load_b(kk, col):
        value = regs('f')
        off = regs('r')
        addr = regs('rd')
        if shared:
            local_col = regs('r')
            emit(f'sub.u32 {local_col}, {col}, {col0};')
            emit(f'mul.lo.u32 {off}, {kk}, {plan.b_smem_shape[1]};')
            emit(f'add.u32 {off}, {off}, {local_col};')
            emit(f'mul.lo.u32 {off}, {off}, 4;')
            emit(f'add.u32 {off}, {off}, {plan.smem_offsets[1]};')
            emit(f'add.u32 {off}, {shared_base}, {off};')
            emit(f'cvt.u64.u32 {addr}, {off};')
            emit(f'ld.shared.f32 {value}, [{addr}];')
            return value
        if plan.b_transpose:
            emit(f'mul.lo.u32 {off}, {col}, {b_shape[1]};')
            emit(f'add.u32 {off}, {off}, {kk};')
        else:
            emit(f'mul.lo.u32 {off}, {kk}, {b_shape[1]};')
            emit(f'add.u32 {off}, {off}, {col};')
        emit(f'mul.wide.u32 {addr}, {off}, 4;')
        emit(f'add.u64 {addr}, {pointers[plan.b_buffer]}, {addr};')
        emit(f'ld.global.f32 {value}, [{addr}];')
        return value

    av = [load_a(r0, kt0), load_a(r1, kt0), load_a(r0, kt1), load_a(r1, kt1)]
    bcol = regs('r')
    emit(f'add.u32 {bcol}, {col0}, {g};')
    bv = [load_b(kt0, bcol), load_b(kt1, bcol)]
    ah = [regs('r') for _ in range(4)]
    bh = [regs('r') for _ in range(2)]
    al = [regs('r') for _ in range(4)]
    bl = [regs('r') for _ in range(2)]
    for reg, value in zip(ah, av):
        emit(f'cvt.rna.tf32.f32 {reg}, {value};')
    for reg, value in zip(bh, bv):
        emit(f'cvt.rna.tf32.f32 {reg}, {value};')
    for reg, value, high in zip(al, av, ah):
        lowf = regs('f')
        emit(f'mov.b32 {lowf}, {high};')
        emit(f'sub.f32 {lowf}, {value}, {lowf};')
        emit(f'cvt.rna.tf32.f32 {reg}, {lowf};')
    for reg, value, high in zip(bl, bv, bh):
        lowf = regs('f')
        emit(f'mov.b32 {lowf}, {high};')
        emit(f'sub.f32 {lowf}, {value}, {lowf};')
        emit(f'cvt.rna.tf32.f32 {reg}, {lowf};')

    def mma(cfrag, afrag, bfrag):
        emit(f'{MMA} {{{cfrag[0]}, {cfrag[1]}, {cfrag[2]}, {cfrag[3]}}}, '
             f'{{{afrag[0]}, {afrag[1]}, {afrag[2]}, {afrag[3]}}}, '
             f'{{{bfrag[0]}, {bfrag[1]}}}, '
             f'{{{cfrag[0]}, {cfrag[1]}, {cfrag[2]}, {cfrag[3]}}};')
    mma(c[0], ah, bh)
    mma(c[1], ah, bl)
    mma(c[2], al, bh)
    emit(f'add.u32 {k}, {k}, 8;')
    emit(f'bra {loop};')
    lines.append(done + ':')

    for i in range(4):
        emit(f'add.f32 {c[0][i]}, {c[0][i]}, {c[1][i]};')
        emit(f'add.f32 {c[0][i]}, {c[0][i]}, {c[2][i]};')

    # C stores: c0/c1 are row g, c2/c3 row g+8, columns 2t and 2t+1.
    out_base = pointers[plan.output]
    for i, (rreg, coff) in enumerate(((r0, 0), (r0, 1), (r1, 0), (r1, 1))):
        off = regs('r')
        col = regs('r')
        addr = regs('rd')
        emit(f'mul.lo.u32 {off}, {rreg}, {plan.n};')
        emit(f'mul.lo.u32 {col}, {t}, 2;')
        if coff:
            emit(f'add.u32 {col}, {col}, 1;')
        emit(f'add.u32 {col}, {col}, {col0};')
        emit(f'add.u32 {off}, {off}, {col};')
        emit(f'mul.wide.u32 {addr}, {off}, 4;')
        emit(f'add.u64 {addr}, {out_base}, {addr};')
        emit(f'st.global.f32 [{addr}], {c[0][i]};')

    decl = [f'    .reg .b32 %r<{regs.count["r"]+1}>;',
            f'    .reg .b64 %rd<{regs.count["rd"]+1}>;',
            f'    .reg .f32 %f<{regs.count["f"]+1}>;',
            f'    .reg .pred %p<{regs.count["p"]+1}>;']
    return '\n'.join(['.version 8.0', '.target sm_89', '.address_size 64', '',
                      *externals,
                      f'.visible .entry {plan.entry}(', ',\n'.join(params), ')', '{',
                      *decl, *lines, '    ret;', '}', ''])
