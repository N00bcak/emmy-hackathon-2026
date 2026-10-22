"""Typed Kernel IR matmul recognition and one-atom TF32 MMA planning.

This module is deliberately separate from both Emmy's renderer and the generic
SIMT emitter in ``compile.py``.  It reads typed nodes, proves a dense
contraction, and either returns a fully validated ``MatmulPlan`` or a
fail-closed classification.  No fixture name, entry name, or accumulator name
participates in recognition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import (
    Accum, Assign, CpAsyncCommit, CpAsyncCopy, CpAsyncWait, KernelOp, Load, Loop,
    Smem, StridedLoop, Sync, Tile, Write,
)
from emmy.compiler.ir.stmt.leaves import Init


PLAN = 'Plan'
UNSUPPORTED = 'UnsupportedMatch'
NO_MATCH = 'NoMatch'


class ProofError(ValueError):
    """A candidate has the right shape but a proof obligation failed."""


@dataclass(frozen=True)
class PipelineCell:
    """One valid shared cell after a typed copy has executed."""
    source: tuple[int, int]
    group: int


@dataclass(frozen=True)
class AffineAtom:
    key: tuple
    variables: frozenset[str]


@dataclass(frozen=True)
class LinearForm:
    """A canonical linear form whose non-linear subterms are opaque atoms.

    This is not a numerical affine model for arbitrary IR.  It is an equality
    certificate over the already-typed expression tree: additions and constant
    scaling are expanded, while ``a / c`` and ``a % c`` remain named atoms.  A
    same-atom cancellation therefore proves only that the two expressions are
    structurally the same typed subexpression, which is exactly the local-row /
    local-column relation the recognizer needs.
    """

    terms: dict[tuple, int] = field(default_factory=dict)
    constant: int = 0
    variables: frozenset[str] = frozenset()

    def scaled(self, factor):
        return LinearForm({k: v * factor for k, v in self.terms.items() if v * factor},
                          self.constant * factor, self.variables)

    def plus(self, other):
        terms = dict(self.terms)
        for key, value in other.terms.items():
            terms[key] = terms.get(key, 0) + value
            if not terms[key]:
                del terms[key]
        return LinearForm(terms, self.constant + other.constant,
                          self.variables | other.variables)

    def minus(self, other):
        return self.plus(other.scaled(-1))

    def zero(self):
        return not self.terms and self.constant == 0


def _stage_factor(key, phase_vars):
    """Return the tile extent contributed by a proved stage-phase atom.

    A shared-memory pipeline writes into one of a small number of rotating
    stage slots.  The typed index then contains ``slot * tile_extent`` in
    addition to the local coordinate.  ``_is_stage_phase`` proves the slot is
    ``(outer / stages + c) % stages`` over declared reduction variables; this
    helper accepts that atom optionally multiplied by a literal tile extent.
    """
    if key[0] != 'atom':
        return None
    expr = key[1]
    factor = 1
    if (isinstance(expr, tuple) and len(expr) == 4 and expr[0] == 'b'
            and expr[1] == '*'):
        left, right = expr[2], expr[3]
        if isinstance(left, tuple) and len(left) == 2 and left[0] == 'l' and isinstance(left[1], int):
            factor, expr = left[1], right
        elif isinstance(right, tuple) and len(right) == 2 and right[0] == 'l' and isinstance(right[1], int):
            factor, expr = right[1], left
        else:
            return None
    if factor > 0 and _is_stage_phase(('atom', expr), phase_vars):
        return factor
    return None


def _without_stage_terms(form, phase_vars):
    """Remove proved rotating-slot terms, preserving every other term."""
    terms = {}
    saw_stage = False
    for key, coefficient in form.terms.items():
        factor = _stage_factor(key, phase_vars)
        if factor is not None:
            saw_stage = True
            continue
        terms[key] = coefficient
    variables = {name for key in terms for name in _vars_from_key(key)}
    return LinearForm(terms, form.constant, frozenset(variables)), saw_stage


def _literal_int(expr):
    if isinstance(expr, Literal) and isinstance(expr.value, int):
        return expr.value
    return None


def linear_form(expr):
    if isinstance(expr, Var):
        return LinearForm({('var', expr.name): 1}, 0, frozenset({expr.name}))
    if isinstance(expr, Literal):
        if not isinstance(expr.value, int):
            return LinearForm({('literal', repr(expr.value)): 1}, 0, frozenset())
        return LinearForm({}, int(expr.value), frozenset())
    if isinstance(expr, BinaryExpr):
        if expr.op == '+':
            return linear_form(expr.left).plus(linear_form(expr.right))
        if expr.op == '-':
            return linear_form(expr.left).minus(linear_form(expr.right))
        if expr.op == '*':
            left, right = linear_form(expr.left), linear_form(expr.right)
            if not left.terms and not right.terms:
                return LinearForm({}, left.constant * right.constant, frozenset())
            if not left.terms:
                return right.scaled(left.constant)
            if not right.terms:
                return left.scaled(right.constant)
        # Division, modulo, and non-constant products are equality atoms.
        key = ('atom', _structural(expr))
        return LinearForm({key: 1}, 0, frozenset(_vars(expr)))
    key = ('atom', _structural(expr))
    return LinearForm({key: 1}, 0, frozenset(_vars(expr)))


def _structural(expr):
    if isinstance(expr, Var):
        return ('v', expr.name)
    if isinstance(expr, Literal):
        return ('l', expr.value)
    if isinstance(expr, BinaryExpr):
        return ('b', expr.op, _structural(expr.left), _structural(expr.right))
    if isinstance(expr, TernaryExpr):
        return ('t', _structural(expr.cond), _structural(expr.if_true),
                _structural(expr.if_false))
    return ('o', type(expr).__name__, repr(expr))


def _vars(expr):
    found = set()
    def walk(e):
        if e is None:
            return
        if isinstance(e, Var):
            found.add(e.name)
        elif isinstance(e, BinaryExpr):
            walk(e.left)
            walk(e.right)
        elif isinstance(e, TernaryExpr):
            walk(e.cond)
            walk(e.if_true)
            walk(e.if_false)
    walk(expr)
    return found


def _static(value):
    if isinstance(value, int):
        return value
    if hasattr(value, 'as_static'):
        return int(value.as_static())
    if hasattr(value, 'value'):
        return int(value.value)
    raise ProofError(f'non-static extent {value!r}')


def eval_expr(expr, env):
    if isinstance(expr, Literal):
        return expr.value
    if isinstance(expr, Var):
        return env[expr.name]
    if isinstance(expr, BinaryExpr):
        left, right = eval_expr(expr.left, env), eval_expr(expr.right, env)
        if expr.op == '+': return left + right
        if expr.op == '-': return left - right
        if expr.op == '*': return left * right
        if expr.op in ('/', '//'): return int(left / right)
        if expr.op == '%': return left % right
        if expr.op == '<': return int(left < right)
        if expr.op == '<=': return int(left <= right)
        if expr.op == '>': return int(left > right)
        if expr.op == '>=': return int(left >= right)
        if expr.op == '==': return int(left == right)
        if expr.op == '!=': return int(left != right)
        raise ProofError(f'unsupported arithmetic operator {expr.op!r}')
    if isinstance(expr, TernaryExpr):
        return eval_expr(expr.if_true, env) if eval_expr(expr.cond, env) else eval_expr(expr.if_false, env)
    raise ProofError(f'unsupported expression {type(expr).__name__}')


@dataclass(frozen=True)
class MatmulPlan:
    entry: str
    output: str
    a_buffer: str
    b_buffer: str
    m: int
    n: int
    k: int
    block_m: int
    block_n: int
    warps_m: int
    warps_n: int
    active_warps: int
    grid_m: int
    grid_n: int
    block_threads: int
    a_transpose: bool
    b_transpose: bool
    staging: str
    a_smem: str | None = None
    b_smem: str | None = None
    a_smem_shape: tuple[int, int] | None = None
    b_smem_shape: tuple[int, int] | None = None
    smem_offsets: tuple[int, int] | None = None
    atom: tuple[int, int, int] = (16, 8, 8)
    k_splits: int = 1
    atomic_output: bool = False
    precision: str = 'tf32x3'
    epilogue: str = 'copy'
    proof: dict[str, Any] = field(default_factory=dict)

    def validate(self):
        if self.m <= 0 or self.n <= 0 or self.k <= 0:
            raise ProofError('non-positive contraction extent')
        if self.atom != (16, 8, 8):
            raise ProofError('unsupported atom')
        if self.k % self.atom[2] or self.k_splits != 1:
            raise ProofError('K does not reduce to one exact TF32 atom grid')
        if self.block_m % self.atom[0] or self.block_n % self.atom[1]:
            raise ProofError('block tile is not atom-aligned')
        if self.warps_m != self.block_m // self.atom[0] or self.warps_n != self.block_n // self.atom[1]:
            raise ProofError('warp grid does not factor the block tile')
        if self.warps_m * self.warps_n != self.active_warps:
            raise ProofError('atom count differs from active warps')
        if self.active_warps != self.block_threads // 32 or self.block_threads % 32:
            raise ProofError('active warps differ from blockDim.x / 32')
        if self.grid_m * self.block_m != self.m or self.grid_n * self.block_n != self.n:
            raise ProofError('grid/block tile does not cover the output exactly')
        if self.grid_m * self.grid_n <= 0:
            raise ProofError('invalid CTA grid')
        if self.atomic_output or self.k_splits != 1:
            raise ProofError('atomic split-K is outside the one-atom emitter')
        if self.staging not in ('global', 'pipeline', 'shared') or self.epilogue != 'copy':
            raise ProofError('unsupported plan variant')
        # An output that aliases a declared input would make a fused matmul an
        # in-place update whose result depends on overwrite order.  Reject it
        # instead of emitting stores through an operand pointer.
        if self.output in (self.a_buffer, self.b_buffer):
            raise ProofError('output buffer aliases a contraction operand')
        if self.staging == 'shared':
            if (not self.a_smem or not self.b_smem
                    or self.a_smem_shape is None or self.b_smem_shape is None
                    or self.smem_offsets is None):
                raise ProofError('shared plan lacks an exact shared layout')
            if self.a_smem == self.b_smem:
                raise ProofError('shared operands alias one buffer')
            for shape in (self.a_smem_shape, self.b_smem_shape):
                if len(shape) != 2 or min(shape) <= 0:
                    raise ProofError('invalid shared tile shape')
            if min(self.smem_offsets) < 0:
                raise ProofError('invalid shared tile offset')
        elif any(value is not None for value in (
                self.a_smem, self.b_smem, self.a_smem_shape,
                self.b_smem_shape, self.smem_offsets)):
            raise ProofError('non-shared plan carries a shared layout')
        # Lane partition: four C cells per lane, all 128 atom cells exactly once.
        cells = []
        for warp_m in range(self.warps_m):
            for warp_n in range(self.warps_n):
                atom_cells = set()
                for lane in range(32):
                    g, t = lane >> 2, lane & 3
                    atom_cells.update({(g, 2*t), (g, 2*t+1),
                                       (g+8, 2*t), (g+8, 2*t+1)})
                    for row, col in ((g, 2*t), (g, 2*t+1),
                                     (g+8, 2*t), (g+8, 2*t+1)):
                        cells.append((warp_m*16 + row, warp_n*8 + col))
                if len(atom_cells) != 128:
                    raise ProofError('lane map does not cover one 16x8 atom')
        expected = {(r, c) for r in range(self.block_m) for c in range(self.block_n)}
        if set(cells) != expected or len(cells) != len(expected):
            raise ProofError('lane partition does not cover the block tile exactly once')
        total = [(cta_m*self.block_m + r, cta_n*self.block_n + c)
                 for cta_m in range(self.grid_m) for cta_n in range(self.grid_n)
                 for r, c in expected]
        if len(total) != self.m*self.n or len(set(total)) != self.m*self.n:
            raise ProofError('CTA coverage is not an exact output partition')
        return True


@dataclass(frozen=True)
class Classification:
    verdict: str
    reason: str
    plan: MatmulPlan | None = None
    facts: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Walk:
    folds: dict[str, list] = field(default_factory=dict)
    assigns: dict[str, list] = field(default_factory=dict)
    loads: dict[str, list] = field(default_factory=dict)
    writes: list = field(default_factory=list)
    copies: dict[str, list] = field(default_factory=dict)
    syncs: list = field(default_factory=list)
    commits: list = field(default_factory=list)
    waits: list = field(default_factory=list)
    smem: dict[str, Smem] = field(default_factory=dict)
    inits: dict[str, list] = field(default_factory=dict)
    sequence: int = 0


def _walk(statements, chain=(), out=None):
    out = out or _Walk()
    for s in statements:
        out.sequence += 1
        seq = out.sequence
        kind = type(s).__name__
        if kind in ('Loop', 'StridedLoop'):
            _walk(s.body, chain + (s,), out)
        elif kind == 'Load':
            for name in s.names:
                out.loads.setdefault(name, []).append((s, chain, seq))
        elif kind == 'Assign':
            out.assigns.setdefault(s.name, []).append((s, chain, seq))
        elif kind == 'Accum':
            out.folds.setdefault(s.name, []).append((s, chain, seq))
        elif kind == 'Write':
            out.writes.append((s, chain, seq))
        elif kind == 'CpAsyncCopy':
            out.copies.setdefault(s.smem, []).append((s, chain, seq))
        elif kind == 'Sync':
            out.syncs.append((s, chain, seq))
        elif kind == 'CpAsyncCommit':
            out.commits.append((s, chain, seq))
        elif kind == 'CpAsyncWait':
            out.waits.append((s, chain, seq))
        elif kind == 'Smem':
            out.smem[s.name] = s
        elif kind == 'Init':
            out.inits.setdefault(s.name, []).append((s, chain, seq))
        elif kind in ('Smem', 'Tile'):
            pass
        else:
            raise ProofError(f'unsupported statement in matmul region: {kind}')
    return out


def _block_and_thread_axes(tile):
    extents = [_static(a.extent) for a in tile.axes]
    total = math.prod(extents)
    cells = tile.block_threads
    if cells is None or cells <= 0 or cells % 32:
        raise ProofError('Tile block_threads is not a positive warp multiple')
    # Threads are innermost; choose the unique smallest suffix that supplies
    # exactly one cell per launched thread.  Outer axes are block-uniform even
    # when their product happens to also divide block_threads (m01 has 16x8
    # block axes and a 512-thread suffix).
    cuts = [i for i in range(len(extents) + 1) if math.prod(extents[i:]) == cells]
    if not cuts:
        raise ProofError('Tile does not factor into one thread suffix')
    cut = max(cuts)
    block_axes = tile.axes[:cut]
    thread_axes = tile.axes[cut:]
    return extents, block_axes, thread_axes, total


def _decode_env(tile, graph_meta, bid, tid):
    extents, block_axes, thread_axes, total = _block_and_thread_axes(tile)
    block_size = math.prod(_static(a.extent) for a in thread_axes)
    env = {}
    gid = bid * block_size + tid
    for i, axis in enumerate(tile.axes):
        extent = extents[i]
        stride = math.prod(extents[i+1:])
        env[axis.name] = (gid // stride) % extent
    env['tid'] = tid
    env['bid'] = bid
    env['lane'] = tid & 31
    env['warp'] = tid >> 5
    return env, gid, total


def _shape_of(name, graph, meta):
    for arg in meta['args']:
        if arg['name'] == name and 'shape' in arg:
            return tuple(int(x) for x in arg['shape'])
    for node in graph.nodes.values():
        for tensor in node.outputs:
            if tensor.name == name:
                return tuple(_static(d) for d in tensor.shape)
    raise ProofError(f'buffer {name!r} has no static metadata shape')


def _write_cells(write, env, output_shape):
    cells = []
    for offset, _ in enumerate(write.values):
        index = list(write.index)
        if offset:
            if not index:
                raise ProofError('vector store on scalar output')
            index[-1] = BinaryExpr('+', index[-1], Literal(offset, 'int'))
        values = tuple(int(eval_expr(e, env)) for e in index)
        if len(values) != 2:
            raise ProofError('matmul output is not rank two')
        row, col = values
        if not (0 <= row < output_shape[0] and 0 <= col < output_shape[1]):
            raise ProofError('write coordinate leaves the output')
        cells.append((row, col))
    return cells


def _expanded_coordinates(write, output_shape):
    result = []
    for offset, value in enumerate(write.values):
        index = list(write.index)
        if offset:
            index[-1] = BinaryExpr('+', index[-1], Literal(offset, 'int'))
        if len(index) != 2:
            raise ProofError('matmul output is not rank two')
        result.append((value, index[0], index[1]))
    return result


def _copy_dest(copy):
    dest = tuple(copy.smem_index)
    if copy.lane_index is not None:
        dest = tuple(BinaryExpr('+', a, b) for a, b in zip(dest, copy.lane_index))
    return dest


def _copy_loop_env(tile, chain, bid, tid, *, force=None):
    """Decode a typed copy loop and return its concrete environments.

    Only a single static StridedLoop participates in the simple staging proof.
    ``force`` is used to bind the loop coordinate to one chosen iteration, so
    the caller can compare a source/destination pair without relying on an
    unproved bijection between two independently enumerated index maps.
    """
    env, _, _ = _decode_env(tile, {}, bid, tid)
    if len(chain) == 1 and type(chain[0]).__name__ == 'StridedLoop':
        loop = chain[0]
        start, step = int(eval_expr(loop.start, {})), int(eval_expr(loop.step, {}))
        end = int(eval_expr(loop.end, {})) if loop.end is not None else _static(loop.axis.extent)
        if start != 0 or step <= 0 or end < start:
            raise ProofError('copy loop is not a zero-based positive stride')
        values = list(range(start, end, step))
        if force is not None:
            env[loop.axis.name] = force
            return [env], values
        for value in values:
            iteration_env = dict(env)
            iteration_env[loop.axis.name] = value
            yield iteration_env
        return
    if chain:
        raise ProofError('staged copy uses a nested or non-strided loop')
    yield env


def _simple_staged_candidate(copies):
    """Whether a copy is in the intentionally narrow simple-transport class."""
    if len(copies) != 1:
        return False
    copy, chain, _ = copies[0]
    return (len(chain) == 1 and type(chain[0]).__name__ == 'StridedLoop'
            and copy.lane_index is None and not copy.lane_rows
            and copy.nbytes == 4 and copy.swizzle in ('NONE', None))


def _smem_layout(walk):
    """Return byte offsets for the typed shared buffers in declaration order."""
    offsets = {}
    total = 0
    for name, statement in walk.smem.items():
        align = int(statement.align or 1)
        total = (total + align - 1) // align * align
        offsets[name] = total
        extents = [_static(x) for x in statement.extents]
        dtype_bytes = {'float': 4, 'f32': 4}.get(statement.dtype, 4)
        total += math.prod(extents) * dtype_bytes
    return offsets, total


def _staged_copy_proof(tile, walk, smem, copies, source_shape, output_shape,
                       grid_size, block_threads):
    """Prove a simple full-tile copy is an exact source-to-shared bijection.

    A cooperative ``StridedLoop`` is indexed by ``start + tid + n*step`` over
    the ancestor Tile axes.  For each element of that declared iteration
    space this enumerates every launched CTA and thread.  Every CTA must fill
    its shared tile exactly once; the same tile may be reloaded by another CTA
    only when its source rectangle is an exact local tile of the operand.
    """
    if not copies or len(copies) != 1:
        raise ProofError('simple staging requires exactly one copy producer')
    copy, chain, _ = copies[0]
    if not chain or len(chain) != 1 or type(chain[0]).__name__ != 'StridedLoop':
        raise ProofError('staged copy is not one typed strided loop')
    if copy.lane_index is not None or copy.lane_rows:
        raise ProofError('simple staging does not support lane-indexed copies')
    if copy.nbytes != 4 or copy.swizzle not in ('NONE', None):
        raise ProofError('simple staging requires scalar nonswizzled copies')
    extents = tuple(_static(x) for x in walk.smem[smem].extents)
    if len(extents) != 2 or len(copy.src_index) != 2:
        raise ProofError('simple staging requires rank-two tiles')
    loop = chain[0]
    cooperative = False
    if isinstance(loop.start, Literal):
        start = int(loop.start.value)
    elif isinstance(loop.start, Var) and loop.start.name == 'tid':
        start, cooperative = 0, True
    else:
        raise ProofError('copy loop start is not a thread coordinate')
    step = int(eval_expr(loop.step, {}))
    end = int(eval_expr(loop.end, {})) if loop.end is not None else _static(loop.axis.extent)
    if start < 0 or step <= 0 or end < start or step != block_threads:
        raise ProofError('copy loop is not a zero-based block-strided iteration')
    per_cta = []
    for bid in range(grid_size):
        seen_dest = {}
        seen_source = {}
        for tid in range(block_threads):
            env, _, _ = _decode_env(tile, {}, bid, tid)
            for value in range(start + (tid if cooperative else 0), end, step):
                env[loop.axis.name] = value
                target = tuple(int(eval_expr(x, env)) for x in _copy_dest(copy))
                source = tuple(int(eval_expr(x, env)) for x in copy.src_index)
                if len(target) != 2 or len(source) != 2:
                    raise ProofError('copy coordinate is not rank two')
                if not (0 <= target[0] < extents[0] and 0 <= target[1] < extents[1]):
                    raise ProofError('copy destination leaves its shared tile')
                if not (0 <= source[0] < source_shape[0] and 0 <= source[1] < source_shape[1]):
                    raise ProofError('copy source leaves the global operand')
                if target in seen_dest:
                    raise ProofError('copy writes a shared cell more than once')
                if source in seen_source:
                    raise ProofError('copy reads a global cell more than once')
                seen_dest[target] = source
                seen_source[source] = target
        if set(seen_dest) != {(r, c) for r in range(extents[0]) for c in range(extents[1])}:
            raise ProofError('copy does not cover its shared tile exactly once')
        rows = sorted({r for r, _ in seen_source})
        cols = sorted({c for _, c in seen_source})
        expected = {(r, c) for r in rows for c in cols}
        if (not rows or not cols or len(expected) != len(seen_source)
                or set(seen_source) != expected):
            raise ProofError('copy source is not one exact operand tile rectangle')
        per_cta.append(seen_dest)
    return per_cta

def _prove_staged_reads(tile, walk, chain, value_roles, coord_map, maps,
                        grid_size, block_threads, a_smem, b_smem,
                        a_transpose, b_transpose):
    """Bind each shared compute load to the global cell it must consume."""
    loop_ranges = []
    for loop in chain:
        count, start, step = _loop_count(loop)
        loop_ranges.append((loop.axis.name, range(start, start + count * step, step)))
    for bid in range(grid_size):
        for tid in range(block_threads):
            base, _, _ = _decode_env(tile, {}, bid, tid)
            if loop_ranges:
                from itertools import product
                environments = product(*(values for _, values in loop_ranges))
            else:
                environments = [()]
            for values in environments:
                env = dict(base)
                for (name, _), value in zip(loop_ranges, values):
                    env[name] = value
                k_global = sum(env[name] for name, _ in loop_ranges)
                for value, pair in value_roles.items():
                    out_row = int(eval_expr(coord_map[value][0], env))
                    out_col = int(eval_expr(coord_map[value][1], env))
                    for operand, role, _, _, _ in pair:
                        if operand.input not in (a_smem, b_smem):
                            continue
                        target = tuple(int(eval_expr(x, env)) for x in operand.index)
                        copying = maps[operand.input]
                        if target not in copying[bid]:
                            raise ProofError('shared compute load is outside its proven tile')
                        source = copying[bid][target]
                        if operand.input == a_smem:
                            expected = ((out_row, k_global) if not a_transpose
                                        else (k_global, out_row))
                        else:
                            expected = ((k_global, out_col) if not b_transpose
                                        else (out_col, k_global))
                        if source != expected:
                            raise ProofError('shared compute load disagrees with the copied operand cell')


def _trace_pipeline(tile, walk, chain, value_roles, coord_map, a_smem, b_smem,
                    a_buffer, b_buffer,
                    a_shape, b_shape, grid_size, block_threads,
                    a_transpose, b_transpose):
    """Execute the typed copy/compute order symbolically for every CTA.

    The simple-staging proof covers one complete before-compute copy.  A
    rotating pipeline has multiple copy groups, a bounded wait window and
    stage-slot reuse, so coverage cannot be established from a single copy.
    This trace replays the typed statement order for every launched CTA and
    thread.  It records which globally indexed cell each shared cell currently
    holds, the commit group that produced it, and the groups made visible by
    ``cp.async.wait_group``.  Every compute read must hit a cell filled by a
    visible group and must name exactly the global cell required by the MMA
    coordinate proof.  The trace is independent of fixture and buffer names.
    """
    smem_extents = {
        name: tuple(_static(x) for x in statement.extents)
        for name, statement in walk.smem.items()
    }
    for name, extents in smem_extents.items():
        if len(extents) != 2 or min(extents) <= 0:
            raise ProofError('pipeline shared tile is not rank two')

    # The replay below samples the first and last CTA.  That is a complete
    # proof only when every coordinate involved is affine in CTA variables;
    # otherwise an interior CTA (or a conditional) can select a different
    # cell while both endpoints match.
    _, block_axes, _, _ = _block_and_thread_axes(tile)
    axis_extents = [_static(axis.extent) for axis in tile.axes]
    cta_vars = {'bid'} | {axis.name for axis in block_axes}
    for i, axis in enumerate(tile.axes):
        period = axis_extents[i] * math.prod(axis_extents[i + 1:])
        if block_threads % period:
            cta_vars.add(axis.name)

    def flatten(coords, shape, what):
        if len(coords) != len(shape):
            raise ProofError(f'{what} has the wrong rank')
        value = 0
        for coordinate, extent in zip(coords, shape):
            if not (0 <= coordinate < extent):
                raise ProofError(f'{what} leaves its declared buffer')
            value = value * extent + coordinate
        return value

    def loop_values(loop, env):
        if isinstance(loop, StridedLoop):
            start = int(eval_expr(loop.start, env))
            step = int(eval_expr(loop.step, env))
            end = (int(eval_expr(loop.end, env)) if loop.end is not None
                   else _static(loop.axis.extent))
            if step <= 0 or start < 0 or end < start:
                raise ProofError('invalid pipeline loop extent')
            return range(start, end, step)
        return range(_static(loop.axis.extent))

    def thread_environment(block_env, bid, tid):
        env, _, _ = _decode_env(tile, {}, bid, tid)
        env.update(block_env)
        env['bid'], env['tid'] = bid, tid
        return env

    expected = {}
    for value, pair in value_roles.items():
        for operand, role, source, global_k_dim, _ in pair:
            expected[id(operand)] = (value, role, source, global_k_dim, operand)

    bids = (0, grid_size - 1) if grid_size > 1 else (0,)
    for bid in bids:
        cells = {name: {} for name in smem_extents}
        group = 0
        visible = set()
        stats = {'copy_groups': 0, 'copy_cells': 0, 'reads': 0}

        def process(statements, parent_env):
            nonlocal group
            for statement in statements:
                kind = type(statement).__name__
                if kind in ('Loop', 'StridedLoop'):
                    values = loop_values(statement, parent_env)
                    # The innermost reduction loop is affine with coefficient
                    # one in every proved compute index (the K certificate
                    # above).  Check the endpoints and one interior point for
                    # every thread; any non-affine use of this loop variable in
                    # a read is rejected here rather than sampled silently.
                    if (statement.axis.name == chain[-1].axis.name
                            and any(type(item).__name__ == 'Load'
                                    for item in statement.body)):
                        forms = []
                        def collect_loads(items):
                            for item in items:
                                if type(item).__name__ == 'Load':
                                    forms.extend(linear_form(x) for x in item.index)
                                for field in ('body', 'else_body', 'combine_states'):
                                    nested = getattr(item, field, None)
                                    if nested:
                                        collect_loads(nested)
                        collect_loads(statement.body)
                        # This sweep samples this loop, so every term containing
                        # the loop variable must be the exact unit variable
                        # itself; otherwise an interior-only gather could hide
                        # between sampled iterations.
                        for form in forms:
                            if any(statement.axis.name in _vars_from_key(key)
                                   and key != ('var', statement.axis.name)
                                   for key in form.terms):
                                raise ProofError('pipeline compute index has a non-affine K term')
                        values = list(values)
                        values = sorted({values[0], values[1] if len(values) > 1 else values[0],
                                         values[-1]})
                    for iteration in values:
                        env = dict(parent_env)
                        env[statement.axis.name] = iteration
                        process(statement.body, env)
                elif kind == 'CpAsyncCopy':
                    name = statement.smem
                    if name not in smem_extents:
                        raise ProofError('copy targets no declared shared buffer')
                    if statement.nbytes not in (4, 8, 16) or statement.nbytes % 4:
                        raise ProofError('unsupported cp.async vector width')
                    width = statement.nbytes // 4
                    if statement.swizzle not in ('NONE', None):
                        raise ProofError('pipeline copy is swizzled')
                    destination = _copy_dest(statement)
                    if len(destination) != 2 or len(statement.src_index) != 2:
                        raise ProofError('pipeline copy is not rank two')
                    if not all(_affine_in_variables(linear_form(expr), cta_vars)
                               for expr in (*destination, *statement.src_index)):
                        raise ProofError('pipeline copy has a non-affine CTA coordinate')
                    source_shape = a_shape if name == a_smem else (
                        b_shape if name == b_smem else None)
                    if source_shape is None:
                        raise ProofError('pipeline copy uses an unproved source')
                    expected_source = (a_buffer if name == a_smem
                                       else b_buffer if name == b_smem else None)
                    if expected_source is None or statement.src != expected_source:
                        raise ProofError('pipeline copy source is not its operand source')
                    written = set()
                    for tid in range(block_threads):
                        env = thread_environment(parent_env, bid, tid)
                        try:
                            dest = (int(eval_expr(destination[0], env)),
                                    int(eval_expr(destination[1], env)))
                            source = (int(eval_expr(statement.src_index[0], env)),
                                      int(eval_expr(statement.src_index[1], env)))
                        except (KeyError, TypeError, ValueError) as exc:
                            raise ProofError(f'pipeline copy coordinate is not finite: {exc}') from exc
                        if not (0 <= dest[0] < smem_extents[name][0]
                                and 0 <= dest[1] <= smem_extents[name][1] - width):
                            raise ProofError('pipeline copy destination leaves its shared tile')
                        if not (0 <= source[0] < source_shape[0]
                                and 0 <= source[1] <= source_shape[1] - width):
                            raise ProofError('pipeline copy source leaves its global operand')
                        for offset in range(width):
                            target = (dest[0], dest[1] + offset)
                            target_flat = flatten(target, smem_extents[name], 'pipeline destination')
                            if target_flat in written:
                                raise ProofError('pipeline copy writes a shared cell twice in one group iteration')
                            written.add(target_flat)
                            cells[name][target_flat] = PipelineCell(
                                (source[0], source[1] + offset), group + 1)
                            stats['copy_cells'] += 1
                elif kind == 'CpAsyncCommit':
                    group += 1
                    stats['copy_groups'] += 1
                elif kind == 'CpAsyncWait':
                    count = statement.group
                    if type(count) is not int or count < 0 or count > group:
                        raise ProofError('cp.async wait depth is not within the committed groups')
                    visible.update(range(1, group - count + 1))
                elif kind == 'Sync':
                    if statement.warp or statement.count is not None:
                        raise ProofError('pipeline barrier is not a full CTA collective')
                elif kind == 'Load':
                    entry = expected.get(id(statement))
                    if entry is None:
                        continue
                    value, role, source, global_k_dim, operand = entry
                    name = statement.input
                    if name not in smem_extents:
                        raise ProofError('pipeline compute reads a non-shared operand')
                    if not all(_affine_in_variables(linear_form(expr), cta_vars)
                               for expr in statement.index):
                        raise ProofError('pipeline read has a non-affine CTA coordinate')
                    for tid in range(block_threads):
                        env = thread_environment(parent_env, bid, tid)
                        try:
                            target = flatten(
                                (int(eval_expr(statement.index[0], env)),
                                 int(eval_expr(statement.index[1], env))),
                                smem_extents[name], 'pipeline shared read')
                        except (KeyError, TypeError, ValueError) as exc:
                            raise ProofError(f'pipeline read coordinate is not finite: {exc}') from exc
                        cell = cells[name].get(target)
                        if cell is None:
                            raise ProofError('pipeline read is not supplied by any copy')
                        if cell.group not in visible:
                            raise ProofError('pipeline read consumes an uncommitted or unawaited copy')
                        out_row = int(eval_expr(coord_map[value][0], env))
                        out_col = int(eval_expr(coord_map[value][1], env))
                        k_global = sum(env[loop.axis.name] for loop in chain)
                        if name == a_smem:
                            expected_cell = ((out_row, k_global) if not a_transpose
                                             else (k_global, out_row))
                            expected_shape = a_shape
                        else:
                            expected_cell = ((k_global, out_col) if not b_transpose
                                             else (out_col, k_global))
                            expected_shape = b_shape
                        if cell.source != expected_cell:
                            raise ProofError('pipeline read receives the wrong global cell')
                        flatten(cell.source, expected_shape, 'pipeline source')
                        stats['reads'] += 1
                elif kind in ('Init', 'Assign', 'Accum', 'Write', 'Smem', 'Tile'):
                    pass
                else:
                    raise ProofError(f'unsupported statement in pipeline: {kind}')

        process(tile.body, {'bid': bid})
        if stats['copy_groups'] == 0 or stats['reads'] == 0:
            raise ProofError('pipeline has no copy or compute read')
        for name, extents in smem_extents.items():
            if len(cells[name]) == 0:
                raise ProofError('pipeline shared buffer is never filled')
    return True

def _copy_cells(tile, copies, smem, bid, tid):
    """Enumerate (source element, destination element) for a copy producer."""
    if len(copies) != 1:
        raise ProofError('simple staging requires one typed copy producer')
    copy, chain, _ = copies[0]
    dest = _copy_dest(copy)
    if len(dest) != 2 or len(copy.src_index) != 2:
        raise ProofError('staged copy is not a rank-two element transport')
    if copy.nbytes != 4 or copy.swizzle not in ('NONE', None):
        raise ProofError('simple staging requires scalar nonswizzled copies')
    for env in _copy_loop_env(tile, chain, bid, tid):
        try:
            source = tuple(int(eval_expr(x, env)) for x in copy.src_index)
            target = tuple(int(eval_expr(x, env)) for x in dest)
        except (KeyError, TypeError, ValueError) as exc:
            raise ProofError(f'copy coordinate is not finite: {exc}') from exc
        yield source, target


def _lambda_form(form, constant_key):
    """Separate one coordinate uniform across a thread from the lane terms."""
    terms = {key: coefficient for key, coefficient in form.terms.items()
             if constant_key not in _vars_from_key(key)}
    return LinearForm(terms, form.constant, form.variables)


def _write_form(write, output_shape):
    """Return lambda row, column expressions for a write's first value."""
    if len(write.values) != 1 or len(write.index) != 2:
        raise ProofError('per-cell output write must be scalar and rank two')
    return write.index


def _atom_lane_offsets():
    """The fixed TF32 m16n8 C fragment map, independent of IR spelling."""
    result = []
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        result.append(((g, 2*t), (g, 2*t+1), (g+8, 2*t), (g+8, 2*t+1)))
    return result


def _prove_accumulator_binding(write, lane, warp_m, warp_n, grid_m, block_m,
                               block_n, *, row, col, lane_cell):
    """Bind one written accumulator to the lane cell that emits it.

    This closes the swap-the-two-writes hole: it is not enough that every
    accumulator reaches some output cell; each accumulator must reach the
    specific cell selected by its own lane fragment slot.
    """
    if lane < 0 or lane >= 32 or warp_m < 0 or warp_n < 0:
        raise ProofError('invalid lane or warp coordinate')
    expected_row, expected_col = lane_cell
    expected_global = (warp_m * 16 + expected_row, warp_n * 8 + expected_col)
    actual_block = (row, col)
    if actual_block != expected_global:
        raise ProofError('accumulator write does not match its MMA C lane slot')
    return expected_global


def _orientation_for_copies(copies, row_grid_vars, col_grid_vars):
    """Return (swap, source, local_kind) for a staged operand's copies."""
    if not copies:
        raise ProofError('shared operand has no copy producer')
    sources = {c[0].src for c in copies}
    if len(sources) != 1:
        raise ProofError('shared operand is filled from multiple global buffers')
    source = next(iter(sources))
    decisions = set()
    for copy, chain, _ in copies:
        dest = _copy_dest(copy)
        src = tuple(copy.src_index)
        if len(dest) != 2 or len(src) != 2:
            raise ProofError('staged copy is not a rank-two tile transport')
        source_forms = [linear_form(x) for x in src]
        row_dims = [i for i, form in enumerate(source_forms) if form.variables & row_grid_vars]
        col_dims = [i for i, form in enumerate(source_forms) if form.variables & col_grid_vars]
        if len(row_dims) == 1 and not col_dims:
            swap = row_dims[0] != 0
        elif len(col_dims) == 1 and not row_dims:
            swap = col_dims[0] != 1
        else:
            raise ProofError('staged copy does not prove a row/column global orientation')
        decisions.add(swap)
    if len(decisions) != 1:
        raise ProofError('staged copies disagree on operand orientation')
    return decisions.pop(), source


def _role(index, row_form, col_form, k_atom_key, block_vars):
    """Classify a rank-two operand index as (row, k) or (k, col)."""
    forms = [linear_form(x) for x in index]
    row = []
    col = []
    for i, form in enumerate(forms):
        diff = row_form.minus(form)
        if not (diff.variables - set(block_vars)):
            row.append(i)
        diff = col_form.minus(form)
        if not (diff.variables - set(block_vars)):
            col.append(i)
    k = [i for i, form in enumerate(forms) if k_atom_key in form.terms]
    if len(k) != 1:
        raise ProofError('operand does not have one reduction coordinate')
    k_index = k[0]
    # A row operand must not use the K coordinate; a column operand likewise.
    row = [i for i in row if i != k_index]
    col = [i for i in col if i != k_index]
    if len(row) == 1 and not col:
        return ('row', row[0], k_index)
    if len(col) == 1 and not row:
        return ('col', col[0], k_index)
    raise ProofError('operand coordinate role is ambiguous or non-affine')


def _is_proved_affine(form, phase_vars=frozenset()):
    """Reject data-dependent divides/modulos in operand coordinate roles.

    Lane/warp tiling atoms have a single-variable numerator.  A numerator
    containing products or several variables is a gather, not a coordinate
    factorization, even though it forms an equality atom in ``LinearForm``.
    """
    def atom_ok(expr):
        # Structural atoms are ('v', name), ('l', value), and
        # ('b', op, left, right), as produced by ``_structural``.
        if isinstance(expr, tuple) and len(expr) == 2 and expr[0] in ('v', 'l'):
            return True
        if isinstance(expr, tuple) and len(expr) == 4 and expr[0] == 'b':
            op, left, right = expr[1], expr[2], expr[3]
            if op in ('+', '-'):
                return atom_ok(left) and atom_ok(right)
            if op == '*':
                return ((isinstance(left, tuple) and left[0] == 'l' and atom_ok(right)) or
                        (isinstance(right, tuple) and right[0] == 'l' and atom_ok(left)))
            if op in ('/', '%'):
                return (isinstance(left, tuple) and left[0] == 'v' and
                        isinstance(right, tuple) and right[0] == 'l')
        return False
    # Only an atom with a single coordinate variable (optionally scaled by a
    # literal) may be opaque.  Composite nested modulo/division is a gather;
    # simple lane/warp decoding is legal affine factorization.
    for key in form.terms:
        if key[0] == 'atom':
            if _is_stage_phase(key, phase_vars):
                continue
            if not atom_ok(key[1]):
                return False
    return True


def _orientation_signature(form, ignored_vars=()):
    """Coefficients of thread/lane atoms, excluding CTA-uniform extents."""
    signature = {}
    for key, coefficient in form.terms.items():
        if key[0] != 'atom':
            continue
        key_vars = set(_vars_from_key(key))
        if key_vars & set(ignored_vars):
            continue
        signature[key] = coefficient
    return signature


def _vars_from_key(key):
    if isinstance(key, tuple) and len(key) == 2 and key[0] == 'var':
        return {key[1]}
    found = set()
    def walk(item):
        if isinstance(item, str):
            return
        if isinstance(item, tuple):
            if len(item) == 2 and item[0] == 'v':
                found.add(item[1])
            else:
                for child in item[1:]:
                    walk(child)
    walk(key)
    return found


def _affine_in_variables(form, variables):
    """Whether a form varies only linearly in the named variables.

    Equality at two endpoints proves equality over an entire finite launch
    interval only when both sides are affine in the same launch variables.  A
    conditional, product, or divide involving a CTA coordinate is an opaque
    atom in ``LinearForm``; silently sampling endpoints could then accept an
    interior-only mutation.
    """
    for key in form.terms:
        if key[0] != 'var' and (_vars_from_key(key) & set(variables)):
            return False
    return True


def _is_stage_phase(atom, phase_vars):
    """Prove a copy slot term is ``(stage-loop / slots + constant) % slots``."""
    if atom[0] != 'atom':
        return False
    expr = atom[1]
    def op(node, name):
        return isinstance(node, tuple) and len(node) == 4 and node[0] == 'b' and node[1] == name
    if not op(expr, '%'):
        return False
    modulus = expr[3]
    if not (isinstance(modulus, tuple) and len(modulus) == 2 and modulus[0] == 'l'
            and isinstance(modulus[1], int) and modulus[1] > 0):
        return False
    inner = expr[2]
    if op(inner, '+') and isinstance(inner[3], tuple) and inner[3][0] == 'l':
        inner = inner[2]
    if not op(inner, '/'):
        return False
    divisor = inner[3]
    if not (isinstance(divisor, tuple) and len(divisor) == 2 and divisor[0] == 'l'
            and isinstance(divisor[1], int) and divisor[1] > 0):
        return False
    names = _vars_from_key(('atom', inner[2]))
    return bool(names) and names <= set(phase_vars)


def _loop_count(loop):
    if isinstance(loop, StridedLoop):
        start = int(eval_expr(loop.start, {}))
        step = int(eval_expr(loop.step, {}))
        end = int(eval_expr(loop.end, {})) if loop.end is not None else _static(loop.axis.extent)
        if step <= 0 or start < 0:
            raise ProofError('invalid strided reduction loop')
        return math.ceil((end - start) / step), start, step
    return _static(loop.axis.extent), 0, 1


def _k_certificate(chain, k):
    if not chain:
        raise ProofError('contraction has no reduction loop')
    total = 1
    inner_count = None
    for i, loop in enumerate(reversed(chain)):
        count, start, step = _loop_count(loop)
        if start != 0:
            raise ProofError('reduction loop does not start at the zero coordinate')
        if i == 0:
            if step != 1:
                raise ProofError('innermost reduction is not unit stride')
            total = count
            inner_count = count
        else:
            if step != inner_count:
                raise ProofError('outer reduction loop does not tile the inner loop')
            total *= count
            inner_count = total
    return total


def _block_tile(cells, output_shape):
    rows = sorted({r for r, _ in cells})
    cols = sorted({c for _, c in cells})
    if not rows or not cols:
        raise ProofError('CTA writes no output cells')
    bm, bn = len(rows), len(cols)
    expected = {(r, c) for r in rows for c in cols}
    if set(cells) != expected:
        raise ProofError('CTA output is not a rectangular tile')
    return min(rows), min(cols), bm, bn


def _atomic_groups(write):
    return bool(write.atomic)


def classify(graph, node, meta):
    """Return a fail-closed Plan/UnsupportedMatch/NoMatch classification."""
    try:
        if type(node.op).__name__ != 'KernelOp':
            return Classification(NO_MATCH, 'node is not a KernelOp')
        op = node.op
        if op.serial:
            return Classification(UNSUPPORTED, 'serial runtime axes are not supported')
        tiles = [s for s in op.body if isinstance(s, Tile)]
        if len(tiles) != 1 or len(tiles) != len(op.body):
            return Classification(NO_MATCH, 'body is not one dedicated Tile region')
        tile = tiles[0]
        if tile.aux_threads or len(meta['block']) != 3 or len(meta['grid']) != 3:
            return Classification(UNSUPPORTED, 'auxiliary threads or non-3D launch metadata')
        block_threads = math.prod(int(x) for x in meta['block'])
        grid_size = math.prod(int(x) for x in meta['grid'])
        if any(int(x) <= 0 for x in meta['block'] + meta['grid']):
            return Classification(UNSUPPORTED, 'non-positive launch extent')
        if any(int(x) != 1 for x in meta['block'][1:] + meta['grid'][1:]):
            return Classification(UNSUPPORTED, 'initial emitter requires a 1-D launch')
        if tile.block_threads != block_threads:
            return Classification(UNSUPPORTED, 'Tile block_threads differs from prescribed launch')
        extents, block_axes, thread_axes, tile_cells = _block_and_thread_axes(tile)
        if grid_size * block_threads != tile_cells:
            return Classification(UNSUPPORTED, 'launch does not cover the Tile exactly')
        walk = _walk(tile.body)
        if not walk.writes:
            return Classification(NO_MATCH, 'no output writes')
        outputs = {w[0].output for w in walk.writes}
        if len(outputs) != 1:
            return Classification(NO_MATCH, 'multiple output buffers')
        output = next(iter(outputs))
        output_shape = _shape_of(output, graph, meta)
        if len(output_shape) != 2:
            return Classification(NO_MATCH, 'output is not rank two')
        if any(w[0].swizzle not in ('NONE', None) for w in walk.writes):
            return Classification(NO_MATCH, 'swizzled output')
        atomic = {bool(w[0].atomic) for w in walk.writes}
        if len(atomic) != 1:
            return Classification(NO_MATCH, 'mixed atomic and ordinary writes')
        atomic_output = atomic.pop()
        if atomic_output and output not in meta.get('zero_outputs', []):
            return Classification(UNSUPPORTED, 'atomic output lacks metadata reset')

        # Expand vector writes to the value -> (row, col) map used by the proof.
        coord_map = {}
        for write, _, _ in walk.writes:
            for value, row, col in _expanded_coordinates(write, output_shape):
                coords = (row, col)
                if value in coord_map and coord_map[value] != coords:
                    raise ProofError('one accumulator aliases distinct output cells')
                coord_map[value] = coords
        writes_per_value = {}
        for write, _, _ in walk.writes:
            for value in write.values:
                writes_per_value[value] = writes_per_value.get(value, 0) + 1
        if any(count != 1 for count in writes_per_value.values()):
            raise ProofError('an accumulator is written more than once')

        # Trace every written value to exactly one add-fold over one multiply of
        # two distinct loads.
        folds = {}
        value_operands = {}
        for value in coord_map:
            seeds = walk.inits.get(value, [])
            if len(seeds) > 1:
                raise ProofError('accumulator has multiple seeds')
            if seeds and float(seeds[0][0].identity) != 0.0:
                raise ProofError('accumulator seed is not the zero identity')
            candidates = walk.folds.get(value, [])
            if len(candidates) != 1:
                raise ProofError(f'written value {value!r} is not one reduction fold')
            fold, chain, _ = candidates[0]
            if fold.op.name not in ('add', 'sum') or fold.base not in (None, value):
                raise ProofError('output accumulator is not a plain sum')
            if len(fold.axes) != 1 or len(chain) == 0 or chain[-1].axis.name != fold.axes[0]:
                raise ProofError('fold axis is not the innermost reduction loop')
            assigns = walk.assigns.get(fold.value, [])
            if len(assigns) != 1:
                raise ProofError('fold value is not one multiply assignment')
            assign, assign_chain, _ = assigns[0]
            if assign.op.name != 'multiply' or len(assign.args) != 2:
                raise ProofError('fold is not a binary multiply reduction')
            operands = []
            for name in assign.args:
                loads = walk.loads.get(name, [])
                if len(loads) != 1:
                    raise ProofError('multiply operand is not one load')
                operands.append(loads[0][0])
            if operands[0].input == operands[1].input:
                raise ProofError('both operands load the same buffer')
            folds[value] = (fold, chain, operands[0], operands[1])

        chains = [entry[1] for entry in folds.values()]
        if any(chain != chains[0] for chain in chains):
            raise ProofError('output cells do not share one reduction loop nest')
        chain = chains[0]
        k_atom_key = ('var', chain[-1].axis.name)
        # The reduction chain's loop cardinalities prove how many K elements
        # each participating CTA covers.  A pipeline's slot-index phase is
        # validated separately against typed copy indices.
        k_chain = 1
        for loop in chain:
            count, start, step = _loop_count(loop)
            if start != 0:
                raise ProofError('reduction loop extent is not a zero-based K coverage')
            k_chain *= count
        phase_vars = {loop.axis.name for loop in chain}

        # File the two operand buffers by proving their coordinate roles.  A
        # staged operand first proves a copy orientation, then the shared-load
        # index must have the corresponding local row/column role.
        block_vars = {axis.name for axis in block_axes}
        block_values = {axis.name: _static(axis.extent) for axis in block_axes}
        first_value = next(iter(coord_map))
        _, first_row, first_col = next((value, row, col) for w in walk.writes
                                       for value, row, col in _expanded_coordinates(w[0], output_shape)
                                       if value == first_value)
        row_vars = _vars(first_row) - _vars(first_col)
        col_vars = _vars(first_col) - _vars(first_row)
        if not row_vars or not col_vars:
            raise ProofError('output row and column coordinates are not separable')
        operand_info = []
        value_roles = {}
        global_reads = {}
        for value, (_, _, left_load, right_load) in folds.items():
            if len(left_load.index) != 2 or len(right_load.index) != 2:
                raise ProofError('operand load is not rank two')
            pair = []
            for operand in (left_load, right_load):
                # A nested reduction nest may use several loop variables in
                # one K coordinate; the K certificate above proves their
                # exact product.  A dimension is a K dimension when it
                # contains any variable from that proved nest.  For a shared
                # operand, remove the separately proved rotating-slot term
                # first so a stage index does not masquerade as K.
                dim_forms = []
                for index in operand.index:
                    form = linear_form(index)
                    if operand.input in walk.smem:
                        form, _ = _without_stage_terms(form, phase_vars)
                    dim_forms.append(form)
                k_dims = [i for i, form in enumerate(dim_forms)
                          if form.variables & set(phase_vars)]
                if len(k_dims) != 1:
                    raise ProofError('operand does not have exactly one K coordinate')
                other = 1 - k_dims[0]
                # Staging-slot elimination is meaningful only for a shared
                # operand whose copy producer has already been identified.  A
                # direct global operand has no stage slot; accepting a phase
                # atom there would let a modulo K map masquerade as a slot.
                if operand.input in walk.smem:
                    other_form, _ = _without_stage_terms(
                        linear_form(operand.index[other]), phase_vars)
                    other_proved = _is_proved_affine(other_form, phase_vars)
                else:
                    other_form = linear_form(operand.index[other])
                    other_proved = _is_proved_affine(other_form)
                if not other_proved:
                    raise ProofError('operand coordinate is a non-affine gather')
                form_vars = other_form.variables
                if not (form_vars & (row_vars | col_vars)):
                    # A thread-local shared tile omits the CTA-uniform term; its
                    # orientation is certified by the copy producer below.
                    form_vars = row_vars if other == 0 else col_vars
                row_side = bool(form_vars & row_vars)
                col_side = bool(form_vars & col_vars)
                if row_side == col_side:
                    raise ProofError('operand coordinate does not prove a unique row/column role')
                load_source = operand.input
                if load_source in walk.smem:
                    # The shared buffer's orientation comes from the global copy
                    # producer, never from a buffer name or accumulator spelling.
                    swap, global_source = _orientation_for_copies(walk.copies.get(load_source, []), row_vars & block_vars, col_vars & block_vars)
                    if not swap:
                        global_row_dim, global_col_dim = 0, 1
                    else:
                        global_row_dim, global_col_dim = 1, 0
                    k_global_dim = 1 - global_row_dim
                    # The simple shared-tile interface stores A as (row, k)
                    # and B as (k, column) locally.  Their *global* source may
                    # still be transposed; orientation comes from the copy map
                    # and is checked against the global cell below.
                    role = 'row' if row_side else 'col'
                    if role == 'row' and k_dims[0] != 1:
                        raise ProofError('shared row operand is not a row-major tile')
                    if role == 'col' and k_dims[0] != 0:
                        raise ProofError('shared column operand is not a K-major tile')
                    # Local tile orientation and global source orientation are
                    # independent.  The copy may read a transposed global view
                    # into a canonical local (row,k)/(k,column) tile; the
                    # staged-read trace below then binds every shared cell to
                    # its exact global source cell.
                else:
                    global_source = load_source
                    role = 'row' if row_side else 'col'
                    k_global_dim = k_dims[0]
                    transpose = (role == 'row' and k_global_dim == 0) or (role == 'col' and k_global_dim == 1)
                    if k_global_dim not in (0, 1):
                        raise ProofError('operand has an invalid K dimension')
                    # A direct global load must name the same global row or
                    # column as the output cell it feeds.  Checking only CTA 0
                    # would accept a CTA-dependent offset on the operand while
                    # the emitter stores the canonical output cell.
                    non_k = linear_form(operand.index[1 - k_dims[0]])
                    expected_coord = coord_map[value][0 if role == 'row' else 1]
                    if non_k != linear_form(expected_coord):
                        raise ProofError(
                            'direct operand coordinate drifts from its output cell')
                    # For a direct operand the K dimension's exact product
                    # must be the reduction coordinate.  Non-affine or
                    # conditional K maps are rejected before they can alter
                    # the emitted contraction.
                    k_form = dim_forms[k_dims[0]]
                    if not _is_proved_affine(k_form):
                        raise ProofError('direct operand K coordinate is a non-affine gather')
                pair.append((operand, role, global_source, k_global_dim, k_dims[0]))
            if pair[0][1] == pair[1][1]:
                raise ProofError('loads do not split into row and column operands')
            if pair[0][1] == 'col':
                pair.reverse()
            operand_info.append(pair)
            value_roles[value] = pair

        # Each operand's K coordinate must be exactly the innermost reduction
        # variable after stripping a proved rotating-slot term.  Merely
        # containing the variable would let k+1 or 2*k silently select a
        # different contraction from the one the emitter reads.
        reduction_names = {loop.axis.name for loop in chain}
        split_names = {
            name for name in block_vars
            if name not in reduction_names
            and not ({name} & (row_vars | col_vars | {'warp'}))
        }
        split_names = {
            name for name in split_names
            if any(('var', name) in linear_form(index).terms
                   for _, _, left, right in folds.values()
                   for operand in (left, right)
                   for index in operand.index)
        }
        inner_k = Var(chain[-1].axis.name)
        # A fully enumerated reduction nest may spell K as a sum of its loop
        # variables (for example k1 + k0 where k1 strides by the inner extent).
        # A rotating pipeline instead names only the inner K variable in the
        # compute load; its typed copy replay proves that the stage slot adds
        # the outer reduction coordinate.  Keep those two exact forms separate
        # rather than accepting arbitrary expressions that merely contain K.
        expected_k_full = LinearForm({}, 0, frozenset())
        counts = [_loop_count(loop)[0] for loop in chain]
        for i, loop in enumerate(chain):
            outer_stride = math.prod(counts[i + 1:])
            expected_k_full = expected_k_full.plus(
                LinearForm({('var', loop.axis.name): 1}, 0,
                           frozenset({loop.axis.name})).scaled(outer_stride))
        expected_k_inner = LinearForm({('var', inner_k.name): 1}, 0,
                                      frozenset({inner_k.name}))
        split_offset = LinearForm({}, 0, frozenset())
        for name in sorted(split_names):
            split_offset = split_offset.plus(
                LinearForm({('var', name): 1}, 0, frozenset({name})).scaled(
                    math.prod(_loop_count(loop)[0] for loop in chain)))
        expected_k_full = expected_k_full.plus(split_offset)
        expected_k_inner = expected_k_inner.plus(split_offset)
        for value, pair in value_roles.items():
            for operand, role, _, _, k_dim in pair:
                form = linear_form(operand.index[k_dim])
                if operand.input in walk.smem:
                    form, _ = _without_stage_terms(form, phase_vars)
                accepted = (form == expected_k_full
                            or (operand.input in walk.smem
                                and form == expected_k_inner))
                if not accepted:
                    raise ProofError('operand K coordinate is not the reduction coordinate')

        # Bind every accumulator to its own MMA C fragment slot.  The write
        # coverage proof alone only says the four values partition the tile;
        # it does not say c0 (which reads a0/b0) is stored at lane offset
        # (g, 2t), not at c1's neighboring cell.  Compare the typed write
        # coordinates against the exact operand offsets that define each
        # fragment.  The fixed lane map then makes the binding complete.
        def non_k(operand):
            k_dims = [i for i, index in enumerate(operand.index)
                      if k_atom_key in linear_form(index).terms]
            if len(k_dims) != 1:
                raise ProofError('operand lost its K coordinate')
            return operand.index[1 - k_dims[0]]

        # Bind each accumulator to its own C fragment slot.  The proof is
        # evaluated over every launched thread in CTA 0: for each accumulator,
        # its typed write-coordinate delta from the base fragment must equal
        # the delta of that accumulator's own A-row/B-column loads.  A write
        # swap therefore fails even though the four writes still partition the
        # tile.  The four observed (row, column) deltas must form one complete
        # 2x2 fragment grid; both m16n8 layouts used by the corpus (adjacent
        # rows, or g/g+8) are accepted, no other lane map is.
        ordered = {}
        for value, pair in value_roles.items():
            row_entry = next(entry for entry in pair if entry[1] == 'row')
            col_entry = next(entry for entry in pair if entry[1] == 'col')
            ordered[value] = (non_k(row_entry[0]), non_k(col_entry[0]))
        base_value = next(iter(ordered))
        base_row_expr, base_col_expr = ordered[base_value]
        base_write_row, base_write_col = coord_map[base_value]
        observed = set()
        for tid in range(block_threads):
            env, _, _ = _decode_env(tile, meta, 0, tid)
            for loop in chain:
                if type(loop).__name__ == 'StridedLoop':
                    env[loop.axis.name] = int(eval_expr(loop.start, {}))
            base_row = int(eval_expr(base_row_expr, env))
            base_col = int(eval_expr(base_col_expr, env))
            base_out_row = int(eval_expr(base_write_row, env))
            base_out_col = int(eval_expr(base_write_col, env))
            if (base_row, base_col) != (base_out_row, base_out_col):
                raise ProofError('base accumulator is not at its A/B lane coordinate')
            for value, (row_expr, col_expr) in ordered.items():
                write_row, write_col = coord_map[value]
                if (int(eval_expr(row_expr, env)) - base_row,
                        int(eval_expr(col_expr, env)) - base_col) != (
                        int(eval_expr(write_row, env)) - base_out_row,
                        int(eval_expr(write_col, env)) - base_out_col):
                    raise ProofError('accumulator write is bound to another C fragment slot')
                observed.add((int(eval_expr(row_expr, env)) - base_row,
                              int(eval_expr(col_expr, env)) - base_col))
        row_values = {row for row, _ in observed}
        col_values = {col for _, col in observed}
        fragment_tail = None
        if len(observed) != 4 or len(row_values) != 2 or len(col_values) != 2:
            # Defer a partial C fragment until after launch factorization.  A
            # forced-warp near miss (m02) must report the atom/warp mismatch,
            # not an incidental fragment-tail reason.  The plan is still
            # rejected before construction below.
            fragment_tail = 'output tail does not cover an m16n8 C fragment grid'
        else:
            rows = sorted(row_values)
            cols = sorted(col_values)
            if rows not in ([0, 1], [0, 8]) or cols != [0, 1]:
                raise ProofError('accumulator fragment offsets are not an m16n8 C map')

        first_a = operand_info[0][0][0]
        first_b = operand_info[0][1][0]
        first_a_global = operand_info[0][0][2]
        first_b_global = operand_info[0][1][2]
        if any(pair[0][2] != first_a_global or pair[1][2] != first_b_global for pair in operand_info):
            raise ProofError('output cells use different contraction operands')
        a_source, b_source = first_a.input, first_b.input
        staged = a_source in walk.smem or b_source in walk.smem
        if staged:
            if a_source not in walk.smem or b_source not in walk.smem:
                raise ProofError('mixed global and shared contraction operands')
            if first_a.input == first_b.input:
                raise ProofError('the same shared buffer is both operands')
            row_grid_vars = row_vars & block_vars
            col_grid_vars = col_vars & block_vars
            a_swap, a_global = _orientation_for_copies(walk.copies.get(a_source, []), row_grid_vars, col_grid_vars)
            b_swap, b_global = _orientation_for_copies(walk.copies.get(b_source, []), row_grid_vars, col_grid_vars)
            if a_global == b_global:
                raise ProofError('staged operands share one global source')
            if a_global != first_a_global or b_global != first_b_global:
                raise ProofError('staged copy source disagrees with the contraction operand role')
            a_transpose, b_transpose = a_swap, b_swap
            a_buffer, b_buffer = a_global, b_global
            if (not _simple_staged_candidate(walk.copies.get(a_source, []))
                    or not _simple_staged_candidate(walk.copies.get(b_source, []))):
                # A rotating full pipeline is still a recognized contraction,
                # but it is not the simple staging interface proved here.  It
                # keeps the frozen SIMT fallback rather than being mislabeled.
                staging = 'pipeline'
                a_buffer = a_global
                b_buffer = b_global
                a_transpose, b_transpose = a_swap, b_swap
            else:
                staging = 'shared'
                # The typed copy and typed compute loads must describe the same
                # global-to-shared bijection.  Their full enumeration is deferred
                # until the launch rectangle and operand shapes are known below.
            # Shared staging must commit, wait, and have a CTA barrier before
            # compute and another before a later overwrite.
            if not walk.commits or not walk.waits:
                raise ProofError('staged plan lacks cp.async commit/wait ordering')
            fold_loops = set()
            for loop in chain:
                fold_loops.add(id(loop))
            fold_records = [record for records in walk.folds.values() for record in records]
            compute_seq = min(seq for _, _, seq in fold_records)
            after_compute_seq = max(seq for _, _, seq in fold_records)
            cta_syncs = [seq for s, c, seq in walk.syncs
                         if not s.warp and s.count is None]
            sync_before = cta_syncs
            if not any(seq < compute_seq for seq in sync_before):
                raise ProofError('staged plan lacks a pre-compute CTA barrier')
            if not any(seq > after_compute_seq for seq in sync_before):
                # This is the explicit partial-collective near miss.
                raise ProofError('staged plan lacks a post-compute CTA barrier')
            for s, _, _ in walk.syncs:
                if s.warp or s.count is not None:
                    raise ProofError('staged barrier is not a full CTA collective')
        else:
            if a_source == b_source:
                raise ProofError('both operands are the same global buffer')
            a_buffer, b_buffer = a_source, b_source
            a_transpose = operand_info[0][0][3] == 0
            b_transpose = operand_info[0][1][3] == 1
            staging = 'global'

        a_shape = _shape_of(a_buffer, graph, meta)
        b_shape = _shape_of(b_buffer, graph, meta)
        metadata_args = {arg['name']: arg for arg in meta['args']}
        if any(metadata_args.get(name, {}).get('dtype') != 'f32'
               for name in (a_buffer, b_buffer, output)):
            return Classification(UNSUPPORTED, 'TF32 plan requires f32 operands and output')
        m, n = output_shape
        if a_transpose:
            k, a_m = a_shape
        else:
            a_m, k = a_shape
        if b_transpose:
            b_n, b_k = b_shape
        else:
            b_k, b_n = b_shape
        if a_m != m or b_n != n or a_shape == b_shape and a_buffer != b_buffer and not (a_transpose or b_transpose):
            # A square A/B pair is legal; only the dimension checks matter.
            pass
        if a_m != m or b_n != n:
            raise ProofError('operand shape does not match the output extents')
        if b_k != k:
            raise ProofError('operand K extents differ')
        if k % 8:
            return Classification(UNSUPPORTED, 'K is not TF32-atom aligned')

        # K coverage is a proof over the exact reduction loop nest.
        k_per_cta = _k_certificate(chain, k)
        if k_per_cta <= 0 or k % k_per_cta:
            raise ProofError('reduction nest does not divide K')
        split = k // k_per_cta

        # Enumerate the complete finite launch and prove output coverage and
        # the CTA tile extent from typed Write coordinates.
        coverage = {}
        per_block = {}
        for bid in range(grid_size):
            block_cells = []
            for tid in range(block_threads):
                env, gid, total = _decode_env(tile, meta, bid, tid)
                if gid >= tile_cells:
                    raise ProofError('inactive Tile thread in a Plan candidate')
                for write, _, _ in walk.writes:
                    try:
                        block_cells.extend(_write_cells(write, env, output_shape))
                    except ProofError as exc:
                        if 'leaves the output' in str(exc):
                            raise ProofError('launch output coverage leaves the output') from exc
                        raise
            per_block[bid] = block_cells
            for cell in block_cells:
                coverage[cell] = coverage.get(cell, 0) + 1
        expected_cells = {(r, c) for r in range(m) for c in range(n)}
        if set(coverage) != expected_cells:
            if set(coverage) < expected_cells:
                return Classification(UNSUPPORTED, 'output coverage has a tail or missing cells')
            raise ProofError('output writes escape the declared output')
        active_warps = block_threads // 32
        if per_block and len({len(cells) for cells in per_block.values()}) == 1:
            cells_per_cta = len(next(iter(per_block.values())))
            unique_per_cta = len(set(next(iter(per_block.values()))))
            if unique_per_cta and cells_per_cta != unique_per_cta:
                atoms = unique_per_cta // 128
                if unique_per_cta % 128 == 0 and atoms != active_warps:
                    return Classification(UNSUPPORTED,
                                          f'launch factorization has {atoms} atoms for {active_warps} warps')
        if not atomic_output and any(count != 1 for count in coverage.values()):
            return Classification(UNSUPPORTED, 'duplicate non-atomic output coverage (split-K without atomics)')
        if atomic_output and split == 1 and any(count != 1 for count in coverage.values()):
            raise ProofError('atomic output is duplicated without a split-K proof')
        if split > 1 and not atomic_output:
            return Classification(UNSUPPORTED, 'split-K requires atomic accumulation')
        tiles = [_block_tile(cells, output_shape) for cells in per_block.values()]
        if len({(t[2], t[3]) for t in tiles}) != 1:
            return Classification(UNSUPPORTED, 'CTA tiles have nonuniform extents')
        _, _, block_m, block_n = tiles[0]
        if block_m % 16 or block_n % 8:
            return Classification(UNSUPPORTED, 'CTA tile is not an exact atom multiple')
        atoms_m, atoms_n = block_m // 16, block_n // 8
        active_warps = block_threads // 32
        if atoms_m * atoms_n != active_warps:
            return Classification(UNSUPPORTED,
                                  f'launch factorization has {atoms_m*atoms_n} atoms for {active_warps} warps')
        grid_m, grid_n = m // block_m, n // block_n
        if grid_m * grid_n != grid_size:
            return Classification(UNSUPPORTED, 'CTA grid does not exactly factor the output tile grid')
        if rows_cells := [tiles[i][0] for i in range(len(tiles))]:
            # Verify the CTA order is the row-major (M outer, N inner) order the
            # emitter derives from bid, not a fixture-name convention.
            for bid, (r0, c0, bm, bn) in enumerate(tiles):
                if (r0, c0) != ((bid // grid_n) * block_m, (bid % grid_n) * block_n):
                    return Classification(UNSUPPORTED, 'CTA coordinate order is not proven row-major')
        if split > 1:
            return Classification(UNSUPPORTED, 'atomic split-K is outside the one-atom emitter')
        if atomic_output:
            return Classification(UNSUPPORTED, 'atomic output is outside the one-atom emitter')
        if fragment_tail is not None:
            return Classification(UNSUPPORTED, fragment_tail)
        if staging == 'shared':
            a_maps = _staged_copy_proof(tile, walk, a_source,
                                        walk.copies.get(a_source, []), a_shape,
                                        output_shape, grid_size, block_threads)
            b_maps = _staged_copy_proof(tile, walk, b_source,
                                        walk.copies.get(b_source, []), b_shape,
                                        output_shape, grid_size, block_threads)
            try:
                _prove_staged_reads(tile, walk, chain, value_roles, coord_map,
                                    {a_source: a_maps, b_source: b_maps},
                                    grid_size, block_threads, a_source, b_source,
                                    a_transpose, b_transpose)
            except ProofError as exc:
                return Classification(UNSUPPORTED, str(exc))
        elif staging == 'pipeline':
            try:
                _trace_pipeline(
                    tile, walk, chain, value_roles, coord_map,
                    a_source, b_source, a_buffer, b_buffer, a_shape, b_shape, grid_size,
                    block_threads, a_transpose, b_transpose)
            except ProofError as exc:
                return Classification(UNSUPPORTED, str(exc))
        offsets, smem_bytes = _smem_layout(walk)
        plan = MatmulPlan(
            entry=node.op.name,
            output=output,
            a_buffer=a_buffer,
            b_buffer=b_buffer,
            m=m, n=n, k=k,
            block_m=block_m, block_n=block_n,
            warps_m=atoms_m, warps_n=atoms_n,
            active_warps=active_warps,
            grid_m=grid_m, grid_n=grid_n,
            block_threads=block_threads,
            a_transpose=a_transpose,
            b_transpose=b_transpose,
            staging=staging,
            a_smem=a_source if staging == 'shared' else None,
            b_smem=b_source if staging == 'shared' else None,
            a_smem_shape=(tuple(_static(x) for x in walk.smem[a_source].extents)
                          if staging == 'shared' else None),
            b_smem_shape=(tuple(_static(x) for x in walk.smem[b_source].extents)
                          if staging == 'shared' else None),
            smem_offsets=((offsets[a_source], offsets[b_source])
                          if staging == 'shared' else None),
            proof={'output_partition': 'exactly-once', 'k_coverage': k,
                   'reduction_nest': [loop.axis.name for loop in chain],
                   'lane_partition': 'all 128 atom cells exactly once',
                   'collectives': 'typed CTA barriers' if staged else 'no shared collective',
                   'simple_staging': staging == 'shared',
                   'smem_bytes': smem_bytes},
        )
        plan.validate()
        return Classification(PLAN, 'validated one-atom TF32 matmul plan', plan, plan.proof)
    except ProofError as exc:
        reason = str(exc)
        # Structural near misses remain NoMatch; a recognized contraction that
        # fails a plan/launch obligation is explicitly unsupported.
        if any(token in reason for token in ('launch', 'split-K', 'tail', 'CTA', 'atomic')):
            return Classification(UNSUPPORTED, reason)
        return Classification(NO_MATCH, reason)
