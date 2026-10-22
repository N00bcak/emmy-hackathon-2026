#!/usr/bin/env python3
"""Adversarial typed-IR probes for tensor_core.classify. Read-only w.r.t. compiler."""
from __future__ import annotations
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.tensor_core_corpus import Case, add, div, lit, mod, mul, sub, v
from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import (
    Accum, Assign, CpAsyncCopy, KernelOp, Load, Loop, Smem, StridedLoop, Sync, Tile, Write,
)
from emmy.compiler.ir.axis import Axis
from emmy.compiler.dtype import F32
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.stmt.leaves import Init
from emmy.compiler.tensor import Tensor
import tensor_core
from mutate_helpers import (
    copies, mutate_statements, replace_tile, replace_tile_statements, save_case,
)

OUT = ROOT / 'experiments/tensor-core-audit' / 'probes'
ROWS = []


def mutate_copy(fn):
    def walk(statement):
        nested = {}
        for field in dataclasses.fields(statement):
            if field.name in ('body', 'else_body', 'combine_states'):
                value = getattr(statement, field.name)
                if value:
                    nested[field.name] = tuple(walk(x) for x in value)
        if nested:
            statement = dataclasses.replace(statement, **nested)
        if type(statement).__name__ == 'CpAsyncCopy':
            return fn(statement)
        return statement
    return walk


def case(ident, m, n, k, **kwargs):
    graph, meta, inputs, expected = Case(ident, 'Plan', m, n, k, **kwargs).build()
    return graph, meta, inputs, expected


def classify(graph, meta, node=None):
    if node is None:
        node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
    return tensor_core.classify(graph, node, meta['kernels'][0])


def row(ident, expected, why, graph, meta, *, node=None, compile_it=True):
    decision = classify(graph, meta, node)
    # Persist and compile the *same* typed node that was classified.  The
    # builder's graph object still holds the pre-mutation node.
    if node is None:
        node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
    graph_to_save = graph
    if graph.nodes[node.id] is not node:
        graph_to_save = Graph.from_dict(graph.to_dict())
        graph_to_save.nodes[node.id] = node

    result = {
        'id': ident,
        'expected': expected,
        'actual': decision.verdict,
        'reason': decision.reason,
        'moral_semantics': why,
    }
    if decision.plan is not None:
        result['plan'] = {
            'staging': decision.plan.staging,
            'output': decision.plan.output,
            'a_buffer': decision.plan.a_buffer,
            'b_buffer': decision.plan.b_buffer,
            'm': decision.plan.m, 'n': decision.plan.n, 'k': decision.plan.k,
            'block_m': decision.plan.block_m, 'block_n': decision.plan.block_n,
            'grid': [decision.plan.grid_m, decision.plan.grid_n],
            'active_warps': decision.plan.active_warps,
        }
    if compile_it:
        folder = OUT / ident
        if folder.exists():
            import shutil
            shutil.rmtree(folder)
        save_case(folder, graph_to_save, meta)
        path = folder / 'case.ir.json'
        done = subprocess.run([sys.executable, str(ROOT / 'compile.py'), str(path)],
                              capture_output=True, text=True, timeout=120)
        result['compile_rc'] = done.returncode
        result['ptx_mma_count'] = done.stdout.count('mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32')
        result['ptx_has_cp_async'] = 'cp.async' in done.stdout
        result['stderr'] = done.stderr.strip()[:300]
    ROWS.append(result)
    print(json.dumps(result, sort_keys=True))
    return decision


# ---- baseline sanity -----------------------------------------------------
graph, meta, inputs, expected = case('a00-baseline-direct', 16, 8, 8)
row('a00-baseline-direct', 'Plan', 'ordinary direct 16x8x8 contraction', graph, meta)

# ---- cta-dependent affine row offset (global path) -----------------------
graph, meta, inputs, expected = case('a01-cta-row-offset', 32, 8, 8)
node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
tile = node.op.body[0]
def cta_row(stmt):
    if type(stmt).__name__ == 'Load' and stmt.input == 'lhs':
        index = list(stmt.index)
        index[0] = add(index[0], mul(v('cta_m'), lit(1)))
        return dataclasses.replace(stmt, index=tuple(index))
    return stmt
node2 = replace_tile(node, replace_tile_statements(tile, cta_row))
row('a01-cta-row-offset', 'NoMatch', 'CTA 1 reads A rows 17..32 but writes C rows 16..31; output is not A@B',
    graph, meta, node=node2)

# ---- CTA-dependent row offset crossing a negative bound ------------------
graph, meta, inputs, expected = case('a02-cta-negative-offset', 32, 8, 8)
node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
tile = node.op.body[0]
def cta_negative(stmt):
    if type(stmt).__name__ == 'Load' and stmt.input == 'lhs':
        index = list(stmt.index)
        # CTA 0 reads row -1; CTA 1 reads rows 15..30.  This must be rejected
        # either for escaping the operand or for wrong logical contraction.
        index[0] = sub(index[0], mul(v('cta_m'), lit(1)))
        return dataclasses.replace(stmt, index=tuple(index))
    return stmt
node2 = replace_tile(node, replace_tile_statements(tile, cta_negative))
row('a02-cta-negative-offset', 'NoMatch', 'CTA 0 reads A row -1; global read escapes declared operand',
    graph, meta, node=node2)

# ---- output aliases the RHS input ---------------------------------------
graph, meta, inputs, expected = case('a03-output-aliases-rhs', 16, 16, 16)
node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
tile = node.op.body[0]
def alias_write(stmt):
    if type(stmt).__name__ == 'Write':
        return dataclasses.replace(stmt, output='rhs')
    return stmt
node2 = replace_tile(node, replace_tile_statements(tile, alias_write))
row('a03-output-aliases-rhs', 'NoMatch', 'output write aliases RHS input; stores mutate an operand',
    graph, meta, node=node2)

# ---- output aliases the LHS input ---------------------------------------
graph, meta, inputs, expected = case('a04-output-aliases-lhs', 16, 16, 16)
node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
tile = node.op.body[0]
def alias_lhs(stmt):
    if type(stmt).__name__ == 'Write':
        return dataclasses.replace(stmt, output='lhs')
    return stmt
node2 = replace_tile(node, replace_tile_statements(tile, alias_lhs))
row('a04-output-aliases-lhs', 'NoMatch', 'output write aliases LHS input; stores mutate an operand',
    graph, meta, node=node2)

# ---- pipeline copy source is shifted, reads still target the same cells ---
graph, meta, inputs, expected = case('a05-pipeline-copy-mismatch', 16, 8, 8, staged=True)
node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
tile = node.op.body[0]
def lane_mutate(stmt):
    if type(stmt).__name__ == 'CpAsyncCopy':
        return dataclasses.replace(stmt, lane_index=(lit(0), lit(1)))
    return stmt
node2 = replace_tile(node, replace_tile_statements(tile, lane_mutate))
row('a05-pipeline-copy-mismatch', 'NoMatch',
    'cp.async copies (r,k) into shared (r+0,k+1), but compute reads (r,k); staged tensor differs',
    graph, meta, node=node2)

# ---- pipeline copy source shifted, cover read outside filled cell -------
graph, meta, inputs, expected = case('a06-pipeline-read-outside', 16, 8, 8, staged=True)
node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
tile = node.op.body[0]
def copy_src_shift(stmt):
    if type(stmt).__name__ == 'CpAsyncCopy':
        index = list(stmt.src_index)
        index[0] = add(index[0], lit(1))
        return dataclasses.replace(stmt, src_index=tuple(index))
    return stmt
node2 = replace_tile(node, replace_tile_statements(tile, copy_src_shift))
row('a06-pipeline-read-outside', 'NoMatch',
    'copy loads A row r+1, but compute reads shared row r; copy no longer defines the read value',
    graph, meta, node=node2)

# ---- structurally valid but unconventional: extra epilogue after accumulation
graph, meta, inputs, expected = case('a07-noop-epilogue', 16, 8, 8)
node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
tile = node.op.body[0]
# Add a global scratch assignment that is not read by any output.  It should
# not change the contraction, though the narrow recognizer may reject it.
def add_unused_epilogue(stmt):
    return stmt
# Use a directly built typed body so this is not a fixture-name mutation.
body = list(tile.body)
body.insert(-5, Assign('unused_epilogue', ElementwiseImpl('add'), ('a0', 'b0'), F32))
node2 = replace_tile(node, dataclasses.replace(tile, body=tuple(body)))
row('a07-noop-epilogue', 'Plan', 'unused elementwise statement cannot change the matmul result',
    graph, meta, node=node2, compile_it=False)

# ---- output coverage appears exact but pointer is wrong -----------------
graph, meta, inputs, expected = case('a08-accumulator-slot-swap', 16, 8, 8)
node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
tile = node.op.body[0]
def swap_c0_c1(stmt):
    if type(stmt).__name__ == 'Write':
        # The builder emits separate Writes in c0,c1,c2,c3 order.  Swap value
        # names for the first two while keeping coordinates.
        if stmt.values == ('acc_c0',):
            return dataclasses.replace(stmt, values=('acc_c1',))
        if stmt.values == ('acc_c1',):
            return dataclasses.replace(stmt, values=('acc_c0',))
    return stmt
node2 = replace_tile(node, replace_tile_statements(tile, swap_c0_c1))
row('a08-accumulator-slot-swap', 'NoMatch',
    'adjacent C fragment slots carry the other accumulator; output cells are permuted',
    graph, meta, node=node2)

# ---- shared staging: copy source transposed but compute reads local -----
graph, meta, inputs, expected = case('a09-shared-copy-swap', 16, 16, 16, staged=True)
node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
tile = node.op.body[0]
def swap_copy_dims(stmt):
    if type(stmt).__name__ == 'CpAsyncCopy' and stmt.smem == '_a_smem':
        return dataclasses.replace(stmt, src_index=tuple(reversed(stmt.src_index)))
    return stmt
node2 = replace_tile(node, replace_tile_statements(tile, swap_copy_dims))
row('a09-shared-copy-swap', 'NoMatch',
    'A copy reads transposed global cells while compute indexes shared as row-major A',
    graph, meta, node=node2)

# ---- launch factorization: 2x2 atoms but 3 warps ------------------------
graph, meta, inputs, expected = case('a10-launch-3-warps', 32, 16, 16, warps=3)
row('a10-launch-3-warps', 'UnsupportedMatch',
    '32x16 output tile is four atoms but launch has three warps (near miss)',
    graph, meta, compile_it=False)

# ---- extra unrelated global load inside reduction -----------------------
# A load that does not feed any fold should be harmless to semantics; the
# recognizer may reject it as an unsupported statement shape.
graph, meta, inputs, expected = case('a11-unused-load', 16, 8, 8)
node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
tile = node.op.body[0]
body = list(tile.body)
body.insert(-3, Load('unused_load', 'lhs', (v('k'), lit(0))))
node2 = replace_tile(node, dataclasses.replace(tile, body=tuple(body)))
row('a11-unused-load', 'Plan', 'unused global load is semantically inert; narrow recognizer may reject',
    graph, meta, node=node2, compile_it=False)

# ---- same source buffer used for A and B (should not be a pure dense matmul)
graph, meta, inputs, expected = case('a12-same-source', 16, 16, 16, same_buffer=True)
row('a12-same-source', 'NoMatch', 'both operands load the same graph buffer; must fail closed',
    graph, meta, compile_it=False)

# ---- broadcast-looking valid matmul: one operand has size-1 K? ----------
# 16x1 @ 1x8 is a legal contraction but not an m16n8k8 atom.  Expect
# UnsupportedMatch (recognized contraction, wrong atom alignment) not Plan.
graph, meta, inputs, expected = case('a13-k1', 16, 8, 1)
row('a13-k1', 'UnsupportedMatch', 'K=1 is a legal matmul but not TF32-atom aligned',
    graph, meta, compile_it=False)

# ---- K=8 but M=8 tail ------------------------------------------------
graph, meta, inputs, expected = case('a14-m8', 8, 8, 8)
row('a14-m8', 'UnsupportedMatch', 'M=8 is below the 16-row atom; tail/atom mismatch',
    graph, meta, compile_it=False)

# ---- direct typed nodes, not the builder template: simple 1-warp case --
# Build the same semantic matmul with nonstandard axis/buffer/SSA names and a
# different but proved lane spelling.  This bypasses Case.build entirely.
g = Graph()
g.add_node(InputOp(), [], Tensor('A', (16, 8), F32), node_id='A')
g.add_node(InputOp(), [], Tensor('B', (8, 8), F32), node_id='B')
gvar = div(v('lane_axis'), lit(4))
tt = mod(v('lane_axis'), lit(4))
row0 = add(mul(v('cta_m'), lit(16)), mul(v('warp'), lit(16)))
col0 = add(mul(v('cta_n'), lit(8)), mul(v('warp'), lit(8)))
rr0, rr1 = add(row0, gvar), add(add(row0, gvar), lit(8))
cc0, cc1 = add(col0, mul(tt, lit(2))), add(add(col0, mul(tt, lit(2))), lit(1))
loop_body = (
    Load('left_low', 'A', (rr0, v('red'))), Load('left_high', 'A', (rr1, v('red'))),
    Load('right_low', 'B', (v('red'), cc0)), Load('right_high', 'B', (v('red'), cc1)),
    Assign('term0', ElementwiseImpl('multiply'), ('left_low', 'right_low'), F32),
    Accum('sum0', 'term0', ElementwiseImpl('add'), F32, axes=('red',), base=None),
    Assign('term1', ElementwiseImpl('multiply'), ('left_low', 'right_high'), F32),
    Accum('sum1', 'term1', ElementwiseImpl('add'), F32, axes=('red',), base=None),
    Assign('term2', ElementwiseImpl('multiply'), ('left_high', 'right_low'), F32),
    Accum('sum2', 'term2', ElementwiseImpl('add'), F32, axes=('red',), base=None),
    Assign('term3', ElementwiseImpl('multiply'), ('left_high', 'right_high'), F32),
    Accum('sum3', 'term3', ElementwiseImpl('add'), F32, axes=('red',), base=None),
)
body = (
    Init('sum0', 0.0, dtype=F32), Init('sum1', 0.0, dtype=F32),
    Init('sum2', 0.0, dtype=F32), Init('sum3', 0.0, dtype=F32),
    Loop(Axis('red', 8), body=loop_body, unroll=True, seed=False),
    Write('C', (rr0, cc0), 'sum0', value_dtype=F32),
    Write('C', (rr0, cc1), 'sum1', value_dtype=F32),
    Write('C', (rr1, cc0), 'sum2', value_dtype=F32),
    Write('C', (rr1, cc1), 'sum3', value_dtype=F32),
)
g.add_node(KernelOp(body=(Tile((Axis('cta_m', 1), Axis('cta_n', 1),
                                Axis('warp', 1), Axis('lane_axis', 32)),
                           body, block_threads=32),),
           name='typed_direct'), ['A', 'B'], Tensor('C', (16, 8), F32), node_id='C')
meta2 = {'id': 'a15-typed-direct', 'rtol': 1e-3, 'atol': 1e-3,
         'graph_inputs': ['A', 'B'], 'graph_outputs': ['C'],
         'kernels': [{'node': 'C', 'entry': 'typed_direct', 'arg_order': ['A', 'B', 'C'],
                      'grid': [1, 1, 1], 'block': [32, 1, 1], 'smem_bytes': 0,
                      'zero_outputs': [],
                      'args': [{'name': 'A', 'dtype': 'f32', 'shape': [16, 8]},
                               {'name': 'B', 'dtype': 'f32', 'shape': [8, 8]},
                               {'name': 'C', 'dtype': 'f32', 'shape': [16, 8], 'output': True}]}]}
row('a15-typed-direct', 'Plan', 'explicit typed nodes with nonstandard names/axis spellings',
    g, meta2, node=g.nodes['C'], compile_it=False)

# ---- direct typed nodes with a CTA-dependent wrong A row ----------------
g_bad = Graph()
g_bad.add_node(InputOp(), [], Tensor('A', (32, 8), F32), node_id='A')
g_bad.add_node(InputOp(), [], Tensor('B', (8, 8), F32), node_id='B')
gvar = div(v('lane_axis'), lit(4))
tt = mod(v('lane_axis'), lit(4))
row0 = add(mul(v('cta_m'), lit(16)), mul(v('warp'), lit(16)))
col0 = add(mul(v('cta_n'), lit(8)), mul(v('warp'), lit(8)))
wrong_row = add(add(row0, gvar), mod(v('cta_m'), lit(2)))
rr0, rr1 = wrong_row, add(wrong_row, lit(8))
cc0, cc1 = add(col0, mul(tt, lit(2))), add(add(col0, mul(tt, lit(2))), lit(1))
loop_body = (
    Load('left_low', 'A', (rr0, v('red'))), Load('left_high', 'A', (rr1, v('red'))),
    Load('right_low', 'B', (v('red'), cc0)), Load('right_high', 'B', (v('red'), cc1)),
    Assign('term0', ElementwiseImpl('multiply'), ('left_low', 'right_low'), F32),
    Accum('sum0', 'term0', ElementwiseImpl('add'), F32, axes=('red',), base=None),
    Assign('term1', ElementwiseImpl('multiply'), ('left_low', 'right_high'), F32),
    Accum('sum1', 'term1', ElementwiseImpl('add'), F32, axes=('red',), base=None),
    Assign('term2', ElementwiseImpl('multiply'), ('left_high', 'right_low'), F32),
    Accum('sum2', 'term2', ElementwiseImpl('add'), F32, axes=('red',), base=None),
    Assign('term3', ElementwiseImpl('multiply'), ('left_high', 'right_high'), F32),
    Accum('sum3', 'term3', ElementwiseImpl('add'), F32, axes=('red',), base=None),
)
body = (
    Init('sum0', 0.0, dtype=F32), Init('sum1', 0.0, dtype=F32),
    Init('sum2', 0.0, dtype=F32), Init('sum3', 0.0, dtype=F32),
    Loop(Axis('red', 8), body=loop_body, unroll=True, seed=False),
    Write('C', (add(row0, gvar), cc0), 'sum0', value_dtype=F32),
    Write('C', (add(row0, gvar), cc1), 'sum1', value_dtype=F32),
    Write('C', (add(add(row0, gvar), lit(8)), cc0), 'sum2', value_dtype=F32),
    Write('C', (add(add(row0, gvar), lit(8)), cc1), 'sum3', value_dtype=F32),
)
g_bad.add_node(KernelOp(body=(Tile((Axis('cta_m', 2), Axis('cta_n', 1),
                                    Axis('warp', 1), Axis('lane_axis', 32)),
                               body, block_threads=32),),
               name='typed_bad'), ['A', 'B'], Tensor('C', (32, 8), F32), node_id='C')
meta_bad = {'id': 'a21-typed-cta-row-offset', 'rtol': 1e-3, 'atol': 1e-3,
            'graph_inputs': ['A', 'B'], 'graph_outputs': ['C'],
            'kernels': [{'node': 'C', 'entry': 'typed_bad', 'arg_order': ['A', 'B', 'C'],
                         'grid': [2, 1, 1], 'block': [32, 1, 1], 'smem_bytes': 0,
                         'zero_outputs': [],
                         'args': [{'name': 'A', 'dtype': 'f32', 'shape': [32, 8]},
                                  {'name': 'B', 'dtype': 'f32', 'shape': [8, 8]},
                                  {'name': 'C', 'dtype': 'f32', 'shape': [32, 8], 'output': True}]}]}
row('a21-typed-cta-row-offset', 'NoMatch',
    'direct typed IR: CTA 1 reads A rows 17..32 for output rows 16..31',
    g_bad, meta_bad, node=g_bad.nodes['C'], compile_it=False)

# ---- output coverage exact but duplicate one value, missing another -----
graph, meta, inputs, expected = case('a16-duplicate-output-cell', 16, 8, 8)
node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
tile = node.op.body[0]
def duplicate_coord(stmt):
    if type(stmt).__name__ == 'Write' and stmt.values == ('acc_c0',):
        # Rewrite c0's coordinate to c1's, leaving c1 in place.  Coverage is
        # then 127 distinct cells with one duplicate, not a full atom.
        index = list(stmt.index)
        index[1] = add(index[1], lit(1))
        return dataclasses.replace(stmt, index=tuple(index))
    return stmt
node2 = replace_tile(node, replace_tile_statements(tile, duplicate_coord))
row('a16-duplicate-output-cell', 'UnsupportedMatch',
    'coverage has a duplicate C cell and a missing C cell; no exact output partition',
    graph, meta, node=node2, compile_it=False)

# ---- output write outside declared shape --------------------------------
graph, meta, inputs, expected = case('a17-write-oob', 16, 8, 8)
node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
tile = node.op.body[0]
def oob_write(stmt):
    if type(stmt).__name__ == 'Write' and stmt.values == ('acc_c3',):
        index = list(stmt.index)
        index[0] = add(index[0], lit(16))
        return dataclasses.replace(stmt, index=tuple(index))
    return stmt
node2 = replace_tile(node, replace_tile_statements(tile, oob_write))
row('a17-write-oob', 'NoMatch', 'one C store escapes the declared output extent',
    graph, meta, node=node2, compile_it=False)

# ---- scalar seed (non-zero accumulator) ---------------------------------
graph, meta, inputs, expected = case('a18-nonzero-seed', 16, 8, 8)
node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
tile = node.op.body[0]
def nonzero_init(stmt):
    if type(stmt).__name__ == 'Init':
        return dataclasses.replace(stmt, identity=1.0)
    return stmt
node2 = replace_tile(node, replace_tile_statements(tile, nonzero_init))
row('a18-nonzero-seed', 'NoMatch', 'accumulator starts at 1, so result is not just A@B',
    graph, meta, node=node2, compile_it=False)

print(json.dumps(ROWS, indent=2))
