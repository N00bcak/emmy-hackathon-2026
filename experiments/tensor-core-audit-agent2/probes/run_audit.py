#!/usr/bin/env python3
"""Independent adversarial probes for the typed TF32 tensor-core checker.

Read-only with respect to compile.py, tensor_core.py, tf32_emitter.py, fixtures,
and the frozen evidence. Each probe is built from typed Emmy nodes, named with a
non-fixture-specific semantic expectation, then classified by the current
checker and compiled with the current compile.py. Outputs are persisted under
this experiment directory. No benchmark is run.
"""
from __future__ import annotations
import dataclasses
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from construct import (  # noqa: E402
    add, base_body, cond, div, kernel_node, lit, map_statements, metadata,
    mod, mul, replace_tile, sub, v,
)
from tools.tensor_core_corpus import Case  # noqa: E402
from emmy.compiler.graph import Graph  # noqa: E402
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var  # noqa: E402
from emmy.compiler.ir.kernel.ir import (  # noqa: E402
    Axis, CpAsyncWait, Loop, StridedLoop,
)
import tensor_core  # noqa: E402

OUT = ROOT / 'experiments' / 'tensor-core-audit-agent2' / 'probes'
MMA = 'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32'
ROWS = []


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def effective_graph(graph, node):
    """Return a graph whose serialized version contains the replacement node."""
    if graph.nodes.get(node.id) is node:
        return graph
    saved = Graph.from_dict(graph.to_dict())
    saved.nodes[node.id] = node
    return saved


def execute(ident, expected, moral, graph, meta, node=None, compile_it=False,
            why=None):
    if node is None:
        node = next(n for n in graph.nodes.values()
                    if type(n.op).__name__ == 'KernelOp')
    decision = tensor_core.classify(graph, node, meta['kernels'][0])
    row = {
        'id': ident,
        'expected': expected,
        'actual': decision.verdict,
        'reason': decision.reason,
        'moral_semantics': moral,
        'agrees': decision.verdict == expected,
    }
    if decision.plan is not None:
        row['plan'] = {
            'm': decision.plan.m, 'n': decision.plan.n, 'k': decision.plan.k,
            'block': [decision.plan.block_m, decision.plan.block_n],
            'grid': [decision.plan.grid_m, decision.plan.grid_n],
            'staging': decision.plan.staging,
        }
    if compile_it:
        folder = OUT / ident
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        saved = effective_graph(graph, node)
        (folder / 'case.ir.json').write_text(
            json.dumps(saved.to_dict(), sort_keys=True, default=repr) + '\n')
        (folder / 'meta.json').write_text(
            json.dumps(meta, sort_keys=True, default=repr) + '\n')
        result = subprocess.run(
            [sys.executable, str(ROOT / 'compile.py'), str(folder / 'case.ir.json')],
            capture_output=True, text=True, timeout=180)
        ptx = result.stdout
        (folder / 'out.ptx').write_text(ptx)
        (folder / 'stderr.txt').write_text(result.stderr)
        row.update({
            'compile_rc': result.returncode,
            'ptx_sha256': hashlib.sha256(ptx.encode()).hexdigest(),
            'mma_count': ptx.count(MMA),
            'ptx_has_cp_async': 'cp.async' in ptx,
            'stdout_is_ptx_only': ptx.startswith('.version') and 'compile.py:' not in ptx,
            'stderr': result.stderr.strip()[:500],
        })
        if why:
            row['why'] = why
    ROWS.append(row)
    print(json.dumps(row, sort_keys=True), flush=True)
    return decision


def replace_node(graph, node, tile):
    return replace_tile(node, tile)


def m01_case():
    folder = ROOT / 'emmy-hackathon-2026' / 'problems' / 'm01_matmul_512'
    meta = json.loads((folder / 'meta.json').read_text())
    graph = Graph.from_dict(json.loads(next(folder.glob('*.ir.json')).read_text()))
    node = graph.nodes[meta['kernels'][0]['node']]
    return graph, meta, node, node.op.body[0]


def mutate_statements(tile, fn):
    return dataclasses.replace(
        tile, body=tuple(map_statements(s, fn) for s in tile.body))


# 1. A valid direct contraction with deliberately unusual names and ABI order.
g1, t1 = base_body(16, 8, 16, a_source='leftish', b_source='rightish',
                   output='outland', axes=('gx', 'gy', 'unit_warp', 'lanes'))
m1 = metadata('P01-unusual-direct', 16, 8, 16, a='leftish', b='rightish',
              output='outland', order=(2, 1, 0))
execute('P01-unusual-names-direct', 'Plan',
        'Direct 16x8x16 contraction with arbitrary SSA/buffer names and ABI order.',
        g1, m1, kernel_node(g1, 'outland'), compile_it=True)

# 2. The same semantic contraction through shared staging with B transposed.
g2, m2, *_ = Case('P02-shared-b-transpose', 'Plan', 32, 16, 16,
                  staged=True, b_transpose=True).build()
execute('P02-shared-b-transpose', 'Plan',
        'Legal B-transposed operand staged through a proved shared tile.',
        g2, m2, g2.nodes['result'], compile_it=True)

# 3. A differently factored 2x1 atom grid (2 warps, 32x8 tile).
g3, t3 = base_body(32, 8, 8, tile=(32, 8, 2), atom_grid=(2, 1))
m3 = metadata('P03-2x1-atoms', 32, 8, 8, block=64)
execute('P03-2x1-atom-grid', 'Plan',
        'A 32x8 tile is exactly two 16x8 atoms and two launched warps.',
        g3, m3, kernel_node(g3), compile_it=True)

# 4. A differently factored 1x2 atom grid (2 warps, 16x16 tile).
g4, t4 = base_body(16, 16, 8, tile=(16, 16, 2), atom_grid=(1, 2))
m4 = metadata('P04-1x2-atoms', 16, 16, 8, block=64)
execute('P04-1x2-atom-grid', 'Plan',
        'A 16x16 tile is exactly two 16x8 atoms and two launched warps.',
        g4, m4, kernel_node(g4), compile_it=True)

# 5. CTA-dependent direct operand drift: moral result is not the written cells.
g5, t5 = base_body(32, 8, 8)
def cta_row(stmt):
    if type(stmt).__name__ == 'Load' and stmt.input == 'A':
        idx = list(stmt.index)
        idx[0] = add(idx[0], v('cta_m'))
        return dataclasses.replace(stmt, index=tuple(idx))
    return stmt
n5 = replace_node(g5, kernel_node(g5), mutate_statements(t5, cta_row))
m5 = metadata('P05-cta-drift', 32, 8, 8, grid=2)
execute('P05-cta-dependent-direct-read', 'NoMatch',
        'CTA 1 reads A rows shifted by one but writes the canonical C rows.',
        g5, m5, n5, compile_it=True)

# 6. A semantically inert unused load.
g6, t6 = base_body(16, 8, 8)
def unused(stmt):
    return stmt
body = list(t6.body)
from emmy.compiler.ir.kernel.ir import Load  # local import keeps construction explicit
t6b = dataclasses.replace(t6, body=tuple(body[:4] + [Load('unused_alpha', 'A', (lit(0), lit(0)))] + body[4:]))
execute('P06-inert-unused-load', 'Plan',
        'An unused load has no effect on the contraction and is legal to ignore.',
        g6, metadata('P06', 16, 8, 8), replace_node(g6, kernel_node(g6), t6b),
        compile_it=True)

# 7. Alias output to an input with a square tile; output overwrite semantics differ.
g7, t7 = base_body(16, 16, 16, tile=(16, 16, 2), atom_grid=(1, 2))
t7 = dataclasses.replace(
    t7, body=tuple(dataclasses.replace(s, output='A')
                   if type(s).__name__ == 'Write' else s for s in t7.body))
execute('P07-output-aliases-A', 'NoMatch',
        'Writing C through A changes operand values while the reduction is running.',
        g7, metadata('P07', 16, 16, 16, block=64), replace_node(g7, kernel_node(g7), t7),
        compile_it=True)

# 8. Valid nested-K spelling: an outer literal-spaced loop and inner unit loop.
g8, t8 = base_body(16, 8, 16)
flat = next(s for s in t8.body if type(s).__name__ == 'Loop')
def rename_k(stmt):
    if type(stmt).__name__ == 'Load':
        return dataclasses.replace(
            stmt, index=tuple(BinaryExpr('+', Var('k1'), x)
                              if isinstance(x, Var) and x.name == 'k' else x
                              for x in stmt.index))
    if type(stmt).__name__ == 'Accum':
        return dataclasses.replace(stmt, axes=('k0',))
    return stmt
inner = Loop(Axis('k0', 8),
             body=tuple(map_statements(s, rename_k) for s in flat.body),
             unroll=True, seed=False)
outer = StridedLoop(Axis('k1', 16), start=lit(0), step=lit(8), body=(inner,),
                    seed=False)
t8 = dataclasses.replace(t8, body=tuple(outer if s is flat else s for s in t8.body))
execute('P08-nested-k-spelling', 'Plan',
        'The nested loops enumerate exactly (k0,k1)=(0..7,0,8), i.e. K=0..15.',
        g8, metadata('P08', 16, 8, 16), replace_node(g8, kernel_node(g8), t8),
        compile_it=True)

# 9. A non-affine but semantically coherent alternate K phase, outside any
#    staging term. Repeating K=0..7 is still a valid symmetric contraction in
#    this asymmetric sample only if the expected reference includes it; here it
#    does not, so it must not be accepted as plain A@B.
phase = sub(v('k'), mul(mod(div(v('k'), lit(8)), lit(2)), lit(8)))
g9, t9 = base_body(16, 8, 16, a_k=phase)
execute('P09-phase-remapped-k', 'NoMatch',
        'A-phase repeats K=0..7 for k=8..15; the accepted plan would read K=0..15.',
        g9, metadata('P09', 16, 8, 16), kernel_node(g9), compile_it=True)

# 10. m01's output aliases an input: never a Plan.
g10, m10, n10, t10 = m01_case()
t10b = dataclasses.replace(
    t10, body=tuple(dataclasses.replace(s, output='x0')
                    if type(s).__name__ == 'Write' else s for s in t10.body))
execute('P10-m01-output-alias', 'NoMatch',
        'The m01 output pointer aliases x0, so stores corrupt the A operand.',
        g10, m10, replace_node(g10, n10, t10b), compile_it=False)

# 11. m01 pipeline wait_group(3) in the loop: the producer's group depth is 3,
#     so no useful group is guaranteed complete when compute begins.
g11, m11, n11, t11 = m01_case()
body = []
for s in t11.body:
    if type(s).__name__ == 'StridedLoop':
        s = dataclasses.replace(
            s, body=tuple(dataclasses.replace(q, group=3)
                          if type(q).__name__ == 'CpAsyncWait' else q
                          for q in s.body))
    body.append(s)
execute('P11-m01-wait-group-3', 'UnsupportedMatch',
        'wait_group(3) leaves all three committed groups in flight; compute may race copies.',
        g11, m11, replace_node(g11, n11, dataclasses.replace(t11, body=tuple(body))),
        compile_it=True)

# 12. m01 same-value duplicate fill: legal but redundant. It must remain Plan.
g12, m12, n12, t12 = m01_case()
body = []
for s in t12.body:
    if type(s).__name__ == 'StridedLoop':
        sub = []
        for q in s.body:
            sub.append(q)
            if type(q).__name__ == 'CpAsyncCopy' and q.smem == '_a_smem':
                sub.append(q)
        s = dataclasses.replace(s, body=tuple(sub))
    body.append(s)
execute('P12-m01-duplicate-fill', 'Plan',
        'Duplicating a copy of the same source into the same slot is semantically inert.',
        g12, m12, replace_node(g12, n12, dataclasses.replace(t12, body=tuple(body))),
        compile_it=True)

# 13. m01 B-copy source shifted +1. This is an actual wrong operand.
g13, m13, n13, t13 = m01_case()
def shift_b(stmt):
    if type(stmt).__name__ == 'CpAsyncCopy' and stmt.smem == '_b_smem':
        idx = list(stmt.src_index)
        idx[1] = add(idx[1], lit(1))
        return dataclasses.replace(stmt, src_index=tuple(idx))
    return stmt
execute('P13-m01-shifted-b-copy', 'UnsupportedMatch',
        'Every B copy is shifted one column, so the MMA consumes the wrong B cells.',
        g13, m13, replace_node(g13, n13, mutate_statements(t13, shift_b)),
        compile_it=True)

# 14. An interior-CTA-only B-copy source shift. Endpoint tracing cannot see it.
g14, m14, n14, t14 = m01_case()
def interior_b(stmt):
    if type(stmt).__name__ == 'CpAsyncCopy' and stmt.smem == '_b_smem':
        idx = list(stmt.src_index)
        # Keep the mutation in bounds: absolute column modulo 64 is the local
        # copy column, and shifting only columns 0..62 never leaves the row.
        where = BinaryExpr('*', BinaryExpr('==', Var('bid'), lit(1)),
                           BinaryExpr('<', mod(idx[1], lit(64)), lit(63)))
        idx[1] = TernaryExpr(where, add(idx[1], lit(1)), idx[1])
        return dataclasses.replace(stmt, src_index=tuple(idx))
    return stmt
execute('P14-m01-interior-cta-only-copy', 'UnsupportedMatch',
        'Only CTA 1 shifts B; both B ports and all CTAs must be proved, not just endpoints.',
        g14, m14, replace_node(g14, n14, mutate_statements(t14, interior_b)),
        compile_it=True)

# 15. A partial named barrier (32 participants) cannot order a 512-thread tile.
g15, m15, n15, t15 = m01_case()
body = []
for s in t15.body:
    if type(s).__name__ == 'StridedLoop':
        s = dataclasses.replace(
            s, body=tuple(dataclasses.replace(q, count=32)
                          if type(q).__name__ == 'Sync' else q for q in s.body))
    body.append(s)
execute('P15-m01-partial-barrier', 'UnsupportedMatch',
        'bar.sync with count=32 only collects one warp; it cannot order a 512-thread CTA.',
        g15, m15, replace_node(g15, n15, dataclasses.replace(t15, body=tuple(body))),
        compile_it=True)

# 16. Pipeline wait_group(1) on a 2-group stage window can read a reused slot
#     before the newest group has landed. Expect no Plan.
g16, m16, n16, t16 = m01_case()
body = []
for s in t16.body:
    if type(s).__name__ == 'StridedLoop':
        s = dataclasses.replace(
            s, body=tuple(dataclasses.replace(q, group=1)
                          if type(q).__name__ == 'CpAsyncWait' else q for q in s.body))
    body.append(s)
execute('P16-m01-wait-group-1-safe', 'Plan',
        'A two-group wait window still supplies each three-stage slot before its compute read.',
        g16, m16, replace_node(g16, n16, dataclasses.replace(t16, body=tuple(body))),
        compile_it=True)

# 17. A split-K contraction without atomics is a non-atomic write race.
for ident, kw, expected, moral in (
        ('P17-split-no-atomics', dict(split=2), 'UnsupportedMatch',
         'Two CTAs write the same non-atomic output cells.'),
        ('P18-split-with-atomic', dict(split=2, atomic=True), 'UnsupportedMatch',
         'Atomic split-K is outside the one-atom emitter even with reset metadata.')):
    g, meta, *_ = Case(ident, expected, 16, 8, 16, **kw).build()
    execute(ident, expected, moral, g, meta, g.nodes['result'], compile_it=True)

# 19. Direct A-transpose is valid but must still prove its global coordinate map.
g19, meta19, *_ = Case('P19-direct-a-transpose', 'Plan', 16, 16, 16,
                       a_transpose=True).build()
execute('P19-direct-a-transpose', 'Plan',
        'A is a transposed (K,M) view; the contraction is still A^T @ B.',
        g19, meta19, g19.nodes['result'], compile_it=True)

# 20. A valid shared-staged A-transpose. The shared tile is row-major (M,K)
#     even though the global A view is (K,M); the checker must orient the copy
#     and the compute load independently.
g20, meta20, *_ = Case('P20-staged-a-transpose', 'Plan', 16, 16, 16,
                       staged=True, a_transpose=True).build()
execute('P20-staged-a-transpose', 'Plan',
        'A is a transposed (K,M) global view staged into a row-major (M,K) shared tile.',
        g20, meta20, g20.nodes['result'], compile_it=True)

summary = {
    'probe_count': len(ROWS),
    'expected_counts': {k: sum(r['expected'] == k for r in ROWS)
                        for k in ('Plan', 'UnsupportedMatch', 'NoMatch')},
    'actual_counts': {k: sum(r['actual'] == k for r in ROWS)
                      for k in ('Plan', 'UnsupportedMatch', 'NoMatch')},
    'disagreements': [r['id'] for r in ROWS if not r['agrees']],
}
(OUT / 'results.json').write_text(json.dumps({'summary': summary, 'rows': ROWS},
                                              indent=2) + '\n')
print(json.dumps(summary, indent=2))
