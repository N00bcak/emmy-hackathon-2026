"""Correctness tests for the fail-closed one-atom TF32 tensor-core path."""
from __future__ import annotations

import hashlib
import dataclasses
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import BinaryExpr, TernaryExpr, Var
import tensor_core
from tf32_emitter import MMA, lane_map, simulate, simulate_matmul, tf32_rna
from tools.tensor_core_corpus import Case, add, div, lit, mod, mul, sub, v


PROBLEMS = HERE / 'emmy-hackathon-2026' / 'problems'
CORPUS = HERE / 'experiments' / 'tensor-core-v1' / 'corpus-v2'
BASELINE = HERE / 'experiments' / 'tensor-core-v1' / 'baseline' / 'baseline.json'
BASELINE_V2 = HERE / 'experiments' / 'tensor-core-v1' / 'baseline' / 'baseline-v2.json'
BASELINE_COMPILER = HERE / 'experiments' / 'tensor-core-v1' / 'baseline' / 'compile.py'


def load_case(folder):
    meta = json.loads((folder / 'meta.json').read_text())
    kernel = meta['kernels'][0]
    ir = next(folder.glob('*.ir.json'), None) or folder / 'case.ir.json'
    return ir, meta, kernel, Graph.from_dict(json.loads(ir.read_text()))


def compile_ptx(ir):
    result = subprocess.run([sys.executable, str(HERE / 'compile.py'), str(ir)],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise AssertionError(result.stderr)
    return result.stdout


def tf32_ptx(graph, kernel, decision):
    """The TF32 emitter's PTX for a Plan; compile.py may prefer SIMT."""
    from tf32_emitter import production_ptx
    return production_ptx(decision.plan, kernel, node=graph.nodes[kernel['node']], graph=graph)


def compile_nodes(graph, meta):
    """Compile one already-mutated graph through the production entry point."""
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        ir = folder / 'case.ir.json'
        ir.write_text(json.dumps(graph.to_dict(), sort_keys=True,
                                 separators=(',', ':'), default=repr) + '\n')
        (folder / 'meta.json').write_text(json.dumps(meta))
        result = subprocess.run([sys.executable, str(HERE / 'compile.py'), str(ir)],
                                capture_output=True, text=True, timeout=120)
    return result


def map_statements(statement, fn):
    changes = {}
    for field in dataclasses.fields(statement):
        if field.name in ('body', 'else_body', 'combine_states'):
            nested = getattr(statement, field.name)
            if nested:
                changes[field.name] = tuple(map_statements(x, fn) for x in nested)
    if changes:
        statement = dataclasses.replace(statement, **changes)
    return fn(statement)


def replace_tile(node, tile):
    return dataclasses.replace(
        node, op=dataclasses.replace(node.op, body=(tile,)))


def replace_statements(node, fn):
    tile = node.op.body[0]
    return replace_tile(
        node, dataclasses.replace(
            tile, body=tuple(map_statements(x, fn) for x in tile.body)))


class TensorCoreTests(unittest.TestCase):
    def test_frozen_baseline_source_identity(self):
        baseline = json.loads(BASELINE.read_text())
        corpus_baseline = json.loads(BASELINE_V2.read_text())
        if BASELINE_COMPILER.exists():
            compiler_sha = hashlib.sha256(BASELINE_COMPILER.read_bytes()).hexdigest()
            self.assertEqual(compiler_sha, baseline['compiler_sha256'])
            self.assertEqual(compiler_sha, corpus_baseline['compiler_sha256'])
        # The baseline PTX for all non-Plan cases must remain reproducible from
        # the unchanged source identity recorded in the evidence bundle.
        self.assertEqual(len(baseline['rows']), 36)
        self.assertEqual(len(corpus_baseline['rows']), 23)

    def test_frozen_corpus_files_match_manifest_hashes(self):
        manifest = json.loads((CORPUS / 'manifest.json').read_text())
        for row in manifest['cases']:
            folder = CORPUS / row['id']
            for name, expected in row['files'].items():
                self.assertEqual(
                    hashlib.sha256((folder / name).read_bytes()).hexdigest(),
                    expected, f'{row["id"]}/{name}')

    def test_m01_is_plan_and_m02_is_unsupported_launch_mismatch(self):
        m01 = PROBLEMS / 'm01_matmul_512'
        ir, meta, kernel, graph = load_case(m01)
        decision = tensor_core.classify(graph, graph.nodes[kernel['node']], kernel)
        self.assertEqual(decision.verdict, tensor_core.PLAN)
        self.assertEqual((decision.plan.m, decision.plan.n, decision.plan.k), (512, 512, 512))
        ptx = tf32_ptx(graph, kernel, decision)
        self.assertEqual(ptx.count(MMA), 3)
        self.assertIn('cvt.rna.tf32.f32', ptx)

        m02 = PROBLEMS / 'm02_matmul_rect'
        _, _, kernel, graph = load_case(m02)
        decision = tensor_core.classify(graph, graph.nodes[kernel['node']], kernel)
        self.assertEqual(decision.verdict, tensor_core.UNSUPPORTED)
        self.assertIn('4 atoms for 8 warps', decision.reason)

    def test_frozen_corpus_classifications_are_fail_closed(self):
        manifest = json.loads((CORPUS / 'manifest.json').read_text())
        self.assertEqual(manifest['version'], 'tensor-core-corpus-v2')
        self.assertEqual((manifest['rtol'], manifest['atol']), (1e-3, 1e-3))
        expected = {row['id']: row['expected'] for row in manifest['cases']}
        self.assertGreaterEqual(sum(v == tensor_core.PLAN for v in expected.values()), 12)
        self.assertGreaterEqual(sum(v != tensor_core.PLAN for v in expected.values()), 6)
        actual = {}
        for folder in sorted(CORPUS.iterdir()):
            if not folder.is_dir():
                continue
            ir, meta, kernel, graph = load_case(folder)
            decision = tensor_core.classify(graph, graph.nodes[kernel['node']], kernel)
            actual[folder.name] = decision.verdict
            self.assertEqual(decision.verdict, expected[folder.name],
                             f'{folder.name}: {decision.reason}')
        self.assertEqual(actual['n05-partial-collective'], tensor_core.UNSUPPORTED)
        self.assertEqual(actual['n06-split-without-atomics'], tensor_core.UNSUPPORTED)
        self.assertEqual(actual['n07-split-atomic'], tensor_core.UNSUPPORTED)

    def test_every_non_plan_keeps_frozen_simt_bytes(self):
        baseline = json.loads(BASELINE.read_text())
        corpus_baseline = json.loads(BASELINE_V2.read_text())
        rows = {row['case']: row for row in baseline['rows']}
        corpus_rows = {row['case']: row for row in corpus_baseline['rows']}
        published = [PROBLEMS / name for name in (
            'a01_attention', 'a02_attention_128', 'f01_rmsnorm', 'f02_row_softmax',
            'f03_l2norm', 'm01_matmul_512', 'm02_matmul_rect', 'p00_scalar_add',
            'p01_vector_add', 'p02_scale_shift', 'p03_relu_mul', 'r01_array_sum',
            'r02_big_sum', 'r03_mean', 'r04_row_sum')]
        checked = 0
        for folder in [*published, *sorted(CORPUS.iterdir())]:
            if not folder.is_dir():
                continue
            ir, meta, kernel, graph = load_case(folder)
            decision = tensor_core.classify(graph, graph.nodes[kernel['node']], kernel)
            if decision.verdict == tensor_core.PLAN:
                continue
            ptx = compile_ptx(ir)
            expected = (corpus_rows if folder.parent == CORPUS else rows)[folder.name]
            self.assertEqual(hashlib.sha256(ptx.encode()).hexdigest(),
                             expected['ptx_sha256'], folder.name)
            checked += 1
        self.assertEqual(checked, 23)

    def test_shared_prologue_decodes_block_axes_without_thread_stride(self):
        folder = CORPUS / 'p04-staged-32x16'
        ir, meta, kernel, graph = load_case(folder)
        decision = tensor_core.classify(graph, graph.nodes[kernel['node']], kernel)
        self.assertEqual(decision.verdict, tensor_core.PLAN)
        self.assertEqual(decision.plan.staging, 'shared')
        ptx = tf32_ptx(graph, kernel, decision)
        # CTA strides cover only block axes.  With a 2x2 CTA grid, block 3
        # must decode as (1, 1); a stride through the thread suffix would
        # decode every block as tile (0, 0).
        self.assertNotIn('div.u32 %r19, %r7, 64;', ptx)
        self.assertIn('div.u32', ptx)
        self.assertIn('bar.sync 0;', ptx)

    def test_tf32_lane_map_is_an_exact_partition(self):
        cells = []
        for lane in range(32):
            mapped = lane_map(lane)
            self.assertEqual(len(mapped['a']), 4)
            self.assertEqual(len(mapped['b']), 2)
            self.assertEqual(len(mapped['c']), 4)
            cells.extend(mapped['c'])
        self.assertEqual(len(cells), 128)
        self.assertEqual(set(cells), {(r, c) for r in range(16) for c in range(8)})

    def test_three_term_model_has_separate_hh_hl_lh_fragments(self):
        rng = np.random.default_rng(71)
        a = rng.uniform(-2, 2, (16, 8)).astype(np.float32)
        b = rng.uniform(-2, 2, (8, 8)).astype(np.float32)
        result, ah, al, bh, bl = simulate(a, b)
        self.assertFalse(np.array_equal(a, ah))
        self.assertTrue(np.array_equal(al, np.vectorize(tf32_rna, otypes=[np.float32])(a - ah)))
        self.assertTrue(np.array_equal(bl, np.vectorize(tf32_rna, otypes=[np.float32])(b - bh)))
        hh = ah.astype(np.float64) @ bh.astype(np.float64)
        hl = ah.astype(np.float64) @ bl.astype(np.float64)
        lh = al.astype(np.float64) @ bh.astype(np.float64)
        expected = np.float32(np.float32(np.float32(hh) + np.float32(hl)) + np.float32(lh))
        self.assertTrue(np.array_equal(result, expected))
        # One-pass TF32 must be measurably worse than the three-term model.
        one_pass = np.float32(hh)
        reference = np.asarray(a.astype(np.float64) @ b.astype(np.float64), dtype=np.float32)
        three_error = np.abs(result.astype(np.float64) - reference.astype(np.float64)).max()
        one_error = np.abs(one_pass.astype(np.float64) - reference.astype(np.float64)).max()
        self.assertLess(three_error, one_error / 100)

    def test_cpu_model_matches_reference_within_tolerance_across_corpus(self):
        for folder in sorted(CORPUS.iterdir()):
            if not folder.is_dir():
                continue
            ir, meta, kernel, graph = load_case(folder)
            decision = tensor_core.classify(graph, graph.nodes[kernel['node']], kernel)
            if decision.verdict != tensor_core.PLAN:
                continue
            with np.load(folder / 'inputs.npz') as inputs:
                a = inputs[decision.plan.a_buffer].astype(np.float32)
                b = inputs[decision.plan.b_buffer].astype(np.float32)
            if decision.plan.a_transpose:
                a = a.T
            if decision.plan.b_transpose:
                b = b.T
            got = simulate_matmul(a, b)
            with np.load(folder / 'expected.npz') as expected:
                reference = expected['result']
            tolerance = meta['atol'] + meta['rtol'] * np.abs(reference)
            self.assertTrue(np.all(np.abs(got.astype(np.float64) - reference.astype(np.float64))
                                   <= tolerance.astype(np.float64)), folder.name)

    def test_plan_ptx_preserves_metadata_abi_and_has_no_compiler_delegation(self):
        folder = CORPUS / 'p12-staged-b-transpose'
        ir, meta, kernel, graph = load_case(folder)
        decision = tensor_core.classify(graph, graph.nodes[kernel['node']], kernel)
        self.assertEqual(decision.verdict, tensor_core.PLAN)
        ptx = tf32_ptx(graph, kernel, decision)
        self.assertEqual(ptx.count(MMA), 3)
        # Per K atom: six high-part and six low-part conversions.  The low
        # parts are formed by f32 subtraction of the stored high fragments.
        self.assertEqual(ptx.count('cvt.rna.tf32.f32'), 12)
        self.assertEqual(ptx.count('sub.f32'), 6)
        self.assertNotIn('nvcc', ptx.lower())
        self.assertNotIn('.externfunc', ptx)
        for banned in ('tma', 'ldmatrix', 'stmatrix', 'wgmma', 'cp.async.bulk'):
            self.assertNotIn(banned, ptx.lower())

    def test_adversarial_typed_mutations_fail_closed(self):
        # A direct global operand may not hide a modulo K map as a pipeline
        # stage term.  This is a complete contraction but not a plain A@B.
        graph, meta, _, _ = Case('typed-phase-k', 'Plan', 16, 8, 16).build()
        node = graph.nodes['result']
        phase = sub(v('k'), mul(mod(div(v('k'), lit(8)), lit(2)), lit(8)))
        def phase_k(statement):
            if type(statement).__name__ == 'Load' and statement.input == 'lhs':
                index = list(statement.index)
                index[1] = phase
                return dataclasses.replace(statement, index=tuple(index))
            return statement
        mutated = replace_statements(node, phase_k)
        decision = tensor_core.classify(graph, mutated, meta['kernels'][0])
        self.assertNotEqual(decision.verdict, tensor_core.PLAN)

        # A CTA-dependent direct read must not drift from its output row.
        graph, meta, _, _ = Case('typed-cta-drift', 'Plan', 32, 8, 8).build()
        node = graph.nodes['result']
        def cta_row(statement):
            if type(statement).__name__ == 'Load' and statement.input == 'lhs':
                index = list(statement.index)
                index[0] = add(index[0], v('cta_m'))
                return dataclasses.replace(statement, index=tuple(index))
            return statement
        mutated = replace_statements(node, cta_row)
        decision = tensor_core.classify(graph, mutated, meta['kernels'][0])
        self.assertNotEqual(decision.verdict, tensor_core.PLAN)

    def test_pipeline_replay_rejects_interior_cta_and_partial_barrier(self):
        ir, meta, kernel, graph = load_case(PROBLEMS / 'm01_matmul_512')
        node = graph.nodes[kernel['node']]
        tile = node.op.body[0]

        # Both endpoint CTAs read the correct cells, but CTA 1 alone shifts B
        # by one in-bounds column.  Endpoint-only sampling must not accept it.
        def interior_shift(statement):
            if type(statement).__name__ == 'CpAsyncCopy' and statement.smem == '_b_smem':
                index = list(statement.src_index)
                where = BinaryExpr(
                    '*', BinaryExpr('==', Var('bid'), lit(1)),
                    BinaryExpr('<', mod(index[1], lit(64)), lit(63)))
                index[1] = TernaryExpr(where, add(index[1], lit(1)), index[1])
                return dataclasses.replace(statement, src_index=tuple(index))
            return statement
        mutated = replace_statements(node, interior_shift)
        decision = tensor_core.classify(graph, mutated, kernel)
        self.assertEqual(decision.verdict, tensor_core.UNSUPPORTED)
        self.assertIn('non-affine CTA', decision.reason)
        saved = Graph.from_dict(graph.to_dict())
        saved.nodes[node.id] = mutated
        # The SIMT fallback itself cannot lower a predicate multiplication, so
        # compile fails closed.  The checker must never hand this IR to the
        # tensor emitter, regardless of whether the fallback succeeds.
        done = compile_nodes(saved, meta)
        self.assertEqual(done.stdout.count(MMA), 0)

        # A named barrier over 32 threads cannot order a 512-thread CTA.
        _, meta, kernel, graph = load_case(PROBLEMS / 'm01_matmul_512')
        node = graph.nodes[kernel['node']]
        tile = node.op.body[0]
        def partial_barrier(statement):
            if type(statement).__name__ == 'Sync':
                return dataclasses.replace(statement, count=32)
            return statement
        body = []
        for statement in tile.body:
            if type(statement).__name__ == 'StridedLoop':
                statement = dataclasses.replace(
                    statement, body=tuple(map_statements(x, partial_barrier)
                                           for x in statement.body))
            body.append(statement)
        mutated = replace_tile(node, dataclasses.replace(tile, body=tuple(body)))
        decision = tensor_core.classify(graph, mutated, kernel)
        self.assertEqual(decision.verdict, tensor_core.UNSUPPORTED)
        self.assertNotIn('Plan', decision.verdict)

    def test_valid_staged_transpose_and_alias_checks(self):
        graph, meta, _, _ = Case('staged-a-transpose', 'Plan', 16, 16, 16,
                                 staged=True, a_transpose=True).build()
        decision = tensor_core.classify(graph, graph.nodes['result'],
                                        meta['kernels'][0])
        self.assertEqual(decision.verdict, tensor_core.PLAN)
        self.assertEqual(decision.plan.staging, 'shared')
        self.assertTrue(decision.plan.a_transpose)

        graph, meta, _, _ = Case('alias-output', 'Plan', 16, 16, 16).build()
        node = graph.nodes['result']
        def alias(statement):
            if type(statement).__name__ == 'Write':
                return dataclasses.replace(statement, output='lhs')
            return statement
        mutated = replace_statements(node, alias)
        decision = tensor_core.classify(graph, mutated, meta['kernels'][0])
        self.assertNotEqual(decision.verdict, tensor_core.PLAN)
        self.assertIn('aliases', decision.reason)


if __name__ == '__main__':
    unittest.main()
