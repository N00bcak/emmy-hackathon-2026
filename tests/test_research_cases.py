"""Independent CPU validation of generated research cases and mathematical oracles.

The small interpreter below deliberately does not import compile.py or call an Emmy
renderer. It interprets scalar serial Kernel IR to catch disagreement between the
generated program and its independently calculated expected output.
"""
import importlib.util
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from emmy.compiler.graph import Graph


ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ROOT / "emmy-hackathon-2026" / "problems"


def load_cases_module():
    spec = importlib.util.spec_from_file_location("research_cases", ROOT / "tools/research_cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mathematical_reference(problem, inputs):
    """Direct equations, separate from the production reference implementation."""
    if problem.startswith("m"):
        return inputs["x0"].astype(np.float64) @ inputs["x1"].astype(np.float64)
    if problem.startswith("f"):
        x = inputs["x"].astype(np.float64)
        weights = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return weights / np.sum(weights, axis=-1, keepdims=True)
    scores = (inputs["q"].astype(np.float64) @ inputs["kt"].astype(np.float64)) / float(inputs["div_c1"][0])
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probabilities = weights / np.sum(weights, axis=-1, keepdims=True)
    return probabilities @ inputs["v"].astype(np.float64)


def _operation(name, args):
    if name in {"add", "plus"}:
        return args[0] + args[1]
    if name == "subtract":
        return args[0] - args[1]
    if name == "multiply":
        return args[0] * args[1]
    if name in {"divide", "true_divide"}:
        return args[0] / args[1]
    if name in {"maximum", "max"}:
        return max(args)
    if name in {"minimum", "min"}:
        return min(args)
    if name in {"negative", "negate"}:
        return -args[0]
    if name == "reciprocal":
        return 1.0 / args[0]
    if name == "sqrt":
        return math.sqrt(args[0])
    if name in {"rsqrt", "reciprocal_sqrt"}:
        return 1.0 / math.sqrt(args[0])
    if name == "exp":
        return math.exp(args[0])
    if name in {"identity", "copy"}:
        return args[0]
    raise AssertionError("unhandled scalar reference operation: " + name)


def _expr(expr, env):
    if isinstance(expr, (float, int)):
        return expr
    kind = type(expr).__name__
    if kind == "Var":
        return env[expr.name]
    if kind == "Literal":
        return expr.value
    if kind == "TernaryExpr":
        return _expr(expr.if_true if _expr(expr.cond, env) else expr.if_false, env)
    if kind == "BinaryExpr":
        left, right = _expr(expr.left, env), _expr(expr.right, env)
        ops = {
            "+": lambda: left + right, "-": lambda: left - right,
            "*": lambda: left * right, "/": lambda: int(left / right),
            "//": lambda: left // right, "%": lambda: left % right,
            "<": lambda: left < right, "<=": lambda: left <= right,
            ">": lambda: left > right, ">=": lambda: left >= right,
            "==": lambda: left == right, "!=": lambda: left != right,
            "&&": lambda: bool(left and right), "||": lambda: bool(left or right),
            "and": lambda: bool(left and right), "or": lambda: bool(left or right),
        }
        return ops[expr.op]()
    raise AssertionError("unhandled reference expression: " + kind)


def _carried(body):
    """Names initialized by this loop; nested loops initialize their own names."""
    found = {}
    for statement in body:
        kind = type(statement).__name__
        if kind == "Accum":
            found[statement.name] = statement.op
        elif kind == "Cond":
            found.update(_carried(statement.body))
            found.update(_carried(statement.else_body))
    return found


def _execute(body, env, buffers):
    for statement in body:
        kind = type(statement).__name__
        if kind == "Tile":
            axes = statement.axes
            for coordinates in itertools.product(*(range(a.extent.as_static()) for a in axes)):
                local = dict(env)
                local.update(zip((a.name for a in axes), coordinates))
                _execute(statement.body, local, buffers)
        elif kind in {"Loop", "StridedLoop"}:
            if statement.seed:
                for name, op in _carried(statement.body).items():
                    if op.identity is None:
                        raise AssertionError("accumulator lacks an identity")
                    env[name] = op.identity
            start = 0 if kind == "Loop" else int(_expr(statement.start, env))
            step = 1 if kind == "Loop" else int(_expr(statement.step, env))
            end = statement.axis.extent.as_static()
            if kind == "StridedLoop" and statement.end is not None:
                end = int(_expr(statement.end, env))
            for value in range(start, end, step):
                env[statement.axis.name] = value
                _execute(statement.body, env, buffers)
        elif kind == "Load":
            index = tuple(int(_expr(i, env)) for i in statement.index)
            for offset, name in enumerate(statement.names):
                location = index[:-1] + (index[-1] + offset,)
                env[name] = float(buffers[statement.input][location])
        elif kind == "Write":
            index = tuple(int(_expr(i, env)) for i in statement.index)
            for offset, name in enumerate(statement.values):
                location = index[:-1] + (index[-1] + offset,)
                if statement.atomic:
                    buffers[statement.output][location] += env[name]
                else:
                    buffers[statement.output][location] = env[name]
        elif kind == "Assign":
            env[statement.name] = _operation(statement.op.name, [env[name] for name in statement.args])
        elif kind == "Accum":
            base = statement.base or statement.name
            env[statement.name] = _operation(statement.op.name, [env[base], env[statement.value]])
        elif kind == "Init":
            env[statement.name] = _expr(statement.identity, env)
        elif kind == "Let":
            env[statement.name] = _expr(statement.value, env)
        elif kind == "Cond":
            _execute(statement.body if _expr(statement.cond, env) else statement.else_body, env, buffers)
        else:
            raise AssertionError("unhandled scalar reference statement: " + kind)


def interpret_case(directory):
    meta = json.loads((directory / "meta.json").read_text())
    with np.load(directory / "inputs.npz") as archive:
        buffers = {name: archive[name].astype(np.float64) for name in archive.files}
    kernels = {}
    for path in directory.glob("*.ir.json"):
        graph = Graph.from_dict(json.loads(path.read_text()))
        kernels.update({node.op.name: node.op for node in graph.nodes.values() if type(node.op).__name__ == "KernelOp"})
    for launch in meta["kernels"]:
        for arg in launch["args"]:
            if arg["name"] not in buffers:
                buffers[arg["name"]] = np.zeros(arg["shape"], dtype=np.float64)
        for name in launch.get("zero_outputs", []):
            buffers[name].fill(0)
        _execute(kernels[launch["entry"]].body, {}, buffers)
    return {name: buffers[name] for name in meta["graph_outputs"]}


def interpret_online_softmax(directory):
    """Lockstep CPU interpretation of the published coupled-state schedule.

    This interprets each combine program from IR, not a second implementation of
    the online recurrence. Expected results remain direct stable NumPy softmax.
    """
    record = json.loads((directory / "case.json").read_text())
    graph = Graph.from_dict(json.loads((directory / record["ir_paths"][0]).read_text()))
    kernel = next(node.op for node in graph.nodes.values() if type(node.op).__name__ == "KernelOp")
    tile = kernel.body[0]
    xname, outname = record["spec"]["names"]["x"], record["spec"]["names"]["out"]
    with np.load(directory / "inputs.npz") as archive:
        x = archive[xname].astype(np.float64)
    output = np.zeros(x.shape, dtype=np.float64)
    for row in range(x.shape[0]):
        buffers = {xname: x, outname: output}
        environments = [{"a0": row, "a1_co": tid, "lane": tid % 32, "warp": tid // 32} for tid in range(128)]
        for statement in tile.body:
            kind = type(statement).__name__
            if kind == "WarpShuffle":
                offset = statement.length // 2
                while offset:
                    old = [dict(e) for e in environments]
                    for tid, env in enumerate(environments):
                        other = (tid // 32) * 32 + ((tid % 32) ^ offset)
                        for name, partner in zip(statement.state, statement.state_b, strict=True):
                            env[partner] = old[other][name]
                        _execute(statement.combine_states, env, buffers)
                    offset //= 2
            elif kind == "Smem":
                buffers[statement.name] = np.zeros(statement.extents, dtype=np.float64)
            elif kind == "Sync":
                pass  # Every preceding statement completed for all 128 lanes.
            elif kind == "TreeHalve":
                stride = statement.length // 2
                while stride:
                    old = {name: buffers[name].copy() for name in statement.bufs}
                    for warp in range(stride):
                        env = dict(environments[warp * 32])
                        for buffer, name, partner in zip(statement.bufs, statement.state, statement.state_b, strict=True):
                            env[name], env[partner] = old[buffer][warp], old[buffer][warp + stride]
                        _execute(statement.combine_states, env, buffers)
                        for buffer, name in zip(statement.bufs, statement.state, strict=True):
                            buffers[buffer][warp] = env[name]
                    stride //= 2
                for env in environments:
                    for buffer, name in zip(statement.bufs, statement.state, strict=True):
                        env[name] = float(buffers[buffer][0])
            else:
                for env in environments:
                    _execute((statement,), env, buffers)
    return output


class ResearchCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases_module()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.destination = Path(self.temporary.name)

    def generate(self, family, seed=13, **kwargs):
        return Path(self.cases.generate_case(self.destination / f"{family}-{seed}", family, seed, **kwargs))

    def test_original_five_gap_references_match_direct_equations(self):
        policies = {
            "m01_matmul_512": (0.001, 0.001), "m02_matmul_rect": (0.001, 0.001),
            "f02_row_softmax": (0.001, 0.00001),
            "a01_attention": (0.001, 0.0001), "a02_attention_128": (0.001, 0.0001),
        }
        for problem, tolerance in policies.items():
            with self.subTest(problem=problem):
                directory = PROBLEMS / problem
                meta = json.loads((directory / "meta.json").read_text())
                self.assertEqual((meta["rtol"], meta["atol"]), tolerance)
                inputs = {}
                for arg in meta["kernels"][0]["args"]:
                    if "seed" in arg:
                        inputs[arg["name"]] = np.random.default_rng(arg["seed"]).standard_normal(arg["shape"]).astype(np.float32)
                    elif "data" in arg:
                        inputs[arg["name"]] = np.asarray(arg["data"], dtype=np.float32).reshape(arg["shape"])
                expected = np.load(directory / "expected.npz")[meta["graph_outputs"][0]]
                direct = mathematical_reference(problem, inputs)
                np.testing.assert_allclose(expected, direct, rtol=tolerance[0], atol=tolerance[1])

    def test_all_families_typed_roundtrip_and_case_validation(self):
        for family in ["pointwise", "reduction", "matmul", "atomic_matmul", "softmax", "attention", "rmsnorm", "l2norm"]:
            with self.subTest(family=family):
                directory = self.generate(family)
                self.assertTrue(self.cases.validate_case(directory)["valid"])
                meta = json.loads((directory / "meta.json").read_text())
                launches = {launch["entry"]: launch for launch in meta["kernels"]}
                for path in directory.glob("*.ir.json"):
                    graph = Graph.from_dict(json.loads(path.read_text()))
                    second = Graph.from_dict(graph.to_dict())
                    self.assertEqual(graph.to_dict(), second.to_dict())
                    for node in graph.nodes.values():
                        if type(node.op).__name__ != "KernelOp":
                            continue
                        launch = launches[node.op.name]
                        self.assertEqual(launch["arg_order"], [arg["name"] for arg in launch["args"]])
                        self.assertEqual(len(launch["arg_order"]), len(set(launch["arg_order"])))
                        for tile in node.op.body:
                            if type(tile).__name__ != "Tile":
                                continue
                            cells = math.prod(axis.extent.as_static() for axis in tile.axes)
                            threads = math.prod(launch["block"])
                            self.assertEqual(launch["grid"], [(cells + threads - 1) // threads, 1, 1])
                            def contains_collective(body):
                                return any(type(s).__name__ in {"Sync", "WarpShuffle", "TreeHalve"} or any(contains_collective(b) for b in s.nested()) for s in body)
                            if contains_collective(tile.body):
                                self.assertEqual(cells % threads, 0, "collective cannot sit behind a partial CTA guard")

    def test_generated_serial_ir_matches_independent_interpretation(self):
        for family, shape in [("pointwise", (14,)), ("matmul", (3, 2, 5)), ("atomic_matmul", (2, 3, 7)), ("softmax", (3, 7)), ("attention", (2, 3, 4, 2)), ("rmsnorm", (2, 7)), ("l2norm", (2, 7))]:
            with self.subTest(family=family):
                directory = self.generate(family, shape=shape)
                meta = json.loads((directory / "meta.json").read_text())
                actual = interpret_case(directory)
                with np.load(directory / "expected.npz") as expected:
                    for name in expected.files:
                        np.testing.assert_allclose(actual[name], expected[name], rtol=meta["rtol"], atol=meta["atol"])

    def test_hand_calculated_generated_references_and_numerical_policies(self):
        for family in ["matmul", "atomic_matmul", "softmax", "attention"]:
            with self.subTest(family=family):
                shape = (2, 3, 5) if "matmul" in family else (2, 3, 4, 2) if family == "attention" else (2, 3)
                directory = self.generate(family, seed=19, shape=shape)
                record = json.loads((directory / "case.json").read_text())
                spec = record["spec"]
                names = spec["names"]
                with np.load(directory / "inputs.npz") as archive:
                    inputs = {name: archive[name].copy() for name in archive.files}
                if "matmul" in family:
                    inputs[names["x"]].fill(1)
                    inputs[names["y"]].fill(1)
                    expected = np.full((2, 3), 5)
                elif family == "softmax":
                    inputs[names["x"]].fill(0)
                    expected = np.full((2, 3), 1 / 3)
                else:
                    inputs[names["x"]].fill(0)
                    inputs[names["v"]][:] = [[1, 2], [3, 4], [5, 6]]
                    expected = np.array([[3, 4], [3, 4]])
                answer = self.cases.reference(spec, inputs)[names["out"]]
                np.testing.assert_allclose(answer, expected, rtol=1e-7, atol=1e-7)
                self.assertEqual(answer.dtype, np.dtype("float32"))
                meta = json.loads((directory / "meta.json").read_text())
                expected_policy = (0.001, 0.001) if "matmul" in family else (0.001, 0.00001) if family == "softmax" else (0.001, 0.0001)
                self.assertEqual((meta["rtol"], meta["atol"]), expected_policy)
                self.assertTrue(record["version"])
                self.assertTrue(record["tolerance_justification"])
                if family == "atomic_matmul":
                    self.assertTrue(record["requires_sanitizer"])
                    self.assertTrue(record["requires_isolated_first"])
                    self.assertTrue(meta["kernels"][0]["zero_outputs"])

    def test_pointwise_operation_width_tail_and_domain_semantics(self):
        for operation, width in itertools.product(["add", "subtract", "multiply", "relu_mul", "scale_shift"], [1, 2, 4]):
            with self.subTest(operation=operation, width=width):
                seed = ["add", "subtract", "multiply", "relu_mul", "scale_shift"].index(operation) * 10 + width
                directory = self.generate("pointwise", seed=seed, shape=(7,), operation=operation, vector_width=width, pattern="cancellation")
                actual = interpret_case(directory)
                with np.load(directory / "expected.npz") as expected:
                    for name in expected.files:
                        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-6, atol=1e-6)

    def test_softmax_domain_properties_and_shift_invariance(self):
        for seed, pattern in enumerate(["normal", "zeros", "impulse", "cancellation", "bounded"]):
            with self.subTest(pattern=pattern):
                directory = self.generate("softmax", seed=seed, pattern=pattern, shape=(3, 7))
                spec = json.loads((directory / "case.json").read_text())["spec"]
                names = spec["names"]
                with np.load(directory / "inputs.npz") as archive:
                    inputs = {name: archive[name].copy() for name in archive.files}
                original = self.cases.reference(spec, inputs)[names["out"]]
                self.assertTrue(np.isfinite(original).all())
                self.assertTrue((original > 0).all())
                np.testing.assert_allclose(original.sum(axis=1), np.ones(3), rtol=1e-6, atol=1e-6)
                inputs[names["x"]] += np.float32(8)
                shifted = self.cases.reference(spec, inputs)[names["out"]]
                np.testing.assert_allclose(original, shifted, rtol=1e-6, atol=1e-7)

    def test_cooperative_online_softmax_preserves_schedule_and_semantics(self):
        original_path = next((PROBLEMS / "f02_row_softmax").glob("*.ir.json"))
        original = Graph.from_dict(json.loads(original_path.read_text()))
        original_tile = next(node.op.body[0] for node in original.nodes.values() if type(node.op).__name__ == "KernelOp")
        for cols in [1, 7, 31, 33, 127, 129, 257]:
            with self.subTest(cols=cols):
                directory = self.generate("softmax", seed=100 + cols, shape=(3, cols), schedule="cooperative")
                self.assertTrue(self.cases.validate_case(directory)["valid"])
                record = json.loads((directory / "case.json").read_text())
                meta = json.loads((directory / "meta.json").read_text())
                self.assertEqual(record["spec"]["schedule"], "cooperative-online-softmax")
                self.assertEqual((meta["rtol"], meta["atol"]), (0.001, 0.00001))
                self.assertEqual(meta["kernels"][0]["block"], [128, 1, 1])
                self.assertEqual(meta["kernels"][0]["grid"], [3, 1, 1])
                self.assertTrue(all(arg["shape"] == [3, cols] for arg in meta["kernels"][0]["args"]))
                graph = Graph.from_dict(json.loads((directory / record["ir_paths"][0]).read_text()))
                self.assertEqual(graph.to_dict(), Graph.from_dict(graph.to_dict()).to_dict())
                tile = next(node.op.body[0] for node in graph.nodes.values() if type(node.op).__name__ == "KernelOp")
                self.assertEqual(tile.block_threads, 128)
                self.assertEqual(tuple(a.extent.as_static() for a in tile.axes), (3, 128))
                self.assertEqual(len(tile.body), len(original_tile.body))
                for cloned, source in zip(tile.body, original_tile.body, strict=True):
                    if type(source).__name__ == "StridedLoop":
                        self.assertEqual(cloned.axis.extent.as_static(), cols)
                        self.assertEqual(cloned.axis.name, source.axis.name)
                        self.assertEqual(cloned.start, source.start)
                        self.assertEqual(cloned.step, source.step)
                        self.assertEqual(cloned.body, source.body)
                        self.assertIsNone(cloned.end)
                    else:
                        self.assertEqual(cloned, source)
                max_ops = [op for op in _carried(tile.body[0].body).values() if op.name == "maximum"]
                self.assertTrue(max_ops)
                self.assertTrue(all(math.isfinite(op.identity) and op.identity < -1e10 for op in max_ops))
                actual = interpret_online_softmax(directory)
                with np.load(directory / "expected.npz") as archive:
                    expected = archive[meta["graph_outputs"][0]]
                np.testing.assert_allclose(actual, expected, rtol=meta["rtol"], atol=meta["atol"])
                np.testing.assert_allclose(actual.sum(axis=-1), np.ones(3), rtol=1e-6, atol=1e-6)

    def test_reproducible_semantics_inputs_and_references(self):
        first = self.generate("attention", seed=47)
        second = Path(self.cases.generate_case(self.destination / "replay", "attention", 47))
        for filename in ["meta.json", "case.json"]:
            self.assertEqual(json.loads((first / filename).read_text()), json.loads((second / filename).read_text()))
        for filename in ["inputs.npz", "expected.npz"]:
            with np.load(first / filename) as a, np.load(second / filename) as b:
                self.assertEqual(a.files, b.files)
                for name in a.files:
                    np.testing.assert_array_equal(a[name], b[name])

    def test_import_preserves_published_contract_and_expected(self):
        source = PROBLEMS / "m02_matmul_rect"
        destination = self.destination / "imported"
        directory = Path(self.cases.import_problem(source, destination))
        self.assertEqual(json.loads((source / "meta.json").read_text()), json.loads((directory / "meta.json").read_text()))
        with np.load(source / "expected.npz") as a, np.load(directory / "expected.npz") as b:
            for name in a.files:
                np.testing.assert_array_equal(a[name], b[name])
        self.assertTrue(self.cases.validate_case(directory)["valid"])

    def test_tampering_is_not_a_new_valid_case(self):
        directory = self.generate("matmul")
        meta_path = directory / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["atol"] *= 10
        meta_path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "changed"):
            self.cases.validate_case(directory)

    def test_argument_declarations_may_differ_from_positional_order(self):
        directory = self.generate('pointwise')
        meta_path = directory / 'meta.json'
        meta = json.loads(meta_path.read_text())
        meta['kernels'][0]['args'].reverse()
        meta_path.write_text(json.dumps(meta))
        # Create a new frozen record; changing an existing record remains invalid.
        record_path = directory / 'case.json'
        record = json.loads(record_path.read_text())
        record['files']['meta.json'] = self.cases.digest(meta_path.read_bytes())
        record['id'] = self.cases.digest(self.cases.canonical(record['files']))
        record_path.write_text(json.dumps(record))
        self.assertTrue(self.cases.validate_case(directory)['valid'])


if __name__ == "__main__":
    unittest.main()
