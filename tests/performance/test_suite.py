import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


suite = load("performance_suite", HERE / "generators/suite.py")
runner = load("performance_runner", HERE / "runner.py")


class ManifestTests(unittest.TestCase):
    def test_frozen_manifest_has_both_tiers_and_unique_ids(self):
        data = suite.load_manifest()
        self.assertEqual(set(data["tiers"]), {"perf-a", "perf-b"})
        cases = suite.all_cases()
        self.assertEqual(len(cases), 22)
        self.assertEqual(len({case["id"] for _, case in cases}), 22)
        for _, case in cases:
            self.assertLessEqual(max(case["shape"]), 2**26)
            self.assertEqual(len(case["tolerance"]), 2)
            self.assertGreaterEqual(case["tolerance"][0], 0)
            self.assertGreaterEqual(case["tolerance"][1], 0)

    def test_large_outputs_are_marked_local_only(self):
        for _, case in suite.all_cases():
            exceeds_return_limit = suite.output_elements(case) > 2**22
            self.assertEqual(case["local_only"], exceeds_return_limit, case["id"])

    def test_pointwise_scalar_alignment_preserves_published_vector_width(self):
        case = next(c for _, c in suite.all_cases("perf-a")
                    if c["id"] == "p00_scalar_add_4m")
        with tempfile.TemporaryDirectory() as temporary:
            directory = suite.generate_case(case, Path(temporary) / case["id"])
            meta = json.loads((directory / "meta.json").read_text())
        self.assertEqual(meta["kernels"][0]["grid"], [7813, 1, 1])
        self.assertEqual(meta["kernels"][0]["block"], [256, 1, 1])

    def test_pointwise_widths_match_the_published_ir(self):
        cases = {c["id"]: c for _, c in suite.all_cases("perf-a")}
        self.assertEqual(cases["p00_scalar_add_4m"]["vector_width"], 2)
        self.assertEqual(cases["p01_vector_add_4m"]["vector_width"], 2)
        self.assertEqual(cases["p02_scale_shift_8m"]["vector_width"], 4)
        self.assertEqual(cases["p03_relu_mul_4m"]["vector_width"], 4)

    def test_row_loops_follow_the_scaled_width(self):
        from emmy.compiler.graph import Graph
        from emmy.compiler.ir.kernel.ir import KernelOp, StridedLoop

        def loops(statements):
            for statement in statements:
                if isinstance(statement, StridedLoop):
                    yield statement
                yield from loops(getattr(statement, "body", None) or ())

        for _, case in suite.all_cases():
            if case["family"] not in ("row_sum", "rmsnorm", "l2norm", "softmax"):
                continue
            meta = suite.scaled_metadata(case)
            with tempfile.TemporaryDirectory() as temporary:
                directory = suite.generate_case(case, Path(temporary) / case["id"])
                generated = Graph.from_dict(json.loads(
                    (directory / (meta["kernels"][0]["entry"] + ".ir.json")).read_text()))
            node = generated.nodes[meta["kernels"][0]["node"]]
            self.assertIsInstance(node.op, KernelOp)
            extents = {loop.axis.name: loop.axis.extent.as_static()
                       for loop in loops(node.op.body)}
            self.assertTrue(extents, case["id"])
            for name, extent in extents.items():
                if name in ("a1", "a2", "a3"):
                    self.assertEqual(extent, case["shape"][1],
                                     f"{case['id']}:{name}")

    def test_metadata_constants_follow_scaled_shapes(self):
        cases = {c["id"]: c for _, c in suite.all_cases("perf-a")}
        rms = suite.scaled_metadata(cases["f01_rmsnorm_3072x2048"])["kernels"][0]["args"]
        self.assertEqual(next(a for a in rms if a["name"] == "mean_count")["data"],
                         [2048.0])
        attention = suite.scaled_metadata(
            cases["a01_attention_256x64"])["kernels"][0]["args"]
        self.assertEqual(next(a for a in attention if a["name"] == "div_c1")["data"],
                         [8.0])


class GenerationTests(unittest.TestCase):
    def test_seed_plan_matches_runner_generation_exactly(self):
        for _, case in suite.all_cases():
            plan = suite.seeded_input_plan(case)
            generated = suite.materialize_seeded(plan)
            for name, (dtype, shape, seed) in plan.items():
                expected = np.random.default_rng(seed).standard_normal(shape).astype(
                    suite.DTYPES[dtype])
                np.testing.assert_array_equal(generated[name], expected,
                                              err_msg=case["id"])
                self.assertEqual(generated[name].dtype, expected.dtype)

    def test_inputs_npz_contains_the_exact_seeded_payloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            for tier, case in suite.all_cases("perf-a"):
                directory = suite.generate_case(case, Path(temporary) / case["id"])
                with np.load(directory / "inputs.npz", allow_pickle=False) as archive:
                    saved = {name: np.array(archive[name], copy=True)
                             for name in archive.files}
                plan = json.loads((directory / "case.json").read_text())["input_seed_plan"]
                for name, row in plan.items():
                    expected = np.random.default_rng(row["seed"]).standard_normal(
                        row["shape"]).astype(suite.DTYPES[row["dtype"]])
                    np.testing.assert_array_equal(saved[name], expected,
                                                  err_msg=f"{case['id']}:{name}")

    def test_pointwise_reference_is_independent_and_exact(self):
        case = next(c for _, c in suite.all_cases("perf-a")
                    if c["id"] == "p00_scalar_add_4m")
        inputs = suite.materialize_seeded(suite.seeded_input_plan(case))
        expected = suite._reference(case, inputs)
        np.testing.assert_array_equal(expected["add"], inputs["x0"] + inputs["x1"])

    def test_softmax_and_attention_references_are_stable(self):
        soft = next(c for _, c in suite.all_cases("perf-a")
                    if c["family"] == "softmax")
        inputs = suite.materialize_seeded(suite.seeded_input_plan(soft))
        expected = suite._reference(soft, inputs)
        np.testing.assert_allclose(expected["softmax"].sum(axis=1), 1,
                                   rtol=1e-5, atol=1e-5)
        att = next(c for _, c in suite.all_cases("perf-a")
                   if c["family"] == "attention")
        inputs = suite.materialize_seeded(suite.seeded_input_plan(att))
        self.assertEqual(inputs["kt"].shape, (att["shape"][1], att["shape"][0]))
        expected = suite._reference(att, inputs)
        self.assertTrue(np.isfinite(expected["matmul_1"]).all())

    def test_cases_roundtrip_and_preserve_launch_contracts(self):
        with tempfile.TemporaryDirectory() as temporary:
            for tier, case in suite.all_cases("perf-a"):
                directory = suite.generate_case(case, Path(temporary) / case["id"])
                meta = json.loads((directory / "meta.json").read_text())
                record = suite.validate_case(directory)
                case_record = json.loads((directory / "case.json").read_text())
                self.assertEqual(record["valid"], True)
                if "block" in case:
                    self.assertEqual(meta["kernels"][0]["block"],
                                     [case["block"], 1, 1])
                self.assertGreaterEqual(meta["kernels"][0]["grid"][0], 1)
                self.assertEqual(case_record["hosted_eligible"],
                                 not case["local_only"])

    def test_buffer_footprints_stay_within_the_published_launch_limit(self):
        """The 2^26 limit is per launch, not per buffer or per tier."""
        for tier, case in suite.all_cases():
            meta = suite.scaled_metadata(case)
            for kernel in meta["kernels"]:
                total = sum(math.prod(arg["shape"]) for arg in kernel["args"]
                            if not arg.get("scalar") and "shape" in arg)
                self.assertLessEqual(total, 2**26, case["id"])


class AdmissionTests(unittest.TestCase):
    def base(self):
        return {
            "static_ready": True, "full_outputs_pass": True,
            "timing_ok": True, "planned_pairs": 9, "complete_pairs": 9,
            "above_floor": True, "floor": {"valid": True},
            "noise": {"valid": True, "resolved_aa_effect": False,
                      "deadband": 0.01, "maximum_deadband": 0.05},
        }

    def test_admission_requires_every_gate(self):
        record = self.base()
        self.assertTrue(runner.admit(record))
        for key in ("static_ready", "full_outputs_pass", "timing_ok",
                    "above_floor"):
            broken = self.base()
            broken[key] = False
            self.assertFalse(runner.admit(broken), key)
        for change in ({"complete_pairs": 8},
                       {"noise": {"valid": False,
                                  "resolved_aa_effect": False,
                                  "deadband": 0.01,
                                  "maximum_deadband": 0.05}},
                       {"noise": {"valid": True, "resolved_aa_effect": True}},
                       {"noise": {"valid": True, "resolved_aa_effect": False,
                                  "deadband": 0.2,
                                  "maximum_deadband": 0.05}}):
            broken = self.base()
            broken.update(change)
            self.assertFalse(runner.admit(broken), change)

    def test_noise_cap_is_fixed_not_derived_from_observed_ratio(self):
        stable = runner.noise_summary([10.0] * 9, [1.0, 1.01] * 4 + [0.99], 0.02)
        self.assertTrue(stable["valid"])
        self.assertFalse(stable["resolved_aa_effect"])
        noisy = runner.noise_summary([10.0] * 9, [0.90, 1.10] * 4 + [1.0], 0.02)
        self.assertFalse(noisy["valid"])
        missing = runner.noise_summary([], [], 0.02)
        self.assertFalse(missing["valid"])

    def test_small_fixtures_cannot_be_performance_timing_cases(self):
        self.assertEqual(runner.timing_role({"id": "p00_scalar_add", "tier": "floor"}),
                         "launch_floor_calibration")
        with self.assertRaises(ValueError):
            runner.timing_role({"id": "p00_scalar_add", "tier": "perf-a"})
        with self.assertRaises(ValueError):
            runner.timing_role({"id": "A86", "tier": "floor"})
        for tier, case in suite.all_cases():
            self.assertEqual(runner.timing_role({**case, "tier": tier}),
                             "performance_case")


if __name__ == "__main__":
    unittest.main()
