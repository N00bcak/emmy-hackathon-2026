"""Calibration unit tests use fake transport only: never contact the endpoint."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import research_calibrate as calibration
from tools.research_store import Store, Invalid
from tools.published import PUBLISHED_PROBLEMS


class FakeClock:
    def __init__(self):
        self.value = 1000.0
        self.sleeps = []

    def __call__(self):
        return self.value

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.value += delay


class Transport:
    def __init__(self, root, run, clock):
        self.root, self.run, self.clock = root, run, clock
        self.calls, self.modules = [], {}
        self.failure = None
        self.corrupt = False
        self.gpu = "NVIDIA GeForce RTX 5090 (sm_120)"

    def __call__(self, method, path, body):
        attempt_path = self.root / "calibration" / self.run / "requests" / f"{len(self.calls) + 1:03d}.json"
        charged = json.loads(attempt_path.read_text())
        assert charged["status"] == "charged"  # Journal is committed before transport.
        self.calls.append((self.clock(), method, path, copy.deepcopy(body)))
        if self.failure and len(self.calls) == self.failure[0]:
            if isinstance(self.failure[1], Exception):
                raise self.failure[1]
            return self.failure[1:]
        if path == "/health":
            return 200, {"ok": True, "can_compile": True, "can_run": True,
                         "gpu": self.gpu, "default_arch": "sm_120", "ptxas": "fake"}
        if path == "/compile":
            assert body["arch"] == "sm_120"
            module = str(len(self.modules))
            self.modules[module] = body["ptx"]
            return 200, {"ok": True, "cubin_id": module, "entry_points": ["k_copy"]}
        args = {arg["name"]: arg for arg in body["args"]}
        input_arg = args["x"] if "x" in args else args["y"]
        values = np.asarray(input_arg["data"], dtype=np.float32)
        if "scale" in args:
            values = values * args["scale"]["data"][0] + args["bias"]["value"]
        outputs = {}
        for arg in args.values():
            if arg.get("output"):
                outputs[arg["name"]] = {"shape": arg["shape"], "dtype": arg["dtype"],
                                        "values": (values + (1 if self.corrupt else 0)).tolist(),
                                        "truncated": False}
        return 200, {"ok": True, "outputs": outputs,
                     "median_us": 90.0 if "candidate" in self.modules[body["cubin_id"]] else 100.0}


class Calibration(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        source = self.base / "source"
        source.mkdir()
        (source / "compile.py").write_text("print('baseline')\n")
        self.root = self.base / "research"
        self.store = Store(self.root, self.base / "work")
        self.store.init(source)
        brief = {"id": "test", "mechanism": "test mechanism", "applicability": "pointwise",
                 "prediction": "less work", "probe": "inspect SASS", "opportunity": "test",
                 "budget": {"variants": 1}, "semantic_overlap_decision": "distinct"}
        self.store.propose(brief)
        (source / "compile.py").write_text("print('candidate')\n")
        self.manifest = self.store.freeze("test", source)
        self.pairs = [self.case("first"), self.case("second")]
        self.issue_shortlist(self.pairs)
        self.clock = FakeClock()

    def tearDown(self):
        for path in sorted(self.base.rglob("*"), reverse=True):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o755)
        self.temporary.cleanup()

    def case(self, name, pipeline=False):
        folder = self.base / name
        folder.mkdir()
        inputs = np.asarray([1, 2, 3, 4], dtype=np.float32)
        np.savez(folder / "inputs.npz", x=inputs, scale=np.asarray([2.0], dtype=np.float32))
        np.savez(folder / "expected.npz", **{"z" if pipeline else "y": inputs * 2 + 1})
        args = [{"name": "x", "dtype": "f32", "shape": [4], "seed": 0},
                {"name": "scale", "dtype": "f32", "shape": [1], "data": [2.0]},
                {"name": "bias", "dtype": "f32", "scalar": True, "value": 1.0},
                {"name": "y", "dtype": "f32", "shape": [4], "output": True}]
        kernel = {"entry": "k_copy", "args": args, "arg_order": ["y", "scale", "bias", "x"],
                  "grid": [1, 1, 1], "block": [4, 1, 1], "zero_outputs": ["y"], "smem_bytes": 0}
        kernels = [kernel]
        if pipeline:
            kernels.append({**kernel, "args": [{"name": "y", "dtype": "f32", "shape": [4], "intermediate": True},
                                                 {"name": "z", "dtype": "f32", "shape": [4], "output": True}],
                            "arg_order": ["z", "y"], "zero_outputs": ["z"]})
        meta = {"id": name, "rtol": 1e-6, "atol": 1e-6, "kernels": kernels,
                "graph_outputs": ["z" if pipeline else "y"]}
        (folder / "meta.json").write_text(json.dumps(meta))
        for arm in ("baseline", "candidate"):
            (folder / f"{arm}.ptx").write_text(f".version 8.0\n.target sm_89\n.address_size 64\n.visible .entry k_copy() {{ret;}}\n// {arm}\n")
        return {"experiment_id": "test", "case_dir": str(folder),
                "baseline_ptx": [str(folder / "baseline.ptx")] * len(kernels),
                "candidate_ptx": [str(folder / "candidate.ptx")] * len(kernels),
                "local_baseline_us": 10.0, "local_candidate_us": 8.0}

    def issue_shortlist(self, pairs, baseline_verified=True):
        files = {}
        for index, descriptor in enumerate(pairs):
            folder = Path(descriptor["case_dir"])
            for path in folder.iterdir():
                if path.is_file():
                    files[f"case-{index}/{path.name}"] = path.read_bytes()
        artifact = self.store.put_artifact("local-evaluation", {}, files)
        suite = self.store.register_suite("suite-v1", {"rtol": 1e-6, "atol": 1e-6})
        confirmation = self.store.reserve_confirmation("test", suite)
        gates = {}
        for source_id in (self.manifest["source_id"], self.manifest["baseline_id"]):
            gate_report = {"source_id": source_id, "suite_id": suite,
                "cases": [{"problem": name, "case_id": name, "status": "pass", "isolated": True,
                           "guards_ok": True, "inputs_ok": True} for name in PUBLISHED_PROBLEMS],
                "sanitizer_passed": True, "artifact_ids": [artifact]}
            gates[source_id] = self.store.issue_published_gate(source_id, suite, gate_report)

        report = {"source_id": self.manifest["source_id"], "baseline_id": self.manifest["baseline_id"],
                  "suite_id": suite, "confirmation_id": confirmation["id"], "fresh_confirmation": True,
                  "cases": [{"case_id": "test", "status": "pass", "isolated": True,
                             "guards_ok": True, "inputs_ok": True}],
                  "baseline_gpu_verified": baseline_verified, "local_device": {"name": "RTX 4090", "uuid": "test"},
                  "artifact_ids": [artifact],
                  "published_gate_id": gates[self.manifest["source_id"]],
                  "baseline_published_gate_id": gates[self.manifest["baseline_id"]]}
        report['calibration_cases'] = {}
        report['timing'] = {}
        for descriptor in pairs:
            folder = Path(descriptor['case_dir'])
            case_id = folder.name
            report['calibration_cases'][case_id] = {
                'case_files_sha256': {name: calibration.sha(folder/name) for name in ('meta.json','inputs.npz','expected.npz')},
                'baseline_ptx_sha256': [calibration.sha(p) for p in descriptor['baseline_ptx']],
                'candidate_ptx_sha256': [calibration.sha(p) for p in descriptor['candidate_ptx']]}
            report['timing'][case_id] = {'ok': True, 'pairs': [
                {'complete': True, 'burst_id': 'fixture-block', 'samples_us': {'baseline': 10.0, 'candidate': 8.0}}
                for _ in range(16)]}
        evidence = self.store.issue_evidence("test", report)
        self.store.decide("test", "shortlist", "mocked test fixture", evidence)

    def run_batch(self, transport, pairs=None, run="run-1", **kwargs):
        return calibration.run_calibration(self.root, run, "vector mechanism", pairs or self.pairs,
                                           transport=transport, sleep=self.clock.sleep,
                                           clock=self.clock, wall_clock=self.clock, **kwargs)

    def test_default_17_requests_full_outputs_exact_inputs_and_spacing(self):
        transport = Transport(self.root, "run-1", self.clock)
        result = self.run_batch(transport)
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(result["requests_attempted"], 17)
        self.assertEqual(len(result["comparisons"]), 2)
        self.assertTrue(all(row["direction_agreement"] for row in result["comparisons"]))
        self.assertTrue(all(right[0] - left[0] >= 2 for left, right in zip(transport.calls, transport.calls[1:])))
        for _stamp, _method, path, body in transport.calls:
            if path == "/run":
                self.assertEqual([arg["name"] for arg in body["args"]], ["y", "scale", "bias", "x"])
                self.assertEqual(body["args"][-1]["data"], [1, 2, 3, 4])
                self.assertEqual(body["zero_outputs"], ["y"])
                self.assertEqual(body["max_return"], 4)
                self.assertEqual((body["warmup"], body["iters"]), (5, 20))
        self.assertEqual(len(calibration.history(self.root)), 1)
        with self.assertRaisesRegex(calibration.CalibrationError, "singleton"):
            self.run_batch(transport)
        self.assertEqual(len(transport.calls), 17)

    def test_busy_429_service_and_transport_errors_stop_without_retry(self):
        failures = [(200, {"ok": False, "stage": "busy"}),
                    (429, {"ok": False}), (503, {"ok": False}), RuntimeError("network unavailable")]
        for index, failure in enumerate(failures):
            run = f"failure-{index}"
            transport = Transport(self.root, run, self.clock)
            transport.failure = (2, failure) if isinstance(failure, Exception) else (2, *failure)
            result = self.run_batch(transport, run=run)
            self.assertEqual(result["status"], "stopped")
            self.assertEqual(result["requests_attempted"], 2)
            self.assertEqual(len(transport.calls), 2)
            attempts = list((self.root / "calibration" / run / "requests").glob("*.json"))
            self.assertEqual(len(attempts), 2)

    def test_spacing_persists_across_failed_run_boundaries(self):
        first = Transport(self.root, "failure-one", self.clock)
        first.failure = (1, 503, {"ok": False})
        self.run_batch(first, run="failure-one")
        first_time = first.calls[0][0]
        second = Transport(self.root, "failure-two", self.clock)
        second.failure = (1, 503, {"ok": False})
        self.run_batch(second, run="failure-two")
        self.assertGreaterEqual(second.calls[0][0] - first_time, 2.0)
        self.assertEqual(len(calibration.history(self.root)), 2)

    def test_cli_requires_explicit_execute_flag_for_network(self):
        pairs_path = self.base / "prepared-pair.json"
        pairs_path.write_text(json.dumps(self.pairs))
        from unittest import mock
        command = ["research_calibrate", "--root", str(self.root), "--run-id", "dry-run",
                   "--mechanism", "vector mechanism", "--prepared-pair", str(pairs_path)]
        with mock.patch.object(calibration.sys if hasattr(calibration, "sys") else sys, "argv", command), \
             mock.patch.object(calibration, "HTTPTransport") as transport, \
             mock.patch("builtins.print"):
            self.assertEqual(calibration.main(), 0)
            transport.assert_not_called()
        self.assertFalse((self.root / "calibration/dry-run").exists())

    def test_wrong_gpu_stops_after_health(self):
        transport = Transport(self.root, "run-1", self.clock)
        transport.gpu = "RTX 4090 (sm_89)"
        result = self.run_batch(transport)
        self.assertEqual(result["requests_attempted"], 1)
        self.assertEqual(result["status"], "stopped")

    def test_reference_failure_stops_and_retains_response(self):
        transport = Transport(self.root, "run-1", self.clock)
        transport.corrupt = True
        result = self.run_batch(transport)
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["requests_attempted"], 4)  # health + two compiles + failing run
        attempt = json.loads((self.root / "calibration/run-1/requests/004.json").read_text())
        self.assertIn("outputs", attempt["response"])

    def test_unverified_baseline_and_swapped_ptx_never_contact_endpoint(self):
        transport = Transport(self.root, "run-1", self.clock)
        self.issue_shortlist(self.pairs, baseline_verified=False)
        with self.assertRaisesRegex(calibration.CalibrationError, "both arms"):
            self.run_batch(transport)
        self.assertFalse(transport.calls)
        self.issue_shortlist(self.pairs)
        Path(self.pairs[0]["candidate_ptx"][0]).write_text(".version 8.0\n// swapped")
        with self.assertRaisesRegex(calibration.CalibrationError, "not retained"):
            self.run_batch(transport)
        self.assertFalse(transport.calls)

    def test_swapped_arms_and_edited_local_medians_never_contact_endpoint(self):
        transport = Transport(self.root, 'run-1', self.clock)
        pairs = copy.deepcopy(self.pairs)
        pairs[0]['baseline_ptx'], pairs[0]['candidate_ptx'] = pairs[0]['candidate_ptx'], pairs[0]['baseline_ptx']
        with self.assertRaisesRegex(calibration.CalibrationError, 'paired measurement'):
            self.run_batch(transport, pairs)
        pairs = copy.deepcopy(self.pairs)
        pairs[0]['local_candidate_us'] = 0.1
        with self.assertRaisesRegex(calibration.CalibrationError, 'local medians'):
            self.run_batch(transport, pairs)
        self.assertFalse(transport.calls)

    def test_missing_baseline_all15_gate_never_contacts_endpoint(self):
        evidence = self.store._manifest("test")["decision"]["evidence_id"]
        report = self.store.verify_object(evidence)["payload"]["report"]
        index = self.root / "control/published-gates" / self.manifest["baseline_id"] / (report["suite_id"] + ".json")
        index.unlink()
        transport = Transport(self.root, "run-1", self.clock)
        with self.assertRaisesRegex(calibration.CalibrationError, "all15"):
            self.run_batch(transport)
        self.assertFalse(transport.calls)

    def test_pipeline_reduces_cases_to_preserve_headroom(self):
        pipeline = self.case("pipeline", pipeline=True)
        self.issue_shortlist([pipeline, self.pairs[1]])
        transport = Transport(self.root, "run-1", self.clock)
        result = self.run_batch(transport, [pipeline, self.pairs[1]])
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(result["requests_attempted"], 17)
        self.assertEqual(result["local_provenance"]["omitted_cases"], 1)
        self.assertEqual(len(result["comparisons"]), 1)

    def test_global_exclusive_lock_and_hard_cap(self):
        transport = Transport(self.root, "run-1", self.clock)
        with calibration.exclusive(self.root / "control/calibration.lock"):
            with self.assertRaisesRegex(calibration.CalibrationError, "exclusive"):
                self.run_batch(transport)
        self.assertFalse(transport.calls)
        batch = self.root / "calibration/cap-test"
        requests = batch / "requests"
        requests.mkdir(parents=True)
        for index in range(25):
            (requests / f"{index + 1:03d}.json").write_text('{"status":"charged"}')
        client = calibration.ChargedRequests(self.root, batch, transport, self.clock.sleep, self.clock, self.clock)
        with self.assertRaisesRegex(calibration.CalibrationError, "hard cap"):
            client.request("GET", "/health")
        self.assertFalse(transport.calls)

    def test_history_trigger_accumulates_across_runs_and_mechanisms(self):
        result = self.run_batch(Transport(self.root, "run-1", self.clock))
        self.assertEqual(result["status"], "completed")
        self.assertFalse(calibration.calibration_due(self.root, "vector mechanism")["due"])
        self.assertTrue(calibration.calibration_due(self.root, "different mechanism", architecture_sensitive=True)["due"])
        state_path = self.root / "control/state.json"
        state = json.loads(state_path.read_text())
        state["baseline_history"] = [{"test": index} for index in range(5)]
        state_path.write_text(json.dumps(state))
        self.assertTrue(calibration.calibration_due(self.root, "vector mechanism")["due"])
        self.assertEqual(len(list((self.root / "knowledge/mechanisms").glob("calibration-*.jsonl"))), 1)


if __name__ == "__main__":
    unittest.main()
