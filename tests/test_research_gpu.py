"""CPU tests for worker contracts; fake driver tests do NOT claim CUDA execution."""
import ctypes as C
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

spec = importlib.util.spec_from_file_location("research_gpu", Path(__file__).resolve().parents[1] / "tools/research_gpu.py")
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)


class FakeDriver:
    """Host-memory simulation to verify launch/reset/copy bookkeeping only."""
    def __init__(self):
        self.snapshot = {"name": "CPU fake: not GPU verification"}
        self.allocations, self.functions = {}, {}
        self.launches, self.resets, self.events = [], [], []
        self.next_handle = 1

    def handle(self, out):
        out._obj.value = self.next_handle
        self.next_handle += 1

    def alloc(self, out, size):
        memory = C.create_string_buffer(size)
        address = C.addressof(memory)
        self.allocations[address] = memory
        out._obj.value = address

    host_alloc = lambda self, out, size, _flags: self.alloc(out, size)
    free = lambda self, pointer: self.allocations.pop(pointer.value)
    host_free = free

    def htod(self, destination, source, size, stream):
        assert stream.value
        C.memmove(destination.value, source.value, size)

    def dtoh(self, destination, source, size, stream):
        assert stream.value
        C.memmove(destination.value, source.value, size)

    def memset(self, destination, value, size, stream):
        self.resets.append((destination.value, size, stream.value))
        C.memset(destination.value, value, size)

    stream_create = lambda self, out, flags: self.handle(out)
    stream_sync = lambda self, stream: None
    stream_destroy = lambda self, stream: None
    module_load = lambda self, out, path: self.handle(out)
    module_unload = lambda self, module: None
    function_attr = lambda self, *args: None
    function_get_attr = lambda self, out, enum, fn: setattr(out._obj, "value", 1024 if enum == 0 else 0)
    occupancy = lambda self, out, fn, threads, smem: setattr(out._obj, "value", 2)
    event_create = lambda self, out, flags: self.handle(out)
    event_record = lambda self, event, stream: self.events.append((event.value, stream.value))
    event_record_external = lambda self, event, stream, flags: self.events.append((event.value, stream.value))
    capture_begin = lambda self, stream, mode: None
    capture_end = lambda self, stream, out: self.handle(out)
    graph_instantiate = lambda self, out, graph, node, log, size: self.handle(out)
    graph_launch = lambda self, graph, stream: None
    graph_exec_destroy = lambda self, graph: None
    graph_destroy = lambda self, graph: None
    event_sync = lambda self, event: None
    event_elapsed = lambda self, out, start, stop: setattr(out._obj, "value", 0.001)
    event_destroy = lambda self, event: None
    close = lambda self: None

    def function_get(self, out, module, entry):
        self.handle(out)
        self.functions[out._obj.value] = entry.decode()

    def launch(self, function, gx, gy, gz, bx, by, bz, smem, stream, params, extra):
        assert stream.value  # All launches must use explicit non-default streams.
        entry = self.functions[function.value]
        pointers = [C.cast(pointer, C.POINTER(gpu.D)).contents.value for pointer in params]
        self.launches.append((entry, stream.value, pointers))
        # Test kernels use arg_order y,x; add kernel emulates atomic accumulation.
        y, x = (C.cast(address, C.POINTER(C.c_float)) for address in pointers)
        for index in range(4):
            y[index] = y[index] + x[index] if entry == "add" else x[index]
        if entry == "bad_guard":
            C.c_ubyte.from_address(pointers[0] + 16).value ^= 255
        if entry == "bad_input":
            x[0] += 1


class CaseFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, name="case", entry="copy", zero=False):
        root = self.root / name
        root.mkdir()
        (root / "module.cubin").write_bytes(b"not a real GPU binary")
        meta = {"id": name, "rtol": 1e-6, "atol": 1e-6, "graph_outputs": ["y"],
                "kernels": [{"entry": entry, "grid": [1, 1, 1], "block": [4, 1, 1],
                             "smem_bytes": 0, "arg_order": ["y", "x"],
                             "zero_outputs": ["y"] if zero else [],
                             "args": [{"name": "x", "dtype": "f32", "shape": [4], "seed": 99},
                                      {"name": "y", "dtype": "f32", "shape": [4], "output": True}]}]}
        (root / "meta.json").write_text(json.dumps(meta))
        inputs = np.asarray([1, -2, 3, 0], dtype=np.float32)
        np.savez(root / "inputs.npz", x=inputs)
        np.savez(root / "expected.npz", y=inputs)
        return {"id": name, "variant": name, "case_dir": str(root),
                "cubins": [str(root / "module.cubin")]}

    def test_load_exact_inputs_tolerances_and_identity(self):
        case = self.fixture()
        data = gpu.load_case(case)
        self.assertEqual(data["immutable"], {"x"})
        np.testing.assert_array_equal(data["arrays"]["x"], [1, -2, 3, 0])
        self.assertEqual(data["meta"]["atol"], 1e-6)
        self.assertEqual(len(data["identities"]), 4)

    def test_published_singleton_reference_shape_is_normalized(self):
        case = self.fixture()
        root = Path(case["case_dir"])
        meta = json.loads((root / "meta.json").read_text())
        for arg in meta["kernels"][0]["args"]:
            arg["shape"] = [1]
        (root / "meta.json").write_text(json.dumps(meta))
        np.savez(root / "inputs.npz", x=np.array([1], dtype=np.float32))
        np.savez(root / "expected.npz", y=np.array(1, dtype=np.float32))
        data = gpu.load_case(case)
        self.assertEqual(data["expected"]["y"].shape, (1,))

    def test_missing_reference_and_invalid_abi_rejected_before_cuda(self):
        case = self.fixture()
        root = Path(case["case_dir"])
        np.savez(root / "expected.npz", wrong=np.zeros(4))
        with self.assertRaisesRegex(ValueError, "not a produced"):
            gpu.load_case(case)
        meta = json.loads((root / "meta.json").read_text())
        meta["kernels"][0]["arg_order"] = ["y", "y"]
        (root / "meta.json").write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "arg_order"):
            gpu.load_case(case)

    def test_repeated_launch_resets_positional_abi_and_input_checks(self):
        case = self.fixture(entry="add", zero=True)
        driver = FakeDriver()
        result = gpu.correctness(driver, {"width": 1, "repetitions": 3}, [gpu.load_case(case)])
        self.assertTrue(result["ok"])
        self.assertEqual(len(driver.resets), 3)
        self.assertEqual(result["launch_count"], 3)
        self.assertEqual(driver.allocations, {})
        self.assertFalse(result["promotion_eligible"])

    def test_guard_failure_quarantines_good_batchmate(self):
        cases = [self.fixture("bad", "bad_guard"), self.fixture("good", "copy")]
        driver = FakeDriver()
        result = gpu.correctness(driver, {"width": 2, "repetitions": 3}, list(map(gpu.load_case, cases)))
        self.assertFalse(result["ok"])
        batch = result["batches"][0]
        self.assertEqual(len(batch["membership"]), 2)
        self.assertTrue(batch["requires_all_batchmates_isolated"])
        bad, good = batch["results"]
        self.assertFalse(bad["checks"][0]["buffers"]["y"]["suffix_ok"])
        self.assertTrue(good["ok"])
        self.assertFalse(good["trusted"])
        self.assertEqual(len(driver.launches), 2)  # Abort after first bad repetition.
        self.assertEqual(len({launch[1] for launch in driver.launches}), 2)

    def test_mutated_input_detected(self):
        driver = FakeDriver()
        result = gpu.correctness(driver, {"width": 1}, [gpu.load_case(self.fixture(entry="bad_input"))])
        check = result["batches"][0]["results"][0]["checks"][0]
        self.assertFalse(check["buffers"]["x"]["input_ok"])
        self.assertFalse(result["ok"])

    def test_pipeline_shares_intermediate_and_checks_every_buffer(self):
        case = self.fixture()
        root = Path(case["case_dir"])
        meta = json.loads((root / "meta.json").read_text())
        first = meta["kernels"][0]
        second = {**first, "args": [{"name": "y", "shape": [4], "dtype": "f32", "intermediate": True},
                                    {"name": "z", "shape": [4], "dtype": "f32", "output": True}],
                  "arg_order": ["z", "y"]}
        meta["kernels"].append(second)
        meta["graph_outputs"] = ["z"]
        (root / "meta.json").write_text(json.dumps(meta))
        np.savez(root / "expected.npz", z=np.asarray([1, -2, 3, 0], dtype=np.float32))
        case["cubins"] *= 2
        driver = FakeDriver()
        result = gpu.correctness(driver, {"width": 1}, [gpu.load_case(case)])
        self.assertTrue(result["ok"])
        self.assertEqual(driver.launches[0][2][0], driver.launches[1][2][1])
        self.assertEqual(set(result["batches"][0]["results"][0]["checks"][0]["buffers"]), {"x", "y", "z"})

    def test_timing_pairs_separate_arms_partial_checkpoint_and_offsets(self):
        case = self.fixture(entry="add", zero=True)
        case["baseline_cubins"] = case["cubins"]
        driver = FakeDriver()
        job = {"pairs": 8, "pair_offset": 8, "seed": 17, "warmup": 2,
               "burst_id": "burst-2", "out": str(self.root / "timing.json")}
        result = gpu.timing(driver, job, gpu.load_case(case), gpu.load_case(case, "baseline_cubins"))
        self.assertTrue(result["ok"])
        self.assertEqual(result["completed_pairs"], 8)
        self.assertEqual(len(driver.resets), 20)
        self.assertEqual([row["order"] for row in result["pairs"]], gpu.paired_orders(17, 8, 8))
        self.assertEqual({row["burst_id"] for row in result["pairs"]}, {"burst-2"})
        self.assertEqual([row["pair_index"] for row in result["pairs"]], list(range(8, 16)))
        self.assertEqual(result["launch_count"], 20)
        checkpoint = json.loads(Path(job["out"] + ".partial.json").read_text())
        self.assertTrue(checkpoint["incomplete"])
        self.assertEqual(len(checkpoint["pairs"]), 8)
        self.assertEqual(driver.allocations, {})
        self.assertNotEqual(driver.launches[0][2][0], driver.launches[1][2][0])

    def test_preflight_invalid_job_never_loads_cuda(self):
        case = self.fixture()
        with mock.patch.object(gpu, "Driver") as driver:
            with self.assertRaisesRegex(ValueError, "width"):
                gpu.run_job({"mode": "correctness", "cases": [case], "width": 3})
            with self.assertRaisesRegex(ValueError, "pairs"):
                gpu.run_job({"mode": "timing", "case": case, "pairs": 201})
            driver.assert_not_called()

    def test_missing_intermediate_producer_rejected(self):
        case = self.fixture()
        root = Path(case["case_dir"])
        meta = json.loads((root / "meta.json").read_text())
        meta["kernels"][0]["args"][0]["intermediate"] = True
        (root / "meta.json").write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "no earlier producer"):
            gpu.load_case(case)

    def test_serve_reuses_context_and_acknowledges_atomic_results(self):
        case = self.fixture()
        jobs = [{"mode": "correctness", "cases": [case], "width": 1,
                 "out": str(self.root / f"result-{index}.json")} for index in range(2)]
        driver, output = FakeDriver(), io.StringIO()
        with mock.patch.object(gpu, "Driver", return_value=driver) as constructor, \
             mock.patch.object(gpu, "parent_death_signal"), \
             mock.patch.object(driver, "close", wraps=driver.close) as close, \
             mock.patch.object(gpu.sys, "stdin", io.StringIO("\n".join(map(json.dumps, jobs)) + "\n")), \
             mock.patch.object(gpu.sys, "stdout", output):
            self.assertEqual(gpu.serve(), 0)
            constructor.assert_called_once()
            close.assert_called_once()
        acks = list(map(json.loads, output.getvalue().splitlines()))
        self.assertEqual(len(acks), 2)
        self.assertTrue(all(ack["ok"] and ack["status"] == 0 for ack in acks))
        for job in jobs:
            self.assertTrue(json.loads(Path(job["out"]).read_text())["ok"])
        self.assertEqual(driver.allocations, {})

    def test_serve_cuda_error_retires_without_processing_next_job(self):
        driver, output = FakeDriver(), io.StringIO()
        jobs = [{"out": str(self.root / f"result-{index}.json")} for index in range(2)]
        with mock.patch.object(gpu, "Driver", return_value=driver), \
             mock.patch.object(gpu, "parent_death_signal"), \
             mock.patch.object(driver, "close", wraps=driver.close) as close, \
             mock.patch.object(gpu, "run_job", side_effect=gpu.CUDAError("controlled error")) as run, \
             mock.patch.object(gpu.sys, "stdin", io.StringIO("\n".join(map(json.dumps, jobs)) + "\n")), \
             mock.patch.object(gpu.sys, "stdout", output):
            self.assertEqual(gpu.serve(), 2)
            run.assert_called_once()
            close.assert_not_called()
        self.assertTrue(json.loads(output.getvalue())["retire_worker"])
        self.assertFalse(Path(jobs[1]["out"]).exists())

    def test_timing_error_keeps_pair_incomplete(self):
        case = self.fixture(entry="bad_guard")
        baseline = gpu.load_case(case)
        candidate = gpu.load_case(case)
        driver = FakeDriver()
        result = gpu.timing(driver, {"warmup": 0, "pairs": 8, "timing_protocol": "legacy-single-launch-v1", "out": str(self.root / "timing.json")},
                            candidate, baseline)
        self.assertFalse(result["ok"])
        self.assertEqual(result["completed_pairs"], 0)
        self.assertFalse(result["pairs"][0]["complete"])
        self.assertNotIn("candidate_over_baseline", result["pairs"][0])


class PureHelpers(unittest.TestCase):
    def test_prefix_suffix_and_input_corruption_are_distinguished(self):
        image = gpu.guard_image(b"abcd", seed=22)
        self.assertEqual(len(image), 516)
        self.assertNotEqual(image[:256], image[-256:])
        self.assertNotEqual(image, gpu.guard_image(b"abcd", seed=23))
        for index, key in ((0, "prefix_ok"), (256, "input_ok"), (260, "suffix_ok")):
            changed = bytearray(image)
            changed[index] ^= 1
            checks = gpu.inspect_guard_image(changed, image, 4, immutable=True)
            self.assertEqual([name for name, good in checks.items() if not good], [key])
        with self.assertRaises(ValueError):
            gpu.guard_image(b"abcd", 22, 257)

    def test_full_output_no_nan_equality_or_tolerant_integer_check(self):
        result = gpu.compare_output(np.array([1., 2., 4.]), np.array([1., 2., 3.]), 1e-6, 1e-6)
        self.assertEqual(result["mismatched"], 1)
        self.assertEqual(result["max_abs_error"], 1)
        self.assertFalse(gpu.compare_output(np.array([np.nan]), np.array([np.nan]), 1., 1.)["ok"])
        self.assertFalse(gpu.compare_output(np.array([100], dtype=np.int64), np.array([101]), 1., 1.)["ok"])

    def test_reproducible_pair_order_resume(self):
        full = gpu.paired_orders(333, 32)
        self.assertEqual(full[16:], gpu.paired_orders(333, 16, 16))
        self.assertTrue(all(set(order) == {"baseline", "candidate"} for order in full))
        self.assertEqual(len({tuple(order) for order in full}), 2)

    def test_quarantine_and_atomic_json(self):
        rows = [{"ok": True}, {"ok": False}, {"ok": True}]
        self.assertTrue(gpu.quarantine_batch(rows))
        self.assertTrue(all(row["quarantined"] and not row["trusted"] for row in rows))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            gpu.atomic_json(path, {"ok": True})
            self.assertEqual(json.loads(path.read_text()), {"ok": True})
            self.assertEqual(list(Path(temporary).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
