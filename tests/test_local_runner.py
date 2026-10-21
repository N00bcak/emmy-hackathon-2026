"""CPU-only contract checks; no CUDA is loaded and no public endpoint is used."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest import mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

import numpy as np

spec = importlib.util.spec_from_file_location("local_runner", Path(__file__).resolve().parents[1] / "tools/local_runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def valid_request():
    return {"cubin_id": "a" * 64, "entry": "kernel", "grid": [1, 1, 1],
            "block": [128, 1, 1], "args": [
                {"name": "x", "dtype": "f32", "shape": [8], "seed": 0},
                {"name": "y", "dtype": "f32", "shape": [8], "output": True}],
            "zero_outputs": ["y"], "warmup": 2, "iters": 3}


class Validation(unittest.TestCase):
    def test_existing_metadata(self):
        problems = Path(__file__).resolve().parents[1] / "emmy-hackathon-2026/problems"
        count = 0
        for path in problems.glob("*/meta.json"):
            meta = json.loads(path.read_text())
            for kernel in meta["kernels"]:
                args = copy.deepcopy(kernel["args"])
                for arg in args:
                    if arg.pop("intermediate", False):
                        arg["data"] = np.zeros(arg["shape"]).reshape(-1).tolist()
                request = {**kernel, "args": args, "cubin_id": "a" * 64}
                runner.run_request(request)
            count += 1
        self.assertEqual(count, 15)

    def test_defaults_and_dimensions(self):
        request = {"cubin_id": "a" * 64, "entry": "kernel", "args": [{"name": "x", "shape": [1], "seed": 0}]}
        normalized = runner.run_request(request)
        self.assertEqual(normalized["grid"], [1, 1, 1])
        self.assertEqual(normalized["block"], [128, 1, 1])
        self.assertEqual((normalized["warmup"], normalized["iters"]), (5, 20))
        self.assertEqual(normalized["args"][0]["dtype"], "f32")
        self.assertNotIn("dtype", request["args"][0])
        with self.assertRaises(runner.Invalid):
            runner.run_request({**request, "args": [{"name": str(i), "shape": [1], "seed": 0} for i in range(33)]})
        with self.assertRaises(runner.Invalid):
            runner.run_request({**request, "args": [{"name": "x", "shape": [1] * 9, "seed": 0}]})

    def test_seed_and_data(self):
        arg = valid_request()["args"][0]
        np.testing.assert_array_equal(runner.array_for(arg), np.random.default_rng(0).standard_normal([8]).astype(np.float32))
        arg = {"name": "x", "dtype": "i32", "shape": [2, 2], "data": [1, 2, 3, 4]}
        np.testing.assert_array_equal(runner.array_for(arg), [[1, 2], [3, 4]])
        self.assertTrue(np.all(runner.array_for(valid_request()["args"][1]) == 0))

    def test_sum_matches_observed_f32_reduction(self):
        array = np.random.default_rng(0).standard_normal(1024).astype(np.float32)
        result = runner.output_summary(array, {"shape": [1024], "dtype": "f32"}, 3)
        self.assertEqual(result["sum"], float(array.sum()))
        self.assertNotEqual(result["sum"], float(array.astype(np.float64).sum()))
        self.assertEqual(len(result["values"]), 3)
        self.assertTrue(result["truncated"])

    def test_worker_rejects_orphan_before_cuda(self):
        with tempfile.TemporaryDirectory() as folder:
            request, result = Path(folder) / "request.json", Path(folder) / "result.json"
            request.write_text(json.dumps({"_parent_pid": -1}))
            completed = subprocess.run([__import__("sys").executable, runner.__file__, "--worker", str(request), str(result)], capture_output=True, text=True, timeout=10)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("parent exited", json.loads(result.read_text())["stderr"])

    def test_limits_and_abi(self):
        base = valid_request()
        for field, value in [("warmup", 21), ("iters", 201), ("iters", 0),
                             ("max_return", 2**22 + 1), ("smem_bytes", 99 * 1024 + 1),
                             ("block", [1024, 2, 1]), ("grid", [0, 1, 1]),
                             ("zero_outputs", ["x"]), ("cubin_id", "../oops")]:
            with self.subTest(field=field, value=value), self.assertRaises(runner.Invalid):
                runner.run_request({**base, field: value})
        base["args"][0]["shape"] = [2**26 + 1]
        with self.assertRaises(runner.Invalid):
            runner.run_request(base)
        with self.assertRaises(runner.Invalid):
            runner.compile_request({"ptx": ".version 8.0", "flags": ["--output-file=/tmp/oops"]}, "sm_89")
        self.assertIsNone(runner.run_request({}))


class Service(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        fake = root / "fake-ptxas"
        fake.write_text('''#!/usr/bin/env python3
import pathlib,sys
if '--version' in sys.argv:
 print('test ptxas 1.0');sys.exit()
source=pathlib.Path(sys.argv[sys.argv.index('-o')-1]).read_text()
if 'BAD' in source:
 print('test assembly rejection',file=sys.stderr);sys.exit(1)
pathlib.Path(sys.argv[sys.argv.index('-o')+1]).write_bytes(b'cubin')
print('ptxas info : Function properties for kernel\\nptxas info : Used 4 registers',file=sys.stderr)
''')
        fake.chmod(0o700)
        self.runner = runner.Runner(root / "cache", ptxas=str(fake))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), runner.handler_for(self.runner))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def post(self, path, data):
        request = Request(self.url + path, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
        try:
            with urlopen(request) as response:
                return response.status, json.load(response)
        except HTTPError as exc:
            return exc.code, json.load(exc)

    def test_compile_cache_and_failures(self):
        ptx = '.version 8.0\n.target sm_89\n.address_size 64\n.visible .entry kernel() { ret; }'
        status, first = self.post("/compile", {"ptx": ptx})
        self.assertEqual(status, 200)
        self.assertEqual(first["entry_points"], ["kernel"])
        self.assertEqual(first["registers"], {"kernel": 4})
        with mock.patch.object(runner.subprocess, "run", side_effect=AssertionError("cache miss")):
            self.assertEqual(self.post("/compile", {"ptx": ptx})[1], first)
        self.assertNotEqual(self.post("/compile", {"ptx": ptx, "flags": ["-O0"]})[1]["cubin_id"], first["cubin_id"])
        status, result = self.post("/compile", {"ptx": "BAD"})
        self.assertEqual((status, result["stage"], result["stderr"]), (200, "ptxas", "test assembly rejection\n"))
        self.assertEqual(self.post("/compile", {"ptx": ptx, "flags": ["-evil"]})[0], 422)

    def test_request_lookup_and_health(self):
        with urlopen(self.url + "/health") as response:
            self.assertEqual(json.load(response)["default_arch"], "sm_89")
        self.assertEqual(self.post("/run", {})[1]["stage"], "request")
        self.assertEqual(self.post("/run", valid_request())[1]["stage"], "lookup")
        self.assertEqual(self.post("/run", {**valid_request(), "iters": 201})[0], 422)

    def test_isolated_worker_timeout_and_busy(self):
        (self.runner.cache / ("a" * 64)).mkdir()
        (self.runner.cache / ("a" * 64) / "module.cubin").write_bytes(b"cubin")
        with mock.patch.object(runner.subprocess, "run", side_effect=subprocess.TimeoutExpired("worker", 20)):
            result = self.runner.run(valid_request())
        self.assertEqual(result["stage"], "run")
        self.assertIn("20 seconds", result["stderr"])
        self.assertTrue(self.runner.run_lock.acquire(blocking=False))
        try:
            self.assertEqual(self.runner.run(valid_request())["stage"], "busy")
        finally:
            self.runner.run_lock.release()


if __name__ == "__main__":
    unittest.main()
