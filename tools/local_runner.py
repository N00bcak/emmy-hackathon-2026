#!/usr/bin/env python3
"""Loopback-only local implementation of the documented Emmy runner API.

Run inside an admitted CUDA job, never as a public service. This is a
trusted research tool, not a sandbox for hostile PTX. No expected-output files are
read. Timing uses CUDA events around each launch (excluding reset/copies/context
startup); backend timing parity with the hosted service is NOT claimed.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

DTYPES = {"f32": np.float32, "f16": np.float16, "f64": np.float64,
          "i32": np.int32, "i64": np.int64}
MAX_PTX = 2_000_000
MAX_ELEMENTS = 2**26
MAX_RETURN = 2**22
MAX_BODY = 96 * 1024 * 1024


class Invalid(ValueError):
    pass


def integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise Invalid(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def object_body(body):
    if not isinstance(body, dict):
        raise Invalid("request must be an object")


def compile_request(body, default_arch):
    object_body(body)
    ptx = body.get("ptx")
    if not isinstance(ptx, str) or not 1 <= len(ptx) <= MAX_PTX:
        raise Invalid(f"ptx must contain 1..{MAX_PTX} characters")
    arch = body.get("arch", default_arch)
    if not isinstance(arch, str) or not re.fullmatch(r"sm_[0-9]{2,3}[af]?", arch):
        raise Invalid("arch must be an sm_ target")
    flags = body.get("flags", [])
    if not isinstance(flags, list) or len(flags) > 8 or any(
        not isinstance(f, str) or not re.fullmatch(r"-O[0-3]|-lineinfo|--maxrregcount=[0-9]+", f)
        for f in flags
    ):
        raise Invalid("flags: at most eight -O0..-O3, -lineinfo, --maxrregcount=N")
    return ptx, arch, flags


def run_request(body):
    object_body(body)
    if "ptx" not in body and "cubin_id" not in body:
        return None
    body = {**body, "args": [dict(a) if isinstance(a, dict) else a for a in body.get("args", [])]} if isinstance(body.get("args", []), list) else dict(body)
    for key, value in (("grid", [1, 1, 1]), ("block", [128, 1, 1]), ("warmup", 5), ("iters", 20)):
        body.setdefault(key, value)
    if "cubin_id" in body and (not isinstance(body["cubin_id"], str) or not
                              re.fullmatch(r"[a-f0-9]{64}", body["cubin_id"])):
        raise Invalid("cubin_id must be a SHA256 hex digest")
    if not isinstance(body.get("entry"), str) or not re.fullmatch(r"[A-Za-z_$][\w$]*", body["entry"]):
        raise Invalid("entry must be a PTX entry identifier")
    for field in ("grid", "block"):
        dims = body.get(field)
        if not isinstance(dims, list) or len(dims) != 3:
            raise Invalid(f"{field} must have three dimensions")
        for i, dim in enumerate(dims):
            integer(dim, f"{field}[{i}]", 1, 2**31 - 1 if field == "grid" and i == 0 else 65535 if field == "grid" else 1024)
    if math.prod(body["block"]) > 1024:
        raise Invalid("block has more than 1024 threads")
    integer(body.get("smem_bytes", 0), "smem_bytes", 0, 99 * 1024)
    integer(body.get("warmup", 5), "warmup", 0, 20)
    integer(body.get("iters", 20), "iters", 1, 200)
    integer(body.get("max_return", 4096), "max_return", 0, MAX_RETURN)
    args = body.get("args")
    if not isinstance(args, list) or len(args) > 32:
        raise Invalid("args must be a list of at most 32 positional arguments")
    names, outputs, total = set(), set(), 0
    for arg in args:
        if not isinstance(arg, dict) or not isinstance(arg.get("name"), str) or not arg["name"]:
            raise Invalid("each arg needs a name")
        if arg["name"] in names:
            raise Invalid("argument names must be unique")
        names.add(arg["name"])
        arg.setdefault("dtype", "f32")
        if arg.get("dtype") not in DTYPES:
            raise Invalid("unsupported argument dtype")
        for field in ("scalar", "output"):
            if field in arg and type(arg[field]) is not bool:
                raise Invalid(f"{field} must be boolean")
        if arg.get("scalar"):
            if arg.get("output") or "data" in arg or "seed" in arg:
                raise Invalid("scalar cannot be output, data, or seed")
            if type(arg.get("value")) not in (int, float) or not math.isfinite(arg["value"]):
                raise Invalid("scalar needs a finite numeric value")
            try:
                np.asarray(arg["value"], dtype=DTYPES[arg["dtype"]]).item()
            except (ValueError, OverflowError, TypeError) as exc:
                raise Invalid("scalar does not fit dtype") from exc
            continue
        shape = arg.get("shape")
        if not isinstance(shape, list) or len(shape) > 8:
            raise Invalid("buffer needs shape of at most 8 dimensions")
        for dim in shape:
            integer(dim, "shape dimension", 0, MAX_ELEMENTS)
        size = math.prod(shape)
        total += size
        if total > MAX_ELEMENTS:
            raise Invalid("total buffer elements exceed 2^26")
        if arg.get("output"):
            if "data" in arg or "seed" in arg:
                raise Invalid("output cannot also have data or seed")
            outputs.add(arg["name"])
        elif "data" in arg:
            if "seed" in arg or not isinstance(arg["data"], list):
                raise Invalid("data must be a list and cannot also have seed")
            try:
                data = np.asarray(arg["data"], dtype=DTYPES[arg["dtype"]])
                if data.size != size:
                    raise Invalid("data length does not match shape")
            except (ValueError, TypeError, OverflowError) as exc:
                raise Invalid("data incompatible with dtype or shape") from exc
        elif "seed" in arg:
            integer(arg["seed"], "seed", 0, 2**64 - 1)
        else:
            raise Invalid("input buffer requires seed or data")
    zero = body.get("zero_outputs", [])
    if not isinstance(zero, list) or any(not isinstance(n, str) or n not in outputs for n in zero):
        raise Invalid("zero_outputs must name output buffers")
    return body


def array_for(arg):
    dtype, shape = DTYPES[arg["dtype"]], arg["shape"]
    if arg.get("output"):
        return np.zeros(shape, dtype=dtype)
    if "data" in arg:
        return np.ascontiguousarray(np.asarray(arg["data"], dtype=dtype).reshape(shape))
    return np.ascontiguousarray(np.random.default_rng(arg["seed"]).standard_normal(shape).astype(dtype))


def output_summary(array, arg, limit):
    returned = min(array.size, limit)
    return {"shape": arg["shape"], "dtype": arg["dtype"],
            "sum": float(array.sum()),
            "values": array.reshape(-1)[:returned].tolist(),
            "truncated": returned < array.size}


class Runner:
    def __init__(self, cache, arch="sm_89", ptxas="ptxas"):
        self.cache, self.arch = Path(cache), arch
        self.cache.mkdir(parents=True, exist_ok=True)
        self.ptxas = shutil.which(ptxas) or ptxas
        self.version = subprocess.run([self.ptxas, "--version"], capture_output=True,
                                      text=True, timeout=10, check=True).stdout
        self.compile_lock = threading.Lock()
        self.run_lock = threading.Lock()

    def compile(self, body):
        ptx, arch, flags = compile_request(body, self.arch)
        identity = json.dumps([ptx, arch, self.version, flags], ensure_ascii=True).encode()
        key = hashlib.sha256(identity).hexdigest()
        folder = self.cache / key
        with self.compile_lock:
            metadata = folder / "compile.json"
            if metadata.exists() and (folder / "module.cubin").exists():
                return json.loads(metadata.read_text())
            folder.mkdir(exist_ok=True)
            (folder / "source.ptx").write_text(ptx)
            command = [self.ptxas, f"-arch={arch}", "-v", *flags,
                       str(folder / "source.ptx"), "-o", str(folder / "module.cubin")]
            try:
                done = subprocess.run(command, capture_output=True, text=True, timeout=30)
            except subprocess.TimeoutExpired:
                return {"ok": False, "stage": "ptxas", "stderr": "ptxas exceeded 30 seconds"}
            (folder / "assembly.log").write_text(done.stdout + done.stderr)
            (folder / "command.json").write_text(json.dumps(command))
            if done.returncode:
                return {"ok": False, "stage": "ptxas", "stderr": done.stderr}
            entries = re.findall(r"\.entry\s+([A-Za-z_$][\w$]*)\s*\(", re.sub(r"/\*.*?\*/|//[^\n]*", "", ptx, flags=re.S))
            registers = {}
            current = None
            for line in done.stderr.splitlines():
                match = re.search(r"Function properties for\s+(\S+)", line)
                if match:
                    current = match.group(1)
                usage = re.search(r"Used\s+(\d+)\s+registers", line)
                if usage and current:
                    registers[current] = int(usage.group(1))
            result = {"ok": True, "cubin_id": key, "bytes": (folder / "module.cubin").stat().st_size,
                      "entry_points": list(dict.fromkeys(entries)), "registers": registers}
            metadata.write_text(json.dumps(result))
            return result

    def run(self, body):
        body = run_request(body)
        if body is None:
            return {"ok": False, "stage": "request", "stderr": "give ptx or cubin_id"}
        if "ptx" in body:
            compiled = self.compile(body)
            if not compiled["ok"]:
                return compiled
            body = {**body, "cubin_id": compiled["cubin_id"]}
        cubin = self.cache / body["cubin_id"] / "module.cubin"
        if not cubin.is_file():
            return {"ok": False, "stage": "lookup", "stderr": "unknown cubin_id"}
        if not self.run_lock.acquire(blocking=False):
            return {"ok": False, "stage": "busy", "stderr": "one local launch already active"}
        try:
            with tempfile.TemporaryDirectory(prefix="run-", dir=self.cache) as tmp:
                request, result = Path(tmp) / "request.json", Path(tmp) / "result.json"
                request.write_text(json.dumps({**body, "_cubin": str(cubin), "_parent_pid": os.getpid()}))
                try:
                    done = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                           "--worker", str(request), str(result)],
                                          capture_output=True, text=True, timeout=20)
                except subprocess.TimeoutExpired:
                    return {"ok": False, "stage": "run", "stderr": "worker exceeded 20 seconds"}
                if done.returncode or not result.is_file():
                    return {"ok": False, "stage": "run", "stderr": done.stderr or f"worker exit {done.returncode}"}
                return json.loads(result.read_text())
        finally:
            self.run_lock.release()


def gpu_worker(body):
    """A fresh process owns its entire CUDA context; all errors retire it."""
    # Retire a launch worker if the admitted server is terminated externally.
    # Check ppid after prctl to close the parent-exit race before touching CUDA.
    libc = C.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [C.c_int, C.c_ulong, C.c_ulong, C.c_ulong, C.c_ulong]
    libc.prctl.restype = C.c_int
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0):
        raise OSError(C.get_errno(), "cannot set worker parent-death signal")
    if os.getppid() != body["_parent_pid"]:
        raise RuntimeError("runner parent exited before CUDA initialization")
    driver = C.CDLL("libcuda.so.1")
    P, I, U, D, Z = C.c_void_p, C.c_int, C.c_uint, C.c_uint64, C.c_size_t
    driver.cuGetErrorString.argtypes = [I, C.POINTER(C.c_char_p)]
    driver.cuGetErrorString.restype = I
    def bind(name, args):
        function = getattr(driver, name)
        function.argtypes, function.restype = args, I
        def checked(*values):
            status = function(*values)
            if status:
                error = C.c_char_p()
                driver.cuGetErrorString(I(status), C.byref(error))
                raise RuntimeError(f"{name}: CUDA {status}: {(error.value or b'unknown').decode()}")
        return checked
    bind("cuInit", [U])(0)
    count = I()
    bind("cuDeviceGetCount", [C.POINTER(I)])(C.byref(count))
    if count.value != 1:
        raise RuntimeError(f"expected one admitted visible GPU, found {count.value}")
    device = I()
    bind("cuDeviceGet", [C.POINTER(I), I])(C.byref(device), 0)
    context = P()
    bind("cuCtxCreate_v2", [C.POINTER(P), U, I])(C.byref(context), 0, device)
    # No recovery is attempted inside this process: OS/driver cleanup follows exit.
    module, function = P(), P()
    bind("cuModuleLoad", [C.POINTER(P), C.c_char_p])(C.byref(module), body["_cubin"].encode())
    bind("cuModuleGetFunction", [C.POINTER(P), P, C.c_char_p])(C.byref(function), module, body["entry"].encode())
    smem = body.get("smem_bytes", 0)
    if smem > 48 * 1024:
        bind("cuFuncSetAttribute", [P, I, I])(function, 8, smem)
    alloc = bind("cuMemAlloc_v2", [C.POINTER(D), Z])
    htod = bind("cuMemcpyHtoD_v2", [D, P, Z])
    dtoh = bind("cuMemcpyDtoH_v2", [P, D, Z])
    memset = bind("cuMemsetD8_v2", [D, C.c_ubyte, Z])
    launch = bind("cuLaunchKernel", [P, U, U, U, U, U, U, U, P, P, P])
    sync = bind("cuCtxSynchronize", [])
    values, arrays, buffers = [], {}, {}
    scalar_types = {"f32": C.c_float, "f64": C.c_double, "i32": C.c_int32,
                    "i64": C.c_int64, "f16": C.c_uint16}
    for arg in body["args"]:
        if arg.get("scalar"):
            value = arg["value"]
            if arg["dtype"] == "f16":
                value = np.asarray(value, dtype=np.float16).view(np.uint16).item()
            values.append(scalar_types[arg["dtype"]](value))
        else:
            array, pointer = array_for(arg), D()
            alloc(C.byref(pointer), max(1, array.nbytes))
            if array.nbytes:
                htod(pointer, array.ctypes.data, array.nbytes)
            arrays[arg["name"]], buffers[arg["name"]] = array, pointer
            values.append(pointer)
    params = (P * len(values))(*(C.cast(C.pointer(value), P) for value in values))
    event_create = bind("cuEventCreate", [C.POINTER(P), U])
    record = bind("cuEventRecord", [P, P])
    event_sync = bind("cuEventSynchronize", [P])
    elapsed = bind("cuEventElapsedTime", [C.POINTER(C.c_float), P, P])
    start, stop = P(), P()
    event_create(C.byref(start), 0)
    event_create(C.byref(stop), 0)
    times = []
    warmup, iters = body.get("warmup", 5), body.get("iters", 20)
    for rep in range(warmup + iters):
        for name in body.get("zero_outputs", []):
            if arrays[name].nbytes:
                memset(buffers[name], 0, arrays[name].nbytes)
        record(start, None)
        launch(function, *body["grid"], *body["block"], smem, None, params, None)
        record(stop, None)
        event_sync(stop)
        milliseconds = C.c_float()
        elapsed(C.byref(milliseconds), start, stop)
        if rep >= warmup:
            times.append(milliseconds.value * 1000)
    sync()
    outputs, remaining = {}, body.get("max_return", 4096)
    for arg in body["args"]:
        if arg.get("output"):
            name, array = arg["name"], arrays[arg["name"]]
            if array.nbytes:
                dtoh(array.ctypes.data, buffers[name], array.nbytes)
            outputs[name] = output_summary(array, arg, remaining)
            remaining -= min(array.size, remaining)
    return {"ok": True, "median_us": statistics.median(times), "iters": iters,
            "outputs": outputs, "timing_method": "CUDA events per launch, excludes reset and copies; indicative",
            "samples_us": times}


def handler_for(runner):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            if self.path != "/health":
                return self.reply(404, {"ok": False, "stage": "request", "stderr": "unknown path"})
            self.reply(200, {"ok": True, "can_compile": True, "can_run": True,
                             "ptxas": runner.ptxas, "default_arch": runner.arch,
                             "gpu": "admitted CUDA device 0; verified on first run",
                             "implementation": "local research runner; not exact hosted-backend parity"})
        def do_POST(self):
            if self.path not in ("/compile", "/run"):
                return self.reply(404, {"ok": False, "stage": "request", "stderr": "unknown path"})
            try:
                length = integer(int(self.headers.get("Content-Length", "0")), "body bytes", 1, MAX_BODY)
                self.connection.settimeout(10)
                body = json.loads(self.rfile.read(length))
                result = runner.compile(body) if self.path == "/compile" else runner.run(body)
                self.reply(200, result)
            except (Invalid, ValueError, TypeError, OverflowError) as exc:
                self.reply(422, {"ok": False, "stage": "request", "stderr": str(exc)})
            except Exception as exc:
                self.reply(500, {"ok": False, "stage": "internal", "stderr": str(exc)})
    return Handler


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        try:
            result = gpu_worker(json.loads(Path(sys.argv[2]).read_text()))
        except Exception as exc:
            result = {"ok": False, "stage": "run", "stderr": str(exc)}
        Path(sys.argv[3]).write_text(json.dumps(result))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--port", type=int, default=48702)
    parser.add_argument("--arch", default="sm_89")
    parser.add_argument("--ptxas", default="ptxas")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(Runner(args.cache, args.arch, args.ptxas)))
    print(f"Local runner at http://127.0.0.1:{server.server_port}; no GPU admission performed here", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
