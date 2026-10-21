#!/usr/bin/env python3
"""Bounded CUDA-driver comparison worker; run only inside an admitted GPU job.

No admission, compilation, Git, HTTP, or background service is performed here.
A CUDA error retires this process. Concurrent correctness is screening only:
callers must isolate every survivor before promotion and retry all batch-mates
when a concurrent discrepancy disappears in isolation. JSON job protocol is
specified in ``run_job``; generated case directories contain meta.json,
inputs.npz (optional for published seeded cases), and expected.npz.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np

DTYPES = {"f32": np.float32, "f16": np.float16, "f64": np.float64,
          "i32": np.int32, "i64": np.int64}
SCALARS = {"f32": C.c_float, "f16": C.c_uint16, "f64": C.c_double,
           "i32": C.c_int32, "i64": C.c_int64}
GUARD_BYTES = 256
MAX_ELEMENTS = 2**26
P, I, U, D, Z = C.c_void_p, C.c_int, C.c_uint, C.c_uint64, C.c_size_t


class CUDAError(RuntimeError):
    """An error after CUDA initialization: no same-context recovery is safe."""


def bounded_int(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def guard_image(payload, seed, guard_bytes=GUARD_BYTES):
    """CPU-visible image with reproducible, buffer-specific random canaries."""
    bounded_int(guard_bytes, "guard bytes", GUARD_BYTES, 2**20)
    if guard_bytes % 256:
        raise ValueError("guard bytes must preserve 256-byte pointer alignment")
    payload = bytes(payload)
    rng = np.random.default_rng(seed)
    prefix = rng.integers(0, 256, guard_bytes, dtype=np.uint8).tobytes()
    suffix = rng.integers(0, 256, guard_bytes, dtype=np.uint8).tobytes()
    return prefix + payload + suffix


def inspect_guard_image(actual, initial, payload_size, immutable=False, guard_bytes=GUARD_BYTES):
    actual, initial = bytes(actual), bytes(initial)
    size = payload_size + 2 * guard_bytes
    if len(actual) != size or len(initial) != size:
        raise ValueError("guard image length mismatch")
    return {
        "prefix_ok": actual[:guard_bytes] == initial[:guard_bytes],
        "suffix_ok": actual[guard_bytes + payload_size:] == initial[guard_bytes + payload_size:],
        "input_ok": not immutable or actual[guard_bytes:guard_bytes + payload_size]
            == initial[guard_bytes:guard_bytes + payload_size],
    }


def compare_output(got, expected, rtol, atol):
    """Check every value; integer outputs never receive floating tolerances."""
    got, expected = np.asarray(got), np.asarray(expected)
    if got.shape != expected.shape:
        return {"ok": False, "error": "output shape differs from reference",
                "shape": list(got.shape), "expected_shape": list(expected.shape)}
    if got.dtype.kind in "iu":
        same = got == expected
    else:
        same = np.isclose(got, expected, rtol=rtol, atol=atol, equal_nan=False)
    error = np.abs(got.astype(np.float64) - expected.astype(np.float64))
    maximum = float(np.max(error)) if error.size else 0.0
    return {"ok": bool(np.all(same)), "elements": int(got.size),
            "mismatched": int(np.count_nonzero(~same)),
            "max_abs_error": maximum if math.isfinite(maximum) else None,
            "rtol": rtol, "atol": atol}


def quarantine_batch(results):
    """A single bad result invalidates trust in every shared-context batchmate."""
    bad = any(not row.get("ok", False) for row in results)
    for row in results:
        row["quarantined"] = bad
        row["trusted"] = bool(row.get("ok")) and not bad
    return bad


def paired_orders(seed, pairs, offset=0):
    rng = random.Random(seed)
    orders = []
    for index in range(offset + pairs):
        order = ["baseline", "candidate"]
        rng.shuffle(order)
        if index >= offset:
            orders.append(order)
    return orders


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_case(spec, cubin_field="cubins"):
    """Validate independent case artifacts without opening CUDA."""
    folder = Path(spec["case_dir"]).resolve()
    meta = json.loads((folder / "meta.json").read_text())
    for key in ("rtol", "atol"):
        value = meta.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"case requires fixed finite nonnegative {key}")
    kernels = meta.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        raise ValueError("case requires ordered kernels")
    paths = spec[cubin_field]
    if not isinstance(paths, list) or len(paths) != len(kernels):
        raise ValueError("cubins must align with meta.kernels")
    paths = [Path(value) for value in paths]
    if any(not value.is_absolute() or not value.is_file() for value in paths):
        raise ValueError("cubin paths must be absolute existing files")
    if (folder / "inputs.npz").is_file():
        with np.load(folder / "inputs.npz", allow_pickle=False) as archive:
            inputs = {name: np.array(archive[name], copy=True) for name in archive.files}
    else:
        inputs = {}
    with np.load(folder / "expected.npz", allow_pickle=False) as archive:
        expected = {name: np.array(archive[name], copy=True) for name in archive.files}
    if not expected:
        raise ValueError("expected.npz must contain at least one checked output")
    definitions, arrays, scalars, immutable, produced = {}, {}, {}, set(), set()
    total = 0
    for kernel in kernels:
        args = kernel["args"]
        names = [arg["name"] for arg in args]
        if len(set(names)) != len(names) or set(names) != set(kernel["arg_order"]):
            raise ValueError("arg_order must list each positional argument exactly once")
        if len(kernel["arg_order"]) != len(names):
            raise ValueError("arg_order contains duplicate names")
        for field in ("grid", "block"):
            dims = kernel[field]
            if not isinstance(dims, list) or len(dims) != 3:
                raise ValueError(f"{field} requires three dimensions")
            for axis, value in enumerate(dims):
                upper = (2**31 - 1 if axis == 0 else 65535) if field == "grid" else 1024
                bounded_int(value, field, 1, upper)
        if math.prod(kernel["block"]) > 1024:
            raise ValueError("block exceeds 1024 threads")
        bounded_int(kernel.get("smem_bytes", 0), "smem_bytes", 0, 99 * 1024)
        output_names = {arg["name"] for arg in args if arg.get("output")}
        if any(name not in output_names for name in kernel.get("zero_outputs", [])):
            raise ValueError("zero_outputs must name this kernel's output buffers")
        for arg in args:
            name, dtype = arg["name"], arg.get("dtype", "f32")
            if dtype not in DTYPES:
                raise ValueError(f"unsupported dtype {dtype}")
            if arg.get("scalar"):
                if name in arrays or arg.get("output"):
                    raise ValueError("scalar/buffer argument conflict")
                definition = (dtype, arg["value"])
                if name in scalars and scalars[name] != definition:
                    raise ValueError("scalar definition differs across pipeline")
                if not math.isfinite(float(arg["value"])):
                    raise ValueError("nonfinite scalar")
                scalars[name] = definition
                continue
            shape = arg["shape"]
            if not isinstance(shape, list) or len(shape) > 8:
                raise ValueError("invalid buffer shape")
            for dim in shape:
                bounded_int(dim, "shape dimension", 0, MAX_ELEMENTS)
            definition = (dtype, tuple(shape))
            if arg.get("intermediate") and name not in produced:
                raise ValueError(f"intermediate {name} has no earlier producer")
            if name in definitions:
                if definitions[name] != definition:
                    raise ValueError("buffer dtype/shape differs across pipeline")
                continue
            if name in scalars:
                raise ValueError("scalar/buffer argument conflict")
            total += math.prod(shape)
            if total > MAX_ELEMENTS:
                raise ValueError("case exceeds 2^26 buffer elements")
            definitions[name] = definition
            if arg.get("output"):
                array = np.zeros(shape, dtype=DTYPES[dtype])
            elif name in inputs:
                array = inputs[name]
                if array.shape != tuple(shape) or array.dtype != np.dtype(DTYPES[dtype]):
                    raise ValueError(f"input {name} dtype/shape differs from metadata")
                immutable.add(name)
            elif arg.get("intermediate") or name in produced:
                raise ValueError(f"intermediate {name} has no earlier producer")
            elif "data" in arg:
                array = np.asarray(arg["data"], dtype=DTYPES[dtype]).reshape(shape)
                immutable.add(name)
            elif "seed" in arg:
                array = np.random.default_rng(arg["seed"]).standard_normal(shape).astype(DTYPES[dtype])
                immutable.add(name)
            else:
                raise ValueError(f"input {name} missing from inputs.npz and metadata")
            arrays[name] = np.ascontiguousarray(array)
        # Reject writes into buffers that were declared immutable external inputs.
        if output_names & immutable:
            raise ValueError("in-place mutation of external inputs is not supported")
        produced.update(output_names)
    for name, ref in expected.items():
        if name not in arrays or name not in produced:
            raise ValueError(f"reference {name} is not a produced buffer")
        if ref.shape != arrays[name].shape:
            # Published scalar reductions store a NumPy scalar but expose a
            # one-element output buffer. Keep every nonsingleton shape strict.
            if ref.size == arrays[name].size == 1:
                expected[name] = ref.reshape(arrays[name].shape)
            else:
                raise ValueError(f"reference {name} shape differs from metadata")
    final = set(meta.get("graph_outputs", expected))
    if not final.issubset(expected):
        raise ValueError("expected.npz omits final outputs")
    identities = {}
    for path in [folder / "meta.json", folder / "expected.npz", folder / "inputs.npz", *paths]:
        if path.is_file():
            identities[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"spec": spec, "meta": meta, "paths": paths, "arrays": arrays,
            "scalars": scalars, "immutable": immutable, "expected": expected,
            "identities": identities}


class Driver:
    def __init__(self):
        self.lib = C.CDLL("libcuda.so.1")
        error_string = self.lib.cuGetErrorString
        error_string.argtypes, error_string.restype = [I, C.POINTER(C.c_char_p)], I
        self.functions = {}
        def bind(name, args):
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, I
            def checked(*values):
                status = function(*values)
                if status:
                    description = C.c_char_p()
                    error_string(status, C.byref(description))
                    raise CUDAError(f"{name}: CUDA {status}: {(description.value or b'unknown').decode()}")
            self.functions[name] = checked
            return checked
        self.init = bind("cuInit", [U])
        self.count = bind("cuDeviceGetCount", [C.POINTER(I)])
        self.device_get = bind("cuDeviceGet", [C.POINTER(I), I])
        self.ctx_create = bind("cuCtxCreate_v2", [C.POINTER(P), U, I])
        self.ctx_destroy = bind("cuCtxDestroy_v2", [P])
        self.alloc = bind("cuMemAlloc_v2", [C.POINTER(D), Z])
        self.free = bind("cuMemFree_v2", [D])
        self.host_alloc = bind("cuMemHostAlloc", [C.POINTER(P), Z, U])
        self.host_free = bind("cuMemFreeHost", [P])
        self.htod = bind("cuMemcpyHtoDAsync_v2", [D, P, Z, P])
        self.dtoh = bind("cuMemcpyDtoHAsync_v2", [P, D, Z, P])
        self.memset = bind("cuMemsetD8Async", [D, C.c_ubyte, Z, P])
        self.stream_create = bind("cuStreamCreate", [C.POINTER(P), U])
        self.stream_sync = bind("cuStreamSynchronize", [P])
        self.stream_destroy = bind("cuStreamDestroy_v2", [P])
        self.module_load = bind("cuModuleLoad", [C.POINTER(P), C.c_char_p])
        self.module_unload = bind("cuModuleUnload", [P])
        self.function_get = bind("cuModuleGetFunction", [C.POINTER(P), P, C.c_char_p])
        self.function_attr = bind("cuFuncSetAttribute", [P, I, I])
        self.function_get_attr = bind("cuFuncGetAttribute", [C.POINTER(I), I, P])
        self.occupancy = bind("cuOccupancyMaxActiveBlocksPerMultiprocessor", [C.POINTER(I), P, I, Z])
        self.launch = bind("cuLaunchKernel", [P, U, U, U, U, U, U, U, P, P, P])
        self.event_create = bind("cuEventCreate", [C.POINTER(P), U])
        self.event_record = bind("cuEventRecord", [P, P])
        self.event_record_external = bind("cuEventRecordWithFlags", [P, P, U])
        self.capture_begin = bind("cuStreamBeginCapture_v2" if hasattr(self.lib, "cuStreamBeginCapture_v2") else "cuStreamBeginCapture", [P, I])
        self.capture_end = bind("cuStreamEndCapture", [P, C.POINTER(P)])
        self.graph_instantiate = bind("cuGraphInstantiate_v2", [C.POINTER(P), P, C.POINTER(P), P, Z])
        self.graph_launch = bind("cuGraphLaunch", [P, P])
        self.graph_exec_destroy = bind("cuGraphExecDestroy", [P])
        self.graph_destroy = bind("cuGraphDestroy", [P])
        self.event_sync = bind("cuEventSynchronize", [P])
        self.event_elapsed = bind("cuEventElapsedTime", [C.POINTER(C.c_float), P, P])
        self.event_destroy = bind("cuEventDestroy_v2", [P])
        self.init(0)
        count = I()
        self.count(C.byref(count))
        if count.value != 1:
            raise ValueError(f"expected exactly one admitted visible device, found {count.value}")
        self.device, self.context = I(), P()
        self.device_get(C.byref(self.device), 0)
        self.ctx_create(C.byref(self.context), 0, self.device)
        self.snapshot = {}
        name = C.create_string_buffer(256)
        bind("cuDeviceGetName", [P, I, I])(name, len(name), self.device)
        self.snapshot["name"] = name.value.decode()
        self.snapshot["worker_pid"] = os.getpid()
        self.snapshot["context_identity"] = f"{os.getpid()}:{self.context.value}"
        version = I()
        bind("cuDriverGetVersion", [C.POINTER(I)])(C.byref(version))
        self.snapshot["driver_version"] = version.value
        attribute = bind("cuDeviceGetAttribute", [C.POINTER(I), I, I])
        for label, enum in (("clock_rate_khz", 13), ("multiprocessors", 16),
                            ("memory_clock_khz", 36), ("compute_major", 75), ("compute_minor", 76)):
            value = I()
            attribute(C.byref(value), enum, self.device)
            self.snapshot[label] = value.value
        uuid = (C.c_ubyte * 16)()
        bind("cuDeviceGetUuid_v2", [P, I])(uuid, self.device)
        encoded = bytes(uuid).hex()
        self.snapshot["uuid"] = "GPU-" + "-".join((encoded[:8], encoded[8:12], encoded[12:16], encoded[16:20], encoded[20:]))
        self.snapshot["telemetry_note"] = "Driver attributes are nominal limits, not live clocks or temperature; host wrapper retains admission telemetry."

    def close(self):
        self.ctx_destroy(self.context)


class Buffer:
    def __init__(self, driver, array, stream, immutable, seed):
        self.driver, self.array, self.immutable = driver, array, immutable
        self.initial = guard_image(array.tobytes(), seed)
        self.size = len(self.initial)
        self.base, self.host = D(), P()
        driver.alloc(C.byref(self.base), self.size)
        driver.host_alloc(C.byref(self.host), self.size, 0)
        C.memmove(self.host, self.initial, self.size)
        self.pointer = D(self.base.value + GUARD_BYTES)
        driver.htod(self.base, self.host, self.size, stream)

    def readback(self, stream):
        self.driver.dtoh(self.host, self.base, self.size, stream)

    def inspect(self):
        image = C.string_at(self.host, self.size)
        guards = inspect_guard_image(image, self.initial, self.array.nbytes, self.immutable)
        payload = image[GUARD_BYTES:GUARD_BYTES + self.array.nbytes]
        array = np.frombuffer(payload, dtype=self.array.dtype).reshape(self.array.shape)
        return guards, array

    def close(self):
        self.driver.free(self.base)
        self.driver.host_free(self.host)


class Case:
    def __init__(self, driver, data, arm="candidate"):
        self.driver, self.data, self.arm = driver, data, arm
        self.stream = P()
        driver.stream_create(C.byref(self.stream), 1)  # CU_STREAM_NON_BLOCKING
        self.buffers, self.modules, self.calls, self.events = {}, [], [], []
        self.resources = []
        spec = data["spec"]
        self.id = str(spec["id"])
        for name, array in data["arrays"].items():
            seed = int.from_bytes(hashlib.sha256(
                f"{self.id}\0{spec.get('variant', '')}\0{arm}\0{name}".encode()).digest()[:8], "little")
            self.buffers[name] = Buffer(driver, array, self.stream, name in data["immutable"], seed)
        for kernel, path in zip(data["meta"]["kernels"], data["paths"]):
            module, function = P(), P()
            driver.module_load(C.byref(module), str(path).encode())
            self.modules.append(module)
            driver.function_get(C.byref(function), module, kernel["entry"].encode())
            smem = kernel.get("smem_bytes", 0)
            if smem > 48 * 1024:
                driver.function_attr(function, 8, smem)
            resources = {"entry": kernel["entry"], "dynamic_shared_bytes": smem,
                         "block_threads": math.prod(kernel["block"])}
            for name, enum in (("max_threads_per_block", 0), ("static_shared_bytes", 1),
                               ("local_bytes", 3), ("registers", 4)):
                value = I()
                driver.function_get_attr(C.byref(value), enum, function)
                resources[name] = value.value
            blocks = I()
            driver.occupancy(C.byref(blocks), function, resources["block_threads"], smem)
            resources["max_active_blocks_per_sm"] = blocks.value
            resources["note"] = "Driver occupancy/resource feasibility, not a throughput prediction."
            self.resources.append(resources)
            if blocks.value == 0 or resources["block_threads"] > resources["max_threads_per_block"]:
                raise ValueError(f"launch resources infeasible for {kernel['entry']}: {resources}")
            values = []
            for name in kernel["arg_order"]:
                if name in data["scalars"]:
                    dtype, value = data["scalars"][name]
                    if dtype == "f16":
                        value = np.asarray(value, dtype=np.float16).view(np.uint16).item()
                    values.append(SCALARS[dtype](value))
                else:
                    values.append(self.buffers[name].pointer)
            params = (P * len(values))(*(C.cast(C.pointer(value), P) for value in values))
            start, stop = P(), P()
            driver.event_create(C.byref(start), 0)
            driver.event_create(C.byref(stop), 0)
            self.events.extend((start, stop))
            self.calls.append((kernel, function, params, values, start, stop))
        # Setup synchronization happens before launching any member of a cohort.
        driver.stream_sync(self.stream)
        self.launch_count = 0

    def launch(self, timed=False, stream=None, event_pairs=None, captured=False):
        stream = stream if stream is not None else self.stream
        for index, (kernel, function, params, _values, start, stop) in enumerate(self.calls):
            if event_pairs is not None:
                start, stop = event_pairs[index]
            for name in kernel.get("zero_outputs", []):
                buffer = self.buffers[name]
                if buffer.array.nbytes:
                    self.driver.memset(buffer.pointer, 0, buffer.array.nbytes, stream)
            if timed:
                if captured:
                    self.driver.event_record_external(start, stream, 1)  # explicit event node
                else:
                    self.driver.event_record(start, stream)
            self.driver.launch(function, *kernel["grid"], *kernel["block"],
                               kernel.get("smem_bytes", 0), stream, params, None)
            self.launch_count += 1
            if timed:
                if captured:
                    self.driver.event_record_external(stop, stream, 1)
                else:
                    self.driver.event_record(stop, stream)

    def readback(self):
        for buffer in self.buffers.values():
            buffer.readback(self.stream)

    def synchronize(self):
        self.driver.stream_sync(self.stream)

    def inspect(self):
        guards, outputs = {}, {}
        for name, buffer in self.buffers.items():
            checks, array = buffer.inspect()
            guards[name] = checks
            if name in self.data["expected"]:
                outputs[name] = compare_output(array, self.data["expected"][name],
                                              self.data["meta"]["rtol"], self.data["meta"]["atol"])
        good = all(all(check.values()) for check in guards.values()) and all(
            value["ok"] for value in outputs.values())
        return {"ok": good, "buffers": guards, "outputs": outputs}

    def elapsed_us(self):
        times = []
        for _kernel, _function, _params, _values, start, stop in self.calls:
            milliseconds = C.c_float()
            self.driver.event_elapsed(C.byref(milliseconds), start, stop)
            times.append(milliseconds.value * 1000)
        return sum(times), times

    def close(self):
        self.synchronize()
        for event in self.events:
            self.driver.event_destroy(event)
        for module in self.modules:
            self.driver.module_unload(module)
        for buffer in self.buffers.values():
            buffer.close()
        self.driver.stream_destroy(self.stream)


def correctness(driver, job, prepared):
    width = bounded_int(job.get("width", 1), "width", 1, 4)
    if width not in (1, 2, 4):
        raise ValueError("width must be one, two, or four")
    repetitions = bounded_int(job.get("repetitions", 3), "repetitions", 1, 200)
    batches, all_ok, launch_count = [], True, 0
    for offset in range(0, len(prepared), width):
        subset = prepared[offset:offset + width]
        cases = [Case(driver, data) for data in subset]
        membership = [{"id": case.id, "variant": case.data["spec"].get("variant")} for case in cases]
        rows = [{**member, "ok": True, "checks": [], "resources": case.resources}
                for member, case in zip(membership, cases)]
        batch_start = time.monotonic()
        try:
            for rep in range(repetitions):
                for case in cases:
                    case.launch()
                    case.readback()
                for case, row in zip(cases, rows):
                    case.synchronize()
                    check = case.inspect()
                    row["checks"].append({"repetition": rep, **check})
                    row["ok"] = row["ok"] and check["ok"]
                if not all(row["ok"] for row in rows):
                    break  # Never continue launching a known-corrupted cohort.
            bad = quarantine_batch(rows)
            all_ok = all_ok and not bad
            batches.append({"batch_id": len(batches), "membership": membership,
                            "quarantined": bad, "results": rows,
                            "wall_seconds": time.monotonic() - batch_start,
                            "requires_all_batchmates_isolated": bad,
                            "isolated": len(cases) == 1})
            launch_count += sum(case.launch_count for case in cases)
        finally:
            # CUDA exceptions deliberately skip same-context cleanup/recovery.
            if sys.exc_info()[0] is None:
                for case in cases:
                    case.close()
    return {"ok": all_ok, "batches": batches, "launch_count": launch_count,
            "width": width, "repetitions": repetitions,
            "promotion_eligible": False,
            "note": "Concurrent results are screening only; fresh isolated confirmation is mandatory."}


def legacy_timing(driver, job, candidate_data, baseline_data):
    pairs = bounded_int(job.get("pairs", 8), "pairs", 1, 200)
    warmup = bounded_int(job.get("warmup", 5), "warmup", 0, 20)
    offset = bounded_int(job.get("pair_offset", 0), "pair_offset", 0, 2**31 - 1)
    seed = bounded_int(job.get("seed", 0), "seed", 0, 2**64 - 1)
    burst_id = str(job.get("burst_id", "0"))
    arms = {"candidate": Case(driver, candidate_data),
            "baseline": Case(driver, baseline_data, "baseline")}
    records, checks = [], []
    ok = True
    try:
        for rep in range(warmup):
            for label in ("baseline", "candidate"):
                case = arms[label]
                case.launch()
                case.readback()
                case.synchronize()
                check = case.inspect()
                checks.append({"phase": "warmup", "repetition": rep, "arm": label, **check})
                ok = ok and check["ok"]
            if not ok:
                break
        if ok:
            for pair_index, order in enumerate(paired_orders(seed, pairs, offset), offset):
                pair = {"pair_index": pair_index, "burst_id": burst_id,
                        "order": order, "samples_us": {}, "kernel_samples_us": {}, "complete": False}
                for label in order:
                    case = arms[label]
                    case.launch(timed=True)
                    case.readback()
                    case.synchronize()
                    total, components = case.elapsed_us()
                    pair["samples_us"][label] = total
                    pair["kernel_samples_us"][label] = components
                    check = case.inspect()
                    checks.append({"phase": "timing", "pair_index": pair_index,
                                   "arm": label, **check})
                    ok = ok and check["ok"]
                    if not ok:
                        break
                pair["complete"] = ok and len(pair["samples_us"]) == 2
                if pair["complete"]:
                    baseline = pair["samples_us"]["baseline"]
                    pair["candidate_over_baseline"] = pair["samples_us"]["candidate"] / baseline if baseline > 0 else None
                records.append(pair)
                # Preserve complete blocks if the admitted parent times out later.
                checkpoint = {"ok": False, "incomplete": True, "mode": "timing",
                              "pairs": records, "checks": checks, "burst_id": burst_id,
                              "device": driver.snapshot,
                              "identities": {"candidate": candidate_data["identities"], "baseline": baseline_data["identities"]}}
                atomic_json(str(job["out"]) + ".partial.json", checkpoint)
                if not ok:
                    break
        return {"ok": ok, "pairs": records, "checks": checks, "burst_id": burst_id,
                "pair_offset": offset, "completed_pairs": sum(pair["complete"] for pair in records),
                "launch_count": sum(case.launch_count for case in arms.values()),
                "timing_method": "Sum of per-kernel CUDA event intervals; excludes resets/copies; no other candidate work overlaps.",
                "cache_mode": "warm: same allocations reused; no streaming-throughput claim"}
    finally:
        if sys.exc_info()[0] is None:
            for case in arms.values():
                case.close()


def timing(driver, job, candidate_data, baseline_data):
    """Capture the paired block as explicit event/kernel nodes, then replay once.

    Reset memsets precede *every* physical kernel launch in the graph and remain
    outside its per-kernel timing intervals. Distinct events retain each pair;
    there is no host Python submission gap between graph event/kernel nodes.
    """
    if job.get("timing_protocol") == "legacy-single-launch-v1":
        result = legacy_timing(driver, job, candidate_data, baseline_data)
        result["timing_protocol"] = "legacy-single-launch-v1"
        result["promotion_timing_eligible"] = False
        return result
    if job.get("timing_protocol", "cuda-graph-paired-v2") != "cuda-graph-paired-v2":
        raise ValueError("unknown timing protocol")
    pairs = bounded_int(job.get("pairs", 8), "pairs", 1, 200)
    warmup = bounded_int(job.get("warmup", 5), "warmup", 0, 20)
    offset = bounded_int(job.get("pair_offset", 0), "pair_offset", 0, 2**31 - 1)
    seed = bounded_int(job.get("seed", 0), "seed", 0, 2**64 - 1)
    burst_id = str(job.get("burst_id", "0"))
    arms = {"candidate": Case(driver, candidate_data),
            "baseline": Case(driver, baseline_data, "baseline")}
    records, checks, allocated_events = [], [], []
    graph, executable, stream = P(), P(), P()
    driver.stream_create(C.byref(stream), 1)
    ok = True
    try:
        for rep in range(warmup):
            for label in ("baseline", "candidate"):
                case = arms[label]
                case.launch()
                case.readback()
                case.synchronize()
                check = case.inspect()
                checks.append({"phase": "warmup", "repetition": rep, "arm": label, **check})
                ok = ok and check["ok"]
            if not ok:
                break
        if ok:
            plan = []
            for pair_index, order in enumerate(paired_orders(seed, pairs, offset), offset):
                events = {}
                for label in order:
                    events[label] = []
                    for _call in arms[label].calls:
                        start, stop = P(), P()
                        driver.event_create(C.byref(start), 0)
                        driver.event_create(C.byref(stop), 0)
                        allocated_events.extend((start, stop))
                        events[label].append((start, stop))
                plan.append((pair_index, order, events))
            atomic_json(str(job["out"]) + ".partial.json", {
                "ok": False, "incomplete": True, "pairs": [], "burst_id": burst_id,
                "timing_protocol": "cuda-graph-paired-v2", "planned_pairs": pairs,
                "note": "An interrupted block contributes no paired comparison evidence."})
            driver.capture_begin(stream, 1)  # CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
            for _pair_index, order, events in plan:
                for label in order:
                    arms[label].launch(timed=True, stream=stream, event_pairs=events[label], captured=True)
            driver.capture_end(stream, C.byref(graph))
            driver.graph_instantiate(C.byref(executable), graph, None, None, 0)
            driver.graph_launch(executable, stream)
            driver.stream_sync(stream)
            # Full validation lies outside every timed interval. Separate repeated
            # correctness screening checks intermediate iterations before timing.
            for label, case in arms.items():
                case.readback()
                case.synchronize()
                check = case.inspect()
                checks.append({"phase": "after-paired-block", "arm": label, **check})
                ok = ok and check["ok"]
            for pair_index, order, events in plan:
                samples, components = {}, {}
                for label in order:
                    components[label] = []
                    for start, stop in events[label]:
                        milliseconds = C.c_float()
                        driver.event_elapsed(C.byref(milliseconds), start, stop)
                        components[label].append(milliseconds.value * 1000)
                    samples[label] = sum(components[label])
                baseline = samples["baseline"]
                records.append({"pair_index": pair_index, "burst_id": burst_id, "order": order,
                                "samples_us": samples, "kernel_samples_us": components,
                                "complete": ok,
                                "candidate_over_baseline": samples["candidate"] / baseline if ok and baseline > 0 else None})
            atomic_json(str(job["out"]) + ".partial.json", {
                "ok": False, "incomplete": True, "mode": "timing", "pairs": records,
                "checks": checks, "burst_id": burst_id, "device": driver.snapshot,
                "timing_protocol": "cuda-graph-paired-v2",
                "identities": {"candidate": candidate_data["identities"], "baseline": baseline_data["identities"]}})
        return {"ok": ok, "pairs": records, "checks": checks, "burst_id": burst_id,
                "pair_offset": offset, "completed_pairs": sum(pair["complete"] for pair in records),
                "launch_count": sum(case.launch_count for case in arms.values()),
                "physical_timed_launches": pairs * sum(len(case.calls) for case in arms.values()) if records else 0,
                "timing_protocol": "cuda-graph-paired-v2",
                "resources": {label: case.resources for label, case in arms.items()},
                "timing_method": "Explicit captured CUDA event nodes around each kernel; paired block replayed once; resets/copies excluded.",
                "validation_scope": "Every warmup and after the complete paired block; use independent repeated correctness checks before timing.",
                "cache_mode": "warm: same allocations reused; no streaming-throughput claim"}
    finally:
        if sys.exc_info()[0] is None:
            if executable.value:
                driver.graph_exec_destroy(executable)
            if graph.value:
                driver.graph_destroy(graph)
            for event in allocated_events:
                driver.event_destroy(event)
            driver.stream_destroy(stream)
            for case in arms.values():
                case.close()


def run_job(job, driver=None):
    """Execute correctness or paired timing protocol (only this function loads CUDA).

    correctness: {mode,cases:[{id,variant,case_dir,cubins}],width,repetitions,out}
    timing: {mode,case:{id,variant,case_dir,cubins,baseline_cubins},pairs,warmup,
             seed,burst_id,pair_offset?:0,out}
    Each cubin list is absolute and aligned with meta.kernels. Parent owns GPU
    admission, per-job wall timeout, queue fairness and isolated rechecks.
    """
    started = time.monotonic()
    mode = job["mode"]
    if mode == "correctness":
        width = bounded_int(job.get("width", 1), "width", 1, 4)
        if width not in (1, 2, 4):
            raise ValueError("width must be one, two, or four")
        bounded_int(job.get("repetitions", 3), "repetitions", 1, 200)
        specs = job["cases"]
        if not isinstance(specs, list) or not 1 <= len(specs) <= 8:
            raise ValueError("correctness requires one to eight cases")
        prepared = [load_case(spec) for spec in specs]
    elif mode == "timing":
        bounded_int(job.get("pairs", 8), "pairs", 1, 200)
        bounded_int(job.get("warmup", 5), "warmup", 0, 20)
        bounded_int(job.get("pair_offset", 0), "pair_offset", 0, 2**31 - 1)
        bounded_int(job.get("seed", 0), "seed", 0, 2**64 - 1)
        prepared = [load_case(job["case"]), load_case(job["case"], "baseline_cubins")]
    else:
        raise ValueError("mode must be correctness or timing")
    preparation = time.monotonic() - started
    owns_driver = driver is None
    driver = driver if driver is not None else Driver()
    cuda_ready = time.monotonic()
    try:
        result = correctness(driver, job, prepared) if mode == "correctness" else timing(driver, job, *prepared)
        result.update({"mode": mode, "device": driver.snapshot,
                       "benchmark_qualified": False,
                       "qualification_note": "Raw diagnostics; benchmark validity requires evaluator-issued source/suite all-15 gates.",
                       "phase_seconds": {"case_loading": preparation,
                                         "cuda_initialization": cuda_ready - started - preparation,
                                         "execution_and_checks": time.monotonic() - cuda_ready,
                                         "total": time.monotonic() - started},
                       "identities": [data["identities"] for data in prepared]})
        return result
    finally:
        if owns_driver and sys.exc_info()[0] is None:
            driver.close()


def parent_death_signal():
    if not sys.platform.startswith("linux"):
        return
    parent = os.getppid()
    libc = C.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [I, C.c_ulong, C.c_ulong, C.c_ulong, C.c_ulong]
    libc.prctl.restype = I
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0):
        raise OSError(C.get_errno(), "cannot install parent-death signal")
    if os.getppid() != parent:
        raise RuntimeError("parent exited before CUDA initialization")


def serve():
    """JSON-lines control channel; context persists until EOF or worker error.

    Every line is the same job object as the one-shot CLI. Results go atomically
    to job.out; stdout carries only a compact acknowledgement. The admitted
    caller owns read/write timeouts and must kill/retire this process on timeout.
    A failed result is not a promotion receipt or an isolated fresh-context test.
    """
    parent_death_signal()
    driver = Driver()
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            job = None
            try:
                job = json.loads(line)
                result = run_job(job, driver=driver)
                atomic_json(job["out"], result)
                acknowledgement = {"ok": result["ok"], "out": str(job["out"]),
                                   "status": 0 if result["ok"] else 1}
                print(json.dumps(acknowledgement), flush=True)
            except Exception as exc:
                result = {"ok": False, "stage": "cuda" if isinstance(exc, CUDAError) else "worker",
                          "error": str(exc), "retire_worker": True}
                if isinstance(job, dict) and job.get("out"):
                    atomic_json(job["out"], result)
                print(json.dumps({"ok": False, "status": 2,
                                  "out": job.get("out") if isinstance(job, dict) else None,
                                  "error": str(exc), "worker_error": str(exc), "retire_worker": True}), flush=True)
                # Never attempt same-context cleanup after an exception.
                return 2
        driver.close()
        return 0
    except Exception:
        # Parent must retire on abnormal exit; no CUDA work follows an error.
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path, nargs="?")
    parser.add_argument("--serve", action="store_true", help="Read JSON jobs from stdin; retain one CUDA context")
    args = parser.parse_args()
    if args.serve:
        if args.job is not None:
            parser.error("--serve cannot be combined with a job path")
        return serve()
    if args.job is None:
        parser.error("a job path or --serve is required")
    job = json.loads(args.job.read_text())
    try:
        parent_death_signal()
        result = run_job(job)
        status = 0 if result["ok"] else 1
    except Exception as exc:
        result = {"ok": False, "stage": "cuda" if isinstance(exc, CUDAError) else "worker",
                  "error": str(exc), "retire_worker": True}
        status = 2
    atomic_json(job["out"], result)
    return status


if __name__ == "__main__":
    sys.exit(main())
