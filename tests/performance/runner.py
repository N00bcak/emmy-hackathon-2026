#!/usr/bin/env python3
"""Prepare, verify, and noise-qualify the scaled performance suite.

Full outputs are checked through the guarded CUDA-driver worker before any
timing is admitted. Timing uses the worker's paired CUDA-graph protocol with
the same cubin in both arms. This is local diagnostic evidence, not a hosted
benchmark receipt.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "generators"))
import suite  # noqa: E402

PUBLISHED_FLOOR = "p00_scalar_add"
PUBLISHED_FLOOR_SHAPE = (128,)


def timing_role(case):
    """Reject small fixtures as performance evidence, except floor calibration."""
    if case.get("tier") == "floor":
        if case.get("id") != PUBLISHED_FLOOR:
            raise ValueError("only the published scalar-add case calibrates the launch floor")
        return "launch_floor_calibration"
    if case.get("id") == PUBLISHED_FLOOR:
        raise ValueError("the published launch-floor case is not a performance case")
    return "performance_case"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def run_command(command, *, cwd=ROOT, timeout=300):
    started = time.monotonic()
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                            timeout=timeout)
    return result, time.monotonic() - started


def compiler_and_assembly(case_dir, meta, compiler, out, ptxas, timeout):
    """Compile each ordered module and assemble both supported targets."""
    modules = []
    for index, kernel in enumerate(meta["kernels"]):
        ir = case_dir / (kernel["entry"] + ".ir.json")
        stem = Path(out).resolve() / f"kernel-{index}-{kernel['entry']}"
        stem.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = stem.with_suffix(".module.json")
        if metadata_path.is_file():
            cached = json.loads(metadata_path.read_text())
            if (cached.get("ir_sha256") == sha(ir)
                    and cached.get("compiler_sha256") == sha(compiler)):
                modules.append(cached)
                continue
        compiled, compile_seconds = run_command(
            [sys.executable, str(compiler), str(ir)], timeout=timeout)
        (stem.with_suffix(".compiler.stdout")).write_text(compiled.stdout)
        (stem.with_suffix(".compiler.stderr")).write_text(compiled.stderr)
        module = {
            "index": index,
            "entry": kernel["entry"],
            "ir": ir.name,
            "ir_sha256": sha(ir), "compiler_sha256": sha(compiler),
            "compile_command": [sys.executable, str(compiler), str(ir)],
            "compile_returncode": compiled.returncode,
            "compile_seconds": compile_seconds,
            "compiler_stderr": compiled.stderr[-4000:],
            "targets": {},
        }
        if compiled.returncode or not compiled.stdout.strip():
            module["error"] = "compiler failed or emitted no PTX"
            modules.append(module)
            continue
        ptx = stem.with_suffix(".ptx")
        ptx.write_text(compiled.stdout)
        module["ptx"] = str(ptx)
        module["ptx_sha256"] = sha(ptx)
        if len(compiled.stdout) > 2_000_000 or not compiled.stdout.isascii():
            module["error"] = "PTX exceeds the published text/ASCII limit"
            modules.append(module)
            continue
        for arch in ("sm_89", "sm_120"):
            cubin = stem.with_suffix(f".{arch}.cubin")
            log = stem.with_suffix(f".{arch}.ptxas.log")
            command = [ptxas, f"-arch={arch}", "-v", str(ptx), "-o", str(cubin)]
            try:
                assembled, seconds = run_command(command, timeout=min(timeout, 60))
            except subprocess.TimeoutExpired:
                module["targets"][arch] = {"ok": False, "error": "ptxas timeout"}
                continue
            log.write_text(assembled.stdout + assembled.stderr)
            registers = [int(value) for value in re.findall(
                r"Used\s+(\d+)\s+registers", assembled.stderr)]
            spill_stores = [int(value) for value in re.findall(
                r"(\d+)\s+bytes spill stores", assembled.stderr)]
            spill_loads = [int(value) for value in re.findall(
                r"(\d+)\s+bytes spill loads", assembled.stderr)]
            module["targets"][arch] = {
                "ok": assembled.returncode == 0 and cubin.is_file(),
                "command": command,
                "returncode": assembled.returncode,
                "seconds": seconds,
                "log": str(log),
                "registers": max(registers, default=None),
                "spill_store_bytes": max(spill_stores, default=0),
                "spill_load_bytes": max(spill_loads, default=0),
                "cubin": str(cubin) if cubin.is_file() else None,
                "cubin_sha256": sha(cubin) if cubin.is_file() else None,
                "stderr": assembled.stderr[-4000:],
            }
            if not module["targets"][arch]["ok"]:
                module["error"] = f"ptxas failed for {arch}"
        modules.append(module)
        write_json(metadata_path, module)
    return modules


def prepare(case, tier, case_root, compiler, out, ptxas, timeout):
    directory = case_root / tier / case["id"]
    if not directory.exists():
        suite.generate_case(case, directory)
    record = json.loads((directory / "case.json").read_text())
    if record["spec"] != suite.canonical_case(case):
        raise ValueError(f"case identity mismatch: {case['id']}")
    if suite.validate_case(directory)["valid"] is not True:
        raise ValueError(f"case validation failed: {case['id']}")
    meta = json.loads((directory / "meta.json").read_text())
    modules = compiler_and_assembly(
        directory, meta, compiler, out / "artifacts" / tier / case["id"], ptxas, timeout)
    cubins = []
    for module in modules:
        target = module.get("targets", {}).get("sm_89", {})
        if not target.get("ok"):
            cubins = []
            break
        cubins.append(target["cubin"])
    smoke_target = all(module.get("targets", {}).get("sm_120", {}).get("ok")
                       for module in modules)
    case_hash = sha(directory / "case.json")
    return {
        "id": case["id"], "tier": tier, "family": case["family"],
        "local_only": case["local_only"], "manifest_spec": suite.canonical_case(case),
        "case_dir": str(directory.resolve()), "case_sha256": case_hash,
        "payload_hashes": record["files"],
        "meta_sha256": sha(directory / "meta.json"),
        "inputs_sha256": sha(directory / "inputs.npz"),
        "expected_sha256": sha(directory / "expected.npz"),
        "launch_contract": meta["kernels"], "cubins": cubins,
        "modules": modules, "sm_89_assembly": bool(cubins),
        "sm_120_assembly": smoke_target, "static_ready": bool(cubins and smoke_target),
    }


def worker_job(job, out, timeout):
    job = {**job, "out": str(out)}
    job_path = out.with_suffix(".job.json")
    write_json(job_path, job)
    command = [sys.executable, "-m", "tools.research_gpu", str(job_path)]
    try:
        process, seconds = run_command(command, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"GPU worker timed out after {timeout}s") from error
    if not out.is_file():
        raise RuntimeError(f"GPU worker produced no result: {process.stderr[-2000:]}")
    result = json.loads(out.read_text())
    result["process"] = {"command": command, "returncode": process.returncode,
                         "stderr": process.stderr[-4000:], "seconds": seconds}
    return result


def correctness(case, record, out, repetitions, timeout):
    result = worker_job({
        "mode": "correctness",
        "cases": [{"id": case["id"], "variant": record["tier"],
                   "case_dir": record["case_dir"], "cubins": record["cubins"]}],
        "width": 1, "repetitions": repetitions,
    }, out, timeout)
    with np.load(Path(record["case_dir"]) / "expected.npz", allow_pickle=False) as expected:
        expected_names = set(expected.files)
    checks = [check for batch in result.get("batches", [])
              for row in batch.get("results", []) for check in row.get("checks", [])]
    covered = {name for check in checks for name in check.get("outputs", {})}
    result["expected_output_names"] = sorted(expected_names)
    result["covered_output_names"] = sorted(covered)
    result["full_outputs_pass"] = bool(
        result.get("ok") is True and checks and all(check.get("ok") for check in checks)
        and covered == expected_names)
    return result


class LocalRunner:
    """Admitted loopback server used only to collect per-launch medians."""

    def __init__(self, cache):
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1",
                       "OMP_NUM_THREADS": "1"}
        self.process = subprocess.Popen(
            [sys.executable, "tools/local_runner.py", "--cache", str(cache),
             "--port", "0", "--arch", "sm_89"],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True, env=environment)
        self.url = None
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            line = self.process.stdout.readline()
            match = re.search(r"http://127\.0\.0\.1:(\d+)", line)
            if match:
                self.url = f"http://127.0.0.1:{match.group(1)}"
                break
            if self.process.poll() is not None:
                break
        if self.url is None:
            stderr = self.process.stderr.read() if self.process.poll() is not None else ""
            self.close()
            raise RuntimeError(f"local timing server did not start: {stderr[-2000:]}")
        with urllib.request.urlopen(self.url + "/health", timeout=20) as response:
            health = json.load(response)
        if health.get("ok") is not True:
            self.close()
            raise RuntimeError("local timing server health check failed")

    def post(self, path, body, timeout=300):
        request = urllib.request.Request(
            self.url + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def close(self):
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, 15)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, 9)
                except ProcessLookupError:
                    pass
                self.process.wait()


def timing_launch(server, case, record, meta, ptx, warmup, iterations, timeout):
    args_by_name = {arg["name"]: arg for arg in meta["args"]}
    args = [args_by_name[name] for name in meta["arg_order"]]
    body = {
        "ptx": ptx, "entry": meta["entry"], "grid": meta["grid"],
        "block": meta["block"], "smem_bytes": meta.get("smem_bytes", 0),
        "zero_outputs": meta.get("zero_outputs", []), "args": args,
        "warmup": warmup, "iters": iterations, "max_return": 0,
    }
    result = server.post("/run", body, timeout=timeout)
    if result.get("ok") is not True:
        raise RuntimeError(f"timing launch failed for {case['id']}: "
                           + str(result.get("stderr", result))[:2000])
    return result


def timing(server, case, record, out, pairs, warmup, iterations, timeout):
    """A/A timing with paired alternating requests; correctness is separate."""
    role = timing_role(case)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        cached = json.loads(out.read_text())
        if (cached.get("warmup") == warmup and cached.get("iterations") == iterations
                and cached.get("complete_pairs") == pairs):
            return cached
    meta = record.get("launch_contract") or json.loads(
        (Path(record["case_dir"]) / "meta.json").read_text())["kernels"]
    ptx = [Path(module["ptx"]).read_text() for module in record["modules"]]
    if len(meta) != len(ptx):
        raise ValueError("timing modules do not align with launch metadata")
    arms = {"candidate": [], "baseline": []}
    requests = []
    for pair_index in range(pairs):
        order = ("candidate", "baseline") if pair_index % 2 == 0 else ("baseline", "candidate")
        for label in order:
            total = 0.0
            kernel_us = []
            for kernel, kernel_ptx in zip(meta, ptx, strict=True):
                result = timing_launch(server, case, record, kernel, kernel_ptx,
                                       warmup, iterations, timeout)
                total += float(result["median_us"])
                kernel_us.append({"entry": kernel["entry"],
                                  "median_us": float(result["median_us"]),
                                  "samples_us": result.get("samples_us", [])})
            arms[label].append(total)
            requests.append({"pair_index": pair_index, "arm": label,
                             "order": list(order), "total_us": total,
                             "kernels": kernel_us})
    candidate, baseline = arms["candidate"], arms["baseline"]
    ratios = [b / a for a, b in zip(candidate, baseline, strict=True)]
    cached = {"ok": True, "timing_role": role, "timing_protocol": "loopback-per-launch-median-v1",
            "correctness_verified_separately": True, "timing_outputs_truncated": True,
            "candidate_us": candidate, "baseline_us": baseline, "ratios": ratios,
            "complete_pairs": len(ratios), "requests": requests,
            "warmup": warmup, "iterations": iterations,
            "timing_method": "Per-launch CUDA-event medians on the admitted loopback runner; no output check here."}
    write_json(out, cached)
    return cached


def latency_summary(times):
    if not times:
        return {"count": 0}
    return {"count": len(times), "min": float(min(times)),
            "p10": float(np.percentile(times, 10)),
            "median": float(statistics.median(times)),
            "p90": float(np.percentile(times, 90)), "max": float(max(times))}


def noise_summary(times, ratios, maximum_deadband):
    if not times or not ratios:
        return {"valid": False, "reason": "no complete paired samples"}
    spread = {"min": min(ratios), "max": max(ratios),
              "median": statistics.median(ratios),
              "p10": float(np.percentile(ratios, 10)),
              "p90": float(np.percentile(ratios, 90))}
    median_bias = abs(spread["median"] - 1.0)
    tail_bias = max(abs(spread["p10"] - 1.0), abs(spread["p90"] - 1.0))
    deadband = max(median_bias, tail_bias)
    latency = latency_summary(times)
    return {"valid": deadband <= maximum_deadband, "deadband": deadband,
            "median_bias": median_bias, "median_us": latency.get("median", 0.0),
            "maximum_deadband": maximum_deadband, "ratio_spread": spread,
            "resolved_aa_effect": bool(spread["p10"] > 1 + maximum_deadband
                                       or spread["p90"] < 1 - maximum_deadband),
            **latency}


def floor_admission(timing_result, multiple):
    """Calibrate the launch floor from absolute latency, not A/A ratio."""
    times = timing_result.get("candidate_us", []) + timing_result.get("baseline_us", [])
    summary = latency_summary(times)
    valid = bool(timing_result.get("ok") is True and len(times) >= 18)
    floor_us = summary.get("p90", 0.0)
    floor_median = summary.get("median", 0.0)
    return {
        "valid": valid,
        "method": "p90 of per-launch medians for published p00 (N=128)",
        "floor_us": floor_us, "floor_median_us": floor_median,
        "threshold_us": floor_us * multiple,
        "multiple": multiple, "sample_summary": summary,
        "timing_protocol": timing_result.get("timing_protocol"),
    }


def admit(record):
    """Fail closed on correctness, full output coverage, noise, and floor."""
    return bool(
        record.get("static_ready")
        and record.get("full_outputs_pass")
        and record.get("timing_ok")
        and record.get("complete_pairs", 0) == record.get("planned_pairs", -1)
        and record.get("noise", {}).get("valid")
        and record.get("noise", {}).get("deadband", float("inf"))
            <= record.get("noise", {}).get("maximum_deadband", -1.0)
        and not record.get("noise", {}).get("resolved_aa_effect", True)
        and record.get("above_floor")
        and record.get("floor", {}).get("valid")
    )


def published_gate(path, compiler):
    path = Path(path)
    results_path = path / "results.json" if path.is_dir() else path
    if not results_path.is_file():
        raise ValueError(f"published all-15 gate is missing: {results_path}")
    report = json.loads(results_path.read_text())
    status = report.get("published_correctness", {})
    if report.get("compiler_sha256") != sha(compiler):
        raise ValueError("published gate is not bound to the current compiler source")
    if status.get("all15_passed") is not True or status.get("source_unchanged") is not True:
        raise ValueError("published gate does not show all 15 cases passing")
    if any(row.get("status") != "pass" for row in report.get("cases", [])):
        raise ValueError("published gate contains a non-passing case")
    return {"results": str(results_path.resolve()), "sha256": sha(results_path),
            "compiler_sha256": report["compiler_sha256"],
            "all15_passed": True, "source_unchanged": True}


def floor_case(case_root, compiler, out, ptxas, timeout):
    source = ROOT / "emmy-hackathon-2026" / "problems" / PUBLISHED_FLOOR
    directory = case_root / "launch-floor" / PUBLISHED_FLOOR
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("meta.json", "expected.npz"):
        shutil.copyfile(source / name, directory / name)
    meta = json.loads((directory / "meta.json").read_text())
    irs = sorted(source.glob("*.ir.json"))
    if len(irs) != len(meta["kernels"]):
        raise ValueError("published floor IR/kernel count differs")
    for ir in irs:
        shutil.copyfile(ir, directory / ir.name)
    modules = compiler_and_assembly(directory, meta, compiler, out / "artifacts", ptxas, timeout)
    if not all(module["targets"]["sm_89"]["ok"] for module in modules):
        raise ValueError("published launch-floor cubin failed to assemble")
    cubins = [module["targets"]["sm_89"]["cubin"] for module in modules]
    return {"id": PUBLISHED_FLOOR, "tier": "floor", "family": "pointwise",
            "shape": list(PUBLISHED_FLOOR_SHAPE),
            "timing_role": timing_role({"id": PUBLISHED_FLOOR, "tier": "floor"}),
            "case_dir": str(directory.resolve()), "cubins": cubins,
            "launch_contract": meta["kernels"], "modules": modules}


def lock_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        stream.close()
        raise RuntimeError(f"GPU lock is busy: {path}") from error
    return stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, default=ROOT / "compile.py")
    parser.add_argument("--tier", choices=("perf-a", "perf-b"), action="append")
    parser.add_argument("--case", action="append")
    parser.add_argument("--published-gate", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timing-iters", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--floor-multiple", type=float, default=10.0)
    parser.add_argument("--max-aa-deadband", type=float, default=0.05)
    parser.add_argument("--budget", type=float, default=1800.0)
    parser.add_argument("--job-timeout", type=float, default=300.0)
    parser.add_argument("--ptxas", default="ptxas")
    parser.add_argument("--lock", type=Path, default=ROOT / ".emmy/research/.locks/gpu.lock")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output directory already exists")
    if not 3 <= args.pairs <= 200 or not 1 <= args.warmup <= 20:
        parser.error("need at least 3 pairs and warmup in [1, 20]")
    if not 1 <= args.timing_iters <= 200:
        parser.error("timing iterations must be in [1, 200]")
    if not 1 <= args.repetitions <= 20 or not 1 < args.floor_multiple <= 100:
        parser.error("repetitions/floor multiple out of bounds")
    if not 0 < args.max_aa_deadband <= 0.25 or not 0 < args.budget <= 7200:
        parser.error("noise cap/budget out of bounds")
    compiler = args.compiler.resolve()
    if not compiler.is_file():
        parser.error(f"compiler does not exist: {compiler}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    gate = published_gate(args.published_gate, compiler)
    selected = [(tier, case) for tier, case in suite.all_cases()
                if (not args.tier or tier in args.tier)
                and (not args.case or case["id"] in args.case)]
    if not selected:
        parser.error("no selected cases")
    if any(timing_role({**case, "tier": tier}) != "performance_case"
           for tier, case in selected):
        parser.error("only scaled suite cases may be timed as performance cases")
    lock = lock_file(args.lock)
    server = LocalRunner(args.case_root / "timing-cache")
    started = time.monotonic()
    ptxas_version = subprocess.check_output([args.ptxas, "--version"], text=True).strip()
    source_hashes = {
        "compiler": sha(compiler), "manifest": sha(suite.MANIFEST),
        "generator": sha(Path(suite.__file__)), "runner": sha(Path(__file__)),
        "research_gpu": sha(ROOT / "tools/research_gpu.py"),
    }
    report = {
        "schema": "emmy-performance-suite-run-v1", "suite_version": suite.VERSION,
        "compiler": str(compiler), "compiler_sha256": source_hashes["compiler"],
        "source_hashes": source_hashes, "published_gate": gate,
        "pairs": args.pairs, "warmup": args.warmup, "timing_iters": args.timing_iters,
        "correctness_repetitions": args.repetitions,
        "floor_multiple": args.floor_multiple, "max_aa_deadband": args.max_aa_deadband,
        "ptxas": args.ptxas, "ptxas_version": ptxas_version,
        "lock": str(args.lock), "cases": [], "timing_is_local_diagnostic": True,
        "timing_server": server.url,
    }
    try:
        prepared = []
        for tier, case in selected:
            row = prepare(case, tier, args.case_root, compiler, args.out,
                          args.ptxas, min(args.job_timeout, 60))
            report["cases"].append(row)
            prepared.append((tier, case, row))
            write_json(args.out / "results.json", report)
            print(f"{case['id']}: prepare={'PASS' if row['static_ready'] else 'FAIL'}",
                  flush=True)
        # No timing is valid until every selected case has passed full-output
        # correctness. A failure stops here rather than producing partial timing.
        for tier, case, row in prepared:
            if not row["static_ready"]:
                row["full_outputs_pass"] = False
                continue
            result = correctness(case, row,
                                 args.out / "correctness" / case["id"] / "result.json",
                                 args.repetitions, args.job_timeout)
            row["correctness"] = result
            row["full_outputs_pass"] = result["full_outputs_pass"]
            write_json(args.out / "results.json", report)
            print(f"{case['id']}: correctness="
                  f"{'PASS' if row['full_outputs_pass'] else 'FAIL'}", flush=True)
        if not all(row["full_outputs_pass"] for _, _, row in prepared):
            report["suite_admission"] = "FAIL"
            report["qualification_note"] = (
                "Full-output correctness failed; no timing admitted.")
            write_json(args.out / "results.json", report)
            return 1

        floor = floor_case(args.case_root, compiler, args.out / "floor",
                           args.ptxas, min(args.job_timeout, 60))
        floor_timing = timing(server, {"id": PUBLISHED_FLOOR, "tier": "floor", "seed": 0},
                              floor, args.out / "floor" / "timing.json", max(args.pairs, 9),
                              args.warmup, args.timing_iters, args.job_timeout)
        floor_noise = floor_admission(floor_timing, args.floor_multiple)
        floor_noise["median_us"] = floor_noise["floor_median_us"]
        report["launch_floor"] = {**floor, "timing": floor_timing, "admission": floor_noise}
        write_json(args.out / "results.json", report)
        print(f"launch floor: {floor_noise['median_us']:.3f} us; "
              f"threshold={floor_noise['threshold_us']:.3f} us; "
              f"deadband={floor_noise.get('deadband', float('nan')):.4f}", flush=True)

        for tier, case, row in prepared:
            remaining = args.budget - (time.monotonic() - started)
            if remaining <= 0:
                row.update(above_floor=False, timing_outputs_pass=False,
                           suite_status="unverified_budget")
                continue
            result = timing(server, case, row,
                            args.out / "timing" / case["id"] / "result.json",
                            args.pairs, args.warmup, args.timing_iters,
                            min(args.job_timeout, remaining))
            noise = noise_summary(result["candidate_us"] + result["baseline_us"],
                                  result["ratios"], args.max_aa_deadband)
            row["timing"] = result
            row["planned_pairs"] = args.pairs
            row["complete_pairs"] = result["complete_pairs"]
            row["timing_ok"] = result.get("ok") is True
            row["noise"] = noise
            row["floor"] = {"valid": floor_noise["valid"],
                            "median_us": floor_noise["median_us"],
                            "threshold_us": floor_noise["threshold_us"],
                            "multiple": args.floor_multiple}
            row["above_floor"] = bool(
                noise.get("median_us", 0) >= floor_noise["threshold_us"])
            row["floor_ratio"] = (noise.get("median_us", 0) / floor_noise["floor_us"]
                                  if floor_noise["floor_us"] > 0 else None)
            row["admitted"] = admit(row)
            row["suite_status"] = (
                "admitted" if row["admitted"] else
                "unverified_floor" if not row["above_floor"] else
                "unverified_noise" if not noise.get("valid") else
                "unverified_correctness" if not row.get("full_outputs_pass") else
                "unverified_contract")
            write_json(args.out / "results.json", report)
            print(f"{case['id']}: median={noise.get('median_us', float('nan')):.2f} us "
                  f"floor_ratio={row['floor_ratio']:.1f}x "
                  f"deadband={noise.get('deadband', float('nan')):.4f} "
                  f"status={row['suite_status']}", flush=True)

        final_hashes = {"compiler": sha(compiler), "manifest": sha(suite.MANIFEST),
                        "generator": sha(Path(suite.__file__)),
                        "runner": sha(__file__),
                        "research_gpu": sha(ROOT / "tools/research_gpu.py")}
        report["source_unchanged"] = final_hashes == source_hashes
        passed = (all(row.get("admitted") for _, _, row in prepared)
                  and report["source_unchanged"] and floor_noise["valid"])
        report["suite_admission"] = "PASS" if passed else "FAIL"
        report["all_admitted"] = passed
        report["qualification_note"] = (
            "Local guarded-driver diagnostic; all outputs were checked at full size, "
            "timing used identical cubins in both A/A arms, and cases had to exceed "
            f"{args.floor_multiple:g}x the measured published p00 launch floor with "
            f"A/A deadband <= {args.max_aa_deadband:.1%}.")
        report["wall_seconds"] = time.monotonic() - started
        write_json(args.out / "results.json", report)
        print(f"suite admission: {report['suite_admission']}", flush=True)
        return 0 if passed else 1
    finally:
        server.close()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
