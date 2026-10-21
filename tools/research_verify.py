#!/usr/bin/env python3
"""Bounded, replayable infrastructure qualification; never a research speed claim.

Use --prepare in a CPU-admitted container job, then --gpu in a GPU-admitted job.
No Git, hosted requests, compiler promotion, or host GPU admission happens here.
The private host wrapper must supply admission, serialization and outer timeout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from tools.research_cases import generate_case, import_problem, save_arrays, validate_case
from tools.research_gpu import atomic_json
from tools.research_prepare import prepare, assemble
from tools.research_pipeline import GPUWorker
from tools.published import PUBLISHED_PROBLEMS

SUPPORTED = PUBLISHED_PROBLEMS
GAPS = ("m01_matmul_512", "m02_matmul_rect", "f02_row_softmax", "a01_attention", "a02_attention_128")


def identity(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(source, destination):
    source, destination = Path(source), Path(destination)
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("qualification source snapshot cannot contain symlinks")
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", ".git", ".venv"))
    return {path.relative_to(destination).as_posix(): identity(path)
            for path in destination.rglob("*") if path.is_file()}


def control_case(root, name, kind, cache):
    root = Path(root) / name
    root.mkdir(parents=True)
    expected = 32.0 if kind == "atomic" else 1.0
    meta = {"id": name, "rtol": 0, "atol": 0, "graph_outputs": ["y"],
            "kernels": [{"entry": "probe", "arg_order": ["y"], "grid": [1, 1, 1],
                         "block": [32 if kind == "atomic" else 1, 1, 1], "smem_bytes": 0,
                         "zero_outputs": ["y"] if kind == "atomic" else [],
                         "args": [{"name": "y", "dtype": "f32", "shape": [1], "output": True}]}]}
    atomic_json(root / "meta.json", meta)
    save_arrays(root / "inputs.npz", {})
    save_arrays(root / "expected.npz", {"y": np.array([expected], dtype=np.float32)})
    header = ".version 8.0\n.target sm_89\n.address_size 64\n.visible .entry probe(.param .u64 out) {\n.reg .u64 %d; .reg .f32 %f,%discard;\nld.param.u64 %d,[out];\n"
    body = "mov.f32 %f, 1.0; st.global.f32 [%d],%f;\n"
    if kind == "guard":
        # This is inside the worker-owned allocation, not an illegal GPU access.
        body += "st.global.f32 [%d+4],%f;\n"
    elif kind == "atomic":
        body = "mov.f32 %f, 1.0; atom.global.add.f32 %discard,[%d],%f;\n"
    elif kind == "slow":
        body = (".reg .u64 %start,%now,%delta; .reg .pred %wait;\n"
                "mov.u64 %start,%clock64; AGAIN: mov.u64 %now,%clock64;\n"
                "sub.u64 %delta,%now,%start; setp.lt.u64 %wait,%delta,100000; @%wait bra AGAIN;\n" + body)
    ptx = root / "probe.ptx"
    ptx.write_text(header + body + "ret; }\n")
    targets = {arch: assemble(ptx, cache, arch) for arch in ("sm_89", "sm_120")}
    return {"id": name, "variant": "infrastructure-control", "case_dir": str(root.resolve()),
            "cubins": [str(Path(targets["sm_89"]["folder"]) / "module.cubin")],
            "targets": targets, "kind": kind,
            "limitation": "Handwritten PTX infrastructure probe, not a challenge compiler or coverage result."}


def prepare_verification(root, source, problems, cache):
    root, source, problems = Path(root), Path(source), Path(problems)
    manifest_path = root / "qualification.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for row in manifest["prepared"]["cases"]:
            validate_case(row["case_dir"])
        return manifest
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "prepare-start.json", {"source": str(source), "problems": str(problems),
                                               "command": sys.argv, "started_at": time.time()})
    source_copy = root / "sources/baseline"
    source_copy.parent.mkdir(parents=True, exist_ok=True)
    source_identity = snapshot(source, source_copy)
    tools_copy = root / "sources/tools"
    tools_copy.mkdir()
    for path in (p for p in Path(__file__).parent.glob("*.py") if p.name.startswith("research") or p.name == "published.py"):
        shutil.copyfile(path, tools_copy / path.name)
    cases = []
    for name in SUPPORTED:
        path = root / "cases" / name
        import_problem(problems / name, path)
        cases.append(path)
    path = root / "cases/generated-pointwise-14"
    generate_case(path, "pointwise", 314159, shape=[14], operation="add")
    cases.append(path)
    path = root / "cases/generated-reduction-257"
    generate_case(path, "reduction", 271828, shape=[257])
    cases.append(path)
    prepared = prepare(source_copy, cases, root / "prepared", cache)
    specs = [{"id": row["case_id"], "variant": "unchanged-baseline-control",
              "case_dir": row["case_dir"],
              "cubins": [str(Path(kernel["targets"]["sm_89"]["folder"]) / "module.cubin")
                         for kernel in row["kernels"]]} for row in prepared["cases"]]
    controls = {kind: control_case(root / "controls", kind, kind, cache)
                for kind in ("copy", "slow", "guard", "atomic")}
    # A no-op PTX module preserves the generated positional ABI and leaves zeroed
    # output allocations unchanged. It must fail the independent generated oracle.
    point = specs[-2]
    meta = json.loads((Path(point["case_dir"]) / "meta.json").read_text())
    kernel = meta["kernels"][0]
    zero = root / "controls/zero.ptx"
    parameters = ",".join(f".param .u64 arg{index}" for index, _ in enumerate(kernel["arg_order"]))
    zero.write_text(f".version 8.0\n.target sm_89\n.address_size 64\n.visible .entry {kernel['entry']}({parameters}) {{ret;}}\n")
    targets = {arch: assemble(zero, cache, arch) for arch in ("sm_89", "sm_120")}
    controls["zero"] = {**point, "id": "zero-output-negative-control", "variant": "safe-noop-control",
                        "cubins": [str(Path(targets["sm_89"]["folder"]) / "module.cubin")], "targets": targets}
    versions = {name: subprocess.check_output([name, "--version"], text=True)
                for name in ("ptxas", "cuobjdump", "nvdisasm", "compute-sanitizer")}
    manifest = {"schema": 1, "source_identity": source_identity, "source_copy": str(source_copy),
                "tools_identity": {path.name: identity(path) for path in tools_copy.iterdir()},
                "prepared": prepared, "specs": specs, "controls": controls, "versions": versions,
                "prepare_job_dir": os.environ.get("EMMY_JOB_DIR"), "prepared_at": time.time(),
                "campaign_gaps_unverified": list(GAPS),
                "purpose": "Infrastructure qualification on replayed baseline controls; not 64 novel compilers or a speedup claim."}
    atomic_json(manifest_path, manifest)
    return manifest


def gpu_verification(root, *, repetitions=3, replay_slots=64, budget_seconds=240):
    root = Path(root)
    manifest = json.loads((root / "qualification.json").read_text())
    specs, controls = manifest["specs"], manifest["controls"]
    results = root / "gpu"
    trial = 1
    while results.exists():
        trial += 1
        results = root / f"gpu-{trial:03d}"
    results.mkdir(exist_ok=False)
    tools_copy = results / "sources/tools"
    tools_copy.mkdir(parents=True)
    for path in (p for p in Path(__file__).parent.glob("*.py") if p.name.startswith("research") or p.name == "published.py"):
        shutil.copyfile(path, tools_copy / path.name)
    atomic_json(root / "latest-gpu.json", {"trial": trial, "directory": str(results), "status": "started"})
    jobs = results / "jobs"
    jobs.mkdir()
    started = time.monotonic()
    summary = {"schema": 1, "started_at": time.time(), "job_dir": os.environ.get("EMMY_JOB_DIR"),
               "baseline_source_identity": manifest["source_identity"],
               "evaluation_tools_identity": {path.name: identity(path) for path in tools_copy.iterdir()},
               "supported_published": list(SUPPORTED), "campaign_gaps_unverified": list(GAPS),
               "replay_slots_requested": replay_slots, "replay_slots_completed": 0,
               "novel_candidates": 0, "measurements": {}, "failures": [], "partial": True}
    atomic_json(results / "summary.json", summary)
    sequence = 0

    def dispatch(worker, name, job):
        nonlocal sequence
        if time.monotonic() - started > budget_seconds:
            raise TimeoutError("qualification budget exhausted; retained evidence is partial")
        sequence += 1
        result_path = jobs / f"{sequence:03d}-{name}.result.json"
        job = {**job, "out": str(result_path)}
        atomic_json(jobs / f"{sequence:03d}-{name}.job.json", job)
        start = time.monotonic()
        result = worker.run(job, timeout=20)
        return result, time.monotonic() - start

    def require(condition, message):
        if not condition:
            summary["failures"].append(message)
        atomic_json(results / "summary.json", summary)

    worker = GPUWorker(results / "persistent-worker")
    summary["persistent_worker_pid"] = worker.process.pid
    atomic_json(results / "summary.json", summary)
    try:
        # <=10 baseline control variants, then the remaining baseline regression
        # cases as isolated confirmation. Context persists across protocol blocks.
        smoke, _ = dispatch(worker, "smoke-eight", {"mode": "correctness", "width": 1,
                             "cases": specs[:8], "repetitions": repetitions})
        require(smoke["ok"], "eight MVP cases failed smoke correctness")
        remaining_results = []
        for offset in range(8, len(specs), 8):
            remaining, _ = dispatch(worker, f"supported-and-generated-{offset}", {
                "mode": "correctness", "width": 1,
                "cases": specs[offset:offset+8], "repetitions": repetitions})
            remaining_results.append(remaining)
            require(remaining["ok"], "fused/generated baseline cases failed correctness")
        remaining_ok = all(row["ok"] for row in remaining_results)
        summary["device"] = smoke.get("device")
        summary["measurements"]["baseline_correctness"] = {
            "published_passed": len(SUPPORTED) if smoke["ok"] and remaining_ok else None,
            "generated_passed": 2 if remaining_ok else None,
            "note": "Counts are pass only when both retained baseline jobs pass; inspect per-case results otherwise."}
        # Infrastructure throughput cannot be qualified using a partial compiler.
        # Correctness debugging above is deliberately still possible when failing.
        if not smoke["ok"] or not remaining_ok:
            raise RuntimeError("all-15 published correctness gate failed; no throughput or timing permitted")
        atomic_spec = specs[list(SUPPORTED).index("m02_matmul_rect")]
        atomic_job = {"mode": "correctness", "width": 1, "cases": [atomic_spec],
                      "repetitions": 5, "out": str(jobs / "m02-sanitizer.result.json")}
        atomic_json(jobs / "m02-sanitizer.job.json", atomic_job)
        command = ["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "86",
                   sys.executable, "-m", "tools.research_gpu", str(jobs / "m02-sanitizer.job.json")]
        process = subprocess.run(command, capture_output=True, text=True, timeout=45)
        (results / "m02-sanitizer.stdout").write_text(process.stdout)
        (results / "m02-sanitizer.stderr").write_text(process.stderr)
        atomic_json(results / "m02-sanitizer.command.json", {"command": command, "exit_code": process.returncode})
        atomic_result = json.loads(Path(atomic_job["out"]).read_text()) if Path(atomic_job["out"]).exists() else {}
        atomic_ok = process.returncode == 0 and atomic_result.get("ok") is True
        require(atomic_ok, "published m02 atomic/reset checks failed Compute Sanitizer")
        summary["published_correctness_gate"] = {"all15_passed": atomic_ok, "source_identity": manifest["source_identity"],
            "isolated": True, "m02_sanitizer_passed": atomic_ok,
            "limitation": "Qualification-local gate; not a research Store promotion receipt."}
        if not atomic_ok:
            raise RuntimeError("m02 sanitizer gate failed; no throughput or timing permitted")
        # Warmed service makes width comparisons independent of context startup.
        width_trials = []
        for trial, widths in enumerate(((1, 2, 4), (4, 1, 2), (2, 4, 1))):
            for width in widths:
                check, seconds = dispatch(worker, f"width-{width}-trial-{trial}", {
                    "mode": "correctness", "width": width, "cases": specs[:8], "repetitions": repetitions})
                count = sum(len(row["checks"]) for batch in check["batches"] for row in batch["results"])
                width_trials.append({"width": width, "trial": trial, "seconds": seconds,
                                     "checks": count, "checks_per_second": count / seconds,
                                     "ok": check["ok"]})
                require(check["ok"], f"width {width} correctness failure")
        eligible = {width: sum(row["checks"] for row in width_trials if row["width"] == width)
                     / sum(row["seconds"] for row in width_trials if row["width"] == width)
                    for width in (1, 2, 4) if all(row["ok"] for row in width_trials if row["width"] == width)}
        chosen = max(eligible, key=eligible.get) if eligible else 1
        summary["measurements"]["width_selection"] = {"trials": width_trials,
                                                        "aggregate_checks_per_second": eligible, "chosen": chosen}
        # Per-case/size A/A evidence; no pooled global noise estimate.
        aa = []
        for index, case in enumerate(specs):
            result, seconds = dispatch(worker, f"aa-{index}", {
                "mode": "timing", "case": {**case, "baseline_cubins": case["cubins"]},
                "pairs": 16, "warmup": 5, "seed": 34001 + index, "burst_id": f"aa-case-{index}"})
            require(result["ok"], f"A/A correctness failure {case['id']}")
            ratios = [pair["candidate_over_baseline"] for pair in result["pairs"] if pair["complete"]]
            aa.append({"case_id": case["id"], "case_dir": case["case_dir"], "seconds": seconds,
                       "complete_pairs": len(ratios), "candidate_over_baseline": ratios})
        summary["measurements"]["aa_by_case"] = aa
        # Explicit replay controls keep hardware/preparation throughput separate
        # from novel hypothesis production and compiler search efficacy.
        replay_start = time.monotonic()
        for offset in range(0, replay_slots, 8):
            subset = [{**specs[index % len(specs)], "id": f"replay-{index:03d}",
                       "variant": "unchanged-baseline-replay"}
                      for index in range(offset, min(offset + 8, replay_slots))]
            result, seconds = dispatch(worker, f"replay-{offset}", {
                "mode": "correctness", "width": chosen, "cases": subset, "repetitions": repetitions})
            require(result["ok"], f"replay cohort {offset} failed")
            if result["ok"]:
                summary["replay_slots_completed"] += len(subset)
            summary["measurements"].setdefault("replay_cohorts", []).append({"slots": len(subset), "seconds": seconds, "ok": result["ok"]})
            atomic_json(results / "summary.json", summary)
        replay_seconds = time.monotonic() - replay_start
        summary["measurements"]["replay"] = {"seconds": replay_seconds,
            "slots_per_hour": summary["replay_slots_completed"] / replay_seconds * 3600,
            "prepared_backlog": True, "unique_compilers": 1,
            "limitation": "Warm prepared baseline controls; excludes idea production and CPU compilation. No scientific speedup claim."}
    finally:
        worker.close()

    # Independent fresh contexts for deliberately failing but memory-safe controls.
    for name in ("guard", "zero"):
        worker = GPUWorker(results / f"{name}-worker")
        try:
            result, _ = dispatch(worker, name, {"mode": "correctness", "width": 1,
                                    "cases": [controls[name]], "repetitions": 3})
            rejected = not result["ok"] and result["batches"][0]["quarantined"]
            require(rejected, f"{name} negative control was not rejected/quarantined")
            summary["measurements"][f"{name}_rejected"] = rejected
        finally:
            worker.close()
    worker = GPUWorker(results / "guard-cohort-worker")
    try:
        result, _ = dispatch(worker, "guard-cohort", {"mode": "correctness", "width": 2,
            "cases": [controls["guard"], controls["copy"]], "repetitions": 3})
        quarantined = (not result["ok"] and result["batches"][0]["quarantined"]
                       and all(not row["trusted"] for row in result["batches"][0]["results"]))
        require(quarantined, "guard corruption did not quarantine its good batchmate")
        summary["measurements"]["guard_cohort_quarantined"] = quarantined
    finally:
        worker.close()
    # Recheck every member independently, including the passing batchmate.
    rechecks = []
    for name in ("guard", "copy"):
        worker = GPUWorker(results / f"guard-batchmate-{name}-isolated")
        try:
            result, _ = dispatch(worker, f"guard-batchmate-{name}", {"mode": "correctness", "width": 1,
                "cases": [controls[name]], "repetitions": 3})
            rechecks.append({"member": name, "ok": result["ok"], "fresh_worker_pid": worker.process.pid})
            require(result["ok"] == (name == "copy"), "isolated guard batchmate attribution was inconsistent")
        finally:
            worker.close()
    summary["measurements"]["guard_all_batchmates_rechecked"] = rechecks
    # Correct deliberately delayed probe demonstrates detection of clear losers.
    worker = GPUWorker(results / "slow-worker")
    try:
        result, _ = dispatch(worker, "slow-correct", {"mode": "timing", "case": {
            **controls["slow"], "baseline_cubins": controls["copy"]["cubins"]},
            "pairs": 16, "warmup": 5, "seed": 554, "burst_id": "deliberately-slow-correct-control"})
        ratios = [pair["candidate_over_baseline"] for pair in result["pairs"] if pair["complete"]]
        slower = result["ok"] and bool(ratios) and float(np.median(ratios)) > 2
        require(slower, "deliberately slowed correct control was not clearly slower")
        summary["measurements"]["deliberately_slow"] = {"ok": result["ok"], "ratios": ratios, "classified_slower": slower}
    finally:
        worker.close()
    # Atomic control tests infrastructure, not m02 compiler support.
    atomic_job = {"mode": "correctness", "width": 1, "cases": [controls["atomic"]],
                  "repetitions": 5, "out": str(jobs / "atomic-sanitizer.result.json")}
    atomic_json(jobs / "atomic-sanitizer.job.json", atomic_job)
    command = ["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "86",
               sys.executable, "-m", "tools.research_gpu", str(jobs / "atomic-sanitizer.job.json")]
    process = subprocess.run(command, capture_output=True, text=True, timeout=45)
    (results / "atomic-sanitizer.stdout").write_text(process.stdout)
    (results / "atomic-sanitizer.stderr").write_text(process.stderr)
    atomic_json(results / "atomic-sanitizer.command.json", {"command": command, "exit_code": process.returncode})
    atomic_result = json.loads(Path(atomic_job["out"]).read_text()) if Path(atomic_job["out"]).exists() else {}
    atomic_ok = process.returncode == 0 and atomic_result.get("ok") is True
    require(atomic_ok, "legal atomic/reset probe failed Compute Sanitizer")
    summary["measurements"]["atomic_sanitizer"] = {"passed": atomic_ok, "exit_code": process.returncode,
          "scope": "Known legal handwritten atomic mechanism/reset probe only; m02 compiler coverage remains unverified."}
    post_identity = {path.name: identity(path) for path in (p for p in Path(__file__).parent.glob("*.py") if p.name.startswith("research") or p.name == "published.py")}
    source_stable = post_identity == summary["evaluation_tools_identity"]
    require(source_stable, "execution tooling changed during qualification; timing evidence invalid")
    summary["evaluation_tools_post_identity"] = post_identity
    summary["execution_source_stable"] = source_stable
    summary["timing_protocol"] = "cuda-graph-paired-v2"
    summary.update(partial=False, completed_at=time.time(), wall_seconds=time.monotonic() - started,
                   passed=not summary["failures"])
    atomic_json(results / "summary.json", summary)
    atomic_json(root / "latest-gpu.json", {"trial": trial, "directory": str(results),
                                           "status": "passed" if summary["passed"] else "failed"})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("/root/bcy/emmy/candidates/correctness15"))
    parser.add_argument("--problems", type=Path, default=Path("emmy-hackathon-2026/problems"))
    parser.add_argument("--cache", type=Path, default=Path("/root/nfs/emmy/research-cache/assembly"))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--gpu", action="store_true")
    parser.add_argument("--replay-slots", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.replay_slots <= 64:
        parser.error("replay slots must be between one and64")
    result = prepare_verification(args.root, args.source, args.problems, args.cache) if args.prepare else gpu_verification(args.root, replay_slots=args.replay_slots)
    print(json.dumps(result, indent=2))
    return 0 if args.prepare or result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
