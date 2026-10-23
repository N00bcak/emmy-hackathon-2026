#!/usr/bin/env python3
"""Measure exact compiler snapshots on the organizer runner, serially.

PTX is generated locally from immutable compile.py snapshots and checked
against every published problem before timing. The request journal is appended
before retrying a throttled request. Only transport errors, HTTP 429/503, and
`busy` are retried.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROBLEMS = ROOT / "emmy-hackathon-2026/problems"
DTYPES = {"f32": np.float32, "f16": np.float16, "f64": np.float64,
          "i32": np.int32, "i64": np.int64}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


class Journal:
    def __init__(self, path: Path, base_url: str, gap: float):
        self.path = path
        self.base_url = base_url.rstrip("/")
        self.gap = gap
        self.rows = []
        if path.exists():
            self.rows = [json.loads(line) for line in path.read_text().splitlines()
                         if line.strip()]
        self.last_request = max((row.get("started", 0.0) for row in self.rows),
                                default=0.0)
        self.completed = {row["key"]: row for row in self.rows if not row["throttled"]}

    def request(self, key, method, path, body=None, retain_output=False):
        if key in self.completed:
            row = self.completed[key]
            return row["status"], row["response"]
        schedule = (2.0, 4.0, 8.0, 16.0, 30.0, 60.0)
        for attempt in range(len(schedule) + 1):
            wait = self.gap - (time.monotonic() - self.last_request)
            if wait > 0:
                time.sleep(wait)
            started = time.monotonic()
            data = None if body is None else canonical(body)
            request = urllib.request.Request(
                self.base_url + path, data=data, method=method,
                headers={"Content-Type": "application/json"})
            status, payload, raw = None, None, b""
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    status = response.status
                    raw = response.read()
            except urllib.error.HTTPError as error:
                status = error.code
                raw = error.read()
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                payload = {"ok": False,
                           "transport_error": f"{type(error).__name__}: {error}"}
            if payload is None:
                try:
                    payload = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    payload = {"ok": False,
                               "raw_body": raw[:2000].decode(errors="replace")}
            self.last_request = time.monotonic()
            stage = payload.get("stage") if isinstance(payload, dict) else None
            throttled = status in (429, 503) or stage == "busy" or status is None
            row = {"key": key, "attempt": attempt + 1, "method": method,
                   "path": path, "status": status, "stage": stage,
                   "throttled": throttled, "started": started,
                   "elapsed_seconds": self.last_request - started,
                   "response": payload if retain_output else concise(payload)}
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(canonical(row).decode() + "\n")
            self.rows.append(row)
            if not throttled:
                self.completed[key] = row
                return status, payload
            if attempt == len(schedule):
                raise RuntimeError(f"rate-limited too many times: {key}")
            delay = schedule[attempt] + random.uniform(0.0, 0.5)
            print(f"backoff {delay:.1f}s after HTTP {status} stage={stage}",
                  file=sys.stderr, flush=True)
            time.sleep(delay)
        raise AssertionError("unreachable")


def snapshot(compiler: Path, out: Path) -> dict:
    source = compiler.read_bytes()
    digest = sha_bytes(source)
    case_rows = {}
    for problem in sorted(path.name for path in PROBLEMS.iterdir()
                          if (path / "meta.json").is_file()):
        folder = out / problem
        folder.mkdir(parents=True, exist_ok=True)
        meta = json.loads((PROBLEMS / problem / "meta.json").read_text())
        kernels = []
        for kernel in meta["kernels"]:
            ir = next((PROBLEMS / problem).glob(f"{kernel['entry']}.ir.json"))
            ptx_path = folder / f"{kernel['entry']}.ptx"
            stderr_path = folder / f"{kernel['entry']}.stderr"
            if not ptx_path.is_file():
                result = subprocess.run(
                    [sys.executable, str(compiler), str(ir)],
                    capture_output=True, text=True, timeout=60)
                stderr_path.write_text(result.stderr)
                if result.returncode != 0 or not result.stdout:
                    raise RuntimeError(
                        f"compile failed for {problem}/{kernel['entry']}: "
                        f"{result.stderr[-2000:]}")
                ptx_path.write_text(result.stdout, encoding="ascii")
            ptx = ptx_path.read_text()
            if not ptx.isascii() or not ptx.lstrip().startswith("."):
                raise RuntimeError(f"invalid PTX for {problem}/{kernel['entry']}")
            kernels.append({"entry": kernel["entry"], "ir": str(ir),
                            "ir_sha256": sha_file(ir), "ptx": str(ptx_path),
                            "ptx_sha256": sha_bytes(ptx.encode())})
        case_rows[problem] = {
            "meta_sha256": sha_file(PROBLEMS / problem / "meta.json"),
            "expected_sha256": sha_file(PROBLEMS / problem / "expected.npz"),
            "kernels": kernels}
    return {"source": str(compiler.resolve()), "source_sha256": digest,
            "compiler_unchanged": digest == sha_bytes(compiler.read_bytes()),
            "cases": case_rows}


def concise(value):
    if isinstance(value, dict):
        result = {}
        for name, item in value.items():
            if name == "values" and isinstance(item, list):
                result[name + "_sha256"] = sha_bytes(canonical(item))
                result[name + "_count"] = len(item)
                result[name + "_head"] = item[:8]
            else:
                result[name] = concise(item)
        return result
    if isinstance(value, list):
        return [concise(item) for item in value]
    return value


def ordered_args(meta):
    kernel = meta["kernels"][0]
    by_name = {arg["name"]: arg for arg in kernel["args"]}
    if len(by_name) != len(kernel["args"]) or set(by_name) != set(kernel["arg_order"]):
        raise ValueError("arg_order does not enumerate each argument once")
    args = [dict(by_name[name]) for name in kernel["arg_order"]]
    need = max([math.prod(arg["shape"]) for arg in args if arg.get("output")] or [1])
    return kernel, args, max(1, need)


def check(payload, meta, expected):
    kernel = meta["kernels"][0]
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("runner omitted outputs")
    checks = []
    for arg in kernel["args"]:
        if not arg.get("output"):
            continue
        name = arg["name"]
        value = outputs.get(name)
        if not isinstance(value, dict) or value.get("truncated") or "values" not in value:
            raise ValueError(f"missing or truncated output {name}")
        got = np.asarray(value["values"], dtype=DTYPES[arg.get("dtype", "f32")])
        got = got.reshape(tuple(arg["shape"]))
        reference = expected[name]
        passed = bool(np.allclose(got, reference, rtol=meta["rtol"],
                                 atol=meta["atol"], equal_nan=False))
        error = (float(np.max(np.abs(got.astype(np.float64) -
                                  reference.astype(np.float64))))
                 if got.size else 0.0)
        checks.append({"name": name, "ok": passed, "max_abs_error": error})
        if not passed:
            raise ValueError(f"output mismatch for {name}: max_abs_error={error}")
    if not checks:
        raise ValueError("no output was checked")
    return checks


def run(client, key, cubin_id, kernel, args, need, warmup, iters, expected, meta,
        retain_output=False):
    body = {"cubin_id": cubin_id, "entry": kernel["entry"],
            "grid": kernel["grid"], "block": kernel["block"],
            "smem_bytes": kernel.get("smem_bytes", 0),
            "zero_outputs": kernel.get("zero_outputs", []), "args": args,
            "warmup": warmup, "iters": iters, "max_return": need}
    status, payload = client.request(key, "POST", "/run", body,
                                     retain_output=retain_output)
    if status != 200 or payload.get("ok") is not True:
        raise RuntimeError(f"/run failed for {key}: HTTP {status}: "
                           f"{canonical(payload)[:2000]}")
    if payload.get("median_us") is None:
        raise RuntimeError(f"runner omitted timing for {key}")
    return payload, check(payload, meta, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True,
                        help="NAME=PATH/TO/compile.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--url", default="http://runner.riftstack.ai:48701")
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--gap", type=float, default=0.35)
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    arms = {}
    for item in args.arm:
        name, sep, compiler = item.partition("=")
        if not sep or not name or name in arms:
            parser.error("--arm must be distinct NAME=PATH")
        arms[name] = Path(compiler).resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    state_path = args.out / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {"schema": 1, "url": args.url.rstrip("/"), "status": "started",
                 "repetitions": args.repetitions, "gap": args.gap,
                 "arms": {}, "cases": {}}
    for name, compiler in arms.items():
        arm = state["arms"].setdefault(
            name, {"source": str(compiler), "source_sha256": sha_file(compiler)})
        if arm["source_sha256"] != sha_file(compiler):
            raise RuntimeError(f"source changed during run: {name}")
        arm["snapshot"] = snapshot(
            compiler, args.out / "ptx" / name / arm["source_sha256"][:12])
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    if args.prepare_only:
        state["status"] = "prepared"
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        return 0
    client = Journal(args.out / "requests.jsonl", args.url, args.gap)
    status, health = client.request("health", "GET", "/health")
    state["health"] = health
    if (status != 200 or health.get("ok") is not True or
            "5090" not in str(health.get("gpu")) or
            health.get("default_arch") != "sm_120"):
        state.update(status="stopped",
                     error="health did not confirm a runnable RTX 5090 / sm_120")
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        return 2
    for case in sorted(state["arms"][next(iter(arms))]["snapshot"]["cases"]):
        folder = PROBLEMS / case
        meta = json.loads((folder / "meta.json").read_text())
        with np.load(folder / "expected.npz", allow_pickle=False) as archive:
            expected = {name: np.array(archive[name]) for name in archive.files}
        kernel, ordered, need = ordered_args(meta)
        row = state["cases"].setdefault(case, {
            "meta_sha256": sha_file(folder / "meta.json"),
            "expected_sha256": sha_file(folder / "expected.npz"),
            "rtol": meta["rtol"], "atol": meta["atol"],
            "entry": kernel["entry"], "need": need, "arms": {}})
        for name in arms:
            arm = row["arms"].setdefault(name, {})
            if "compile" not in arm:
                ptx_path = Path(state["arms"][name]["snapshot"]
                                ["cases"][case]["kernels"][0]["ptx"])
                ptx = ptx_path.read_text()
                cstatus, payload = client.request(
                    f"compile:{case}:{name}", "POST", "/compile",
                    {"ptx": ptx, "arch": "sm_120", "flags": []})
                if cstatus != 200 or payload.get("ok") is not True:
                    arm["compile"] = {"ok": False, "status": cstatus,
                                      "response": payload}
                    state.update(status="stopped",
                                 error=f"compile failed: {name}/{case}")
                    state_path.write_text(json.dumps(state, indent=2) + "\n")
                    return 1
                if kernel["entry"] not in payload.get("entry_points", []):
                    raise RuntimeError(
                        f"module omits {kernel['entry']} for {name}/{case}")
                arm["compile"] = {
                    "ok": True, "cubin_id": payload["cubin_id"],
                    "registers": payload.get("registers"),
                    "ptx_sha256": sha_file(ptx_path)}
            if "gate" not in arm:
                payload, checks = run(
                    client, f"gate:{case}:{name}", arm["compile"]["cubin_id"],
                    kernel, ordered, need, 0, 1, expected, meta,
                    retain_output=True)
                arm["gate"] = {"ok": True, "checks": checks,
                               "median_us": payload.get("median_us")}
            state_path.write_text(json.dumps(state, indent=2) + "\n")
    all_pass = all(
        arm.get("gate", {}).get("ok") is True
        for row in state["cases"].values() for arm in row["arms"].values())
    state["published_correctness"] = {
        "all15_passed": all_pass,
        "scope": "Published cases on hosted RTX 5090; hidden set unverified"}
    if not all_pass:
        state.update(status="stopped", error="hosted all-15 gate failed")
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        return 1
    if args.gate_only:
        state["status"] = "gate-complete"
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        return 0
    names = list(arms)
    for repetition in range(args.repetitions):
        order = names if repetition % 2 == 0 else list(reversed(names))
        for case in sorted(state["cases"]):
            folder = PROBLEMS / case
            meta = json.loads((folder / "meta.json").read_text())
            with np.load(folder / "expected.npz", allow_pickle=False) as archive:
                expected = {name: np.array(archive[name]) for name in archive.files}
            kernel, ordered, need = ordered_args(meta)
            row = state["cases"][case]
            for name in order:
                arm = row["arms"][name]
                samples = arm.setdefault("timing_us", [])
                if len(samples) > repetition:
                    continue
                payload, checks = run(
                    client, f"timing:{case}:{name}:{repetition}",
                    arm["compile"]["cubin_id"], kernel, ordered, need,
                    10, 50, expected, meta)
                samples.append(payload["median_us"])
                arm.setdefault("timing_checks", []).append(checks)
            state_path.write_text(json.dumps(state, indent=2) + "\n")
        print(f"repetition {repetition + 1}/{args.repetitions}",
              file=sys.stderr, flush=True)
    for row in state["cases"].values():
        row["medians"] = {name: statistics.median(arm["timing_us"])
                          for name, arm in row["arms"].items()}
        best = min(row["medians"], key=row["medians"].get)
        row["winner"] = best
        row["ratios"] = {name: row["medians"][name] / row["medians"][best]
                         for name in row["medians"]}
    state["status"] = "completed"
    state["requests"] = len(client.rows)
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    (args.out / "per-case.csv").write_text(
        "problem," + ",".join(names) + ",winner\n" +
        "\n".join(case + "," +
                  ",".join(f"{row['medians'][name]:.6f}" for name in names) +
                  "," + row["winner"]
                  for case, row in sorted(state["cases"].items())) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
