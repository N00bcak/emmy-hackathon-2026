#!/usr/bin/env python3
"""Explicit, budgeted hosted calibration of already GPU-verified shortlists.

Never called as an automatic local-evaluation fallback. CLI requires --execute.
All HTTP attempts are durably charged before network I/O; interrupted batches
cannot be restarted under the same run ID. No Git, CUDA, or compiler execution.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import fcntl
import numpy as np

try:
    from .research_store import Store, component, Invalid
    from .research_gpu import DTYPES, atomic_json, compare_output
except ImportError:
    from research_store import Store, component, Invalid
    from research_gpu import DTYPES, atomic_json, compare_output

DEFAULT_URL = "http://runner.riftstack.ai:48701"
HARD_REQUEST_CAP = 25
PLANNED_REQUEST_CAP = 22  # Leave at least three attempts unallocated.
SPACING_SECONDS = 2.0
REPETITIONS = 3
MAX_RETURN = 2**22


class CalibrationError(ValueError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CalibrationError("another calibration batch holds the exclusive lock") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def append_record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def history(root):
    path = Path(root) / "control/calibration-history.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def calibration_due(root, mechanism, architecture_sensitive=False):
    """Advisory trigger; eligibility is checked separately before any request."""
    root = Path(root)
    state = json.loads((root / "control/state.json").read_text())
    promotions = len(state.get("baseline_history", []))
    previous = [row for row in history(root) if row.get("comparisons")]
    reasons = []
    if not previous:
        reasons.append("first eligible speed shortlist")
    elif promotions - max(row["promotion_count"] for row in previous) >= 5:
        reasons.append("five Linux promotions since last informative calibration")
    if architecture_sensitive and not any(row["mechanism"] == mechanism for row in previous):
        reasons.append("new architecture-sensitive mechanism")
    return {"due": bool(reasons), "reasons": reasons, "promotion_count": promotions}


class HTTPTransport:
    def __init__(self, base_url=DEFAULT_URL):
        self.base_url = base_url.rstrip("/")

    def __call__(self, method, path, body):
        data = None if body is None else canonical(body)
        request = Request(self.base_url + path, data=data, method=method,
                          headers={"Content-Type": "application/json"})
        try:
            response = urlopen(request, timeout=35)
        except HTTPError as exc:
            response = exc
        with response:
            raw = response.read(128 * 1024 * 1024 + 1)
            if len(raw) > 128 * 1024 * 1024:
                raise CalibrationError("response exceeds local retention limit")
            try:
                payload = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                payload = {"ok": False, "raw_body": raw.decode(errors="replace")}
            return response.status, payload


def _boot_id():
    path = Path("/proc/sys/kernel/random/boot_id")
    return path.read_text().strip() if path.is_file() else "non-linux"


class ChargedRequests:
    def __init__(self, root, batch, transport, sleep, clock, wall_clock):
        self.root, self.batch, self.transport = Path(root), Path(batch), transport
        self.sleep, self.clock, self.wall_clock = sleep, clock, wall_clock
        self.spacing = self.root / "control/calibration-spacing.json"
        self.boot = _boot_id()

    def request(self, method, path, body=None):
        attempts = list((self.batch / "requests").glob("*.json"))
        if len(attempts) >= HARD_REQUEST_CAP:
            raise CalibrationError("25-request hard cap reached")
        if self.spacing.is_file():
            prior = json.loads(self.spacing.read_text())
            if prior["boot_id"] == self.boot and self.clock() >= prior["monotonic"]:
                wait = SPACING_SECONDS - (self.clock() - prior["monotonic"])
            else:
                wait = SPACING_SECONDS - (self.wall_clock() - prior["wall"])
            if wait > 0:
                self.sleep(wait)
        # Persist the attempt before I/O. Charge count is derived from attempt files,
        # so a process exit between journal and status writes cannot lose a charge.
        record = {"attempt": len(attempts) + 1, "method": method, "path": path,
                  "request": body, "status": "charged", "wall": self.wall_clock(),
                  "monotonic": self.clock(), "boot_id": self.boot}
        attempt_path = self.batch / "requests" / f"{record['attempt']:03d}.json"
        atomic_json(attempt_path, record)
        atomic_json(self.spacing, {key: record[key] for key in ("wall", "monotonic", "boot_id")})
        try:
            status, response = self.transport(method, path, body)
            record.update(http_status=status, response=response, status="received",
                          completed_wall=self.wall_clock())
            atomic_json(attempt_path, record)
        except Exception as exc:
            record.update(status="transport_error", error=str(exc), completed_wall=self.wall_clock())
            atomic_json(attempt_path, record)
            raise CalibrationError(f"transport stopped calibration: {exc}") from exc
        finally:
            # Space conservatively from completion, not from a pre-I/O journal
            # stamp whose fsync delay could shorten actual start-to-start spacing.
            atomic_json(self.spacing, {"wall": self.wall_clock(), "monotonic": self.clock(), "boot_id": self.boot})
        if status != 200 or not isinstance(response, dict) or response.get("ok") is not True:
            raise CalibrationError(f"service stopped calibration: HTTP {status}: {response}")
        return response


def _ptx_paths(descriptor, key, count):
    value = descriptor[key]
    paths = [Path(value)] if isinstance(value, str) else [Path(path) for path in value]
    if len(paths) != count or any(not path.is_absolute() or not path.is_file() for path in paths):
        raise CalibrationError(f"{key} must contain absolute PTX paths aligned with kernels")
    for path in paths:
        source = path.read_text()
        if not source.isascii() or not 1 <= len(source) <= 2_000_000:
            raise CalibrationError("PTX must be ASCII and at most two million characters")
    return paths


def _bound_local_comparison(report, folder, baseline, candidate):
    """Derive local medians from the exact retained, arm-bound paired block."""
    expected = {name: sha(folder / name) for name in ('meta.json', 'inputs.npz', 'expected.npz')}
    matches = []
    for case_id, binding in report.get('calibration_cases', {}).items():
        if binding.get('case_files_sha256') != expected:
            continue
        if binding.get('baseline_ptx_sha256') != [sha(p) for p in baseline] or \
                binding.get('candidate_ptx_sha256') != [sha(p) for p in candidate]:
            continue
        matches.append(case_id)
    if len(matches) != 1:
        raise CalibrationError('case and PTX arms are not retained in a unique matching paired measurement')
    case_id = matches[0]
    timing = report.get('timing', {}).get(case_id, {})
    pairs = timing.get('pairs', [])
    if timing.get('ok') is not True or not pairs or any(p.get('complete') is not True for p in pairs):
        raise CalibrationError('complete accepted local paired measurements required')
    # Confirmation records one completed block per case. Never pool across bursts.
    bursts = {p.get('burst_id') for p in pairs}
    if len(bursts) != 1 or None in bursts:
        raise CalibrationError('local calibration measurements must come from one paired burst')
    medians = {}
    for arm in ('baseline', 'candidate'):
        values = [p.get('samples_us', {}).get(arm) for p in pairs]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
            raise CalibrationError('positive finite retained paired measurements required')
        medians['local_' + arm + '_us'] = statistics.median(values)
    return case_id, medians


def validate_pairs(root, descriptors):
    """Require accepted local evidence and actual retained artifacts for both arms.

    Descriptors add experiment_id to the public case/PTX/timing fields. That
    experiment must be shortlisted/promoted; its evaluator report must explicitly
    attest baseline_gpu_verified and local_device, and retain every supplied case
    and PTX file in evaluator-issued artifact objects. Hash-only claims do not pass.
    """
    if not isinstance(descriptors, list) or not 1 <= len(descriptors) <= 2:
        raise CalibrationError("supply one or two representative case descriptors")
    store = Store(root)
    prepared, evidence, experiment_ids, suites = [], {}, set(), set()
    with store._lock():
        for descriptor in descriptors:
            experiment_id = component(descriptor["experiment_id"])
            experiment_ids.add(experiment_id)
            manifest = store._manifest(experiment_id)
            decision = manifest.get("decision") or {}
            if decision.get("decision") not in {"shortlist", "promote"}:
                raise CalibrationError("hosted calibration requires an evaluated shortlist")
            report = store._accepted_evidence(experiment_id, manifest, decision["evidence_id"])
            if report.get("baseline_gpu_verified") is not True or not report.get("local_device"):
                raise CalibrationError("retained evidence must verify both arms on an identified local GPU")
            # Hosted comparison also needs exact-source all15 gates for BOTH arms.
            # A correctness-only repair shortlist does not qualify an old baseline.
            try:
                store.require_published_gate(manifest["source_id"], report["suite_id"])
                store.require_published_gate(manifest["baseline_id"], report["suite_id"])
            except Invalid as error:
                raise CalibrationError("all15 published correctness gates required for both arms: " + str(error)) from error
            store.verify_object(manifest["source_id"], "source")
            store.verify_object(manifest["baseline_id"], "source")
            ids = [decision["evidence_id"], *report.get("artifact_ids", []), report["suite_id"]]
            retained_hashes = set()
            for object_id in ids:
                artifact = store.verify_object(object_id)
                retained_hashes.update(entry["sha256"] for entry in artifact["files"].values())
            folder = Path(descriptor["case_dir"]).resolve()
            paths = [folder / name for name in ("meta.json", "inputs.npz", "expected.npz")]
            if any(not path.is_file() for path in paths):
                raise CalibrationError("calibration requires exact meta, inputs.npz, and expected.npz")
            meta = json.loads(paths[0].read_text())
            count = len(meta["kernels"])
            if count < 1:
                raise CalibrationError("empty kernel pipeline")
            baseline = _ptx_paths(descriptor, "baseline_ptx", count)
            candidate = _ptx_paths(descriptor, "candidate_ptx", count)
            identities = {str(path): sha(path) for path in paths + baseline + candidate}
            if not set(identities.values()).issubset(retained_hashes):
                raise CalibrationError("supplied case/PTX bytes were not retained in accepted local evidence")
            case_id, local_medians = _bound_local_comparison(report, folder, baseline, candidate)
            for key, value in local_medians.items():
                if key in descriptor and descriptor[key] != value:
                    raise CalibrationError('caller local medians differ from retained paired evidence')
            descriptor = {**descriptor, **local_medians, 'local_case_id': case_id}
            with np.load(paths[1], allow_pickle=False) as archive:
                inputs = {name: np.array(archive[name]) for name in archive.files}
            with np.load(paths[2], allow_pickle=False) as archive:
                expected = {name: np.array(archive[name]) for name in archive.files}
            if not expected:
                raise CalibrationError("empty expected outputs")
            if not set(meta.get("graph_outputs", expected)).issubset(expected):
                raise CalibrationError("references omit final outputs")
            for tolerance in ("rtol", "atol"):
                value = meta[tolerance]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise CalibrationError("fixed tolerances required")
            # Build all requests now, before eligibility can cause network I/O.
            _arguments(meta, inputs, {}, preflight=True)
            prepared.append({"descriptor": descriptor, "meta": meta, "inputs": inputs,
                             "expected": expected, "baseline": baseline, "candidate": candidate,
                             "identities": identities, "suite_id": report["suite_id"]})
            evidence[experiment_id] = {"source_id": manifest["source_id"], "baseline_id": manifest["baseline_id"],
                                       "evidence_id": decision["evidence_id"], "local_device": report["local_device"]}
            suites.add(report["suite_id"])
    if len(experiment_ids) != 1:
        raise CalibrationError("one candidate compiler per batch")
    # More complex pipelines reduce representative case count, not request safety.
    selected, planned = [], 1
    for case in prepared:
        cost = 2 * len(case["meta"]["kernels"]) * (1 + REPETITIONS)
        if planned + cost <= PLANNED_REQUEST_CAP:
            selected.append(case)
            planned += cost
    if not selected:
        raise CalibrationError("pipeline cannot fit planned request budget with headroom")
    return selected, {"evidence": evidence, "suite_ids": sorted(suites),
                      "planned_requests": planned, "omitted_cases": len(prepared) - len(selected)}


def _arguments(meta, inputs, produced, preflight=False, upto=None):
    all_args = []
    earlier = set()
    for kernel_index, kernel in enumerate(meta["kernels"]):
        if upto is not None and kernel_index > upto:
            break
        by_name = {arg["name"]: arg for arg in kernel["args"]}
        if len(by_name) != len(kernel["args"]) or set(kernel["arg_order"]) != set(by_name) or len(kernel["arg_order"]) != len(by_name):
            raise CalibrationError("arg_order must enumerate every argument once")
        args, need, total = [], 0, 0
        for name in kernel["arg_order"]:
            source = by_name[name]
            arg = {key: value for key, value in source.items() if key != "intermediate"}
            if arg.get("dtype", "f32") not in DTYPES:
                raise CalibrationError("unsupported dtype")
            dtype = DTYPES[arg.get("dtype", "f32")]
            if arg.get("scalar"):
                args.append(arg)
                continue
            shape = tuple(arg["shape"])
            count = math.prod(shape)
            if any(type(dim) is not int or dim < 0 for dim in shape):
                raise CalibrationError("invalid shape")
            total += count
            if arg.get("output"):
                need += count
                arg.pop("data", None)
                arg.pop("seed", None)
            else:
                if name in produced:
                    values = np.asarray(produced[name], dtype=dtype).reshape(shape)
                elif name in earlier and preflight:
                    values = np.zeros(shape, dtype=dtype)
                elif name in inputs:
                    values = inputs[name]
                    if values.shape != shape or values.dtype != np.dtype(dtype):
                        raise CalibrationError("input archive dtype/shape differs from metadata")
                elif "data" in arg:
                    values = np.asarray(arg["data"], dtype=dtype).reshape(shape)
                else:
                    raise CalibrationError(f"exact input {name} missing")
                arg.pop("seed", None)
                arg["data"] = values.reshape(-1).tolist()
            args.append(arg)
        if total > 2**26 or need > MAX_RETURN:
            raise CalibrationError("case exceeds documented runner element/return limits")
        all_args.append((args, need))
        earlier.update(arg["name"] for arg in args if arg.get("output"))
    return all_args


def _check_outputs(response, kernel, case):
    outputs = response.get("outputs")
    if not isinstance(outputs, dict):
        raise CalibrationError("runner omitted outputs")
    produced, diagnostics = {}, {}
    for arg in kernel["args"]:
        if not arg.get("output"):
            continue
        name, shape = arg["name"], tuple(arg["shape"])
        out = outputs.get(name)
        if not isinstance(out, dict) or out.get("truncated") or "values" not in out:
            raise CalibrationError("runner returned missing/truncated outputs")
        if tuple(out["shape"]) != shape or out["dtype"] != arg.get("dtype", "f32"):
            raise CalibrationError("runner output contract mismatch")
        values = np.asarray(out["values"], dtype=DTYPES[arg.get("dtype", "f32")])
        if values.size != math.prod(shape):
            raise CalibrationError("runner returned incomplete values")
        produced[name] = values.reshape(shape)
        if name in case["expected"]:
            reference = case["expected"][name]
            if reference.shape != produced[name].shape and reference.size == produced[name].size == 1:
                reference = reference.reshape(produced[name].shape)
            check = compare_output(produced[name], reference, case["meta"]["rtol"], case["meta"]["atol"])
            diagnostics[name] = check
            if not check["ok"]:
                raise CalibrationError(f"hosted correctness failure for {name}: {check}")
    return produced, diagnostics


def run_calibration(root, run_id, mechanism, prepared_pair, transport=None,
                    sleep=time.sleep, clock=time.monotonic, wall_clock=time.time,
                    architecture_sensitive=False):
    """Explicit executor; transport(method,path,body)->(HTTP status, decoded JSON).

    prepared_pair entries: experiment_id, case_dir, baseline_ptx, candidate_ptx,
    optional local_baseline_us/local_candidate_us assertions. Local medians are
    derived from the exact accepted paired evidence, not caller values. PTX paths
    may be lists aligned with meta.kernels. Validation and budget checks complete before first request.
    """
    root, run_id = Path(root).resolve(), component(run_id)
    if not isinstance(mechanism, str) or not mechanism.strip():
        raise CalibrationError("mechanism required")
    with exclusive(root / "control/calibration.lock"):
        selected, provenance = validate_pairs(root, prepared_pair)
        batch = root / "calibration" / run_id
        if batch.exists():
            raise CalibrationError("this run already consumed its singleton calibration batch")
        trigger = calibration_due(root, mechanism, architecture_sensitive)
        if not trigger["due"]:
            return {"status": "not_due", "requests_attempted": 0, "trigger": trigger}
        (batch / "requests").mkdir(parents=True)
        result = {"schema": 1, "run_id": run_id, "mechanism": mechanism,
                  "status": "started", "created_at": wall_clock(),
                  "promotion_count": trigger["promotion_count"], "trigger": trigger,
                  "endpoint": getattr(transport, "base_url", "injected transport" if transport else DEFAULT_URL),
                  "local_provenance": provenance, "comparisons": [],
                  "cases": [{"identities": case["identities"], "suite_id": case["suite_id"]} for case in selected],
                  "uncertainty": "Indicative timing; few cases and repeats. Accumulate compatible observations across runs; no reliable predictor is established."}
        atomic_json(batch / "result.json", result)
        client = ChargedRequests(root, batch, transport or HTTPTransport(), sleep, clock, wall_clock)
        try:
            health = client.request("GET", "/health")
            gpu = json.dumps(health.get("gpu", ""))
            arch = health.get("default_arch")
            if arch is None and ("sm_120" in gpu or "12.0" in gpu):
                arch = "sm_120"
            if "5090" not in gpu or arch != "sm_120" or not health.get("can_compile") or not health.get("can_run"):
                raise CalibrationError("health does not confirm runnable RTX 5090 / sm_120 endpoint")
            result["environment"] = {"health": health, "assembly_arch": arch,
                                     "measurement": {"warmup": 5, "iters": 20, "repetitions": REPETITIONS}}
            for case in selected:
                compiled = {}
                for arm in ("baseline", "candidate"):
                    compiled[arm] = []
                    for path, kernel in zip(case[arm], case["meta"]["kernels"]):
                        response = client.request("POST", "/compile", {"ptx": path.read_text(), "arch": arch, "flags": []})
                        if kernel["entry"] not in response.get("entry_points", []):
                            raise CalibrationError("compiled module omits required entry")
                        compiled[arm].append(response["cubin_id"])
                samples, diagnostics = {"baseline": [], "candidate": []}, []
                for repetition in range(REPETITIONS):
                    for arm in (("baseline", "candidate") if repetition % 2 == 0 else ("candidate", "baseline")):
                        produced, total_us = {}, 0.0
                        for index, (kernel, cubin_id) in enumerate(zip(case["meta"]["kernels"], compiled[arm])):
                            args, need = _arguments(case["meta"], case["inputs"], produced, upto=index)[index]
                            body = {"cubin_id": cubin_id, "entry": kernel["entry"],
                                    "grid": kernel["grid"], "block": kernel["block"],
                                    "smem_bytes": kernel.get("smem_bytes", 0),
                                    "zero_outputs": kernel.get("zero_outputs", []),
                                    "args": args, "warmup": 5, "iters": 20, "max_return": need}
                            response = client.request("POST", "/run", body)
                            outputs, checks = _check_outputs(response, kernel, case)
                            produced.update(outputs)
                            diagnostics.append({"repetition": repetition, "arm": arm, "kernel": index, "outputs": checks})
                            elapsed = response["median_us"]
                            if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
                                raise CalibrationError("invalid hosted timing")
                            total_us += elapsed
                        if not set(case["expected"]).issubset(produced):
                            raise CalibrationError("pipeline failed to produce checked outputs")
                        samples[arm].append(total_us)
                remote_ratio = statistics.median(samples["candidate"]) / statistics.median(samples["baseline"])
                descriptor = case["descriptor"]
                local_ratio = descriptor["local_candidate_us"] / descriptor["local_baseline_us"]
                # Direction near equality is explicitly unresolved, not a rank win.
                direction = lambda ratio: "faster" if ratio < 0.99 else "slower" if ratio > 1.01 else "unresolved"
                local_direction, remote_direction = direction(local_ratio), direction(remote_ratio)
                comparison = {"case_id": case["meta"].get("id"), "suite_id": case["suite_id"],
                              "identities": case["identities"], "local_ratio": local_ratio,
                              "remote_ratio": remote_ratio, "remote_samples_us": samples,
                              "local_direction": local_direction, "remote_direction": remote_direction,
                              "direction_agreement": local_direction == remote_direction if "unresolved" not in (local_direction, remote_direction) else None,
                              "diagnostics": diagnostics,
                              "note": "Direction uses a descriptive 1% deadband, not a significance test."}
                result["comparisons"].append(comparison)
                atomic_json(batch / "result.json", result)
            result["status"] = "completed"
        except Exception as exc:
            result.update(status="stopped", error=str(exc))
        result.update(requests_attempted=len(list((batch / "requests").glob("*.json"))), completed_at=wall_clock())
        atomic_json(batch / "result.json", result)
        append_record(root / "control/calibration-history.jsonl", result)
        mechanism_key = hashlib.sha256(mechanism.encode()).hexdigest()[:24]
        append_record(root / "knowledge/mechanisms" / f"calibration-{mechanism_key}.jsonl", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mechanism", required=True)
    parser.add_argument("--prepared-pair", type=Path, required=True)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--architecture-sensitive", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Required explicit authorization to contact hosted endpoint")
    args = parser.parse_args()
    descriptors = json.loads(args.prepared_pair.read_text())
    if not args.execute:
        selected, provenance = validate_pairs(args.root, descriptors)
        print(json.dumps({"status": "validated_no_requests", "cases": len(selected), **provenance,
                          "trigger": calibration_due(args.root, args.mechanism, args.architecture_sensitive)}, indent=2))
        return 0
    result = run_calibration(args.root, args.run_id, args.mechanism, descriptors,
                             transport=HTTPTransport(args.url), architecture_sensitive=args.architecture_sensitive)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"completed", "not_due"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
