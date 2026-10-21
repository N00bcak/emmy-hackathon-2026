"""Durable, provider-neutral research records. No CUDA, Git, or network calls.

The active agent is the coordinator. This module serializes its shared writes;
it does not decide research strategy or infer success from a process exit code.
Candidate processes must run without write access to this store.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import tempfile
import time
import unicodedata
import uuid

import fcntl
try:
    from tools.published import PUBLISHED_PROBLEMS
except ModuleNotFoundError:
    from published import PUBLISHED_PROBLEMS


COVERAGE = ("m01_matmul_512", "m02_matmul_rect", "f02_row_softmax",
            "a01_attention", "a02_attention_128")
DEFAULT_ROOT = "/root/nfs/emmy/research"
DEFAULT_WORK_ROOT = "/root/bcy/emmy/research-workspaces"
EXCLUDED = {".git", ".agents", ".claude", ".codex", ".ssh", ".aws", ".venv",
            ".env", ".credentials", "venv", "__pycache__", ".pytest_cache", ".DS_Store", "candidates"}
EVALUATION_TREES = {"emmy-hackathon-2026", "tests", "experiments", "suites", "objects",
                    "control", "knowledge", "assembly-cache", "baseline-preparation",
                    "preparation", "confirmation"}


class Invalid(ValueError):
    """A requested operation lacks valid identities, evidence, or ownership."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalize(value):
    if not isinstance(value, str) or not value.strip():
        raise Invalid("nonempty text required")
    normalized = " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold()))
    if not normalized:
        raise Invalid("nonempty words required")
    return normalized


def component(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise Invalid("invalid record identifier")
    return value


def relative(value):
    p = PurePosixPath(value)
    if not value or p.is_absolute() or any(s in {"", ".", ".."} for s in p.parts):
        raise Invalid("unsafe artifact path")
    if any(s in EXCLUDED or "private" in s.casefold() for s in p.parts):
        raise Invalid("private/control paths cannot enter source snapshots")
    return p.as_posix()


def atomic_json(path, value):
    path = Path(path)
    data = canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


class Scheduler:
    """Small scheduling policy, not a research state machine.

    Call only at synchronized batch/block boundaries. A caller finishes an
    already-started paired block even when it crosses the soft burst limit.
    """
    def __init__(self, state=None):
        self.state = dict(state or {"mode": "correctness", "burst_started": None,
                                    "timed_candidates": 0, "last_hypothesis": None})

    def choose(self, correctness, timing, now):
        if not correctness and not timing:
            return {"mode": "idle", "items": []}
        oldest = min((x.get("waiting_since", now) for x in timing), default=now)
        yielded_to_correctness = False
        if self.state["mode"] == "timing":
            started = self.state.get("burst_started")
            spent = now - (now if started is None else started)
            if not timing or (correctness and (self.state["timed_candidates"] >= 4 or spent >= 60)):
                self.state.update(mode="correctness", burst_started=None, timed_candidates=0)
                yielded_to_correctness = bool(correctness)
        if self.state["mode"] == "correctness":
            if not yielded_to_correctness and timing and (not correctness or len(timing) >= 4 or now - oldest >= 30):
                self.state.update(mode="timing", burst_started=now, timed_candidates=0)
        mode = self.state["mode"]
        pool = timing if mode == "timing" else correctness
        count = min(4 - self.state["timed_candidates"], 4) if mode == "timing" else 8
        if count <= 0:  # Nothing else to run; start another bounded timing burst.
            self.state.update(burst_started=now, timed_candidates=0)
            count = 4
        picked = self.interleave(pool, count)
        if mode == "timing":
            self.state["timed_candidates"] += len(picked)
        return {"mode": mode, "items": picked}

    def interleave(self, items, count):
        pending = list(items)
        output = []
        previous = self.state.get("last_hypothesis")
        while pending and len(output) < count:
            index = next((i for i, v in enumerate(pending)
                          if v.get("hypothesis") != previous), 0)
            chosen = pending.pop(index)
            output.append(chosen)
            previous = chosen.get("hypothesis")
        if output:
            self.state["last_hypothesis"] = previous
        return output


class Store:
    def __init__(self, root=None, work_root=None):
        self.root = Path(root or os.environ.get("EMMY_RESEARCH_ROOT", DEFAULT_ROOT)).absolute()
        self.work_root = Path(work_root or os.environ.get("EMMY_RESEARCH_WORK_ROOT", DEFAULT_WORK_ROOT)).absolute()

    @contextmanager
    def _lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        lock = self.root / ".coordinator.lock"
        if lock.is_symlink():
            raise Invalid("coordinator lock cannot be a symlink")
        with lock.open("a+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _state(self):
        path = self.root / "control/state.json"
        if not path.exists():
            raise Invalid("research store is not initialized")
        return read_json(path)

    def _save(self, state):
        state["updated_at"] = time.time()
        atomic_json(self.root / "control/state.json", state)

    def _experiment(self, experiment_id):
        return self.root / "experiments" / component(experiment_id)

    def _manifest(self, experiment_id):
        return read_json(self._experiment(experiment_id) / "manifest.json")

    def _put(self, kind, payload=None, files=None):
        """Caller holds the coordinator lock. Actual bytes, not hash-only records."""
        entries = {}
        for name, data in sorted((files or {}).items()):
            name = relative(name)
            if not isinstance(data, bytes):
                raise Invalid("artifact values must be bytes")
            entries[name] = {"sha256": digest(data), "bytes": len(data)}
        manifest = {"schema": 1, "kind": kind, "payload": payload or {}, "files": entries}
        object_id = digest(canonical(manifest))
        target = self.root / "objects" / object_id
        if target.exists():
            self.verify_object(object_id)
            return object_id
        objects = self.root / "objects"
        objects.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=".object-", dir=objects))
        try:
            (temp / "files").mkdir()
            for name in entries:
                path = temp / "files" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(files[name])
                path.chmod(0o444)
            (temp / "manifest.json").write_bytes(canonical(manifest) + b"\n")
            (temp / "manifest.json").chmod(0o444)
            for folder, _, _ in os.walk(temp, topdown=False):
                if Path(folder) != temp:
                    Path(folder).chmod(0o555)
            os.rename(temp, target)
            target.chmod(0o555)
        except BaseException:
            if temp.exists():
                for folder, _, _ in os.walk(temp):
                    Path(folder).chmod(0o755)
                shutil.rmtree(temp)
            raise
        return object_id

    def _snapshot(self, source):
        source = Path(source)
        if source.is_symlink() or not source.is_dir():
            raise Invalid("source must be a real directory")
        source = source.resolve()
        # Never recursively ingest the store or a work root's sibling records.
        if source == self.root.resolve() or source in self.root.resolve().parents:
            raise Invalid("source must not contain the evidence store")
        files = {}
        for folder, directories, names in os.walk(source, followlinks=False):
            directories[:] = sorted(d for d in directories if d not in EXCLUDED and "private" not in d.casefold())
            for directory in directories:
                if directory in EVALUATION_TREES:
                    raise Invalid('use a minimal compiler-source directory, not an evaluation tree: '+directory)
            for name in directories + names:
                path = Path(folder) / name
                if name in EXCLUDED or "private" in name.casefold() or name.endswith((".pyc", ".cubin")):
                    continue
                if path.is_symlink():
                    raise Invalid(f"source symlink rejected: {path.relative_to(source)}")
                if path.is_file():
                    if name in {'expected.npz','inputs.npz','case.json'} or name.endswith(('.ir.json','.ref.cu')):
                        raise Invalid('use a minimal compiler-source directory; evaluation labels/fixtures are forbidden: '+name)
                    rel = relative(path.relative_to(source).as_posix())
                    files[rel] = path.read_bytes()
        if not files or "compile.py" not in files:
            raise Invalid("source snapshot must contain root compile.py")
        return self._put("source", files=files)

    def snapshot_tree(self, source):
        with self._lock():
            return self._snapshot(source)

    def verify_object(self, object_id, kind=None):
        if not isinstance(object_id, str) or not re.fullmatch(r"[a-f0-9]{64}", object_id):
            raise Invalid("invalid object hash")
        target = self.root / "objects" / object_id
        if target.is_symlink() or (target / "manifest.json").is_symlink() or (target / "files").is_symlink():
            raise Invalid("object symlink rejected")
        try:
            manifest = read_json(target / "manifest.json")
        except (OSError, ValueError) as error:
            raise Invalid(f"object unavailable: {object_id}") from error
        if digest(canonical(manifest)) != object_id or (kind and manifest["kind"] != kind):
            raise Invalid("object manifest identity mismatch")
        actual = set()
        for folder, directories, files in os.walk(target / "files", followlinks=False):
            for name in directories + files:
                path = Path(folder) / name
                if path.is_symlink():
                    raise Invalid("object symlink rejected")
                if path.is_file():
                    rel = relative(path.relative_to(target / "files").as_posix())
                    actual.add(rel)
                    data = path.read_bytes()
                    entry = manifest["files"].get(rel)
                    if entry != {"sha256": digest(data), "bytes": len(data)}:
                        raise Invalid("object bytes do not match identity")
        if actual != set(manifest["files"]):
            raise Invalid("object files missing")
        return manifest

    def restore(self, object_id, destination):
        manifest = self.verify_object(object_id, "source")
        destination = Path(destination)
        if destination.exists() or destination.is_symlink():
            raise Invalid("replay destination must not exist")
        destination.mkdir(parents=True)
        for name in manifest["files"]:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.root / "objects" / object_id / "files" / name, path)
        return str(destination)

    def init(self, baseline):
        with self._lock():
            if (self.root / "control/state.json").exists():
                raise Invalid("already initialized; use status/resume")
            # Candidate compilation runs under another UID. References and
            # evidence must not become readable merely through predictable paths.
            self.root.chmod(0o700)
            for directory in ("control", "knowledge/mechanisms", "knowledge/decisions",
                              "experiments", "objects", "suites"):
                (self.root / directory).mkdir(parents=True, exist_ok=True)
            baseline_id = self._snapshot(baseline)
            state = {"schema": 1, "baseline_id": baseline_id, "baseline_history": [],
                     "coverage": {x: {"status": "open", "evidence_id": None} for x in COVERAGE},
                     "experiments": [], "queue": [], "preparation": {}, "active_gpu": None,
                     "scheduler": Scheduler().state, "runs": {}, "current_run": None,
                     "discovery_enabled":False,
                     "created_at": time.time()}
            self._save(state)
            (self.root / "START.md").write_text(
                "# Kernel research recovery\n\n"
                "Read control/state.json, the assigned experiment's brief.json, notes.md, "
                "handoff.json and manifest.json, then linked knowledge/evidence objects.\n\n"
                "Verify object identities before reuse. Inspect active_gpu and preparation "
                "jobs before continuing. An old heartbeat is not permission to kill a process "
                "or duplicate its work. No Git on the GPU host.\n\n"
                "The active Codex/Claude session is coordinator and sole shared-record writer. "
                "Two coverage responsibilities persist until all five backlog items close. "
                "Workspaces are mutable; objects are immutable source/evidence copies.\n",
                encoding="utf-8")
            return state

    def status(self):
        with self._lock():
            state = self._state()
            state["coverage_slots"] = 0 if all(x["status"] == "closed" for x in state["coverage"].values()) else 2
            state["speed_slots"] = 3 - state["coverage_slots"]
            return state

    def search(self, query):
        terms = set(normalize(query).split())
        matches = []
        for base in (self.root / "experiments", self.root / "knowledge"):
            for path in sorted(base.rglob("*")):
                if path.is_symlink() or not path.is_file() or path.suffix not in {".json", ".md"}:
                    continue
                text = path.read_text(encoding="utf-8")
                words = set(re.findall(r"\w+", text.casefold()))
                overlap = terms & words
                if overlap:
                    matches.append({"path": path.relative_to(self.root).as_posix(),
                                    "matched_terms": sorted(overlap), "score": len(overlap) / len(terms)})
        return sorted(matches, key=lambda m: (-m["score"], m["path"]))[:30]

    def propose(self, brief):
        required = ("mechanism", "applicability", "prediction", "probe", "opportunity",
                    "budget", "semantic_overlap_decision")
        if any(not brief.get(k) for k in required):
            raise Invalid("brief requires mechanism/applicability/prediction/probe/opportunity/budget/semantic_overlap_decision")
        mechanism_key = digest(canonical([normalize(brief["mechanism"]), normalize(brief["applicability"])]))
        with self._lock():
            state = self._state()
            duplicates = []
            for existing in state["experiments"]:
                prior = read_json(self._experiment(existing) / "brief.json")
                if prior["mechanism_key"] == mechanism_key:
                    duplicates.append(existing)
            if duplicates and not brief.get("repeat_reason"):
                raise Invalid("duplicate mechanism/applicability; repeat_reason required: " + ", ".join(duplicates))
            contradictions = []
            for path in sorted((self.root / "knowledge/mechanisms").glob("*.json")):
                prior = read_json(path)
                if prior.get("mechanism_key") == mechanism_key and prior.get("verdict") == "contradicted":
                    contradictions.append(prior["id"])
            if contradictions and not brief.get("contradiction_response"):
                raise Invalid("address recorded contradiction before implementation: " + ", ".join(contradictions))
            experiment_id = component(brief.get("id") or f"exp-{uuid.uuid4().hex[:12]}")
            target = self._experiment(experiment_id)
            if target.exists():
                raise Invalid("experiment already exists")
            related = self.search(brief["mechanism"] + " " + brief["applicability"])
            brief = {**brief, "id": experiment_id, "mechanism_key": mechanism_key,
                     "related_records": related, "duplicate_experiments": duplicates,
                     "contradicted_mechanisms": contradictions,
                     "created_at": time.time()}
            target.mkdir()
            atomic_json(target / "brief.json", brief)
            atomic_json(target / "manifest.json", {"id": experiment_id, "baseline_id": state["baseline_id"],
                        "source_id": None, "confirmation": None, "decision": None, "evidence_ids": []})
            atomic_json(target / "handoff.json", {"next_action": "implement or probe in isolated workspace",
                        "active_job_ids": [], "dependencies": [], "updated_at": time.time()})
            (target / "notes.md").write_text("# Investigation\n\nRecord observations, interpretations, sources, failures, and next tests.\n")
            state["experiments"].append(experiment_id)
            self._save(state)
            return brief

    def record_mechanism(self, record):
        """Coordinator records evidence-backed findings, including failed mechanisms."""
        required = ("id", "mechanism", "applicability", "findings", "evidence_ids", "verdict")
        if any(not record.get(k) for k in required):
            raise Invalid("mechanism record requires id/mechanism/applicability/findings/evidence_ids/verdict")
        if record["verdict"] not in {"supported", "contradicted", "inconclusive"}:
            raise Invalid("unknown mechanism verdict")
        with self._lock():
            self._state()
            for object_id in record["evidence_ids"]:
                self.verify_object(object_id)
            path = self.root / "knowledge/mechanisms" / f"{component(record['id'])}.json"
            if path.exists():
                raise Invalid("mechanism records are append-only; create a new record linking the prior one")
            value = {**record, "mechanism_key": digest(canonical([normalize(record["mechanism"]),
                                                                    normalize(record["applicability"])])),
                     "recorded_at": time.time()}
            value["object_id"] = self._put("mechanism", payload=value)
            atomic_json(path, value)
            return value

    def freeze(self, experiment_id, source):
        with self._lock():
            manifest = self._manifest(experiment_id)
            if manifest["source_id"]:
                raise Invalid("candidate already frozen; create a new experiment for revised source")
            manifest.update(source_id=self._snapshot(source), frozen_at=time.time())
            atomic_json(self._experiment(experiment_id) / "manifest.json", manifest)
            return manifest

    def workspace(self, experiment_id):
        """Materialize baseline in a new isolated plain directory, without Git."""
        with self._lock():
            manifest = self._manifest(experiment_id)
            if manifest["source_id"]:
                raise Invalid("frozen candidate cannot acquire a new mutable workspace")
            if manifest.get("workspace"):
                raise Invalid("workspace already exists; resume the recorded one")
            destination = self.work_root / component(experiment_id)
            self.restore(manifest["baseline_id"], destination)
            manifest["workspace"] = str(destination)
            atomic_json(self._experiment(experiment_id) / "manifest.json", manifest)
            return str(destination)

    def register_suite(self, version, contract, files=None):
        component(version)
        with self._lock():
            suite_id = self._put("suite", payload=contract, files=files)
            destination = self.root / "suites" / f"{version}.json"
            if destination.exists() and read_json(destination)["suite_id"] != suite_id:
                raise Invalid("evaluation version is immutable; choose a new version")
            atomic_json(destination, {"version": version, "suite_id": suite_id})
            return suite_id

    def reserve_confirmation(self, experiment_id, suite_id):
        with self._lock():
            manifest = self._manifest(experiment_id)
            if not manifest["source_id"]:
                raise Invalid("freeze source before drawing confirmation cases")
            self.verify_object(manifest["source_id"], "source")
            self.verify_object(suite_id, "suite")
            if manifest["confirmation"]:
                if manifest["confirmation"]["suite_id"] != suite_id:
                    raise Invalid("confirmation suite already reserved")
                return manifest["confirmation"]
            confirmation = {"source_id": manifest["source_id"], "suite_id": suite_id,
                            "seed": secrets.randbits(128), "reserved_at": time.time()}
            confirmation["id"] = digest(canonical(confirmation))
            manifest["confirmation"] = confirmation
            atomic_json(self._experiment(experiment_id) / "manifest.json", manifest)
            return confirmation

    def handoff(self, experiment_id, record):
        if not record.get("next_action") or not isinstance(record.get("active_job_ids", []), list):
            raise Invalid("handoff requires next_action and active_job_ids")
        with self._lock():
            manifest = self._manifest(experiment_id)
            value = {**record, "source_id": manifest["source_id"], "updated_at": time.time()}
            atomic_json(self._experiment(experiment_id) / "handoff.json", value)
            return value

    def resume(self, experiment_id=None):
        state = self.status()
        output = {"control": state, "warning": "Inspect recorded jobs before restarting; never kill on heartbeat age alone."}
        if experiment_id:
            path = self._experiment(experiment_id)
            manifest = self._manifest(experiment_id)
            for object_id in [manifest.get("source_id"), manifest["baseline_id"], *manifest["evidence_ids"]]:
                if object_id:
                    self.verify_object(object_id)
            output.update(manifest=manifest, brief=read_json(path / "brief.json"),
                          handoff=read_json(path / "handoff.json"), notes=(path / "notes.md").read_text())
        return output

    def start_run(self, run_id, mode="smoke"):
        component(run_id)
        if mode not in {"smoke", "qualification", "research"}:
            raise Invalid("mode must be smoke, qualification, or research")
        limits = {"smoke": (1800, 10), "qualification": (7200, 64), "research": (1800, 10)}
        with self._lock():
            state = self._state()
            if mode=='research' and not state.get('discovery_enabled'):
                raise Invalid('discovery is opt-in; routine bug repair does not activate a research campaign')
            if mode=='research' and state.get('discovery_authorization',{}).get('run_id'):
                raise Invalid('discovery authorization was already used; a new research run needs fresh explicit opt-in')
            if run_id in state["runs"]:
                raise Invalid("run already exists; resume it")
            if state["active_gpu"] or state["preparation"]:
                raise Invalid("inspect/reconcile existing jobs before starting a run")
            seconds, variants = limits[mode]
            run = {"id": run_id, "mode": mode, "started_at": time.time(),
                   "seconds": seconds, "max_variants": variants, "evaluated": [], "closed": False}
            state["runs"][run_id] = run
            state["current_run"] = run_id
            if mode=='research': state['discovery_authorization']['run_id']=run_id
            self._save(state)
            return run

    def activate_discovery(self,reason):
        if not isinstance(reason,str) or not reason.strip(): raise Invalid('record explicit user authorization for discovery')
        with self._lock():
            state=self._state()
            state['discovery_enabled']=True
            state['discovery_authorization']={'reason':reason,'recorded_at':time.time()}
            self._save(state)
            return state['discovery_authorization']

    def deactivate_discovery(self,reason='explicit stop'):
        with self._lock():
            state=self._state(); state['discovery_enabled']=False
            run=state['runs'].get(state['current_run'])
            if run and run['mode']=='research':
                run.update(closed=True,closed_at=time.time(),closed_reason=reason)
            self._save(state)
            return {'discovery_enabled':False,'reason':reason,
                    'note':'New work is disabled. Recorded live jobs must be inspected and cleaned up; no process was killed.'}

    def finish_run(self,run_id=None,reason='run completed'):
        with self._lock():
            state=self._state(); run_id=run_id or state['current_run']
            if run_id not in state['runs']: raise Invalid('unknown run')
            run=state['runs'][run_id]
            run.update(closed=True,closed_at=time.time(),closed_reason=reason)
            if run['mode']=='research': state['discovery_enabled']=False
            self._save(state)
            return run

    def _budget(self, state, now):
        run = state["runs"].get(state["current_run"])
        if run is None:
            raise Invalid("start an explicit bounded run before claiming work")
        if run and (run["closed"] or now - run["started_at"] >= run["seconds"]):
            return False
        return True

    def enqueue(self, experiment_id, hypothesis=None):
        with self._lock():
            state = self._state()
            manifest = self._manifest(experiment_id)
            if not manifest["source_id"]:
                raise Invalid("freeze source before enqueue")
            self.verify_object(manifest["source_id"], "source")
            hypothesis = component(hypothesis or experiment_id)
            if any(q["experiment_id"] == experiment_id for q in state["queue"]):
                raise Invalid("experiment already enqueued")
            if sum(q["status"] in {"queued", "preparing", "ready", "waiting_timing"} for q in state["queue"]) >= 16:
                raise Invalid("ready/preparation queue capacity is 16")
            if sum(q["hypothesis"] == hypothesis for q in state["queue"]) >= 16:
                raise Invalid("hypothesis variant cap is 16; record a follow-up hypothesis")
            queue = {"id": f"queue-{uuid.uuid4().hex[:12]}", "experiment_id": experiment_id,
                     "source_id": manifest["source_id"], "hypothesis": hypothesis,
                     "status": "queued", "enqueued_at": time.time()}
            state["queue"].append(queue)
            self._save(state)
            return queue

    def claim_preparation(self, owner):
        component(owner)
        with self._lock():
            state = self._state()
            if not self._budget(state, time.time()) or len(state["preparation"]) >= 2:
                return None
            run = state["runs"].get(state["current_run"])
            if run and len(run["evaluated"]) >= run["max_variants"]:
                return None
            policy = Scheduler(state["scheduler"])
            items = policy.interleave([q for q in state["queue"] if q["status"] == "queued"], 1)
            if not items:
                return None
            item = items[0]
            item.update(status="preparing", owner=owner)
            state["preparation"][item["id"]] = {"owner": owner, "started_at": time.time(), "job_ids": []}
            state["scheduler"] = policy.state
            self._save(state)
            return dict(item)

    def preparation_jobs(self, queue_id, owner, job_ids):
        with self._lock():
            state = self._state()
            active = state["preparation"].get(queue_id)
            if not active or active["owner"] != owner:
                raise Invalid("preparation ownership mismatch")
            active["job_ids"] = list(job_ids)
            self._save(state)

    def finish_preparation(self, queue_id, owner, artifact_id, gate):
        with self._lock():
            state = self._state()
            active = state["preparation"].get(queue_id)
            if not active or active["owner"] != owner:
                raise Invalid("preparation ownership mismatch")
            if gate not in {"pass", "inactive", "invalid", "failed"}:
                raise Invalid("unknown static gate result")
            self.verify_object(artifact_id)
            item = next(q for q in state["queue"] if q["id"] == queue_id)
            item.update(status="ready" if gate == "pass" else "screened_out",
                        prepared_id=artifact_id, static_gate=gate)
            del state["preparation"][queue_id]
            self._save(state)
            return item

    def put_artifact(self, kind, payload=None, files=None):
        with self._lock():
            return self._put(kind, payload, files)

    def claim_gpu(self, owner, now=None):
        component(owner)
        now = time.time() if now is None else now
        with self._lock():
            state = self._state()
            if state["active_gpu"]:
                raise Invalid("GPU job already recorded; inspect it before continuing")
            if not self._budget(state, now):
                return None
            run = state["runs"].get(state["current_run"])
            known = set(run["evaluated"]) if run else set()
            capacity = run["max_variants"] - len(known) if run else 16
            eligible = [q for q in state["queue"]
                        if q["experiment_id"] in known or capacity > 0]
            policy = Scheduler(state["scheduler"])
            chosen = policy.choose([q for q in eligible if q["status"] == "ready"],
                                   [q for q in eligible if q["status"] == "waiting_timing"], now)
            if not chosen["items"]:
                return None
            if run:
                remaining = run["max_variants"] - len(run["evaluated"])
                # Existing candidates can resume timing without spending another variant.
                known = set(run["evaluated"])
                limited = []
                for item in chosen["items"]:
                    if item["experiment_id"] in known or remaining > 0:
                        limited.append(item)
                        if item["experiment_id"] not in known:
                            run["evaluated"].append(item["experiment_id"])
                            known.add(item["experiment_id"])
                            remaining -= 1
                chosen["items"] = limited
            if not chosen["items"]:
                return None
            job = {"job_id": f"gpu-{uuid.uuid4().hex[:12]}", "owner": owner, "mode": chosen["mode"],
                   "queue_ids": [q["id"] for q in chosen["items"]], "started_at": now,
                   "external_job_ids": [], "burst_id": f"burst-{uuid.uuid4().hex[:12]}"}
            for item in chosen["items"]:
                item["status"] = "active_gpu"
            state["scheduler"] = policy.state
            state["active_gpu"] = job
            self._save(state)
            return job

    def gpu_job_ids(self, job_id, owner, external_job_ids):
        with self._lock():
            state = self._state()
            active = state["active_gpu"]
            if not active or active["job_id"] != job_id or active["owner"] != owner:
                raise Invalid("GPU ownership mismatch")
            active["external_job_ids"] = list(external_job_ids)
            self._save(state)

    def claim_gpu_auxiliary(self, owner, experiment_id, mode="confirmation"):
        """Record isolated confirmation under the same single-GPU ownership lock."""
        component(owner)
        if mode != "confirmation": raise Invalid("unsupported auxiliary GPU mode")
        with self._lock():
            state=self._state()
            if state["active_gpu"]: raise Invalid("another GPU job is recorded; inspect it before continuing")
            if not self._budget(state,time.time()): return None
            self._manifest(experiment_id)
            run=state['runs'][state['current_run']]
            if experiment_id not in run['evaluated']:
                if len(run['evaluated'])>=run['max_variants']: return None
                run['evaluated'].append(experiment_id)
            job={"job_id":f"gpu-{uuid.uuid4().hex[:12]}","owner":owner,"mode":mode,
                 "queue_ids":[],"experiment_id":experiment_id,"started_at":time.time(),
                 "external_job_ids":[],"burst_id":f"burst-{uuid.uuid4().hex[:12]}"}
            state['active_gpu']=job
            self._save(state)
            return job

    def finish_gpu(self, job_id, owner, outcomes):
        """Release our job only after caller has synchronized/retired its worker.

        Outcomes map queue IDs to waiting_timing, ready, complete, quarantined,
        or failed. These operational labels never constitute promotion evidence.
        """
        with self._lock():
            state = self._state()
            active = state["active_gpu"]
            if not active or active["job_id"] != job_id or active["owner"] != owner:
                raise Invalid("GPU ownership mismatch")
            if set(outcomes) != set(active["queue_ids"]):
                raise Invalid("record an outcome for every batch member")
            permitted = {"waiting_timing", "ready", "complete", "quarantined", "failed"}
            if any(x not in permitted for x in outcomes.values()):
                raise Invalid("invalid job outcome")
            for item in state["queue"]:
                if item["id"] in outcomes:
                    item["status"] = outcomes[item["id"]]
                    item["last_job_id"] = job_id
                    if item["status"] == "waiting_timing":
                        item.setdefault("waiting_since", time.time())
            active.update(outcomes=outcomes, finished_at=time.time())
            atomic_json(self.root / "control" / "jobs" / f"{job_id}.json", active)
            state["active_gpu"] = None
            self._save(state)
            return active

    def record_pair_block(self, experiment_id, case_id, baseline_us, candidate_us, burst_id,
                          complete=True, observations=None):
        component(case_id)
        component(burst_id)
        if len(baseline_us) != len(candidate_us) or not baseline_us:
            raise Invalid("paired blocks require equal nonempty samples")
        if any(not isinstance(x, (float, int)) or not math.isfinite(x) or x <= 0
               for x in list(baseline_us) + list(candidate_us)):
            raise Invalid("samples must be positive finite microseconds")
        with self._lock():
            self._manifest(experiment_id)
            path = self._experiment(experiment_id) / "samples.json"
            samples = read_json(path) if path.exists() else {"cases": {}}
            blocks = samples["cases"].setdefault(case_id, [])
            counted = sum(len(b["baseline_us"]) for b in blocks if b["complete"])
            if complete and counted + len(baseline_us) > 200:
                raise Invalid("timed repetition cap is 200 per case")
            block = {"block": len(blocks), "burst_id": burst_id, "complete": bool(complete),
                     "baseline_us": list(baseline_us), "candidate_us": list(candidate_us),
                     "observations": observations or {}, "recorded_at": time.time()}
            block["artifact_id"] = self._put("paired-block", payload={**block, "experiment_id": experiment_id, "case_id": case_id})
            blocks.append(block)
            atomic_json(path, samples)
            return block

    def issue_evidence(self, experiment_id, report, artifacts=None):
        """Evaluator integration API; deliberately absent from the public CLI.

        The evaluator owns issuance. Store permissions must exclude candidate
        users; same-owner arbitrary Python is not an adversarial trust boundary.
        All referenced artifacts are immutable and reverified before decisions.
        """
        with self._lock():
            manifest = self._manifest(experiment_id)
            if not manifest["source_id"] or report.get("source_id") != manifest["source_id"]:
                raise Invalid("evidence source identity mismatch")
            if report.get("baseline_id") != manifest["baseline_id"]:
                raise Invalid("evidence baseline identity mismatch")
            self.verify_object(manifest["source_id"], "source")
            for artifact_id in report.get("artifact_ids", []):
                self.verify_object(artifact_id)
            receipt = {"experiment_id": experiment_id, "issuer": "local-research-evaluator-v1",
                       "issued_at": time.time(), "report": report}
            evidence_id = self._put("evaluation", payload=receipt, files=artifacts)
            manifest["evidence_ids"].append(evidence_id)
            atomic_json(self._experiment(experiment_id) / "manifest.json", manifest)
            atomic_json(self._experiment(experiment_id) / "results.json", {"latest_evidence_id": evidence_id,
                        "evidence_ids": manifest["evidence_ids"]})
            return evidence_id

    def _validate_published_gate(self,source_id,suite_id,report,files=None):
        self.verify_object(source_id,'source'); self.verify_object(suite_id,'suite')
        if report.get('source_id')!=source_id or report.get('suite_id')!=suite_id:
            raise Invalid('all15 correctness source/suite identity mismatch')
        cases=report.get('cases',[])
        problems=[case.get('problem') for case in cases]
        if len(problems)!=len(PUBLISHED_PROBLEMS) or set(problems)!=set(PUBLISHED_PROBLEMS):
            raise Invalid('all15 published correctness is mandatory: missing or duplicate published case')
        if any(not case.get('case_id') or case.get('status')!='pass' or case.get('isolated') is not True
               or case.get('guards_ok') is not True or case.get('inputs_ok') is not True for case in cases):
            raise Invalid('all15 published cases must pass isolated correctness and integrity checks')
        if report.get('sanitizer_passed') is not True:
            raise Invalid('all15 gate requires passing atomic Compute Sanitizer evidence')
        if not files and not report.get('artifact_ids'):
            raise Invalid('all15 correctness gate requires retained actual evidence')
        for object_id in report.get('artifact_ids',[]): self.verify_object(object_id)

    def issue_published_gate(self,source_id,suite_id,report,artifacts=None):
        """Trusted evaluator only: bind all fifteen published passes to exact bytes.

        Correctness-debug execution may establish this gate. No CLI can issue it,
        and no performance debt or subset result can substitute for it.
        """
        with self._lock():
            self._state()
            self._validate_published_gate(source_id,suite_id,report,artifacts)
            receipt={'issuer':'local-published-correctness-v1','source_id':source_id,'suite_id':suite_id,
                     'report':report,'issued_at':time.time(),
                     'scope':'All published correctness only; hidden-set correctness remains unverified.'}
            gate_id=self._put('published-correctness',receipt,artifacts)
            atomic_json(self.root/'control/published-gates'/source_id/(suite_id+'.json'),{'gate_id':gate_id})
            return gate_id

    def published_gate(self,source_id,suite_id,gate_id=None,required=True):
        """Read and verify a source+suite gate; never upgrades historical evidence."""
        self.verify_object(source_id,'source'); self.verify_object(suite_id,'suite')
        index=self.root/'control/published-gates'/source_id/(suite_id+'.json')
        if index.exists() and read_json(index).get('revoked'):
            if not required: return None
            raise Invalid('all15 correctness gate was invalidated by a later failure')
        if gate_id is None:
            if not index.exists():
                if required: raise Invalid('all15 published correctness gate missing for this exact source and suite')
                return None
            gate_id=read_json(index)['gate_id']
        record=self.verify_object(gate_id,'published-correctness')
        receipt=record['payload']
        if receipt.get('issuer')!='local-published-correctness-v1' or receipt.get('source_id')!=source_id or receipt.get('suite_id')!=suite_id:
            raise Invalid('all15 gate is from a different source or evaluation suite')
        self._validate_published_gate(source_id,suite_id,receipt['report'],record['files'])
        return {'gate_id':gate_id,**receipt}

    def require_published_gate(self,source_id,suite_id,gate_id=None):
        return self.published_gate(source_id,suite_id,gate_id,required=True)

    def invalidate_published_gate(self,source_id,suite_id,failure_id,reason):
        with self._lock():
            self.verify_object(source_id,'source'); self.verify_object(suite_id,'suite')
            self.verify_object(failure_id)
            atomic_json(self.root/'control/published-gates'/source_id/(suite_id+'.json'),
                        {'revoked':True,'failure_id':failure_id,'reason':reason,'invalidated_at':time.time()})

    def _accepted_evidence(self, experiment_id, manifest, evidence_id):
        if evidence_id not in manifest["evidence_ids"]:
            raise Invalid("decision requires evaluator-issued evidence linked to this experiment")
        receipt = self.verify_object(evidence_id, "evaluation")["payload"]
        report = receipt["report"]
        confirmation = manifest.get("confirmation")
        if receipt["experiment_id"] != experiment_id or receipt["issuer"] != "local-research-evaluator-v1":
            raise Invalid("evidence issuer/experiment mismatch")
        if report["source_id"] != manifest["source_id"] or report["baseline_id"] != manifest["baseline_id"]:
            raise Invalid("evidence identity mismatch")
        if not confirmation or report.get("confirmation_id") != confirmation["id"] or report.get("suite_id") != confirmation["suite_id"]:
            raise Invalid("fresh reserved confirmation evidence required")
        self.verify_object(confirmation["suite_id"], "suite")
        if not report.get('published_gate_id'):
            raise Invalid('all15 published correctness receipt required; performance debt cannot bypass it')
        self.require_published_gate(manifest['source_id'],confirmation['suite_id'],report['published_gate_id'])
        cases = report.get("cases", [])
        if not cases or any(c.get("status") != "pass" or c.get("isolated") is not True
                            or c.get("guards_ok") is not True or c.get("inputs_ok") is not True for c in cases):
            raise Invalid("every confirmation case must pass isolated correctness and integrity checks")
        if report.get("fresh_confirmation") is not True:
            raise Invalid("fresh confirmation required")
        if report.get('evaluation_ok') is False:
            raise Invalid('evaluation reported a correctness, timing or sanitizer failure')
        if any(value.get('ok') is not True for value in report.get('timing',{}).values() if isinstance(value,dict) and 'ok' in value):
            raise Invalid('failed timing correctness cannot support a decision')
        if report.get('sanitizer_required') and report.get('sanitizer_passed') is not True:
            raise Invalid('required sanitizer evidence failed or missing')
        if report.get('admission_status') == 'pending_host_postcheck':
            raise Invalid('host post-job contention attestation is required')
        if report.get('required_case_ids'):
            identifiers=[c.get('case_id') for c in cases]
            if len(identifiers)!=len(set(identifiers)) or set(identifiers)!=set(report['required_case_ids']):
                raise Invalid('confirmation dropped or duplicated required cases')
        # Case files, outputs, diagnostics and admission records must actually be retained.
        if not report.get("artifact_ids") and not self.verify_object(evidence_id)["files"]:
            raise Invalid("evaluation must retain actual evidence artifacts")
        for artifact_id in report.get("artifact_ids", []):
            self.verify_object(artifact_id)
        return report

    def decide(self, experiment_id, decision, reason, evidence_id=None, performance_debt=None):
        if decision not in {"shortlist", "promote", "reject", "inconclusive"} or not reason.strip():
            raise Invalid("decision and reason required")
        with self._lock():
            state = self._state()
            manifest = self._manifest(experiment_id)
            if decision in {"shortlist", "promote"}:
                report=self._accepted_evidence(experiment_id, manifest, evidence_id)
                self.verify_object(manifest["source_id"], "source")
                if manifest["baseline_id"] != state["baseline_id"]:
                    raise Invalid("stale parent baseline: integrate and retest before shortlist/promotion")
                if decision=='promote' and manifest['source_id']!=state['baseline_id']:
                    for item,coverage in state['coverage'].items():
                        if coverage['status']=='closed':
                            scopes={c.get('scope') for c in report['cases']
                                    if c.get('coverage_item')==item or item in c.get('coverage_items',[])}
                            if not {'published','generated'}.issubset(scopes):
                                raise Invalid('promotion omitted previously closed coverage: '+item)
            record = {"decision": decision, "reason": reason, "evidence_id": evidence_id,
                      "source_id": manifest["source_id"], "parent_baseline_id": manifest["baseline_id"],
                      "performance_debt": performance_debt, "decided_at": time.time()}
            record["decision_id"] = self._put("decision", payload=record)
            manifest["decision"] = record
            atomic_json(self._experiment(experiment_id) / "manifest.json", manifest)
            if decision == "promote":
                state["baseline_history"].append({"source_id": state["baseline_id"], "replaced_by": record["decision_id"]})
                state["baseline_id"] = manifest["source_id"]
            self._save(state)
            return record

    def coverage(self, item, experiment_id=None, evidence_id=None, reopen_reason=None):
        if item not in COVERAGE:
            raise Invalid("unknown coverage item")
        with self._lock():
            state = self._state()
            if reopen_reason:
                state["coverage"][item] = {"status": "open", "reason": reopen_reason, "evidence_id": None}
            else:
                manifest = self._manifest(experiment_id)
                report = self._accepted_evidence(experiment_id, manifest, evidence_id)
                if manifest['source_id']!=state['baseline_id']:
                    raise Invalid('coverage can close only on the integrated working baseline; promote first')
                scopes = {c.get("scope") for c in report["cases"]
                          if c.get("coverage_item") == item or item in c.get('coverage_items',[])}
                if not {"published", "generated"}.issubset(scopes) or not any(
                        c.get("scope") == "regression" or c.get('regression') is True for c in report["cases"]):
                    raise Invalid("coverage requires supplied case, generated variants, and existing regressions")
                if item == "m02_matmul_rect" and not report.get("sanitizer_passed"):
                    raise Invalid("atomic coverage requires Compute Sanitizer evidence")
                state["coverage"][item] = {"status": "closed", "experiment_id": experiment_id, "evidence_id": evidence_id}
            self._save(state)
            return state["coverage"][item]

    def export(self, experiment_id, inbox):
        """Copy evaluated shortlist to a *new* inbox subdirectory, never compile.py."""
        with self._lock():
            manifest = self._manifest(experiment_id)
            decision = manifest.get("decision") or {}
            if decision.get("decision") not in {"shortlist", "promote"}:
                raise Invalid("only evaluated shortlisted/promoted candidates may be exported")
            self._accepted_evidence(experiment_id, manifest, decision["evidence_id"])
            inbox = Path(inbox)
            if inbox.name != "candidates" or inbox.is_symlink():
                raise Invalid("export destination must be a nonsymlink candidates inbox")
            for ancestor in [inbox, *inbox.parents]:
                if ancestor.is_symlink():
                    raise Invalid("export path cannot traverse symlinks")
            target = inbox / component(experiment_id)
            if target.exists():
                raise Invalid("export already exists; never overwrite review candidates")
            self.restore(manifest["source_id"], target / "source")
            shutil.copytree(self._experiment(experiment_id), target / "experiment")
            for object_id in [decision["evidence_id"], manifest["confirmation"]["suite_id"]]:
                self.verify_object(object_id)
                shutil.copytree(self.root / "objects" / object_id, target / "evidence" / object_id)
            report = self.verify_object(decision["evidence_id"])["payload"]["report"]
            for object_id in report.get("artifact_ids", []):
                if not (target / "evidence" / object_id).exists():
                    shutil.copytree(self.root / "objects" / object_id, target / "evidence" / object_id)
            return str(target)
