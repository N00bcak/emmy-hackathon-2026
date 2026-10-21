"""CPU-only research control checks: no network, Git, compiler or CUDA jobs."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.research_store import COVERAGE, Invalid, Scheduler, Store, read_json
from tools.published import PUBLISHED_PROBLEMS


class ResearchStore(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name).resolve()
        self.source = self.base / "source"
        self.source.mkdir()
        (self.source / "compile.py").write_text("print('baseline')\n")
        self.store = Store(self.base / "nfs", self.base / "work")
        self.store.init(self.source)
        self.store.start_run("test", "qualification")

    def tearDown(self):
        # Read-only snapshots are deliberately immutable under ordinary tools.
        for folder in sorted(self.base.rglob("*"), reverse=True):
            if folder.is_dir() and not folder.is_symlink():
                folder.chmod(0o755)
        self.temp.cleanup()

    def brief(self, name="one", mechanism=None):
        return {"id": name, "mechanism": mechanism or "vectorized loads",
                "applicability": "aligned pointwise vectors", "prediction": "fewer loads",
                "probe": "inspect SASS vector width", "opportunity": "bound: 4 loads to 1",
                "budget": {"variants": 4, "seconds": 60},
                "semantic_overlap_decision": "distinct from prior mechanism or explicit repeat"}

    def frozen(self, name="one", mechanism=None):
        self.store.propose(self.brief(name, mechanism))
        return self.store.freeze(name, self.source)

    def evidence(self, name="one", item=None, isolated=True):
        manifest = self.store._manifest(name)
        suite_id = self.store.register_suite("tests-v1", {"tolerances": {"rtol": .001, "atol": .001}})
        confirmation = self.store.reserve_confirmation(name, suite_id)
        case = {"case_id": "random-1", "status": "pass", "isolated": isolated,
                "guards_ok": True, "inputs_ok": True}
        cases = [{**case, "coverage_item": item, "scope": "published"},
                 {**case, "case_id": "random-2", "coverage_item": item, "scope": "generated"},
                 {**case, "case_id": "old-1", "scope": "regression"}]
        report = {"source_id": manifest["source_id"], "baseline_id": manifest["baseline_id"],
                  "suite_id": suite_id, "confirmation_id": confirmation["id"],
                  "fresh_confirmation": True, "cases": cases, "sanitizer_passed": True,
                  "timing": {"decision": "coverage gain"}}
        # Synthetic evaluator fixture only; no test here claims real GPU passes.
        gate_report={'source_id':manifest['source_id'],'suite_id':suite_id,'sanitizer_passed':True,
                     'cases':[{'problem':p,'case_id':'published-'+p,'status':'pass','isolated':True,
                               'guards_ok':True,'inputs_ok':True} for p in PUBLISHED_PROBLEMS]}
        report['published_gate_id']=self.store.issue_published_gate(manifest['source_id'],suite_id,gate_report,
                                                                  {'all15-test-fixture.json':b'unit test fixture, not GPU evidence'})
        # This helper acts as the trusted evaluator, never as candidate code.
        evidence_id = self.store.issue_evidence(name, report, {"outputs.json": b'{"y":[1.0]}',
                                                              "admission.txt": b"test fixture"})
        return evidence_id, report

    def test_source_replay_preserves_actual_bytes_and_excludes_private(self):
        private = self.source / ".agents" / "skills" / "private-runtime"
        private.mkdir(parents=True)
        (private / "key.txt").write_text("never copy")
        (self.source / ".git").write_text("gitdir: hidden")
        source_id = self.store.snapshot_tree(self.source)
        manifest = self.store.verify_object(source_id, "source")
        self.assertEqual(set(manifest["files"]), {"compile.py"})
        destination = self.base / "replay"
        self.store.restore(source_id, destination)
        self.assertEqual((destination / "compile.py").read_bytes(), (self.source / "compile.py").read_bytes())
        with self.assertRaises(Invalid):
            self.store.restore(source_id, destination)
        self.assertEqual(source_id, self.store.snapshot_tree(self.source))

    def test_source_symlink_and_object_tampering_rejected(self):
        (self.source / "escape.py").symlink_to("/etc/passwd")
        with self.assertRaises(Invalid):
            self.store.snapshot_tree(self.source)
        (self.source / "escape.py").unlink()
        source_id = self.store.snapshot_tree(self.source)
        path = self.store.root / "objects" / source_id / "files" / "compile.py"
        path.chmod(0o644)
        path.write_text("changed")
        with self.assertRaises(Invalid):
            self.store.verify_object(source_id)

    def test_whole_checkout_or_reference_assets_cannot_enter_compiler_snapshot(self):
        for name in ('expected.npz','inputs.npz','case.json','kernel.ir.json','kernel.ref.cu'):
            with self.subTest(name=name):
                path=self.source/name; path.write_bytes(b'reference or fixture')
                with self.assertRaisesRegex(Invalid,'minimal compiler-source'):
                    self.store.snapshot_tree(self.source)
                path.unlink()
        directory=self.source/'emmy-hackathon-2026'; directory.mkdir()
        with self.assertRaisesRegex(Invalid,'minimal compiler-source'):
            self.store.snapshot_tree(self.source)

    def test_reject_missing_files_and_object_file_root_symlink(self):
        source_id = self.store.snapshot_tree(self.source)
        folder = self.store.root / "objects" / source_id
        folder.chmod(0o755)
        files = folder / "files"
        files.chmod(0o755)
        files.rename(folder / "old")
        files.symlink_to(folder / "old", target_is_directory=True)
        with self.assertRaises(Invalid):
            self.store.verify_object(source_id)

    def test_duplicate_check_precedes_candidate_work(self):
        self.store.propose(self.brief())
        duplicate = self.brief("two", "  VECTORIZED  LOADS!!!")
        with self.assertRaisesRegex(Invalid, "duplicate"):
            self.store.propose(duplicate)
        self.assertFalse((self.store.root / "experiments/two").exists())
        duplicate["repeat_reason"] = "new assembler version"
        result = self.store.propose(duplicate)
        self.assertEqual(result["duplicate_experiments"], ["one"])
        self.assertTrue(result["related_records"])
        with self.assertRaises(Invalid):
            self.store.propose({**self.brief("three", "other"), "semantic_overlap_decision": ""})

    def test_concurrent_shared_writes_are_serialized(self):
        def propose(index):
            other = Store(self.store.root, self.store.work_root)
            return other.propose(self.brief(f"p{index}", f"mechanism {index}"))["id"]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(propose, range(8)))
        self.assertEqual(set(self.store.status()["experiments"]), set(results))

    def test_recorded_contradictions_require_response_before_work(self):
        artifact = self.store.put_artifact("probe", {"finding": "no vector instructions"}, {"sass.txt": b"LDG.E"})
        brief = self.brief()
        self.store.record_mechanism({"id": "no-vectors", "mechanism": brief["mechanism"],
                                    "applicability": brief["applicability"], "findings": "compiler scalarizes",
                                    "verdict": "contradicted", "evidence_ids": [artifact]})
        with self.assertRaisesRegex(Invalid, "contradiction"):
            self.store.propose(brief)
        brief["contradiction_response"] = "new lowering removes the scalarization seen in no-vectors"
        result = self.store.propose(brief)
        self.assertEqual(result["contradicted_mechanisms"], ["no-vectors"])

    def test_freeze_then_confirmation_resumes_without_redrawing(self):
        self.store.propose(self.brief())
        suite = self.store.register_suite("v1", {"tolerance": 1e-3})
        with self.assertRaises(Invalid):
            self.store.reserve_confirmation("one", suite)
        frozen = self.store.freeze("one", self.source)
        first = self.store.reserve_confirmation("one", suite)
        self.assertEqual(first["source_id"], frozen["source_id"])
        self.assertEqual(first, Store(self.store.root).reserve_confirmation("one", suite))
        with self.assertRaises(Invalid):
            self.store.freeze("one", self.source)
        with self.assertRaises(Invalid):
            self.store.register_suite("v1", {"tolerance": 1e-2})

    def test_workspace_is_plain_separate_and_never_overwritten(self):
        self.store.propose(self.brief())
        path = Path(self.store.workspace("one"))
        self.assertEqual(path.parent, self.store.work_root)
        self.assertFalse((path / ".git").exists())
        (path / "compile.py").write_text("edited")
        with self.assertRaises(Invalid):
            self.store.workspace("one")
        source = self.store.freeze("one", path)["source_id"]
        self.assertNotEqual(source, self.store.status()["baseline_id"])

    def test_promotion_requires_evaluator_evidence_and_isolated_confirmation(self):
        self.frozen()
        with self.assertRaises(Invalid):
            self.store.decide("one", "promote", "trust me", evidence_id="a" * 64)
        evidence, _ = self.evidence(isolated=False)
        with self.assertRaisesRegex(Invalid, "isolated"):
            self.store.decide("one", "shortlist", "fast", evidence)
        evidence, report = self.evidence()
        report["fresh_confirmation"] = False
        not_fresh = self.store.issue_evidence("one", report, {"outputs.json": b"{}"})
        with self.assertRaisesRegex(Invalid, "fresh"):
            self.store.decide("one", "promote", "fast", not_fresh)
        self.store.decide("one", "shortlist", "verified coverage", evidence)

    def test_stale_baseline_cannot_promote(self):
        self.frozen("old", "old question")
        self.store.propose(self.brief("new", "new question"))
        new_source = self.base / "newsource"
        new_source.mkdir()
        (new_source / "compile.py").write_text("print('improved')\n")
        new_id = self.store.freeze("new", new_source)["source_id"]
        evidence_new, _ = self.evidence("new")
        evidence_old, _ = self.evidence("old")
        self.store.decide("new", "promote", "confirmed", evidence_new)
        self.assertEqual(self.store.status()["baseline_id"], new_id)
        with self.assertRaisesRegex(Invalid, "stale"):
            self.store.decide("old", "promote", "old evidence", evidence_old)

    def test_all_five_gaps_keep_coverage_responsibilities_until_closed(self):
        self.assertEqual(self.store.status()["coverage_slots"], 2)
        for i, item in enumerate(COVERAGE):
            name = f"cover-{i}"
            self.frozen(name, f"lower {item}")
            evidence, _ = self.evidence(name, item)
            self.store.coverage(item, name, evidence)
            self.assertEqual(self.store.status()["coverage_slots"], 0 if i == 4 else 2)
        self.store.coverage(COVERAGE[0], reopen_reason="fresh randomized failure")
        self.assertEqual(self.store.status()["coverage_slots"], 2)

    def test_atomic_coverage_requires_sanitizer(self):
        self.frozen()
        evidence, report = self.evidence(item="m02_matmul_rect")
        report["sanitizer_passed"] = False
        other = self.store.issue_evidence("one", report, {"outputs.json": b"{}"})
        with self.assertRaisesRegex(Invalid, "Sanitizer"):
            self.store.coverage("m02_matmul_rect", "one", other)
        self.store.coverage("m02_matmul_rect", "one", evidence)

    def test_coverage_cannot_close_on_unintegrated_candidate(self):
        self.store.propose(self.brief())
        (self.source/'compile.py').write_text("print('new candidate')\n")
        self.store.freeze('one',self.source)
        evidence,_=self.evidence(item='f02_row_softmax')
        with self.assertRaisesRegex(Invalid,'integrated working baseline'):
            self.store.coverage('f02_row_softmax','one',evidence)

    def test_promotion_cannot_omit_previously_closed_coverage(self):
        self.frozen('covered','softmax lowering')
        evidence,_=self.evidence('covered',item='f02_row_softmax')
        self.store.coverage('f02_row_softmax','covered',evidence)
        self.store.propose(self.brief('speed','faster pointwise'))
        (self.source/'compile.py').write_text("print('speed candidate')\n")
        self.store.freeze('speed',self.source)
        evidence,_=self.evidence('speed')
        with self.assertRaisesRegex(Invalid,'previously closed coverage'):
            self.store.decide('speed','promote','new faster source',evidence)

    def test_export_requires_shortlist_and_does_not_overwrite_owner_compiler(self):
        self.frozen()
        inbox = self.base / "owner" / "candidates"
        with self.assertRaises(Invalid):
            self.store.export("one", inbox)
        evidence, _ = self.evidence()
        self.store.decide("one", "shortlist", "confirmed", evidence)
        destination = Path(self.store.export("one", inbox))
        self.assertTrue((destination / "source/compile.py").exists())
        self.assertTrue((destination / "evidence" / evidence / "files/outputs.json").exists())
        self.assertFalse((inbox.parent / "compile.py").exists())
        with self.assertRaises(Invalid):
            self.store.export("one", inbox)
        with self.assertRaises(Invalid):
            self.store.export("one", self.base / "not-an-inbox")

    def test_two_preparation_slots_and_inactive_static_gate(self):
        artifact = self.store.put_artifact("static-evidence", {"mechanism_active": False}, {"ptxas.txt": b"no change"})
        for i in range(3):
            self.frozen(f"v{i}", f"variant {i}")
            self.store.enqueue(f"v{i}", "h")
        a = self.store.claim_preparation("cpu-1")
        b = self.store.claim_preparation("cpu-2")
        self.assertIsNone(self.store.claim_preparation("cpu-3"))
        with self.assertRaises(Invalid):
            self.store.finish_preparation(a["id"], "not-owner", artifact, "pass")
        self.store.finish_preparation(a["id"], "cpu-1", artifact, "inactive")
        self.store.finish_preparation(b["id"], "cpu-2", artifact, "invalid")
        self.assertIsNone(self.store.claim_gpu("gpu"))
        self.assertIsNotNone(self.store.claim_preparation("cpu-1"))

    def test_gpu_ownership_batch_outcomes_and_recovery(self):
        self.frozen()
        self.store.enqueue("one")
        item = self.store.claim_preparation("cpu")
        static = self.store.put_artifact("static", {}, {"cubin.bin": b"actual test bytes"})
        self.store.finish_preparation(item["id"], "cpu", static, "pass")
        job = self.store.claim_gpu("gpu")
        self.store.gpu_job_ids(job["job_id"], "gpu", ["container-process-123"])
        resumed = Store(self.store.root).resume("one")
        self.assertEqual(resumed["control"]["active_gpu"]["external_job_ids"], ["container-process-123"])
        with self.assertRaises(Invalid):
            self.store.claim_gpu("other")
        with self.assertRaises(Invalid):
            self.store.finish_gpu(job["job_id"], "gpu", {})
        self.store.finish_gpu(job["job_id"], "gpu", {item["id"]: "waiting_timing"})
        timed = self.store.claim_gpu("gpu")
        self.assertEqual(timed["mode"], "timing")

    def test_sample_state_preserves_bursts_and_excludes_incomplete_blocks(self):
        self.frozen()
        self.store.record_pair_block("one", "case", [10.] * 8, [9.] * 8, "burst-a")
        self.store.record_pair_block("one", "case", [11.] * 8, [10.] * 8, "burst-b", complete=False)
        self.store.record_pair_block("one", "case", [10.] * 192, [9.] * 192, "burst-c")
        with self.assertRaises(Invalid):
            self.store.record_pair_block("one", "case", [10.], [9.], "burst-d")
        samples = read_json(self.store.root / "experiments/one/samples.json")["cases"]["case"]
        self.assertEqual([b["burst_id"] for b in samples], ["burst-a", "burst-b", "burst-c"])
        self.assertEqual(sum(len(b["baseline_us"]) for b in samples if b["complete"]), 200)
        for block in samples:
            self.store.verify_object(block["artifact_id"], "paired-block")

    def test_variant_budget_does_not_prevent_admitted_candidate_finishing(self):
        self.store.start_run("smoke")
        static = self.store.put_artifact("static", {}, {"cubin": b"test"})
        queued = []
        for i in range(11):
            name = f"v-{i}"
            self.frozen(name, f"variant {i}")
            queued.append(self.store.enqueue(name, "hypothesis"))
            prepared = self.store.claim_preparation("cpu")
            self.store.finish_preparation(prepared["id"], "cpu", static, "pass")
        admitted = []
        while len(admitted) < 10:
            job = self.store.claim_gpu("gpu")
            self.assertEqual(job["mode"], "correctness")
            admitted += job["queue_ids"]
            self.store.finish_gpu(job["job_id"], "gpu", {q: "complete" for q in job["queue_ids"]})
        self.assertEqual(len(admitted), 10)
        self.assertIsNone(self.store.claim_gpu("gpu"))
        # A continuing candidate may still consume its budgeted timing work.
        with self.store._lock():
            state = self.store._state()
            first = next(q for q in state["queue"] if q["id"] == admitted[0])
            first.update(status="waiting_timing", waiting_since=0)
            self.store._save(state)
        resumed = self.store.claim_gpu("gpu")
        self.assertEqual(resumed["mode"], "timing")
        self.assertEqual(resumed["queue_ids"], [admitted[0]])

    def test_capacity_backpressure_and_source_identity_in_receipts(self):
        for i in range(16):
            name = f"v-{i}"
            self.frozen(name, f"variant {i}")
            self.store.enqueue(name, "hypothesis")
        self.frozen("overflow", "another")
        with self.assertRaisesRegex(Invalid, "capacity"):
            self.store.enqueue("overflow", "another")
        _, report = self.evidence("v-0")
        report["source_id"] = "f" * 64
        with self.assertRaisesRegex(Invalid, "source"):
            self.store.issue_evidence("v-0", report, {"outputs": b"{}"})

    def test_tampered_retained_evidence_prevents_export(self):
        self.frozen()
        evidence, _ = self.evidence()
        self.store.decide("one", "shortlist", "confirmed", evidence)
        path = self.store.root / "objects" / evidence / "files/outputs.json"
        path.chmod(0o644)
        path.write_text("tampered")
        with self.assertRaises(Invalid):
            self.store.export("one", self.base / "owner/candidates")

    def test_cli_does_not_offer_receipt_issuance(self):
        tool = Path(__file__).resolve().parents[1] / "tools/research.py"
        result = subprocess.run([sys.executable, str(tool), "--root", str(self.store.root), "status"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(json.loads(result.stdout)["coverage"]), 5)
        help_result = subprocess.run([sys.executable, str(tool), "--help"], capture_output=True, text=True, timeout=10)
        self.assertNotIn("issue-evidence", help_result.stdout)

    def test_claims_require_explicit_bounded_run(self):
        other = Store(self.base / "no-run", self.base / "other-work")
        other.init(self.source)
        with self.assertRaisesRegex(Invalid, "bounded run"):
            other.claim_preparation("cpu")
        with self.assertRaisesRegex(Invalid, "bounded run"):
            other.claim_gpu("gpu")
        self.assertEqual(other.root.stat().st_mode & 0o777, 0o700)

    def test_all15_gate_rejects_missing_failed_or_foreign_source_suite(self):
        self.frozen()
        evidence,report=self.evidence()
        gate=self.store.require_published_gate(report['source_id'],report['suite_id'])
        body=copy.deepcopy(gate['report'])
        body['cases'].pop()
        with self.assertRaisesRegex(Invalid,'all15'):
            self.store.issue_published_gate(report['source_id'],report['suite_id'],body,{'output':b'fixture'})
        body=copy.deepcopy(gate['report']); body['cases'][-1]['status']='fail'
        with self.assertRaisesRegex(Invalid,'all15'):
            self.store.issue_published_gate(report['source_id'],report['suite_id'],body,{'output':b'fixture'})
        other_suite=self.store.register_suite('different-suite',{'tolerance':.001})
        with self.assertRaisesRegex(Invalid,'different source or evaluation suite'):
            self.store.require_published_gate(report['source_id'],other_suite,gate['gate_id'])
        with self.assertRaisesRegex(Invalid,'gate missing'):
            self.store.require_published_gate(report['source_id'],other_suite)

    def test_missing_all15_receipt_cannot_be_waived_as_performance_debt(self):
        self.frozen(); _,report=self.evidence()
        report.pop('published_gate_id')
        evidence=self.store.issue_evidence('one',report,{'outputs':b'unit fixture'})
        with self.assertRaisesRegex(Invalid,'performance debt cannot bypass'):
            self.store.decide('one','promote','partial correctness is insufficient',evidence,performance_debt='any')

    def test_later_failure_revokes_even_an_explicit_older_gate(self):
        self.frozen(); _,report=self.evidence()
        gate_id=report['published_gate_id']
        failure=self.store.put_artifact('correctness-failure',{'problem':'a02_attention_128'},{'stderr':b'unit fixture'})
        self.store.invalidate_published_gate(report['source_id'],report['suite_id'],failure,'later published failure')
        with self.assertRaisesRegex(Invalid,'invalidated'):
            self.store.require_published_gate(report['source_id'],report['suite_id'],gate_id)
        self.assertIsNone(self.store.published_gate(report['source_id'],report['suite_id'],required=False))

    def test_discovery_requires_fresh_opt_in_and_can_stop_without_killing_jobs(self):
        self.assertFalse(self.store.status()['discovery_enabled'])
        with self.assertRaisesRegex(Invalid,'opt-in'): self.store.start_run('research','research')
        self.store.activate_discovery('explicit unit-test authorization')
        self.store.start_run('research','research')
        with self.assertRaisesRegex(Invalid,'fresh explicit opt-in'):
            self.store.start_run('another','research')
        self.store.finish_run('research')
        self.assertFalse(self.store.status()['discovery_enabled'])
        self.store.activate_discovery('new explicit authorization')
        self.store.start_run('second','research')
        stopped=self.store.deactivate_discovery()
        self.assertFalse(stopped['discovery_enabled'])
        self.assertTrue(self.store.status()['runs']['second']['closed'])


class Scheduling(unittest.TestCase):
    def items(self, count, waiting=0, prefix="x"):
        return [{"id": f"{prefix}{i}", "hypothesis": f"h{i % 3}", "waiting_since": waiting} for i in range(count)]

    def test_timing_is_not_starved_by_continuously_refilled_correctness(self):
        policy = Scheduler()
        self.assertEqual(policy.choose(self.items(20), self.items(1, 0, "t"), 20)["mode"], "correctness")
        selected = policy.choose(self.items(20), self.items(1, 0, "t"), 31)
        self.assertEqual(selected["mode"], "timing")

    def test_correctness_gets_a_turn_after_timing_limit_even_with_many_survivors(self):
        policy = Scheduler()
        self.assertEqual(policy.choose(self.items(20), self.items(20, 0, "t"), 10)["mode"], "timing")
        turn = policy.choose(self.items(20), self.items(20, 0, "t"), 11)
        self.assertEqual(turn["mode"], "correctness")
        self.assertLessEqual(len(turn["items"]), 8)
        self.assertEqual(policy.choose(self.items(20), self.items(20, 0, "t"), 12)["mode"], "timing")

    def test_soft_time_cap_yields_at_next_block_boundary(self):
        policy = Scheduler()
        self.assertEqual(policy.choose(self.items(8), self.items(1, 0, "t"), 31)["mode"], "timing")
        resumed = Scheduler(copy.deepcopy(policy.state))
        self.assertEqual(resumed.choose(self.items(8), self.items(1, 0, "t"), 92)["mode"], "correctness")

    def test_preparation_round_robin_avoids_monopolization(self):
        policy = Scheduler()
        items = [{"id": str(i), "hypothesis": "a" if i < 8 else "b"} for i in range(16)]
        selected = policy.interleave(items, 6)
        self.assertEqual([s["hypothesis"] for s in selected], ["a", "b", "a", "b", "a", "b"])


if __name__ == "__main__":
    unittest.main()
