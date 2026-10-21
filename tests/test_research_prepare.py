"""CPU-only tests for compiler isolation, assembly identity, and static gates."""
import copy
import json
import os
from pathlib import Path
import signal
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

from tools import research_cases as cases
from tools import research_prepare as preparation


ROOT = Path(__file__).resolve().parents[1]
PTX = ".version 8.0\n.target sm_89\n.address_size 64\n.visible .entry test_entry() { ret; }\n"


def running(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.split(")", 1)[1].strip().split()[0] != "Z"


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="emmy-prepare-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.case = cases.import_problem(ROOT / "emmy-hackathon-2026/problems/p02_scale_shift", self.root / "case")
        self.ir_name = json.loads((self.case / "case.json").read_text())["ir_paths"][0]

    def compiler(self, body):
        (self.source / "compile.py").write_text(body + "\nprint(" + repr(PTX) + ")\n")

    def compile(self, **kwargs):
        return preparation.compile_ir(self.source, self.case, self.ir_name, self.root / "compiled", **kwargs)

    def test_compiler_gets_only_readonly_semantic_inputs(self):
        self.compiler("""
import json, os, pathlib, sys
assert os.getuid() == 65534 and os.getgid() == 65534
status = pathlib.Path('/proc/self/status').read_text()
assert 'NoNewPrivs:\\t1' in status
p = pathlib.Path(sys.argv[1])
meta = json.loads(p.with_name('meta.json').read_text())
assert 'id' not in meta and 'code' not in meta
assert all('seed' not in a and 'data' not in a for k in meta['kernels'] for a in k['args'])
assert 'arg_order' in meta['kernels'][0] and 'rtol' in meta
assert {x.name for x in p.parent.iterdir()} == {p.name, 'meta.json'}
for target in (pathlib.Path(__file__), p, p.with_name('meta.json')):
    try: target.write_text('changed')
    except PermissionError: pass
    else: raise AssertionError('write escaped readonly compiler boundary')
""")
        result = self.compile()
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual((self.root / "compiled/stdout.ptx").read_text().strip(), PTX.strip())

    def test_evaluator_references_and_network_are_denied(self):
        secret = self.root / "private-reference.txt"
        secret.write_text("expected output must stay outside compiler view")
        self.compiler("""
import pathlib, socket
try: pathlib.Path(""" + repr(str(secret)) + """).read_text()
except PermissionError: pass
else: raise AssertionError('compiler read private reference')
try: socket.socket()
except PermissionError: pass
else: raise AssertionError('compiler created network socket')
""")
        self.assertEqual(self.compile()["exit_code"], 0)

    def test_real_baseline_can_deserialize_and_compile_inside_boundary(self):
        shutil.copyfile(ROOT / "compile.py", self.source / "compile.py")
        result = self.compile()
        self.assertEqual(result["exit_code"], 0)
        self.assertIn(".visible .entry", (self.root / "compiled/stdout.ptx").read_text())

    def test_gpu_ioctl_is_denied(self):
        self.compiler("""
import errno, fcntl
for request in (0, 0x4600, 0x30000001):
    try: fcntl.ioctl(1, request)
    except OSError as e:
        assert e.errno == errno.EPERM, e
    else: raise AssertionError('ioctl unexpectedly succeeded')
""")
        self.assertEqual(self.compile()["exit_code"], 0)

    def test_source_symlinks_are_rejected_before_dereference(self):
        self.compiler("pass")
        secret = self.root / "secret"
        secret.write_text("private")
        (self.source / "sidecar").symlink_to(secret)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.compile()

    def test_timeout_kills_owned_descendants(self):
        self.compiler("""
import subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])
print('child_pid=' + str(child.pid), file=sys.stderr, flush=True)
time.sleep(300)
""")
        with self.assertRaisesRegex(ValueError, "124"):
            self.compile(timeout=0.5)
        text = (self.root / "compiled/stderr.txt").read_text()
        pid = int(text.split("child_pid=")[1].splitlines()[0])
        self.assertFalse(running(pid), "timed-out compiler left a running descendant")

    def test_successful_compiler_cannot_leave_running_descendants(self):
        self.compiler("""
import subprocess, sys
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])
print('child_pid=' + str(child.pid), file=sys.stderr, flush=True)
""")
        self.assertEqual(self.compile()["exit_code"], 0)
        text = (self.root / "compiled/stderr.txt").read_text()
        pid = int(text.split("child_pid=")[1].splitlines()[0])
        try:
            deadline = time.monotonic() + 1
            while running(pid) and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(running(pid), "successful compiler left a running descendant")
        finally:
            if running(pid):
                os.kill(pid, signal.SIGKILL)

    def test_descendants_cannot_escape_process_group(self):
        self.compiler("""
import subprocess, sys
try:
    subprocess.Popen([sys.executable, '-c', 'pass'], start_new_session=True)
except PermissionError: pass
else: raise AssertionError('compiler child escaped process group')
""")
        self.assertEqual(self.compile()["exit_code"], 0)

    def test_invalid_ptx_output_is_rejected(self):
        (self.source / "compile.py").write_text("print('non ASCII \\u2603')\n")
        with self.assertRaisesRegex(ValueError, "invalid PTX"):
            self.compile()

    def test_sass_normalization_preserves_semantic_immediates(self):
        first = "/*0010*/ MOV R0, 0x1; /* 0x1234567890 */"
        same = "/*00a0*/ MOV R0, 0x1; /* 0xaabbccdde0 */"
        changed = "/*0010*/ MOV R0, 0x2; /* 0x1234567890 */"
        self.assertEqual(preparation.normalize_sass(first), preparation.normalize_sass(same))
        self.assertNotEqual(preparation.normalize_sass(first), preparation.normalize_sass(changed))

    def test_real_assembly_cache_targets_and_corruption(self):
        ptx = self.root / "module.ptx"
        ptx.write_text(PTX)
        first = preparation.assemble(ptx, self.root / "cache", "sm_89")
        repeated = preparation.assemble(ptx, self.root / "cache", "sm_89")
        other = preparation.assemble(ptx, self.root / "cache", "sm_120")
        self.assertFalse(first["cache_hit"])
        self.assertTrue(repeated["cache_hit"])
        self.assertNotEqual(first["key"], other["key"])
        self.assertTrue((Path(other["folder"]) / "nvdisasm.txt").exists())
        metadata = Path(other["folder"]) / "assembly.json"
        altered = json.loads(metadata.read_text())
        altered["key"] = "wrong cache identity"
        metadata.write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError, "metadata corrupt"):
            preparation.assemble(ptx, self.root / "cache", "sm_120")
        cubin = Path(first["folder"]) / "module.cubin"
        cubin.write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "corrupt"):
            preparation.assemble(ptx, self.root / "cache", "sm_89")

    def test_static_gate_checks_activation_and_case_identity(self):
        self.compiler("pass")
        def compile_stub(source, case, ir, out):
            Path(out).mkdir(parents=True)
            (Path(out) / "stdout.ptx").write_text(PTX)
            return {"exit_code": 0}
        def assemble_stub(ptx, cache, arch):
            folder = Path(cache) / arch
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "sass.normalized.txt").write_text("MOV R0, 0x1")
            return {"folder": str(folder), "sass_sha256": "same", "registers": [8], "spills": [], "arch": arch, "resources": "REG:8 SHARED:0"}
        with patch.object(preparation, "compile_ir", compile_stub), patch.object(preparation, "assemble", assemble_stub):
            baseline = preparation.prepare(self.source, [self.case], self.root / "baseline", self.root / "fake-cache")
            inactive = preparation.prepare(self.source, [self.case], self.root / "inactive", self.root / "fake-cache", mechanism={"required_sass": [{"pattern": "LDG\\.E\\.128"}]})
            self.assertEqual(inactive["gate"], "inactive")
            matched = preparation.prepare(self.source, [self.case], self.root / "matched", self.root / "fake-cache", baseline=baseline)
            self.assertTrue(matched["runtime_identical_to_baseline"])
            different = copy.deepcopy(baseline)
            different["cases"][0]["case_id"] = "different case"
            mismatched = preparation.prepare(self.source, [self.case], self.root / "mismatched", self.root / "fake-cache", baseline=different)
            self.assertFalse(mismatched["runtime_identical_to_baseline"])
            changed_launch = copy.deepcopy(baseline)
            changed_launch["cases"][0]["launch_contract"][0]["arg_order"].reverse()
            changed = preparation.prepare(self.source, [self.case], self.root / "changed-launch", self.root / "fake-cache", baseline=changed_launch)
            self.assertFalse(changed["runtime_identical_to_baseline"])
            with self.assertRaisesRegex(ValueError, "scope"):
                preparation.prepare(self.source, [self.case], self.root / "unknown-scope", self.root / "fake-cache", mechanism={"required_sass": [{"pattern": "MOV", "case_ids": ["missing-case"]}]})

    def test_activation_cannot_be_satisfied_by_unrelated_case(self):
        second = cases.import_problem(ROOT / "emmy-hackathon-2026/problems/p00_scalar_add", self.root / "second-case")
        def compile_stub(source, case, ir, out):
            Path(out).mkdir(parents=True)
            (Path(out) / "stdout.ptx").write_text(PTX)
            return {"exit_code": 0}
        def assemble_stub(ptx, cache, arch):
            index = Path(ptx).parent.name
            folder = Path(cache) / arch / index
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "sass.normalized.txt").write_text("SHFL R0" if index.startswith("case-0-") else "MOV R0")
            return {"folder": str(folder), "sass_sha256": "same", "registers": [8], "spills": [], "arch": arch, "resources": "REG:8 SHARED:0"}
        with patch.object(preparation, "compile_ir", compile_stub), patch.object(preparation, "assemble", assemble_stub):
            all_cases = preparation.prepare(self.source, [self.case, second], self.root / "all-cases", self.root / "scope-cache", mechanism={"required_sass": [{"pattern": "SHFL"}]})
            self.assertEqual(all_cases["gate"], "inactive")
            first_id = json.loads((self.case / "case.json").read_text())["id"]
            scoped = preparation.prepare(self.source, [self.case, second], self.root / "scoped", self.root / "scope-cache", mechanism={"required_sass": [{"pattern": "SHFL", "case_ids": [first_id]}]})
            self.assertEqual(scoped["gate"], "pass")

    def test_resource_bounds_are_advisory_except_architectural_impossibility(self):
        for arch in ('sm_89','sm_120'):
            target={'arch':arch,'registers':[128],'resources':'REG:128 SHARED:0'}
            bound=preparation.resource_envelope(target,{'block':[128,1,1],'smem_bytes':0})
            self.assertFalse(bound['architecturally_impossible'])
            self.assertEqual(bound['blocks_per_sm_upper_bound'],4)
            self.assertLess(bound['warp_occupancy_upper_bound'],1)
            self.assertIn('Optimistic',bound['limitation'])
            impossible=preparation.resource_envelope(target,{'block':[128,1,1],'smem_bytes':100*1024})
            self.assertTrue(impossible['architecturally_impossible'])


if __name__ == "__main__":
    unittest.main()
