"""Qualification must obey the worker's eight-case bound after all15 expansion."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from tools import research_verify as verify
from tools.published import PUBLISHED_PROBLEMS


class StopAfterGate(Exception):
    pass


class Qualification(unittest.TestCase):
    def test_preparation_requires_explicit_compiler_and_uses_selected_root_for_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'qualification'
            source=Path(temporary)/'compiler'
            args=['verify','--root',str(root),'--prepare']
            with mock.patch('sys.argv',args), mock.patch('sys.stderr'):
                with self.assertRaises(SystemExit) as stopped: verify.main()
            self.assertEqual(stopped.exception.code,2)
            self.assertFalse(root.exists())
            with mock.patch('sys.argv',args+['--source',str(source)]), \
                 mock.patch.object(verify,'prepare_verification',return_value={}) as prepare, \
                 mock.patch('builtins.print'):
                self.assertEqual(verify.main(),0)
            self.assertEqual(prepare.call_args.args[0:2],(root,source))
            self.assertEqual(prepare.call_args.args[3],root/'assembly-cache')

    def test_full_suite_is_dispatched_in_legal_batches_before_any_timing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            specs=[{'id':name,'case_dir':str(root/name),'cubins':[]} for name in
                   (*PUBLISHED_PROBLEMS,'generated-pointwise','generated-reduction')]
            (root/'qualification.json').write_text(json.dumps({
                'source_identity':{'compile.py':'unit fixture'},'specs':specs,'controls':{}}))
            jobs=[]
            class Worker:
                def __init__(self,*args):self.process=mock.Mock(pid=123)
                def close(self):pass
                def run(self,job,**kwargs):
                    if len(jobs)==3: raise StopAfterGate()
                    self_test.assertLessEqual(len(job['cases']),8)
                    self_test.assertEqual(job['mode'],'correctness')
                    jobs.append(job)
                    return {'ok':True,'device':{'name':'CPU fake'}}
            def sanitizer(command,**kwargs):
                job=json.loads(Path(command[-1]).read_text())
                Path(job['out']).write_text('{"ok":true}')
                return subprocess.CompletedProcess(command,0,'CPU fake sanitizer','')
            self_test=self
            with mock.patch.object(verify,'GPUWorker',Worker), \
                 mock.patch.object(verify.subprocess,'run',side_effect=sanitizer):
                with self.assertRaises(StopAfterGate):verify.gpu_verification(root)
            self.assertEqual([len(j['cases']) for j in jobs],[8,8,1])
            self.assertEqual([c['id'] for j in jobs for c in j['cases']],[s['id'] for s in specs])
            # The deliberate stop happens before the next summary checkpoint.
            # Inspect the completed sanitizer result, not that older snapshot.
            self.assertTrue(json.loads((root/'gpu/jobs/m02-sanitizer.result.json').read_text())['ok'])


if __name__=='__main__':unittest.main()
