"""CPU-only orchestration tests. Fake workers NEVER establish CUDA correctness."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import research_pipeline as pipeline
from tools.research_cases import generate_case
from tools.research_store import Store, Invalid, atomic_json, canonical, digest, read_json
from tools.published import PUBLISHED_PROBLEMS


DEVICE={'name':'CPU fake, not GPU evidence','uuid':'GPU-test','context_identity':'fake-context'}


def correctness_result(ok=True):
    return {'ok':ok,'device':DEVICE,'batches':[{'quarantined':not ok,
            'results':[{'ok':ok,'checks':[{'ok':ok}]}]}]}


class FakeWorker:
    instances=[]
    timing_ok=True

    def __init__(self,*args):
        self.jobs=[]; self.closed=False; self.device=DEVICE
        self.instances.append(self)

    def run(self,job,timeout=20):
        self.jobs.append(copy.deepcopy(job))
        if job['mode']=='correctness': result=correctness_result()
        else:
            same=job['case']['cubins']==job['case']['baseline_cubins']
            ratio=1. if same else 1.5
            result={'ok':self.timing_ok,'device':DEVICE,'cache_mode':'fake test',
                    'pairs':[{'complete':True,'pair_index':i,'order':['baseline','candidate'],
                              'samples_us':{'baseline':10.,'candidate':10.*ratio},
                              'candidate_over_baseline':ratio}
                             for i in range(job.get('pair_offset',0),job.get('pair_offset',0)+job['pairs'])]}
        atomic_json(job['out'],result)
        return result

    def close(self,*args,**kwargs): self.closed=True


class Pipeline(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name).resolve()
        self.source=self.root/'source'; self.source.mkdir()
        (self.source/'compile.py').write_text("print('test source only')\n")
        self.store=Store(self.root/'nfs',self.root/'work')
        self.store.init(self.source); self.store.start_run('unit','qualification')
        self.host=self.root/'host-job'; self.host.mkdir()
        self.env=mock.patch.dict(pipeline.os.environ,{'EMMY_JOB_DIR':str(self.host)})
        self.env.start(); FakeWorker.instances.clear(); FakeWorker.timing_ok=True

    def tearDown(self):
        self.env.stop()
        for p in sorted(self.root.rglob('*'),reverse=True):
            if p.is_dir() and not p.is_symlink(): p.chmod(0o755)
        self.temporary.cleanup()

    def experiment(self,name='candidate'):
        self.store.propose({'id':name,'mechanism':'mechanism '+name,'applicability':'f32 valid IR',
            'prediction':'test','probe':'inspect','opportunity':'unknown','budget':{'variants':1},
            'semantic_overlap_decision':'distinct unit fixture'})
        return self.store.freeze(name,self.source)

    def case(self,name='case',family='pointwise'):
        shape={'pointwise':[7],'matmul':[2,3,4],'atomic_matmul':[2,3,4],'softmax':[3,7]}[family]
        path=self.root/name
        generate_case(path,family,12,shape=shape)
        return str(path)

    def prepared(self,cases,out):
        out=Path(out); out.mkdir(parents=True,exist_ok=False)
        rows=[]
        for ci,case in enumerate(cases):
            info=read_json(Path(case)/'case.json'); kernels=[]
            for ki,ir in enumerate(info['ir_paths']):
                targets={}
                for arch in ('sm_89','sm_120'):
                    folder=out/f'{ci}-{ki}-{arch}'; folder.mkdir()
                    data=f'fake {ci} {ki} {arch}'.encode(); (folder/'module.cubin').write_bytes(data)
                    targets[arch]={'folder':str(folder),'hashes':{'module.cubin':digest(data)}}
                kernels.append({'ir':ir,'targets':targets,
                                'compile':{'ptx_sha256':digest(f'fake PTX {ci} {ki} {out}'.encode())}})
            rows.append({'case_id':info['id'],'case_dir':case,'kernels':kernels,
                         'requires_sanitizer':info.get('requires_sanitizer',False),
                         'requires_isolated_first':info.get('requires_isolated_first',False)})
        value={'cases':rows,'gate':'pass','unsupported':[]}
        atomic_json(out/'prepared.json',value)
        return value

    def prepare_fake(self,source,cases,out,*args,**kwargs): return self.prepared(cases,out)

    def fixture_gate(self,source_id,suite_id):
        report={'source_id':source_id,'suite_id':suite_id,'sanitizer_passed':True,
                'cases':[{'problem':p,'case_id':'unit-'+p,'status':'pass','isolated':True,
                          'guards_ok':True,'inputs_ok':True} for p in PUBLISHED_PROBLEMS]}
        return self.store.issue_published_gate(source_id,suite_id,report,
            {'all15-unit-fixture.json':b'Synthetic unit fixture, not GPU evidence'})

    def confirmation_setup(self,family='pointwise',gated=True):
        self.experiment()
        suite=self.store.register_suite('unit-v1',{'families':[family],'confirmation_cases_per_family':1})
        cases=[self.case(family=family)]
        with mock.patch.object(pipeline,'make_cases',return_value=cases), \
             mock.patch.object(pipeline,'prepare',side_effect=self.prepare_fake):
            plan=pipeline.prepare_confirmation(self.store,'candidate',suite)
        if gated:
            manifest=self.store._manifest('candidate')
            for source_id in {manifest['source_id'],manifest['baseline_id']}: self.fixture_gate(source_id,suite)
        return suite,cases,plan

    def host_records(self):
        monitor={'observed':True,'samples':1,'foreign':[],'errors':[]}
        atomic_json(self.host/'gpu-attestation.json',{'host_job_dir':str(self.host),'monitor':monitor,
            'observed_no_contention':True,'exit_code':0,'gpu_uuid':DEVICE['uuid'],'gpu_model':DEVICE['name']})
        atomic_json(self.host/'monitor-result.json',monitor)
        telemetry={'time':0.,'gpu':[[DEVICE['uuid'],DEVICE['name'],'0','0','0','0','40']],
                   'processes':[],'our_host_pids':[],'foreign':[]}
        (self.host/'telemetry.jsonl').write_bytes(canonical(telemetry)+b'\n')
        for name in ('admission.csv','processes.csv','after.csv','processes-after.csv','container.json','command.txt'):
            (self.host/name).write_text('unit test fixture')

    def attest(self):
        self.host_records()
        return pipeline.attest_confirmation(self.store,'candidate')

    def queued_prepared(self,name,cases,gated=True):
        self.experiment(name); self.store.enqueue(name,'family')
        suite=self.store.register_suite('queue-unit',{'families':[]})
        atomic_json(Path(cases[0]).parent/'case-list.json',{'suite_id':suite,'cases':cases})
        queue=self.store.claim_preparation('cpu')
        with mock.patch.object(pipeline,'prepare',side_effect=self.prepare_fake):
            prepared=pipeline._prepare_one(self.store,queue,cases,'cpu')
        if gated:
            manifest=self.store._manifest(name)
            for source_id in {manifest['source_id'],manifest['baseline_id']}: self.fixture_gate(source_id,suite)
        return prepared

    def test_case_labels_reservations_and_identity(self):
        suite=self.store.register_suite('small',{'families':['pointwise']})
        for label in ('../outside','/tmp/escape','a/b','.'):
            with self.assertRaises(Invalid): pipeline.make_cases(self.store,suite,10,label,published=False)
        cases=pipeline.make_cases(self.store,suite,10,'fixed',count=1,published=False)
        expected=(Path(cases[0])/'expected.npz').read_bytes()
        self.assertEqual(cases,pipeline.make_cases(self.store,suite,10,'fixed',count=1,published=False))
        self.assertEqual(expected,(Path(cases[0])/'expected.npz').read_bytes())
        with self.assertRaises(Invalid): pipeline.make_cases(self.store,suite,11,'fixed',count=1,published=False)
        with self.assertRaises(Invalid): pipeline.make_cases(self.store,suite,10,'fixed',count=2,published=False)
        path=Path(cases[0])/'case.json'; record=read_json(path); record['id']='a'*64; atomic_json(path,record)
        with self.assertRaisesRegex(Invalid,'identity'): pipeline._case_identity(cases[0])

    def test_softmax_suite_contains_cooperative_and_serial_shapes(self):
        suite=self.store.register_suite('softmax',{'families':['softmax']})
        cases=pipeline.make_cases(self.store,suite,22,'softmax',published=False)
        specs=[read_json(Path(case)/'case.json')['spec'] for case in cases]
        self.assertEqual([s['shape'] for s in specs[:2]],[[3,127],[5,257]])
        self.assertEqual([s['schedule'] for s in specs[:2]],['cooperative-online-softmax']*2)
        self.assertTrue(all(s['schedule']=='serial-per-cell' for s in specs[2:]))

    def test_prepared_cubin_and_case_tampering_rejected(self):
        case=self.case(); prepared=self.prepared([case],self.root/'prepared')
        pipeline._module_specs(prepared,'x')
        path=Path(prepared['cases'][0]['kernels'][0]['targets']['sm_120']['folder'])/'module.cubin'
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(Invalid,'assembly'): pipeline._module_specs(prepared,'x')

    def test_gpu_ack_retirement_and_stale_result_are_not_trusted(self):
        output=self.root/'output.json'; atomic_json(output,{'ok':True,'stale':True})
        worker=pipeline.GPUWorker.__new__(pipeline.GPUWorker)
        worker.log=io.StringIO(); worker.process=mock.Mock()
        worker.process.poll.return_value=None; worker.process.stdout=io.StringIO(json.dumps(
            {'ok':False,'status':2,'retire_worker':True,'out':str(output),'error':'controlled fault'})+'\n')
        worker.process.stdin=io.StringIO()
        with mock.patch.object(pipeline.select,'select',return_value=([worker.process.stdout],[],[])), \
             mock.patch.object(worker,'close') as close:
            with self.assertRaisesRegex(RuntimeError,'controlled fault'): worker.run({'out':str(output)})
            close.assert_called_once_with(kill=True)
        self.assertFalse(output.exists())

    def test_budget_near_deadline_never_starts_sample(self):
        worker=mock.Mock()
        with self.assertRaises(pipeline.BudgetExhausted):
            pipeline._sample(worker,{},self.root/'result','burst',deadline=pipeline.time.monotonic()+24)
        worker.run.assert_not_called()

    def test_incomplete_blocks_excluded_from_summary(self):
        blocks=[{'complete':True,'baseline_us':[10.,20.],'candidate_us':[9.,18.],'burst_id':'a'},
                {'complete':False,'baseline_us':[10.],'candidate_us':[1.],'burst_id':'b'}]
        summary=pipeline.paired_summary(blocks)
        self.assertEqual(summary['pairs'],2); self.assertAlmostEqual(summary['median_block_ratio'],.9)

    def test_confirmation_requires_cpu_preparation_and_does_not_compile_on_gpu(self):
        self.experiment(); suite=self.store.register_suite('unit-v1',{'families':['pointwise']})
        with self.assertRaisesRegex(Invalid,'prepare-confirmation'):
            pipeline.confirm(self.store,'candidate',suite)

    def test_confirmation_pending_until_attested_and_baseline_not_recompiled(self):
        suite,cases,plan=self.confirmation_setup()
        with mock.patch.object(pipeline,'GPUWorker',FakeWorker), \
             mock.patch.object(pipeline,'prepare',side_effect=AssertionError('GPU must not compile')), \
             mock.patch.object(pipeline,'_baseline_prepared',side_effect=AssertionError('GPU must not compile')):
            result=pipeline.confirm(self.store,'candidate',suite)
        self.assertTrue(result['report']['evaluation_ok'])
        self.assertEqual(result['report']['required_case_ids'],[pipeline._case_identity(cases[0])])
        self.assertIsNone(self.store.status()['active_gpu'])
        with self.assertRaisesRegex(Invalid,'attestation'):
            self.store.decide('candidate','shortlist','unit',result['evidence_id'])
        attested=self.attest()
        self.assertNotEqual(attested['evidence_id'],result['evidence_id'])
        self.store.decide('candidate','shortlist','unit evidence only',attested['evidence_id'])
        self.assertEqual(len(FakeWorker.instances),1)

    def test_native_runtime_attestation_requires_environment_provenance(self):
        suite,_,_=self.confirmation_setup()
        with mock.patch.object(pipeline,'GPUWorker',FakeWorker):
            pipeline.confirm(self.store,'candidate',suite)
        self.host_records()
        (self.host/'container.json').unlink()
        with self.assertRaisesRegex(Invalid,'environment provenance'):
            pipeline.attest_confirmation(self.store,'candidate')
        atomic_json(self.host/'environment.json',{'runtime':'native','dependencies':'unit fixture'})
        receipt=pipeline.attest_confirmation(self.store,'candidate')
        self.assertEqual(receipt['report']['admission_status'],'observed_no_contention')

    def test_failed_timing_cannot_be_promoted_even_if_correctness_passed(self):
        suite,_,_=self.confirmation_setup(); FakeWorker.timing_ok=False
        with mock.patch.object(pipeline,'GPUWorker',FakeWorker):
            result=pipeline.confirm(self.store,'candidate',suite)
        self.assertEqual(result['report']['cases'][0]['status'],'pass')
        self.assertFalse(result['report']['evaluation_ok'])
        attested=self.attest()
        with self.assertRaisesRegex(Invalid,'all15|failure'):
            self.store.decide('candidate','promote','unit',attested['evidence_id'])

    def test_required_sanitizer_failure_blocks_promotion(self):
        suite,_,_=self.confirmation_setup('atomic_matmul')
        with mock.patch.object(pipeline,'GPUWorker',FakeWorker), \
             mock.patch.object(pipeline,'_run_sanitizer',return_value={'ok':False}):
            result=pipeline.confirm(self.store,'candidate',suite)
        self.assertTrue(result['report']['sanitizer_required'])
        self.assertFalse(result['report']['sanitizer_passed'])
        attested=self.attest()
        with self.assertRaises(Invalid): self.store.decide('candidate','shortlist','unit',attested['evidence_id'])

    def test_attestation_requires_matching_host_gpu_and_telemetry(self):
        suite,_,_=self.confirmation_setup()
        with mock.patch.object(pipeline,'GPUWorker',FakeWorker): pipeline.confirm(self.store,'candidate',suite)
        atomic_json(self.host/'gpu-attestation.json',{'host_job_dir':str(self.host),'observed_no_contention':False,
            'exit_code':0,'gpu_uuid':'wrong','gpu_model':'fake'})
        with self.assertRaisesRegex(Invalid,'observations'):
            pipeline.attest_confirmation(self.store,'candidate')
        atomic_json(self.host/'gpu-attestation.json',{'host_job_dir':str(self.host),'observed_no_contention':True,
            'exit_code':0,'gpu_uuid':'wrong','gpu_model':'fake'})
        with self.assertRaisesRegex(Invalid,'GPU identity'):
            pipeline.attest_confirmation(self.store,'candidate')

    def test_confirmation_full_required_case_set_is_enforced(self):
        suite,_,_=self.confirmation_setup()
        with mock.patch.object(pipeline,'GPUWorker',FakeWorker): pipeline.confirm(self.store,'candidate',suite)
        attested=self.attest(); report=copy.deepcopy(attested['report'])
        report['required_case_ids'].append('another-required-case')
        receipt=self.store.issue_evidence('candidate',report)
        with self.assertRaisesRegex(Invalid,'required cases'):
            self.store.decide('candidate','shortlist','unit',receipt)

    def test_coverage_generated_rows_map_all_attention_items(self):
        directory=self.root/'case'; directory.mkdir()
        atomic_json(directory/'case.json',{'spec':{'family':'attention'}})
        row=pipeline._confirmation_row({'id':'case','case_dir':str(directory)},True)
        self.assertEqual(row['scope'],'generated')
        self.assertEqual(row['coverage_items'],['a01_attention','a02_attention_128'])

    def test_correctness_disagreement_rechecks_all_batchmates(self):
        case=self.case()
        queues=[self.queued_prepared(name,[case]) for name in ('a','b')]
        worker=mock.Mock()
        worker.run.side_effect=[correctness_result(False),correctness_result(),correctness_result()]
        out=self.root/'correctness'; out.mkdir()
        outcomes,rows=pipeline.evaluate_correctness(self.store,worker,queues,{},out,2)
        self.assertEqual(set(outcomes.values()),{'quarantined'})
        self.assertEqual(worker.run.call_count,3)
        self.assertEqual([len(c.args[0]['cases']) for c in worker.run.call_args_list],[2,1,1])

    def test_atomic_screening_runs_sanitizer_before_timing_eligibility(self):
        case=self.case(family='atomic_matmul')
        queue=self.queued_prepared('a',[case]); worker=FakeWorker()
        out=self.root/'correctness'; out.mkdir()
        with mock.patch.object(pipeline,'_run_sanitizer',return_value={'ok':False}) as sanitize:
            outcomes,_=pipeline.evaluate_correctness(self.store,worker,[queue],{},out,2)
        sanitize.assert_called_once()
        self.assertEqual(outcomes[queue['id']],'failed')
        self.assertEqual(len(worker.jobs),1)

    def test_noise_scoped_to_run_device_case_set_and_service(self):
        case=self.case()
        queues=[self.queued_prepared(name,[case]) for name in ('a','b')]
        job=self.store.claim_gpu('manual')
        self.store.finish_gpu(job['job_id'],'manual',{q['id']:'waiting_timing' for q in queues})
        with mock.patch.object(pipeline,'GPUWorker',FakeWorker), \
             mock.patch.object(pipeline,'_baseline_prepared',side_effect=AssertionError('no compilation in GPU service')):
            result=pipeline.gpu_step(self.store)
        self.assertEqual(set(result['outcomes'].values()),{'complete'})
        noise=list((self.store.root/'noise').glob('*.json'))
        self.assertEqual(len(noise),1)
        saved=read_json(noise[0])
        self.assertEqual(saved['run_id'],'unit')
        self.assertEqual(saved['device'],DEVICE)
        self.assertEqual(saved['job_id'],result['job']['job_id'])
        self.assertTrue(saved['case_set_id'])
        jobs=FakeWorker.instances[-1].jobs
        self.assertEqual(len([j for j in jobs if j.get('pairs')==16]),1)
        self.assertEqual(len([j for j in jobs if j.get('pairs')==8]),4)

    def test_confirmation_resumes_complete_case_artifacts_without_redrawing(self):
        self.experiment()
        suite=self.store.register_suite('unit-v1',{'families':['pointwise','softmax'],'confirmation_cases_per_family':1})
        cases=[self.case('point'),self.case('soft',family='softmax')]
        with mock.patch.object(pipeline,'make_cases',return_value=cases), \
             mock.patch.object(pipeline,'prepare',side_effect=self.prepare_fake):
            pipeline.prepare_confirmation(self.store,'candidate',suite)
        manifest=self.store._manifest('candidate')
        for source_id in {manifest['source_id'],manifest['baseline_id']}: self.fixture_gate(source_id,suite)
        calls=0
        original=pipeline._block_budget
        def limited(deadline):
            nonlocal calls
            calls+=1
            if calls==4: raise pipeline.BudgetExhausted('unit safe-boundary stop')
            return original(deadline)
        with mock.patch.object(pipeline,'GPUWorker',FakeWorker),mock.patch.object(pipeline,'_block_budget',side_effect=limited):
            partial=pipeline.confirm(self.store,'candidate',suite)
        self.assertTrue(partial['incomplete']); self.assertEqual(len(partial['completed_case_ids']),1)
        self.host_records()
        self.host=self.root/'second-host-job'; self.host.mkdir()
        with mock.patch.dict(pipeline.os.environ,{'EMMY_JOB_DIR':str(self.host)}), \
             mock.patch.object(pipeline,'GPUWorker',FakeWorker), \
             mock.patch.object(pipeline,'make_cases',side_effect=AssertionError('do not redraw')):
            result=pipeline.confirm(self.store,'candidate',suite)
        self.assertTrue(result['report']['evaluation_ok'])
        self.assertEqual(len(FakeWorker.instances),2)
        self.assertEqual(len(result['report']['host_job_dirs']),2)
        self.attest()

    def test_default_suite_contains_all15_even_without_claimed_families(self):
        suite=self.store.register_suite('all-published',{'families':[]})
        cases=pipeline.make_cases(self.store,suite,18,'all-published',count=1)
        problems=[read_json(Path(c)/'case.json')['spec']['problem'] for c in cases]
        self.assertEqual(set(problems),set(PUBLISHED_PROBLEMS)); self.assertEqual(len(problems),15)

    def test_one_case_timing_without_all15_gate_fails_before_cuda_worker(self):
        case=self.case(); queue=self.queued_prepared('a',[case],gated=False)
        job=self.store.claim_gpu('manual')
        self.store.finish_gpu(job['job_id'],'manual',{queue['id']:'waiting_timing'})
        with mock.patch.object(pipeline,'GPUWorker') as constructor:
            with self.assertRaisesRegex(Invalid,'all15'):
                pipeline.gpu_step(self.store)
        constructor.assert_not_called()

    def test_confirmation_debug_subset_cannot_time_or_advance(self):
        suite,_,_=self.confirmation_setup(gated=False)
        with mock.patch.object(pipeline,'GPUWorker',FakeWorker): result=pipeline.confirm(self.store,'candidate',suite)
        self.assertTrue(all(job['mode']=='correctness' for worker in FakeWorker.instances for job in worker.jobs))
        self.assertFalse(result['report']['valid_gpu_evaluation'])
        self.assertFalse(result['report']['benchmark_qualified'])
        attested=self.attest()
        with self.assertRaisesRegex(Invalid,'all15'):
            self.store.decide('candidate','promote','cannot waive subset',attested['evidence_id'],performance_debt='irrelevant')

    def test_all15_debug_confirmation_establishes_gate_without_timing(self):
        self.experiment()
        suite=self.store.register_suite('all15-unit',{'families':[],'confirmation_cases_per_family':1})
        with mock.patch.object(pipeline,'prepare',side_effect=self.prepare_fake):
            pipeline.prepare_confirmation(self.store,'candidate',suite)
        with mock.patch.object(pipeline,'GPUWorker',FakeWorker), \
             mock.patch.object(pipeline,'_run_sanitizer',return_value={'ok':True}):
            result=pipeline.confirm(self.store,'candidate',suite)
        self.assertEqual(len(result['report']['cases']),15)
        self.assertTrue(result['report']['valid_gpu_evaluation'])
        self.assertFalse(result['report']['benchmark_qualified'])
        self.assertFalse(result['report']['timing'])
        self.assertTrue(self.store.require_published_gate(self.store._manifest('candidate')['source_id'],suite))
        attested=self.attest()
        self.store.decide('candidate','shortlist','synthetic all15 fixture only',attested['evidence_id'])

    def all15_prepared_with_prior_gate(self):
        self.experiment()
        suite=self.store.register_suite('all15-unit',{'families':[],'confirmation_cases_per_family':1})
        with mock.patch.object(pipeline,'prepare',side_effect=self.prepare_fake):
            pipeline.prepare_confirmation(self.store,'candidate',suite)
        manifest=self.store._manifest('candidate')
        for source_id in {manifest['source_id'],manifest['baseline_id']}: self.fixture_gate(source_id,suite)
        return suite,manifest

    def test_failed_timing_does_not_reissue_all15_gate(self):
        suite,manifest=self.all15_prepared_with_prior_gate()
        FakeWorker.timing_ok=False
        with mock.patch.object(pipeline,'GPUWorker',FakeWorker), \
             mock.patch.object(pipeline,'_run_sanitizer',return_value={'ok':True}):
            result=pipeline.confirm(self.store,'candidate',suite)
        self.assertTrue(all(c['status']=='pass' for c in result['report']['cases']))
        self.assertFalse(result['report']['evaluation_ok'])
        self.assertIsNone(result['report']['published_gate_id'])
        self.assertEqual(result['report']['calibration_cases'],{})
        with self.assertRaisesRegex(Invalid,'invalidated'):
            self.store.require_published_gate(manifest['source_id'],suite)

    def resume_failed_timing(self,simulate_missing_invalidation=False):
        suite,manifest=self.all15_prepared_with_prior_gate()
        FakeWorker.timing_ok=False
        calls=0; original=pipeline._block_budget
        def limited(deadline):
            nonlocal calls
            calls+=1
            if calls==4: raise pipeline.BudgetExhausted('unit stop after failed paired timing checkpoint')
            return original(deadline)
        with mock.patch.object(pipeline,'GPUWorker',FakeWorker), \
             mock.patch.object(pipeline,'_run_sanitizer',return_value={'ok':True}), \
             mock.patch.object(pipeline,'_block_budget',side_effect=limited):
            partial=pipeline.confirm(self.store,'candidate',suite)
        self.assertTrue(partial['incomplete'])
        if simulate_missing_invalidation:
            # Simulate a crash after the immutable failure/progress writes but
            # before revocation: the old index still looks valid at restart.
            self.fixture_gate(manifest['source_id'],suite)
            self.assertTrue(self.store.require_published_gate(manifest['source_id'],suite))
        self.host_records()
        self.host=self.root/'resumed-host'; self.host.mkdir(); FakeWorker.timing_ok=True
        previous_workers=len(FakeWorker.instances)
        with mock.patch.dict(pipeline.os.environ,{'EMMY_JOB_DIR':str(self.host)}), \
             mock.patch.object(pipeline,'GPUWorker',FakeWorker), \
             mock.patch.object(pipeline,'_run_sanitizer',return_value={'ok':True}):
            result=pipeline.confirm(self.store,'candidate',suite)
        self.assertFalse(result['report']['evaluation_ok'])
        self.assertIsNone(result['report']['published_gate_id'])
        with self.assertRaisesRegex(Invalid,'invalidated'):
            self.store.require_published_gate(manifest['source_id'],suite)
        self.assertTrue(all(job['mode']=='correctness' for worker in FakeWorker.instances[previous_workers:]
                            for job in worker.jobs))

    def test_resumed_failed_timing_cannot_restore_revoked_gate(self):
        self.resume_failed_timing()

    def test_recovery_replays_revocation_when_crash_left_old_gate_valid(self):
        self.resume_failed_timing(simulate_missing_invalidation=True)

    def test_cached_attestation_rechecks_revoked_gate(self):
        suite,_,_=self.confirmation_setup()
        with mock.patch.object(pipeline,'GPUWorker',FakeWorker): pipeline.confirm(self.store,'candidate',suite)
        attested=self.attest()
        self.assertTrue(attested['report']['benchmark_qualified'])
        source=self.store._manifest('candidate')['source_id']
        failure=self.store.put_artifact('later-failure',{'problem':'a02_attention_128'},{'output':b'unit fixture'})
        self.store.invalidate_published_gate(source,suite,failure,'later observed failure')
        with self.assertRaisesRegex(Invalid,'invalidated'):
            pipeline.attest_confirmation(self.store,'candidate')
        with mock.patch.object(pipeline,'GPUWorker') as constructor:
            with self.assertRaisesRegex(Invalid,'invalidated'):
                pipeline.confirm(self.store,'candidate',suite)
        constructor.assert_not_called()

    def test_calibration_binding_comes_from_matching_prepared_arms(self):
        suite,cases,plan=self.confirmation_setup()
        with mock.patch.object(pipeline,'GPUWorker',FakeWorker):
            result=pipeline.confirm(self.store,'candidate',suite)
        case_id=pipeline._case_identity(cases[0])
        binding=result['report']['calibration_cases'][case_id]
        prepared=self.store.verify_object(plan['prepared_id'],'confirmation-prepared')['payload']
        self.assertEqual(binding['case_files_sha256'],{name:digest((Path(cases[0])/name).read_bytes())
            for name in ('meta.json','inputs.npz','expected.npz')})
        self.assertEqual(binding['candidate_ptx_sha256'],[k['compile']['ptx_sha256']
            for k in prepared['prepared']['cases'][0]['kernels']])
        self.assertEqual(binding['baseline_ptx_sha256'],[k['compile']['ptx_sha256']
            for k in prepared['baseline']['cases'][0]['kernels']])


if __name__=='__main__': unittest.main()
