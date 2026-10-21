"""Execute CPU preparation and bounded GPU research jobs inside the container.

GPU entrypoints MUST be invoked by the private host admission wrapper. Native
agents supply research direction; this module runs and records their experiments.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
import os
from pathlib import Path
import select
import signal
import statistics
import subprocess
import sys
import time
import uuid

import numpy as np
from tools.research_store import Store, Invalid, atomic_json, read_json, component, digest, canonical, COVERAGE
from tools.research_cases import VERSION, TOLERANCES, generate_case, import_problem, validate_case
from tools.research_prepare import prepare
from tools.published import PUBLISHED_PROBLEMS


class BudgetExhausted(RuntimeError):
    """Stop at a safe boundary; preserved work can resume in another job."""


WORKER_SECONDS = 20
JOB_MARGIN_SECONDS = 5


def _block_budget(deadline):
    if deadline is not None and deadline-time.monotonic() < WORKER_SECONDS+JOB_MARGIN_SECONDS:
        raise BudgetExhausted('insufficient admitted job budget to start another paired block/check')


def _deadline(store, seconds):
    if not isinstance(seconds,(float,int)) or not 0 < seconds <= 240:
        raise Invalid('GPU service budget must be positive and at most 240 seconds')
    state=store.status(); run=state['runs'].get(state['current_run'])
    if not run or run['closed']: raise Invalid('start an explicit bounded run before GPU execution')
    remaining=run['seconds']-(time.time()-run['started_at'])
    return time.monotonic()+min(float(seconds),max(0.,remaining))


class GPUWorker:
    """One disposable CUDA process/context per admitted service, never per pair."""
    def __init__(self, logdir):
        logdir=Path(logdir); logdir.mkdir(parents=True,exist_ok=True)
        self.log=(logdir/'worker.stderr').open('ab')
        self.process=subprocess.Popen([sys.executable,'-m','tools.research_gpu','--serve'],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,
            start_new_session=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        self.device=None

    def run(self, job, timeout=20):
        if not 0 < timeout <= WORKER_SECONDS: raise Invalid('worker timeout must be in (0,20] seconds')
        if self.process.poll() is not None: raise RuntimeError('GPU worker retired')
        output=Path(job['out'])
        if output.is_symlink(): raise Invalid('worker output cannot be a symlink')
        output.parent.mkdir(parents=True,exist_ok=True)
        output.unlink(missing_ok=True)
        self.process.stdin.write(json.dumps(job)+'\n'); self.process.stdin.flush()
        ready,_,_=select.select([self.process.stdout],[],[],timeout)
        if not ready:
            self.close(kill=True); raise TimeoutError('GPU job exceeded 20 seconds; worker retired')
        line=self.process.stdout.readline()
        if not line:
            self.close(kill=True); raise RuntimeError('GPU worker exited: inspect stderr')
        ack=json.loads(line)
        if ack.get('retire_worker') or ack.get('status')==2 or ack.get('worker_error'):
            self.close(kill=True); raise RuntimeError(ack.get('error') or ack.get('worker_error') or 'GPU worker retired')
        if ack.get('out')!=str(output) or ack.get('status') not in (0,1):
            self.close(kill=True); raise RuntimeError('GPU acknowledgement identity/status mismatch')
        result=read_json(output)
        if result.get('retire_worker'):
            self.close(kill=True); raise RuntimeError(result.get('error','GPU worker retired'))
        if result.get('ok') != ack.get('ok'):
            self.close(kill=True); raise RuntimeError('GPU result/acknowledgement mismatch')
        self.device=result.get('device')
        return result

    def close(self,kill=False):
        if self.process.poll() is None:
            if kill:
                try: os.killpg(self.process.pid,signal.SIGKILL)
                except ProcessLookupError: pass
            else:
                try: self.process.stdin.close()
                except BrokenPipeError: pass
                try: self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try: os.killpg(self.process.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
            self.process.wait()
        if not self.log.closed: self.log.close()


def artifacts_from_tree(root):
    root=Path(root)
    return {p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file() and not p.is_symlink()}


def register_protocol(store,version='research-all15-v1',families=('pointwise','reduction')):
    # Freeze actual local-evaluator dependencies. CLI/qualification drivers and
    # hosted transport have their own execution records, not local scoring logic.
    names=('research_pipeline.py','research_store.py','research_prepare.py',
           'research_gpu.py','research_cases.py','published.py')
    files={name:Path(__file__).with_name(name).read_bytes() for name in names}
    contract={'generator':VERSION,'families':list(families),'tolerances':TOLERANCES,
              'exploration_cases_per_family':4,'confirmation_cases_per_family':4,
              'confirmation_pairs':32,'screening_blocks':2,'pairs_per_block':8,
              'maximum_pairs':200,'isolated_confirmation_required':True,
              'published_problems':list(PUBLISHED_PROBLEMS),
              'performance_requires_all15':True,
              'published_files':_published_fingerprints(),
              'softmax_schedules':'Two cooperative online cases (3x127,5x257), remaining cases serial.',
              'independence':'Coordinator-owned generators; independently reviewed direct NumPy references.',
              'scope':'Published cooperative/async/fused anchors plus fresh serial schedules. New hidden schedule coverage remains unproven.'}
    return store.register_suite(version,contract,files)


def _published_fingerprints():
    problems=Path(__file__).resolve().parents[1]/'emmy-hackathon-2026/problems'
    return {name:{p.name:digest(p.read_bytes()) for p in
                 [problems/name/'meta.json',problems/name/'expected.npz',*sorted((problems/name).glob('*.ir.json'))]}
            for name in PUBLISHED_PROBLEMS}


def _verify_protocol(store,suite_id):
    suite=store.verify_object(suite_id,'suite')
    for name,entry in suite['files'].items():
        if (name.startswith('research') or name=='published.py') and name.endswith('.py') and Path(name).name==name:
            current=Path(__file__).with_name(name)
            if not current.is_file() or digest(current.read_bytes())!=entry['sha256']:
                raise Invalid('running evaluator differs from frozen suite; create a new evaluation version: '+name)
    return suite


def make_cases(store, suite_id, seed, label, count=4, published=True):
    component(label)
    with (store.root/'suites'/('cases-'+label+'.lock')).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _make_cases_unlocked(store,suite_id,seed,label,count,published)


def _make_cases_unlocked(store, suite_id, seed, label, count=4, published=True):
    component(label)
    if type(count) is not int or not 1<=count<=16: raise Invalid('case count must be between 1 and 16 per family')
    if type(seed) is not int or seed<0: raise Invalid('case seed must be a nonnegative integer')
    suite_manifest=_verify_protocol(store,suite_id); suite=suite_manifest['payload']
    if suite.get('published_files') is not None and suite['published_files']!=_published_fingerprints():
        raise Invalid('published suite fixtures changed; create a new evaluation version')
    # Names/seed are part of an immutable reservation. Never overwrite or redraw.
    root=store.root/'suites'/('cases-'+label)
    reservation={'suite_id':suite_id,'seed':str(seed),'count':count,'published':bool(published)}
    if root.is_symlink(): raise Invalid('case directory cannot be a symlink')
    if (root/'case-list.json').exists():
        saved=read_json(root/'case-list.json')
        if any(saved.get(k)!=v for k,v in reservation.items()): raise Invalid('case reservation changed')
        for case in saved['cases']:
            if Path(case).parent.resolve()!=root.resolve(): raise Invalid('saved case escaped reservation directory')
            _case_identity(case)
        if published:
            names=[read_json(Path(case)/'case.json').get('spec',{}).get('problem') for case in saved['cases']]
            if sorted(n for n in names if n in PUBLISHED_PROBLEMS)!=sorted(PUBLISHED_PROBLEMS):
                raise Invalid('historical subset reservation is not an all15 suite; create a new reservation')
        return saved['cases']
    root.mkdir(parents=True,exist_ok=True)
    cases=[]
    for fi,family in enumerate(suite['families']):
        for i in range(count):
            path=root/f'{family}-{i}'
            if not path.exists():
                # Small semantic factories plus larger pointwise/reduction sizes.
                shape=(1024*(i+1)+i,) if family=='pointwise' else ((257+128*i,) if family=='reduction' else None)
                cooperative=family=='softmax' and i<2
                if cooperative: shape=((3,127),(5,257))[i]
                generate_case(path,family,(int(seed)+fi*100003+i)%(2**64),shape=shape,
                              operation='add' if family=='pointwise' else None,
                              pattern=('normal','cancellation','impulse','bounded')[i%4],
                              schedule='cooperative' if cooperative else 'serial')
            record=read_json(path/'case.json')
            wanted=(int(seed)+fi*100003+i)%(2**64)
            if record.get('spec',{}).get('family')!=family or record['spec'].get('seed')!=wanted:
                raise Invalid('interrupted case reservation identity mismatch')
            _case_identity(path); cases.append(str(path))
    if published:
        problems=Path(__file__).resolve().parents[1]/'emmy-hackathon-2026/problems'
        # Every valid source must satisfy all published problems, even when a
        # research question or benchmark only targets one family or one case.
        for name in PUBLISHED_PROBLEMS:
            path=root/name
            if not path.exists(): import_problem(problems/name,path)
            if read_json(path/'case.json').get('spec',{}).get('problem')!=name:
                raise Invalid('published case reservation identity mismatch')
            _case_identity(path); cases.append(str(path))
    atomic_json(root/'case-list.json',{**reservation,'cases':cases})
    return cases


def _case_identity(case):
    info=validate_case(case); record=read_json(Path(case)/'case.json')
    if record['id']!=digest(canonical(record['files'])):
        raise Invalid('case manifest identity does not match files')
    if Path(case).is_symlink() or any((Path(case)/name).is_symlink() for name in record['files']):
        raise Invalid('case artifacts cannot be symlinks')
    return info['id']


def _prepared_payload(store,queue):
    value=store.verify_object(queue['prepared_id'],'prepared')['payload']
    manifest=store._manifest(queue['experiment_id'])
    if value.get('source_id')!=manifest['source_id'] or value.get('baseline_id')!=manifest['baseline_id']:
        raise Invalid('prepared source/baseline identity mismatch')
    if value.get('suite_id'): _verify_protocol(store,value['suite_id'])
    return value


def _module_specs(prepared,variant):
    specs=[]
    for row in prepared['cases']:
        if _case_identity(row['case_dir'])!=row['case_id']:
            raise Invalid('prepared case identity changed')
        info=read_json(Path(row['case_dir'])/'case.json')
        if [k.get('ir') for k in row['kernels']]!=info['ir_paths']:
            raise Invalid('prepared pipeline omitted or reordered an IR module')
        for kernel in row['kernels']:
            for arch,target in kernel['targets'].items():
                folder=Path(target['folder'])
                if folder.is_symlink() or not folder.is_dir(): raise Invalid('assembly cache folder changed')
                if 'module.cubin' not in target.get('hashes',{}): raise Invalid('assembly lacks cubin identity')
                for name,expected in target['hashes'].items():
                    if Path(name).name!=name: raise Invalid('assembly hash path escaped folder')
                    path=folder/name
                    if path.is_symlink() or digest(path.read_bytes())!=expected:
                        raise Invalid('prepared assembly artifact changed')
        cubins=[str(Path(k['targets']['sm_89']['folder'])/'module.cubin') for k in row['kernels']]
        specs.append({'id':row['case_id'],'variant':variant,'case_dir':row['case_dir'],'cubins':cubins,
                      'requires_sanitizer':bool(info.get('requires_sanitizer')),
                      'requires_isolated_first':bool(info.get('requires_isolated_first'))})
    return specs


def _require_cases(prepared,cases):
    expected=[_case_identity(c) for c in cases]
    actual=[r['case_id'] for r in prepared['cases']]
    if len(set(expected))!=len(expected) or actual!=expected:
        raise Invalid('preparation dropped, duplicated, or reordered mandatory cases')


def _prepare_one(store,queue,cases,owner):
    store.preparation_jobs(queue['id'],owner,[os.getpid()])
    manifest=store._manifest(queue['experiment_id'])
    source=store.root/'objects'/manifest['source_id']/'files'
    store.verify_object(manifest['source_id'],'source')
    baseline=_baseline_prepared(store,cases,manifest['baseline_id'])
    out=store.root/'experiments'/queue['experiment_id']/'preparation'
    if out.exists():
        if not (out/'prepared.json').exists(): raise Invalid('interrupted preparation: inspect retained directory before retrying')
        prepared=read_json(out/'prepared.json')
        if prepared.get('source_id')!=manifest['source_id'] or prepared.get('baseline_id')!=manifest['baseline_id']:
            raise Invalid('cached preparation identity changed')
    else:
        brief=read_json(store._experiment(queue['experiment_id'])/'brief.json')
        prepared=prepare(source,cases,out,store.root/'assembly-cache',brief,baseline)
        prepared.update(source_id=manifest['source_id'],baseline_id=manifest['baseline_id'],
                        case_set_id=digest(canonical([_case_identity(c) for c in cases])))
    prepared['baseline_prepared']=baseline
    reservations=[Path(case).parent/'case-list.json' for case in cases]
    suite_ids={read_json(path)['suite_id'] for path in reservations if path.exists()}
    if len(suite_ids)>1: raise Invalid('case batch mixes evaluation suites')
    prepared['suite_id']=next(iter(suite_ids),None)
    if prepared['suite_id']: _verify_protocol(store,prepared['suite_id'])
    _require_cases(prepared,cases)
    _module_specs(prepared,queue['experiment_id'])
    _module_specs(baseline,'baseline')
    atomic_json(out/'prepared.json',prepared)
    files=artifacts_from_tree(out)
    for side,value in [('candidate',prepared),('baseline',baseline)]:
        for ci,row in enumerate(value['cases']):
            for ki,kernel in enumerate(row['kernels']):
                for arch,target in kernel['targets'].items():
                    for name,data in artifacts_from_tree(target['folder']).items(): files[f'assembly/{side}/{ci}/{ki}/{arch}/{name}']=data
    aid=store.put_artifact('prepared',prepared,files)
    return store.finish_preparation(queue['id'],owner,aid,prepared['gate'])


def prepare_pending(store,cases,*,seconds=240):
    end=time.monotonic()+seconds; output=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        while time.monotonic()<end:
            claimed=[]
            for i in range(2):
                owner='prepare-'+uuid.uuid4().hex[:8]; item=store.claim_preparation(owner)
                if item: claimed.append((item,owner))
            if not claimed: break
            futures=[pool.submit(_prepare_one,store,q,cases,owner) for q,owner in claimed]
            for (q,owner),future in zip(claimed,futures):
                try: output.append(future.result())
                except Exception as error:
                    aid=store.put_artifact('preparation-failure',{'error':str(error),'experiment':q['experiment_id']},
                                           artifacts_from_tree(store._experiment(q['experiment_id'])/'preparation'))
                    output.append(store.finish_preparation(q['id'],owner,aid,'failed'))
    return output


def _correct(worker,specs,out,width=1,repetitions=3):
    return worker.run({'mode':'correctness','cases':specs,'width':width,'repetitions':repetitions,'out':str(out)})


def _run_sanitizer(spec,out,deadline=None):
    _block_budget(deadline)
    out=Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    job={'mode':'correctness','cases':[spec],'width':1,'repetitions':3,'out':str(out)}
    jobpath=out.with_name(out.stem+'-job.json'); atomic_json(jobpath,job)
    command=['compute-sanitizer','--tool','memcheck','--error-exitcode','99',sys.executable,
             '-m','tools.research_gpu',str(jobpath)]
    child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    timed_out=False
    try: stdout,stderr=child.communicate(timeout=WORKER_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out=True
        try: os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        stdout,stderr=child.communicate()
    out.with_suffix('.stdout').write_bytes(stdout); out.with_suffix('.stderr').write_bytes(stderr)
    result=read_json(out) if out.exists() else {}
    return {'ok':not timed_out and child.returncode==0 and b'ERROR SUMMARY: 0 errors' in stderr+stdout
                 and _checks_ok(result), 'exit_code':child.returncode,'timed_out':timed_out,'command':command}


def _checks_ok(result):
    batches=result.get('batches',[])
    return (result.get('ok') is True and bool(batches)
            and all(not b.get('quarantined',True) and bool(b.get('results'))
                    and all(r.get('ok') is True and bool(r.get('checks'))
                            and all(c.get('ok') is True for c in r['checks'])
                            for r in b['results']) for b in batches))


def evaluate_correctness(store,worker,queues,job,out,width,deadline=None):
    outcomes={}; rows=[]
    # Interleave matching case positions across candidates, bounded to eight.
    prepared={q['id']:_prepared_payload(store,q) for q in queues}
    specs={q['id']:_module_specs(prepared[q['id']],q['experiment_id']) for q in queues}
    for q in queues:
        for i,spec in enumerate(specs[q['id']]):
            if spec.get('requires_isolated_first') or spec.get('requires_sanitizer'):
                _block_budget(deadline)
                r=_correct(worker,[spec],out/f'first-{q["id"]}-{i}.json')
                rows.append(r)
                if not _checks_ok(r): outcomes[q['id']]='failed'; break
                if spec.get('requires_sanitizer'):
                    sanitizer=_run_sanitizer(spec,out/f'first-sanitizer-{q["id"]}-{i}.json',deadline)
                    rows.append(sanitizer)
                    if not sanitizer['ok']: outcomes[q['id']]='failed'; break
    longest=max((len(x) for x in specs.values()),default=0)
    for i in range(longest):
        active=[(q,specs[q['id']][i]) for q in queues if q['id'] not in outcomes and i<len(specs[q['id']])]
        if not active: continue
        _block_budget(deadline)
        result=_correct(worker,[spec for q,spec in active],out/f'batch-{i}.json',width)
        rows.append(result)
        if not _checks_ok(result):
            # Recheck every cohort member, not only the apparent victim.
            for q,spec in active:
                _block_budget(deadline)
                solo=_correct(worker,[spec],out/f'recheck-{i}-{q["id"]}.json')
                rows.append(solo)
                outcomes[q['id']]='quarantined' if _checks_ok(solo) else 'failed'
    for q in queues: outcomes.setdefault(q['id'],'waiting_timing')
    return outcomes,rows


def paired_summary(blocks):
    good=[b for b in blocks if b['complete']]
    ratios=[statistics.median(c/b for b,c in zip(x['baseline_us'],x['candidate_us'])) for x in good]
    return {'blocks':len(good),'pairs':sum(len(b['baseline_us']) for b in good),
            'median_block_ratio':statistics.median(ratios) if ratios else None,
            'block_ratios':ratios,'aggregation':'median of within-block paired ratios; bursts remain distinct'}


def _sample(worker,spec,out,burst,offset=0,pairs=8,deadline=None):
    _block_budget(deadline)
    return worker.run({'mode':'timing','case':spec,'pairs':pairs,'warmup':5,'seed':17,
                       'burst_id':burst,'pair_offset':offset,'out':str(out)})


def _record_samples(store,exp,spec,result,burst):
    pairs=result.get('pairs',[])
    complete=result.get('ok') is True and pairs and all(p.get('complete') for p in pairs)
    valid=[p for p in pairs if len(p.get('samples_us',{}))==2]
    if valid:
        return store.record_pair_block(exp,spec['id'],[p['samples_us']['baseline'] for p in valid],
            [p['samples_us']['candidate'] for p in valid],burst,complete=bool(complete),
            observations={'device':result.get('device'), 'pair_order':[p['order'] for p in valid],
                          'pair_offset':pairs[0]['pair_index'],'cache_mode':result.get('cache_mode')})
    return None


def _baseline_prepared(store,cases,baseline_id):
    """CPU-only preparation. GPU entrypoints must never invoke this function."""
    store.verify_object(baseline_id,'source')
    source=store.root/'objects'/baseline_id/'files'; result={'cases':[],'unsupported':[]}
    for case in cases:
        cid=_case_identity(case); out=store.root/'baseline-preparation'/baseline_id/cid
        out.parent.mkdir(parents=True,exist_ok=True)
        with (out.parent/(cid+'.lock')).open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            failure=out/'unsupported.json'
            if failure.exists():
                record=read_json(failure)
                if record['source_id']!=baseline_id or record['case_id']!=cid:
                    raise Invalid('baseline failure cache identity changed')
                result['unsupported'].append(record); continue
            try:
                value=read_json(out/'prepared.json') if (out/'prepared.json').exists() else prepare(source,[case],out,store.root/'assembly-cache')
                if len(value['cases'])!=1 or value['cases'][0]['case_id']!=cid:
                    raise Invalid('baseline preparation case identity changed')
                _module_specs(value,'baseline')
                result['cases']+=value['cases']
            except Invalid:
                raise
            except ValueError as error:
                record={'case_dir':str(case),'error':str(error),'source_id':baseline_id,'case_id':cid}
                atomic_json(failure,record)
                result['unsupported'].append(record)
    return result


def gpu_step(store,*,width=None,seconds=240):
    if width is None:
        settings=store.root/'control/runtime.json'
        width=read_json(settings).get('correctness_width',1) if settings.exists() else 1
    if width not in (1,2,4): raise Invalid('configured correctness width must be one, two, or four')
    if not os.environ.get('EMMY_JOB_DIR'): raise Invalid('GPU execution requires the host admission wrapper')
    deadline=_deadline(store,seconds)
    _block_budget(deadline)
    owner='gpu-'+uuid.uuid4().hex[:8]; job=store.claim_gpu(owner)
    if not job: return {'mode':'idle'}
    out=store.root/'control/jobs'/job['job_id']; out.mkdir(parents=True)
    store.gpu_job_ids(job['job_id'],owner,[os.getpid(),os.environ['EMMY_JOB_DIR']])
    queues=[q for q in store.status()['queue'] if q['id'] in job['queue_ids']]
    worker=None
    outcomes={q['id']:'ready' if job['mode']=='correctness' else 'waiting_timing' for q in queues}
    start=time.monotonic(); soft_deadline=min(deadline,start+60); noise_cache={}
    try:
        if job['mode']=='timing':
            # Check every arm before creating a CUDA context, including focused
            # one-case benchmarks whose all15 passes were retained earlier.
            for q in queues:
                prepared=_prepared_payload(store,q); suite_id=prepared.get('suite_id')
                if not suite_id: raise Invalid('timing requires an exact evaluation suite and all15 correctness gates')
                store.require_published_gate(prepared['source_id'],suite_id)
                store.require_published_gate(prepared['baseline_id'],suite_id)
        worker=GPUWorker(out)
        if job['mode']=='correctness':
            outcomes,_=evaluate_correctness(store,worker,queues,job,out,width,deadline)
        else:
            for q in queues:
                if time.monotonic()>=soft_deadline: break
                exp=q['experiment_id']; prepared=_prepared_payload(store,q)
                suite_id=prepared.get('suite_id')
                if not suite_id: raise Invalid('timing requires an exact evaluation suite and all15 correctness gates')
                candidate_gate=store.require_published_gate(prepared['source_id'],suite_id)
                baseline_gate=store.require_published_gate(prepared['baseline_id'],suite_id)
                baseline=prepared.get('baseline_prepared')
                if baseline is None: raise Invalid('CPU preparation must include baseline; never compile inside GPU timing')
                byid={s['id']:s for s in _module_specs(baseline,'baseline')}; finished=True
                for spec in _module_specs(prepared,exp):
                    if spec['id'] not in byid: continue
                    spec['baseline_cubins']=byid[spec['id']]['cubins']
                    spec['published_correctness']={'source_id':prepared['source_id'],'baseline_id':prepared['baseline_id'],
                        'suite_id':suite_id,'candidate_gate_id':candidate_gate['gate_id'],'baseline_gate_id':baseline_gate['gate_id']}
                    samplepath=store._experiment(exp)/'samples.json'
                    blocks=read_json(samplepath)['cases'].get(spec['id'],[]) if samplepath.exists() else []
                    summary=paired_summary(blocks)
                    key=(prepared['baseline_id'],prepared['case_set_id'],spec['id'])
                    if key not in noise_cache:
                        if time.monotonic()>=soft_deadline: finished=False; break
                        aa={**spec,'cubins':spec['baseline_cubins']}
                        aa_result=_sample(worker,aa,out/('noise-'+spec['id']+'.json'),job['burst_id'],pairs=16,deadline=deadline)
                        if not aa_result.get('ok') or len(aa_result.get('pairs',[]))!=16 or not all(p.get('complete') for p in aa_result['pairs']):
                            outcomes[q['id']]='failed'; raise RuntimeError('baseline A/A correctness or sampling failed')
                        deviations=[abs(p['candidate_over_baseline']-1) for p in aa_result['pairs'] if p['complete']]
                        identity={'case_id':spec['id'],'baseline_id':prepared['baseline_id'],
                                  'case_set_id':prepared['case_set_id'],'device':aa_result['device'],
                                  'run_id':store.status()['current_run'],'job_id':job['job_id']}
                        noise_path=store.root/'noise'/(digest(canonical(identity))+'.json')
                        noise={**identity,
                               'deviation_p90':float(np.quantile(deviations,.9)),'result':aa_result,
                               'scope':'This case, device, run, suite/case set, and admitted service only.'}
                        atomic_json(noise_path,noise)
                        noise_cache[key]=(noise,noise_path)
                    noise,noise_path=noise_cache[key]
                    threshold=max(.01,3*noise['deviation_p90'])
                    while summary['pairs']<200:
                        if summary['pairs']>=16 and (abs(summary['median_block_ratio']-1)>threshold): break
                        if time.monotonic()>=soft_deadline: finished=False; break
                        r=_sample(worker,spec,out/(spec['id']+'-'+str(summary['pairs'])+'.json'),job['burst_id'],summary['pairs'],min(8,200-summary['pairs']),deadline)
                        _record_samples(store,exp,spec,r,job['burst_id'])
                        if not r.get('ok'): outcomes[q['id']]='failed'; raise RuntimeError('timing integrity/correctness failed; worker retired')
                        blocks=read_json(samplepath)['cases'][spec['id']]; summary=paired_summary(blocks)
                    atomic_json(store._experiment(exp)/('screen-'+spec['id']+'.json'),{'summary':summary,'noise':noise_path.as_posix(),'threshold':threshold,'confirmation_required':True})
                    if not finished: break
                if outcomes[q['id']]!='failed': outcomes[q['id']]='complete' if finished else 'waiting_timing'
        return {'job':job,'outcomes':outcomes,'wall_seconds':time.monotonic()-start,
                'benchmark_qualified':False,'admission_status':'pending_host_postcheck',
                'qualification_note':'Timing remains provisional until host post-job observations are retained.'}
    except BudgetExhausted as error:
        return {'job':job,'outcomes':outcomes,'incomplete':True,'reason':str(error)}
    except Exception as error:
        # An exception may have contaminated every member of the shared context.
        # Do not silently requeue the apparent victim or trust its batchmates.
        outcomes={q['id']:'quarantined' for q in queues}
        atomic_json(out/'failure.json',{'error':str(error),'batch_members':job['queue_ids'],
                                      'next_action':'Inspect and independently recheck all batch members before retrying.'})
        raise
    finally:
        if worker is not None: worker.close()
        store.finish_gpu(job['job_id'],owner,outcomes)


def prepare_confirmation(store,experiment,suite_id):
    """Draw reserved fresh cases and compile both arms before GPU admission."""
    state=store.status()
    if not store._budget(state,time.time()): raise Invalid('run budget exhausted')
    reservation=store.reserve_confirmation(experiment,suite_id)
    suite=_verify_protocol(store,suite_id)['payload']
    cases=make_cases(store,suite_id,reservation['seed'],reservation['id'],
                     count=suite.get('confirmation_cases_per_family',4))
    manifest=store._manifest(experiment); out=store._experiment(experiment)/'confirmation'/reservation['id']
    out.mkdir(parents=True,exist_ok=True)
    planpath=out/'cpu-prepared.json'
    if planpath.exists():
        plan=read_json(planpath)
        payload=store.verify_object(plan['prepared_id'],'confirmation-prepared')['payload']
        if payload['source_id']!=manifest['source_id'] or payload['reservation']!=reservation:
            raise Invalid('confirmation preparation identity changed')
        _require_cases(payload['prepared'],cases)
        _module_specs(payload['prepared'],experiment); _module_specs(payload['baseline'],'baseline')
        return plan
    source=store.root/'objects'/manifest['source_id']/'files'
    store.verify_object(manifest['source_id'],'source')
    baseline=_baseline_prepared(store,cases,manifest['baseline_id'])
    prepdir=out/'preparation'
    prepared=read_json(prepdir/'prepared.json') if (prepdir/'prepared.json').exists() else prepare(source,cases,prepdir,store.root/'assembly-cache')
    _require_cases(prepared,cases)
    _module_specs(prepared,experiment); _module_specs(baseline,'baseline')
    files=artifacts_from_tree(out)
    for i,case in enumerate(cases):
        for name,data in artifacts_from_tree(case).items(): files[f'cases/{i}/{name}']=data
    for side,value in [('candidate',prepared),('baseline',baseline)]:
        for i,row in enumerate(value['cases']):
            for j,kernel in enumerate(row['kernels']):
                for arch,target in kernel['targets'].items():
                    for name,data in artifacts_from_tree(target['folder']).items(): files[f'{side}/{i}/{j}/{arch}/{name}']=data
    payload={'reservation':reservation,'source_id':manifest['source_id'],'baseline_id':manifest['baseline_id'],
             'prepared':prepared,'baseline':baseline,'cases':cases}
    plan={'prepared_id':store.put_artifact('confirmation-prepared',payload,files),
          'reservation_id':reservation['id'],'source_id':manifest['source_id']}
    atomic_json(planpath,plan)
    return plan


REGRESSIONS=set(PUBLISHED_PROBLEMS)
FAMILY_COVERAGE={'matmul':['m01_matmul_512'],'atomic_matmul':['m02_matmul_rect'],
                 'softmax':['f02_row_softmax'],'attention':['a01_attention','a02_attention_128']}


def _confirmation_row(spec,good):
    info=read_json(Path(spec['case_dir'])/'case.json'); family=info['spec']['family']
    item=info['spec'].get('problem')
    return {'case_id':spec['id'],'status':'pass' if good else 'fail','isolated':True,
            'guards_ok':good,'inputs_ok':good,
            'scope':'published' if family=='published' else 'generated',
            'problem':item,'regression':item in REGRESSIONS,
            'coverage_item':item,'coverage_items':FAMILY_COVERAGE.get(family,[])}


def _calibration_bindings(prepared,baseline,timings):
    """Bind measured case bytes and PTX arms to the source-bound timing report."""
    candidate_rows={r['case_id']:r for r in prepared['cases']}
    baseline_rows={r['case_id']:r for r in baseline['cases']}
    bindings={}
    for case_id,timing in timings.items():
        pairs=timing.get('pairs',[])
        if timing.get('ok') is not True or not pairs or not all(p.get('complete') is True for p in pairs):
            continue
        for pair in pairs:
            samples=pair.get('samples_us',{})
            if set(samples)!={'baseline','candidate'} or any(
                    type(value) not in (int,float) or not math.isfinite(value) or value<=0
                    for value in samples.values()):
                raise Invalid('paired calibration evidence contains invalid timing samples')
        if case_id not in candidate_rows or case_id not in baseline_rows:
            raise Invalid('paired timing omitted a prepared compiler arm')
        candidate,anchor=candidate_rows[case_id],baseline_rows[case_id]
        if _case_identity(candidate['case_dir'])!=case_id or _case_identity(anchor['case_dir'])!=case_id:
            raise Invalid('calibration case bytes changed after timing')
        def emitted_ptx(row):
            hashes=[k.get('compile',{}).get('ptx_sha256') for k in row['kernels']]
            if any(not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value) for value in hashes):
                raise Invalid('prepared emitted PTX identity missing from paired measurement')
            return hashes
        bindings[case_id]={
            'case_files_sha256':{name:digest((Path(candidate['case_dir'])/name).read_bytes())
                                 for name in ('meta.json','inputs.npz','expected.npz')},
            'baseline_ptx_sha256':emitted_ptx(anchor),'candidate_ptx_sha256':emitted_ptx(candidate)}
    return bindings


def _revalidate_report_gates(store,report):
    if report.get('valid_gpu_evaluation') or report.get('benchmark_qualified'):
        if not report.get('published_gate_id'):
            raise Invalid('cached qualification has no all15 correctness gate')
    for source_key,gate_key in (('source_id','published_gate_id'),('baseline_id','baseline_published_gate_id')):
        if report.get(gate_key):
            store.require_published_gate(report[source_key],report['suite_id'],report[gate_key])


def _revoke_checkpoint_failures(store,manifest,suite_id,item,object_id):
    """Replay durable failures before trusting a gate or skipping a checkpoint.

    The process may have exited after publishing progress.json but before its
    original invalidation writes. Replaying this operation is deliberately safe.
    """
    source_failed=False; baseline_failed=False
    if item['case'].get('problem') in PUBLISHED_PROBLEMS and item['case']['status']!='pass':
        store.invalidate_published_gate(manifest['source_id'],suite_id,object_id,'Published isolated correctness failed')
        source_failed=True
    if item.get('sanitizer') is not None and item['sanitizer'].get('ok') is not True:
        store.invalidate_published_gate(manifest['source_id'],suite_id,object_id,'Required Compute Sanitizer check failed')
        source_failed=True
    if item.get('timing') is not None and item['timing'].get('ok') is not True:
        for source_id in {manifest['source_id'],manifest['baseline_id']}:
            store.invalidate_published_gate(source_id,suite_id,object_id,'Paired timing integrity/correctness failed; attribution unresolved')
        source_failed=True; baseline_failed=True
    return source_failed,baseline_failed


def confirm(store,experiment,suite_id,*,seconds=240):
    """Mandatory fresh isolated confirmation; CPU preparation is separate."""
    if not os.environ.get('EMMY_JOB_DIR'): raise Invalid('confirmation requires host GPU admission')
    deadline=_deadline(store,seconds); _block_budget(deadline)
    manifest=store._manifest(experiment); reservation=manifest.get('confirmation')
    _verify_protocol(store,suite_id)
    if not reservation or reservation['suite_id']!=suite_id:
        raise Invalid('run prepare-confirmation in a CPU job before GPU admission')
    out=store._experiment(experiment)/'confirmation'/reservation['id']
    if (out/'receipt.json').exists():
        receipt=read_json(out/'receipt.json')
        actual=store.verify_object(receipt['evidence_id'],'evaluation')['payload']
        _revalidate_report_gates(store,actual['report'])
        return receipt
    if not (out/'cpu-prepared.json').exists(): raise Invalid('CPU confirmation preparation is missing')
    plan=read_json(out/'cpu-prepared.json'); payload=store.verify_object(plan['prepared_id'],'confirmation-prepared')['payload']
    if payload['source_id']!=manifest['source_id'] or payload['baseline_id']!=manifest['baseline_id'] or payload['reservation']!=reservation:
        raise Invalid('confirmation source/reservation identity changed')
    specs=_module_specs(payload['prepared'],experiment)
    byid={s['id']:s for s in _module_specs(payload['baseline'],'baseline')}
    if len({s['id'] for s in specs})!=len(specs): raise Invalid('confirmation case identities duplicated')
    progresspath=out/'progress.json'
    progress=read_json(progresspath) if progresspath.exists() else {'reservation_id':reservation['id'],'completed':{}}
    if progress['reservation_id']!=reservation['id']: raise Invalid('confirmation checkpoint identity changed')
    expected={s['id'] for s in specs}
    for case_id,object_id in progress['completed'].items():
        old=store.verify_object(object_id,'confirmation-case')['payload']
        if case_id not in expected or old['case']['case_id']!=case_id \
                or old['source_id']!=manifest['source_id'] or old['reservation_id']!=reservation['id']:
            raise Invalid('confirmation case checkpoint identity changed')
        _revoke_checkpoint_failures(store,manifest,suite_id,old,object_id)
    candidate_gate=store.published_gate(manifest['source_id'],suite_id,required=False)
    baseline_gate=store.published_gate(manifest['baseline_id'],suite_id,required=False)
    timing_enabled=bool(candidate_gate and baseline_gate)
    owner='confirm-'+uuid.uuid4().hex[:8]
    job=store.claim_gpu_auxiliary(owner,experiment)
    if not job: raise BudgetExhausted('run budget exhausted')
    store.gpu_job_ids(job['job_id'],owner,[os.getpid(),os.environ['EMMY_JOB_DIR']])
    try:
        for i,spec in enumerate(specs):
            if spec['id'] in progress['completed']:
                old=store.verify_object(progress['completed'][spec['id']],'confirmation-case')['payload']
                if old['source_id']!=manifest['source_id'] or old['reservation_id']!=reservation['id']:
                    raise Invalid('confirmation case checkpoint identity changed')
                # Previous jobs require a successful post-job observation before reuse.
                if old['host_job_dir']==os.environ['EMMY_JOB_DIR']:
                    continue
                post=Path(old['host_job_dir'])/'gpu-attestation.json'
                if post.exists():
                    att=read_json(post)
                    if att.get('observed_no_contention') is True and att.get('exit_code')==0:
                        continue
            _block_budget(deadline)
            worker=GPUWorker(out/f'isolated-{i}')
            caseout=out/f'case-{i}-{job["job_id"]}'; caseout.mkdir()
            item={'source_id':manifest['source_id'],'reservation_id':reservation['id'],
                  'host_job_dir':os.environ['EMMY_JOB_DIR'],'sanitizer':None,'timing':None}
            try:
                result=_correct(worker,[spec],caseout/'correctness.json',1)
                good=_checks_ok(result)
                item['device']=result.get('device'); item['case']=_confirmation_row(spec,good)
                if timing_enabled and good and spec['id'] in byid:
                    spec['baseline_cubins']=byid[spec['id']]['cubins']
                    spec['published_correctness']={'source_id':manifest['source_id'],'baseline_id':manifest['baseline_id'],
                        'suite_id':suite_id,'candidate_gate_id':candidate_gate['gate_id'],'baseline_gate_id':baseline_gate['gate_id']}
                    timing=_sample(worker,spec,caseout/'timing.json',job['burst_id'],pairs=32,deadline=deadline)
                    item['timing']={'ok':timing['ok'] and len(timing.get('pairs',[]))==32
                                              and all(p.get('complete') for p in timing.get('pairs',[])),
                                    'pairs':timing.get('pairs',[]),'cache_mode':timing.get('cache_mode')}
            finally: worker.close()
            if spec.get('requires_sanitizer'):
                item['sanitizer']=_run_sanitizer(spec,caseout/'sanitizer.json',deadline)
            object_id=store.put_artifact('confirmation-case',item,artifacts_from_tree(caseout))
            progress['completed'][spec['id']]=object_id
            atomic_json(progresspath,progress)
            source_failed,baseline_failed=_revoke_checkpoint_failures(store,manifest,suite_id,item,object_id)
            if source_failed: candidate_gate=None; timing_enabled=False
            if baseline_failed: baseline_gate=None; timing_enabled=False
        completed=[store.verify_object(progress['completed'][s['id']],'confirmation-case')['payload'] for s in specs]
        timing={v['case']['case_id']:v['timing'] for v in completed if v['timing'] is not None}
        sanitizers=[v['sanitizer'] for v in completed if v['sanitizer'] is not None]
        report={'source_id':manifest['source_id'],'baseline_id':manifest['baseline_id'],'suite_id':suite_id,
                'confirmation_id':reservation['id'],'fresh_confirmation':True,'cases':[v['case'] for v in completed],
                'required_case_ids':[s['id'] for s in specs],'timing':timing,
                'baseline_gpu_verified':bool(timing) and all(t['ok'] for t in timing.values()),
                'baseline_unsupported':payload['baseline']['unsupported'],
                'sanitizer_required':bool(sanitizers),'sanitizer_passed':all(s['ok'] for s in sanitizers),
                'local_device':completed[-1]['device'],'admission_status':'pending_host_postcheck',
                'case_devices':{v['case']['case_id']:v['device'] for v in completed},
                'case_host_jobs':{v['case']['case_id']:v['host_job_dir'] for v in completed},
                'host_job_dirs':sorted({v['host_job_dir'] for v in completed}),
                'artifact_ids':[plan['prepared_id'],*[progress['completed'][s['id']] for s in specs]]}
        published=[row for row in report['cases'] if row.get('problem') in PUBLISHED_PROBLEMS]
        # Include failures loaded from immutable checkpoints as well as this
        # process's results. A short isolated check passing does not erase a
        # later repeated-launch timing failure or its gate revocation.
        timing_integrity_passed=all(t.get('ok') is True for t in timing.values())
        if len(published)==len(PUBLISHED_PROBLEMS) and set(c['problem'] for c in published)==set(PUBLISHED_PROBLEMS) \
                and all(c['status']=='pass' for c in published) and report['sanitizer_passed'] \
                and timing_integrity_passed:
            gate_report={'source_id':manifest['source_id'],'suite_id':suite_id,'cases':published,
                         'sanitizer_passed':any(v['case'].get('problem')=='m02_matmul_rect'
                              and (v.get('sanitizer') or {}).get('ok') is True for v in completed),
                         'artifact_ids':report['artifact_ids']}
            if gate_report['sanitizer_passed']:
                gate_id=store.issue_published_gate(manifest['source_id'],suite_id,gate_report)
                candidate_gate=store.require_published_gate(manifest['source_id'],suite_id,gate_id)
        report.update(published_gate_id=candidate_gate['gate_id'] if candidate_gate else None,
                      baseline_published_gate_id=baseline_gate['gate_id'] if baseline_gate else None,
                      timing_enabled=timing_enabled,benchmark_qualified=False,
                      timing_blocked_reason=None if timing_enabled else 'Correctness debug only: exact source/suite all15 gate missing before timing.')
        report['evaluation_ok']=all(c['status']=='pass' for c in report['cases']) and all(t['ok'] for t in timing.values()) and report['sanitizer_passed']
        report['valid_gpu_evaluation']=bool(candidate_gate) and report['evaluation_ok']
        report['calibration_cases']=_calibration_bindings(payload['prepared'],payload['baseline'],timing)
        evidence_id=store.issue_evidence(experiment,report)
        result={'evidence_id':evidence_id,'report':report}; atomic_json(out/'receipt.json',result)
        return result
    except BudgetExhausted as error:
        return {'incomplete':True,'reason':str(error),'completed_case_ids':list(progress['completed']),
                'confirmation_id':reservation['id']}
    finally:
        store.finish_gpu(job['job_id'],owner,{})


def attest_confirmation(store,experiment):
    """CPU-only post-job observation; issues a new immutable evaluator receipt."""
    manifest=store._manifest(experiment); reservation=manifest.get('confirmation')
    if not reservation: raise Invalid('confirmation reservation missing')
    out=store._experiment(experiment)/'confirmation'/reservation['id']
    original=read_json(out/'receipt.json')
    receipt=store.verify_object(original['evidence_id'],'evaluation')['payload']
    report=receipt['report']
    if report.get('admission_status')=='observed_no_contention':
        # Cached qualification is not authority to ignore a subsequent failure.
        # Revalidate every referenced gate before returning qualified evidence.
        _revalidate_report_gates(store,report)
        return original
    host_ids=report.get('host_job_dirs',[])
    if not host_ids: raise Invalid('host job identities missing')
    retained={}; observations=[]
    for index,directory in enumerate(host_ids):
        path=Path(directory)
        att=read_json(path/'gpu-attestation.json')
        declared=att.get('host_job_dir')
        if declared is None or Path(declared).resolve()!=path.resolve():
            raise Invalid('host attestation job-directory identity mismatch')
        if att.get('observed_no_contention') is not True or att.get('exit_code')!=0:
            raise Invalid('host observations do not support uncontended timings')
        if not att.get('gpu_uuid') or not att.get('gpu_model'):
            raise Invalid('host attestation device identity missing')
        if report.get('local_device',{}).get('uuid')!=att['gpu_uuid']:
            raise Invalid('host/worker GPU identity mismatch')
        for case_id,host in report.get('case_host_jobs',{}).items():
            if host==directory and report.get('case_devices',{}).get(case_id,{}).get('uuid')!=att['gpu_uuid']:
                raise Invalid('host/case GPU identity mismatch')
        files=artifacts_from_tree(path)
        if 'telemetry.jsonl' not in files or not files['telemetry.jsonl'].strip():
            raise Invalid('host telemetry evidence missing')
        required={'monitor-result.json','admission.csv','processes.csv','after.csv',
                  'processes-after.csv','container.json','command.txt'}
        if not required.issubset(files): raise Invalid('host admission/provenance records missing')
        monitor=att.get('monitor',{})
        if monitor!=json.loads(files['monitor-result.json']) or monitor.get('observed') is not True \
                or type(monitor.get('samples')) is not int or monitor['samples']<1 \
                or monitor.get('foreign')!=[] or monitor.get('errors')!=[]:
            raise Invalid('host monitor observations are incomplete or contaminated')
        samples=[json.loads(line) for line in files['telemetry.jsonl'].decode().splitlines() if line.strip()]
        if len(samples)!=monitor['samples']: raise Invalid('host telemetry sample count mismatch')
        previous=None
        for sample in samples:
            if not {'time','gpu','processes','our_host_pids','foreign'}.issubset(sample):
                raise Invalid('host telemetry fields missing')
            timestamp=sample.get('time')
            if not isinstance(timestamp,(int,float)) or not math.isfinite(timestamp) or (previous is not None and timestamp<previous):
                raise Invalid('host telemetry timestamps invalid')
            previous=timestamp
            gpu=sample.get('gpu',[])
            if sample.get('error') or sample.get('foreign')!=[] or len(gpu)!=1 \
                    or len(gpu[0])<7 or gpu[0][0].strip()!=att['gpu_uuid']:
                raise Invalid('host telemetry did not observe selected GPU cleanly')
            allowed=sample.get('our_host_pids',[])
            if not isinstance(allowed,list) or any(type(pid) is not int for pid in allowed):
                raise Invalid('host telemetry process ownership missing')
            if not isinstance(sample['processes'],list): raise Invalid('host process observations invalid')
            for process in sample['processes']:
                if len(process)<2 or process[0].strip()!=att['gpu_uuid'] or int(process[1].strip()) not in allowed:
                    raise Invalid('foreign process appears in retained host telemetry')
        for name,data in files.items(): retained[f'job-{index}/{name}']=data
        observations.append(att)
    audit=store.put_artifact('host-gpu-observations',{'observations':observations,
                            'prior_evidence_id':original['evidence_id']},retained)
    report={**report,'admission_status':'observed_no_contention',
            'admission_limitation':'Admission and post-job observations are not a GPU reservation.',
            'artifact_ids':[*report['artifact_ids'],audit]}
    if report.get('published_gate_id'):
        store.require_published_gate(report['source_id'],report['suite_id'],report['published_gate_id'])
    report['valid_gpu_evaluation']=bool(report.get('published_gate_id')) and report.get('evaluation_ok') is True
    report['benchmark_qualified']=bool(report['valid_gpu_evaluation'] and report.get('timing_enabled')
        and report.get('baseline_published_gate_id') and report.get('timing'))
    if report['benchmark_qualified']:
        store.require_published_gate(report['baseline_id'],report['suite_id'],report['baseline_published_gate_id'])
    evidence_id=store.issue_evidence(experiment,report)
    result={'evidence_id':evidence_id,'report':report,'prior_evidence_id':original['evidence_id']}
    atomic_json(out/'receipt.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path); p.add_argument('--work-root',type=Path)
    sub=p.add_subparsers(dest='command',required=True)
    suite=sub.add_parser('suite'); suite.add_argument('--version',default='research-all15-v1'); suite.add_argument('--families',nargs='+',choices=sorted(TOLERANCES),default=['pointwise','reduction'])
    cases=sub.add_parser('cases'); cases.add_argument('--suite-id',required=True); cases.add_argument('--seed',type=int,required=True); cases.add_argument('--label',required=True)
    prep=sub.add_parser('prepare'); prep.add_argument('--case-list',type=Path,required=True)
    gpu=sub.add_parser('gpu-step'); gpu.add_argument('--width',type=int,choices=[1,2,4],help='Override the qualified width in control/runtime.json')
    con=sub.add_parser('confirm'); con.add_argument('experiment'); con.add_argument('--suite-id',required=True)
    pc=sub.add_parser('prepare-confirmation'); pc.add_argument('experiment'); pc.add_argument('--suite-id',required=True)
    attest=sub.add_parser('attest-confirmation'); attest.add_argument('experiment')
    args=p.parse_args(); store=Store(args.root,args.work_root)
    if args.command=='suite': result={'suite_id':register_protocol(store,args.version,args.families)}
    elif args.command=='cases': result={'cases':make_cases(store,args.suite_id,args.seed,args.label),
                                       'case_list':str(store.root/'suites'/('cases-'+component(args.label))/'case-list.json'),
                                       'suite_id':args.suite_id}
    elif args.command=='prepare': result=prepare_pending(store,read_json(args.case_list)['cases'])
    elif args.command=='gpu-step': result=gpu_step(store,width=args.width)
    elif args.command=='prepare-confirmation': result=prepare_confirmation(store,args.experiment,args.suite_id)
    elif args.command=='attest-confirmation': result=attest_confirmation(store,args.experiment)
    else: result=confirm(store,args.experiment,args.suite_id)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
