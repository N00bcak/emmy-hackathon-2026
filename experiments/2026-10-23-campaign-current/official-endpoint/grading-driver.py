"""One explicitly requested published-suite grading batch; no discovery or retries."""
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from tools.research_store import Store, read_json, atomic_json
from tools.research_calibrate import (
    ChargedRequests, HTTPTransport, exclusive, append_record,
    _arguments, _check_outputs, sha, MAX_RETURN,
)

ROOT = Path('/root/nfs/emmy/research')
WORK = '/root/bcy/emmy/research-workspaces'
EXP = 'attention-selected-erf-independent-v1'
SOURCE = '549b6047f91818ea876deb76adfc88cabc36f554c05ab2bc51b2c961aa9143f2'
COMPILER = 'f47c21b0bb2818df21cd218573cd3cd1231f87955ee78c43a779f5b0be6ebac0'
mode, run_id = sys.argv[1:3]
assert mode in ('preflight', 'execute')
store = Store(ROOT, WORK)
m = store._manifest(EXP)
assert store._state()['baseline_id'] == m['source_id'] == SOURCE
assert m['decision']['decision'] == 'promote'
source = store.verify_object(SOURCE, 'source')
assert sha(ROOT / 'objects' / SOURCE / 'files' / 'compile.py') == COMPILER
accepted = store._accepted_evidence(EXP, m, m['decision']['evidence_id'])
assert accepted['benchmark_qualified'] and accepted['evaluation_ok']
gate = store.require_published_gate(SOURCE, accepted['suite_id'])
folder = store._experiment(EXP) / 'confirmation' / m['confirmation']['id']
cpu = read_json(folder / 'cpu-prepared.json')
prepared = store.verify_object(cpu['prepared_id'], 'confirmation-prepared')['payload']
assert prepared['source_id'] == SOURCE
published = {c['case_id']: c for c in accepted['cases'] if c['scope'] == 'published'}
assert len(published) == 15 and all(c['status'] == 'pass' for c in published.values())
cases = []
for row in prepared['prepared']['cases']:
    if row['case_id'] not in published:
        continue
    case_dir = Path(row['case_dir'])
    binding = accepted['calibration_cases'][row['case_id']]
    identities = {name: sha(case_dir / name) for name in ('meta.json', 'inputs.npz', 'expected.npz')}
    assert identities == binding['case_files_sha256']
    meta = read_json(case_dir / 'meta.json')
    ptx = []
    for i, kernel in enumerate(row['kernels']):
        path = Path(kernel['targets']['sm_120']['folder']) / 'kernel.ptx'
        assert sha(path) == kernel['compile']['ptx_sha256'] == binding['candidate_ptx_sha256'][i]
        text = path.read_text()
        assert text.isascii() and 0 < len(text) <= 2_000_000
        ptx.append(text)
    assert len(ptx) == len(meta['kernels'])
    with np.load(case_dir / 'inputs.npz', allow_pickle=False) as archive:
        inputs = {name: np.array(archive[name]) for name in archive.files}
    with np.load(case_dir / 'expected.npz', allow_pickle=False) as archive:
        expected = {name: np.array(archive[name]) for name in archive.files}
    assert expected and set(meta.get('graph_outputs', expected)).issubset(expected)
    _arguments(meta, inputs, {}, preflight=True)
    cases.append({'case_id': row['case_id'], 'problem': published[row['case_id']]['problem'],
                  'meta': meta, 'inputs': inputs, 'expected': expected, 'ptx': ptx,
                  'identities': identities, 'ptx_sha256': binding['candidate_ptx_sha256']})
cases.sort(key=lambda c: c['problem'])
assert len(cases) == 15 and len({c['problem'] for c in cases}) == 15
planned = 1 + sum(len(c['ptx']) for c in cases)
assert planned <= 17  # User's standing budget: hard 25, serial, >=2-second spacing.
plan = {'run_id': run_id, 'source_id': SOURCE, 'compiler_sha256': COMPILER,
        'experiment_id': EXP, 'accepted_evidence_id': m['decision']['evidence_id'],
        'published_gate_id': gate['gate_id'], 'suite_id': accepted['suite_id'],
        'planned_http_attempts': planned, 'hard_http_cap': 25,
        'driver_sha256': sha(__file__), 'authorization': 'Explicit user request to evaluate retained compiler against remote endpoint',
        'cases': [{k: c[k] for k in ('case_id', 'problem', 'identities', 'ptx_sha256')} for c in cases],
        'measurement': {'warmup': 5, 'iters': 20, 'limitation': 'Indicative endpoint latency; not paired speedup confirmation or hidden-set grading'}}
out = ROOT / 'calibration' / run_id
if mode == 'preflight':
    assert not out.exists(), 'singleton batch already exists'
    target = ROOT / 'official-preflight' / (run_id + '.json')
    atomic_json(target, plan)
    print(json.dumps({'status': 'validated_no_requests', 'plan': str(target), 'planned_http_attempts': planned,
                      'cases': [c['problem'] for c in cases], 'driver_sha256': plan['driver_sha256']}), flush=True)
    sys.exit(0)
assert read_json(ROOT / 'official-preflight' / (run_id + '.json')) == plan
with exclusive(ROOT / 'control' / 'calibration.lock'):
    assert not out.exists(), 'singleton batch already consumed; do not retry'
    (out / 'requests').mkdir(parents=True)
    shutil.copyfile(__file__, out / 'grading-driver.py')
    result = {**plan, 'status': 'started', 'created_at': time.time(), 'results': [],
              'endpoint': HTTPTransport().base_url, 'research_run_reopened': False}
    atomic_json(out / 'result.json', result)
    client = ChargedRequests(ROOT, out, HTTPTransport(), time.sleep, time.monotonic, time.time)
    try:
        health = client.request('GET', '/health')
        result['health'] = health
        gpu = json.dumps(health.get('gpu', ''))
        arch = health.get('default_arch') or ('sm_120' if 'sm_120' in gpu or '12.0' in gpu else None)
        assert '5090' in gpu and arch == 'sm_120' and health.get('can_compile') and health.get('can_run'), health
        for case in cases:
            produced, kernels, elapsed = {}, [], 0.0
            for i, kernel in enumerate(case['meta']['kernels']):
                args, need = _arguments(case['meta'], case['inputs'], produced, upto=i)[i]
                assert need <= MAX_RETURN
                body = {'ptx': case['ptx'][i], 'entry': kernel['entry'],
                        'grid': kernel['grid'], 'block': kernel['block'],
                        'smem_bytes': kernel.get('smem_bytes', 0),
                        'zero_outputs': kernel.get('zero_outputs', []),
                        'args': args, 'warmup': 5, 'iters': 20, 'max_return': need}
                response = client.request('POST', '/run', body)
                outputs, diagnostics = _check_outputs(response, kernel, case)
                produced.update(outputs)
                latency = response['median_us']
                assert isinstance(latency, (float, int)) and np.isfinite(latency) and latency > 0
                elapsed += latency
                kernels.append({'entry': kernel['entry'], 'median_us': latency, 'diagnostics': diagnostics})
            assert set(case['expected']).issubset(produced)
            row = {'problem': case['problem'], 'case_id': case['case_id'], 'status': 'pass',
                   'median_us': elapsed, 'kernels': kernels}
            result['results'].append(row)
            atomic_json(out / 'result.json', result)
            print(json.dumps(row), flush=True)
        result['status'] = 'completed'
    except Exception as exc:
        result.update(status='stopped', error=str(exc))
    result.update(requests_attempted=len(list((out / 'requests').glob('*.json'))),
                  passed=len(result['results']), required=15, completed_at=time.time())
    atomic_json(out / 'result.json', result)
    append_record(ROOT / 'control' / 'official-grading-history.jsonl', result)
    print(json.dumps({'status': result['status'], 'passed': result['passed'], 'required': 15,
                      'requests_attempted': result['requests_attempted'], 'error': result.get('error'),
                      'output_dir': str(out)}), flush=True)
    sys.exit(0 if result['status'] == 'completed' else 1)
