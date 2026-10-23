"""Continue only unstarted cases after a valid numerical failure; never retry HTTP."""
import sys
from pathlib import Path

original = Path('/root/nfs/emmy/research/campaigns/campaign-20261023-12h-v2/official-20261023-retained-v1/driver.py')
sys.argv = [str(original), 'execute', 'official-20261023-retained-v1']
namespace = {'__file__': str(original), '__name__': 'verified_original_preflight'}
prefix = original.read_text().split("out = ROOT / 'calibration' / run_id\n", 1)[0]
exec(compile(prefix, str(original), 'exec'), namespace)
globals().update({k: v for k, v in namespace.items() if not k.startswith('__')})
from tools.research_calibrate import CalibrationError, DTYPES, compare_output

out = ROOT / 'calibration' / run_id
assert read_json(ROOT / 'official-preflight' / (run_id + '.json')) == plan
with exclusive(ROOT / 'control' / 'calibration.lock'):
    result = read_json(out / 'result.json')
    assert result['status'] == 'stopped' and result['requests_attempted'] == 7
    assert result['error'].startswith('hosted correctness failure for matmul:')
    assert len(result['results']) == 5 and [r['problem'] for r in result['results']] == [c['problem'] for c in cases[:5]]
    assert len(list((out / 'requests').glob('*.json'))) == 7
    last = read_json(out / 'requests' / '007.json')
    failed = cases[5]
    assert failed['problem'] == 'm01_matmul_512' and all(len(c['ptx']) == 1 for c in cases)
    assert last['status'] == 'received' and last['http_status'] == 200 and last['response']['ok'] is True
    assert last['path'] == '/run' and last['request']['ptx'] == failed['ptx'][0]
    kernel = failed['meta']['kernels'][0]
    args, need = _arguments(failed['meta'], failed['inputs'], {}, upto=0)[0]
    assert last['request']['args'] == args and last['request']['entry'] == kernel['entry']
    assert last['request']['grid'] == kernel['grid'] and last['request']['block'] == kernel['block']
    assert not (out / 'result-before-numerical-continuation.json').exists()
    shutil.copyfile(out / 'result.json', out / 'result-before-numerical-continuation.json')
    shutil.copyfile('/root/nfs/emmy/research/campaigns/campaign-20261023-12h-v2/official-20261023-retained-v1/continuation.py', out / 'continuation-driver.py')

    def checks_for(response, case):
        checks = {}
        for name, reference in case['expected'].items():
            output = response['outputs'][name]
            assert not output.get('truncated')
            values = np.asarray(output['values'], dtype=DTYPES[output['dtype']]).reshape(output['shape'])
            if reference.shape != values.shape and reference.size == values.size == 1:
                reference = reference.reshape(values.shape)
            checks[name] = compare_output(values, reference, case['meta']['rtol'], case['meta']['atol'])
        return checks

    failure = {'problem': failed['problem'], 'case_id': failed['case_id'], 'status': 'fail',
               'median_us': last['response']['median_us'], 'request_attempt': 7,
               'diagnostics': checks_for(last['response'], failed), 'error': result['error']}
    result['results'].append(failure)
    result.update(status='continuing_unstarted_cases', continuation={
        'reason': 'Original batch stopped on a valid HTTP200 numerical mismatch, not a service error. Complete the explicitly requested suite using only the nine unstarted cases within the SAME singleton journal and 16-request plan.',
        'source_plan_unchanged': True, 'retried_cases': [], 'retried_requests': [],
        'driver_sha256': sha(out / 'continuation-driver.py'), 'created_at': time.time()})
    atomic_json(out / 'result.json', result)
    client = ChargedRequests(ROOT, out, HTTPTransport(), time.sleep, time.monotonic, time.time)
    try:
        for case in cases[6:]:
            kernel = case['meta']['kernels'][0]
            args, need = _arguments(case['meta'], case['inputs'], {}, upto=0)[0]
            body = {'ptx': case['ptx'][0], 'entry': kernel['entry'], 'grid': kernel['grid'],
                    'block': kernel['block'], 'smem_bytes': kernel.get('smem_bytes', 0),
                    'zero_outputs': kernel.get('zero_outputs', []), 'args': args,
                    'warmup': 5, 'iters': 20, 'max_return': need}
            response = client.request('POST', '/run', body)  # Service errors stop immediately.
            try:
                produced, diagnostics = _check_outputs(response, kernel, case)
                assert set(case['expected']).issubset(produced)
                status, error = 'pass', None
            except CalibrationError as exc:
                if not str(exc).startswith('hosted correctness failure for '):
                    raise
                diagnostics, status, error = checks_for(response, case), 'fail', str(exc)
            row = {'problem': case['problem'], 'case_id': case['case_id'], 'status': status,
                   'median_us': response['median_us'], 'diagnostics': diagnostics,
                   'request_attempt': len(list((out / 'requests').glob('*.json')))}
            if error:
                row['error'] = error
            result['results'].append(row)
            atomic_json(out / 'result.json', result)
            print(json.dumps(row), flush=True)
        result['status'] = 'completed_with_failures'
    except Exception as exc:
        result.update(status='stopped_on_service_or_protocol_error', continuation_error=str(exc))
    result.update(requests_attempted=len(list((out / 'requests').glob('*.json'))),
                  passed=sum(r['status'] == 'pass' for r in result['results']),
                  failed=sum(r['status'] == 'fail' for r in result['results']),
                  evaluated=len(result['results']), required=15, completed_at=time.time())
    atomic_json(out / 'result.json', result)
    append_record(ROOT / 'control' / 'official-grading-history.jsonl', result)
    print(json.dumps({'status': result['status'], 'passed': result['passed'], 'failed': result['failed'],
                      'evaluated': result['evaluated'], 'required': 15,
                      'requests_attempted': result['requests_attempted'], 'output_dir': str(out)}), flush=True)
    sys.exit(0 if len(result['results']) == 15 else 1)
