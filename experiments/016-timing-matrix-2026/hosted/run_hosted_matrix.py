#!/usr/bin/env python3
"""Run the all-15 branch comparison against the organizer's RTX 5090 runner."""
from __future__ import annotations
import argparse, hashlib, json, math, random, statistics, sys, time
from pathlib import Path
import numpy as np
import requests

DEFAULT_URL = "http://runner.riftstack.ai:48701"
DTYPES = {"f32": np.float32, "f16": np.float16, "f64": np.float64, "i32": np.int32, "i64": np.int64}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value): return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

class Client:
    def __init__(self, base_url, log_path, gap_seconds):
        self.base_url, self.log_path = base_url.rstrip('/'), log_path
        self.gap_seconds, self.last_request, self.request_number, self.backoff_count = gap_seconds, 0.0, 0, 0
    def _gap(self):
        wait = self.gap_seconds - (time.monotonic() - self.last_request)
        if wait > 0: time.sleep(wait)
    def _record(self, row):
        with self.log_path.open('a', encoding='utf-8') as stream: stream.write(canonical(row) + '\n')
    def request(self, method, path, body=None):
        schedule = (2, 4, 8, 16, 30)
        for attempt in range(1, len(schedule) + 2):
            self._gap(); self.request_number += 1; started = time.monotonic()
            try:
                response = requests.request(method, self.base_url + path, json=body, timeout=(10, 300))
                status = response.status_code
                try: payload = response.json()
                except ValueError: payload = {'ok': False, 'raw_body': response.text[:2000]}
            except (requests.ConnectionError, requests.Timeout) as exc:
                status, payload = None, {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
            self.last_request = time.monotonic()
            stage = payload.get('stage') if isinstance(payload, dict) else None
            throttled = status in (429, 503) or stage == 'busy' or status is None
            self._record({'attempt': attempt, 'request_number': self.request_number, 'method': method,
                          'path': path, 'status': status, 'ok': payload.get('ok') if isinstance(payload, dict) else None,
                          'stage': stage, 'throttled': throttled, 'elapsed_seconds': self.last_request - started,
                          'cubin_id': payload.get('cubin_id') if isinstance(payload, dict) else None,
                          'entry': body.get('entry') if isinstance(body, dict) else None})
            if not throttled: return status, payload
            if attempt > len(schedule): return status, payload
            self.backoff_count += 1
            delay = schedule[attempt - 1] + random.uniform(0.0, 0.5)
            print(f'backoff {delay:.1f}s after HTTP {status} stage={stage}', file=sys.stderr, flush=True)
            time.sleep(delay)
        raise AssertionError('unreachable')

def ordered_args(meta):
    kernel = meta['kernels'][0]; by_name = {a['name']: a for a in kernel['args']}
    if set(by_name) != set(kernel['arg_order']) or len(by_name) != len(kernel['arg_order']):
        raise ValueError('arg_order must enumerate every argument once')
    args, need = [], 0
    for name in kernel['arg_order']:
        arg = dict(by_name[name])
        if arg.get('output'): need += math.prod(arg['shape'])
        args.append(arg)
    return kernel, args, max(1, need)

def check_outputs(payload, meta, expected):
    kernel = meta['kernels'][0]; outputs = payload.get('outputs')
    if not isinstance(outputs, dict): raise ValueError('runner omitted outputs')
    checks = []
    for arg in kernel['args']:
        if not arg.get('output'): continue
        name = arg['name']; out = outputs.get(name)
        if not isinstance(out, dict) or out.get('truncated') or 'values' not in out:
            raise ValueError(f'missing or truncated output {name}')
        dtype = DTYPES[arg.get('dtype', 'f32')]
        got = np.asarray(out['values'], dtype=dtype).reshape(tuple(arg['shape']))
        reference = expected[name]
        if reference.shape != got.shape:
            if reference.size != got.size or reference.size != 1:
                raise ValueError(f'shape mismatch for {name}: {got.shape} vs {reference.shape}')
            got = got.reshape(reference.shape)
        passed = bool(np.allclose(got, reference, rtol=meta['rtol'], atol=meta['atol'], equal_nan=False))
        error = float(np.max(np.abs(got.astype(np.float64) - reference.astype(np.float64)))) if got.size else 0.0
        checks.append({'name': name, 'ok': passed, 'max_abs_error': error, 'rtol': meta['rtol'], 'atol': meta['atol']})
        if not passed: raise ValueError(f'output mismatch for {name}: max_abs_error={error}')
    if not checks: raise ValueError('no checked outputs')
    return checks

def run_once(client, cubin_id, meta, args, need, warmup, iters, expected):
    kernel = meta['kernels'][0]
    body = {'cubin_id': cubin_id, 'entry': kernel['entry'], 'grid': kernel['grid'], 'block': kernel['block'],
            'smem_bytes': kernel.get('smem_bytes', 0), 'zero_outputs': kernel.get('zero_outputs', []),
            'args': args, 'warmup': warmup, 'iters': iters, 'max_return': need}
    status, payload = client.request('POST', '/run', body)
    if status != 200 or payload.get('ok') is not True:
        raise RuntimeError(f'/run failed: HTTP {status}: {canonical(payload)[:2000]}')
    checks = check_outputs(payload, meta, expected); elapsed = payload.get('median_us')
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0: raise RuntimeError('invalid median_us')
    return payload, checks

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default=DEFAULT_URL); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True); parser.add_argument('--repetitions', type=int, default=7)
    parser.add_argument('--gap', type=float, default=0.35); args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    prepared_root = args.prepared.resolve() / 'prepared'; manifest = json.loads((prepared_root / 'manifest.json').read_text())
    cases = list(manifest['cases']); client = Client(args.url, out / 'requests.jsonl', args.gap)
    summary = {'schema': 1, 'url': args.url, 'started_at': time.time(), 'prepared_source': str(prepared_root),
               'candidate_sha256': manifest['candidate_sha256'], 'baseline_sha256': manifest['baseline_sha256'],
               'repetitions': args.repetitions, 'gap_seconds': args.gap, 'cases': {}, 'status': 'started'}
    status, health = client.request('GET', '/health'); summary['health'] = health
    if status != 200 or health.get('ok') is not True or '5090' not in str(health.get('gpu')) or health.get('default_arch') != 'sm_120':
        summary.update(status='stopped', error='health did not confirm runnable RTX 5090 / sm_120')
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n'); return 2
    for case in cases:
        case_dir = Path('emmy-hackathon-2026/problems') / case; meta = json.loads((case_dir / 'meta.json').read_text())
        with np.load(case_dir / 'expected.npz', allow_pickle=False) as archive: expected = {name: np.array(archive[name]) for name in archive.files}
        kernel, run_args, need = ordered_args(meta)
        row = {'case': case, 'meta_sha256': sha(case_dir / 'meta.json'), 'expected_sha256': sha(case_dir / 'expected.npz'),
               'rtol': meta['rtol'], 'atol': meta['atol'], 'entry': kernel['entry'], 'need': need, 'branches': {}}
        for branch in ('baseline', 'candidate'):
            arm = manifest['cases'][case]['kernels'][0][branch]; ptx_path = Path(arm['ptx']); ptx = ptx_path.read_text()
            if sha(ptx_path) != arm['ptx_sha256']: raise RuntimeError(f'prepared PTX identity changed: {ptx_path}')
            cstatus, compiled = client.request('POST', '/compile', {'ptx': ptx, 'arch': 'sm_120', 'flags': []})
            if cstatus != 200 or compiled.get('ok') is not True:
                row['branches'][branch] = {'compile': {'ok': False, 'status': cstatus, 'response': compiled}}
                summary.update(status='stopped', error=f'compile failed for {branch}/{case}', cases={**summary['cases'], case: row})
                (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n'); return 1
            if kernel['entry'] not in compiled.get('entry_points', []): raise RuntimeError(f"compiled module omits {kernel['entry']}")
            arm_row = {'ptx_sha256': arm['ptx_sha256'], 'cubin_id': compiled['cubin_id'], 'registers': compiled.get('registers'), 'correctness': [], 'timing_us': []}
            payload, checks = run_once(client, compiled['cubin_id'], meta, run_args, need, 0, 1, expected)
            arm_row['correctness'].append({'iteration': 'correctness', 'checks': checks, 'median_us': payload['median_us']})
            row['branches'][branch] = arm_row
        summary['cases'][case] = row; (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    all15 = all(all(check['ok'] for c in row['branches'].values() for item in c['correctness'] for check in item['checks']) for row in summary['cases'].values())
    summary['published_correctness'] = {'required': cases, 'all15_passed': all15, 'scope': 'Published correctness on the hosted RTX 5090; hidden set unverified'}
    if not all15:
        summary.update(status='stopped', error='hosted published correctness did not pass for both branches')
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n'); return 1
    for repetition in range(args.repetitions):
        order = ('baseline', 'candidate') if repetition % 2 == 0 else ('candidate', 'baseline')
        for case in cases:
            case_dir = Path('emmy-hackathon-2026/problems') / case; meta = json.loads((case_dir / 'meta.json').read_text())
            with np.load(case_dir / 'expected.npz', allow_pickle=False) as archive: expected = {name: np.array(archive[name]) for name in archive.files}
            _, run_args, need = ordered_args(meta); row = summary['cases'][case]
            for branch in order:
                arm = row['branches'][branch]
                payload, checks = run_once(client, arm['cubin_id'], meta, run_args, need, 10, 50, expected)
                arm['timing_us'].append(payload['median_us'])
                arm['correctness'].append({'iteration': repetition, 'checks': checks, 'median_us': payload['median_us']})
            (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(f'repetition {repetition + 1}/{args.repetitions} complete', file=sys.stderr, flush=True)
    for case in cases:
        row = summary['cases'][case]; base = row['branches']['baseline']['timing_us']; cand = row['branches']['candidate']['timing_us']
        row['medians'] = {'baseline_us': statistics.median(base), 'candidate_us': statistics.median(cand),
                          'candidate_over_baseline': statistics.median(cand) / statistics.median(base),
                          'baseline_samples': base, 'candidate_samples': cand}
        row['timing_checks_ok'] = all(check['ok'] for b in row['branches'].values() for item in b['correctness'] for check in item['checks'])
    summary['published_correctness']['all15_passed'] = all(row['timing_checks_ok'] for row in summary['cases'].values())
    summary['status'] = 'completed' if summary['published_correctness']['all15_passed'] else 'stopped'
    summary['requests'] = client.request_number; summary['backoff_count'] = client.backoff_count; summary['completed_at'] = time.time()
    summary['note'] = "Hosted median_us is the runner's indicative per-launch median; no clock locking or evaluator speed qualification is claimed."
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return 0 if summary['status'] == 'completed' else 1

if __name__ == '__main__': raise SystemExit(main())
