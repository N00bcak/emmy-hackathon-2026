"""Consolidated compiler gate: unchanged GPU cases plus distinct CPU cases.

Published assembly, m01 hoisting and m02 fallback assertions consume the same
source-bound compiled programs as their GPU cases. They are never skipped.
Planning is CPU-only and rejects an over-budget selection before execution.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from tools.regression_suite import canonical, read, sha, verify_case
from tools.published import PUBLISHED_PROBLEMS

HERE = Path(__file__).resolve().parents[1]
POLICY = HERE / 'tests/experiment-gate.json'
SOURCE_FILES = ('compile.py', 'tensor_core.py', 'tf32_emitter.py')
SHARED_METHODS = {
    'test_all_published_lower_and_assemble_for_both_targets': list(PUBLISHED_PROBLEMS),
    'test_m01_inner_loop_hoists_invariant_index_division': ['m01_matmul_512'],
    'test_vector_copy_has_runtime_fallback_and_scalar_negative': ['m02_matmul_rect'],
}


def plan(policy_path=POLICY, case_dirs=None):
    policy = read(policy_path)
    test_file = HERE / policy['compiler_tests']
    if sha(test_file.read_bytes()) != policy['compiler_tests_sha256']:
        raise ValueError('compiler tests changed: review the leaf-case inventory')
    tree = ast.parse(test_file.read_bytes())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CompilerTests')
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')}
    cpu = policy['additional_cpu_checks']
    shared = policy['shared_cpu_methods']
    if shared != SHARED_METHODS:
        raise ValueError('only the three implemented shared validators may reuse GPU cases')
    names = [row['method'] for row in cpu] + list(shared)
    if len(names) != len(set(names)) or set(names) != methods:
        raise ValueError('unaccounted or duplicated compiler test method')
    cpu_ids = [row['method'] + ':' + canonical(params).decode()
               for row in cpu for params in row['cases']]
    if not cpu_ids or len(cpu_ids) != len(set(cpu_ids)) or any(not row['cases'] for row in cpu):
        raise ValueError('CPU leaf cases must be nonempty and unique')
    gpu_policy = read(HERE / policy['gpu_policy'])
    if sha((HERE / policy['gpu_policy']).read_bytes()) != policy['gpu_policy_sha256']:
        raise ValueError('GPU policy changed: review the combined budget')
    required = {row['id']: row for row in gpu_policy['cases']}
    extra = policy['additional_gpu_cases']
    if case_dirs is None:
        case_dirs = [HERE / 'tests/gpu-cases/cases' / cid for cid in required]
        case_dirs += [HERE / row['path'] for row in extra]
    expected = {**required, **{row['id']: row for row in extra}}
    records = [verify_case(path, expected.get(read(Path(path) / 'case.json')['id'])) for path in case_dirs]
    ids = [row['id'] for row in records]
    if len(ids) != len(set(ids)) or not set(expected) <= set(ids):
        raise ValueError('missing or duplicated required GPU case')
    published = {row['spec']['problem']: row['id'] for row in records
                 if row.get('spec', {}).get('family') == 'published'}
    if set(published) != set(PUBLISHED_PROBLEMS) or sum(
            row.get('spec', {}).get('family') == 'published' for row in records) != 15:
        raise ValueError('exactly all15 published cases are mandatory')
    total = len(ids) + len(cpu_ids)
    limit = policy['max_checks']
    if not isinstance(limit, int) or not 0 < limit <= 100 or total > limit:
        raise ValueError(f'experiment requires {total} actual cases; limit is {limit}')
    identity = {'version': policy['version'], 'policy_sha256': sha(Path(policy_path).read_bytes()),
                'runner_sha256': sha(Path(__file__).read_bytes()),
                'compiler_tests_sha256': policy['compiler_tests_sha256'],
                'gpu_case_ids': ids, 'gpu_record_sha256': {r['id']: sha(canonical(r)) for r in records},
                'cpu_leaf_ids': cpu_ids, 'shared_cpu_methods': shared}
    return {**identity, 'plan_id': sha(canonical(identity)), 'cases': [str(Path(p).resolve()) for p in case_dirs],
            'gpu_cases': len(ids), 'additional_cpu_leaf_checks': len(cpu_ids), 'total_checks': total,
            'shared_published_cases': published, 'additional_cpu_checks': cpu,
            'note': 'Shared CPU assertions execute on their GPU case; six copy-address inputs remain six cases.'}


def prepare_shared(root, evidence_id, out):
    """Freeze a source-bound view of already prepared published programs."""
    from tools.research_store import Store
    from tools.research_pipeline import _module_specs, _require_cases
    p = plan()
    store = Store(Path(root))
    receipt = store.verify_object(evidence_id, 'evaluation')['payload']
    manifest = store._manifest(receipt['experiment_id'])
    report = store._accepted_evidence(manifest['id'], manifest, evidence_id)
    if not report['evaluation_ok'] or report.get('admission_status') != 'observed_no_contention':
        raise ValueError('accepted source-bound GPU correctness required')
    if {r['case_id'] for r in report['cases']} != set(p['gpu_case_ids']):
        raise ValueError('GPU receipt must cover the exact combined selection')
    source_id = report['source_id']
    store.verify_object(source_id, 'source')
    source = store.root / 'objects' / source_id / 'files'
    for name in SOURCE_FILES:
        if (HERE / name).read_bytes() != (source / name).read_bytes():
            raise ValueError('CPU compiler differs from the GPU-verified source: ' + name)
    preparations = [i for i in report['artifact_ids']
                    if store.verify_object(i)['kind'] == 'confirmation-prepared']
    if len(preparations) != 1:
        raise ValueError('one retained confirmation preparation required')
    prepared = store.verify_object(preparations[0], 'confirmation-prepared')['payload']
    if prepared['source_id'] != source_id or prepared['reservation'] != manifest['confirmation']:
        raise ValueError('prepared source or reservation differs')
    _require_cases(prepared['prepared'], prepared['cases'])
    _module_specs(prepared['prepared'], 'combined-gate')
    rows = {row['case_id']: row for row in prepared['prepared']['cases']}
    published = {}
    for problem, cid in p['shared_published_cases'].items():
        row = rows[cid]
        folder = Path(row['case_dir'])
        record = verify_case(folder, {'record_sha256': p['gpu_record_sha256'][cid]})
        original = HERE / 'emmy-hackathon-2026/problems' / problem
        ir = next(original.glob('*.ir.json'))
        if len(row['kernels']) != 1 or read(ir) != read(folder / record['ir_paths'][0]):
            raise ValueError('published CPU/GPU IR differs: ' + problem)
        if read(original / 'meta.json') != read(folder / 'meta.json'):
            raise ValueError('published CPU/GPU metadata differs: ' + problem)
        kernel = row['kernels'][0]
        if set(kernel['targets']) != {'sm_89', 'sm_120'} or kernel['compile']['exit_code'] != 0:
            raise ValueError('both assembly targets required')
        for target in kernel['targets'].values():
            if target['exit_code'] != 0 or target['hashes']['kernel.ptx'] != kernel['compile']['ptx_sha256']:
                raise ValueError('published assembly/PTX identity differs')
        text = (Path(kernel['targets']['sm_89']['folder']) / 'kernel.ptx').read_text()
        published[problem] = {'text': text, 'meta': read(folder / 'meta.json'),
                              'ptx_sha256': sha(text.encode()), 'case_id': cid}
    payload = {'plan_id': p['plan_id'], 'plan': p, 'source_id': source_id,
               'source_files': {n: sha((source / n).read_bytes()) for n in SOURCE_FILES},
               'gpu_evidence_id': evidence_id, 'gpu_prepared_id': preparations[0],
               'gpu_evaluation_reused': True, 'published': published}
    aid = store.put_artifact('combined-experiment-shared-checks', payload,
                             {'policy.json': POLICY.read_bytes(),
                              'compiler-tests.py': (HERE / 'tests/test_compiler.py').read_bytes(),
                              'runner.py': Path(__file__).read_bytes()})
    Path(out).write_text(json.dumps({'shared_checks_id': aid, 'plan_id': p['plan_id'],
                                    'source_id': source_id, 'total_checks': p['total_checks']}, indent=2) + '\n')
    return aid


def worker(input_path):
    """One restricted CPU case, or the six explicitly inventoried address cases."""
    import contextlib
    q = read(input_path)
    spec = importlib.util.spec_from_file_location('gate_compiler_tests', HERE / 'tests/test_compiler.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if q['kind'] == 'published':
        case = module.CompilerTests()
        case.setUp()
        try:
            case.check_published_entries(q['meta'], q['text'])
            if q['problem'] == 'm01_matmul_512':
                source = next((module.ROOT / q['problem']).glob('*.ir.json'))
                case.check_m01_invariant_index_division(source, q['text'])
            if q['problem'] == 'm02_matmul_rect':
                case.check_vector_copy_runtime_fallback(q['text'])
        finally:
            case.doCleanups()
        result = {'ok': True, 'leaf_checks': 1, 'integrated_problem': q['problem']}
    else:
        seen = []

        class BudgetedCase(module.CompilerTests):
            @contextlib.contextmanager
            def subTest(self, msg=None, **params):
                if len(seen) >= len(q['cases']) or params != q['cases'][len(seen)]:
                    raise ValueError('unplanned CPU parameter case; body not executed')
                seen.append(params)
                with super().subTest(msg=msg, **params):
                    yield

        outcome = unittest.TestResult()
        BudgetedCase(q['method']).run(outcome)
        actual = len(seen) if seen else 1
        result = {'ok': outcome.wasSuccessful() and not outcome.skipped and actual == len(q['cases']),
                  'leaf_checks': actual, 'method': q['method'], 'parameters': seen,
                  'failures': [(str(test), detail) for test, detail in outcome.failures],
                  'errors': [(str(test), detail) for test, detail in outcome.errors]}
    result['uid'] = os.getuid()
    print(json.dumps(result))
    return 0 if result['ok'] else 1


def run_cpu(root, shared_id, out, uid=65534):
    from tools.research_store import Store
    if sys.platform != 'linux' or os.geteuid() != 0 or uid == 0:
        raise ValueError('CPU checks require Linux and a separate restricted UID')
    p = plan()
    store = Store(Path(root))
    shared = store.verify_object(shared_id, 'combined-experiment-shared-checks')['payload']
    if shared['plan_id'] != p['plan_id']:
        raise ValueError('shared checks use a different reviewed plan')
    store.verify_object(shared['source_id'], 'source')
    if set(shared['source_files']) != set(SOURCE_FILES) or set(shared['published']) != set(PUBLISHED_PROBLEMS):
        raise ValueError('shared source or published inventory differs')
    gpu = store.verify_object(shared['gpu_evidence_id'], 'evaluation')['payload']
    report = store._accepted_evidence(gpu['experiment_id'], store._manifest(gpu['experiment_id']),
                                      shared['gpu_evidence_id'])
    if report['source_id'] != shared['source_id'] or {r['case_id'] for r in report['cases']} != set(p['gpu_case_ids']):
        raise ValueError('GPU evidence differs from the CPU source or case inventory')
    for name, digest in shared['source_files'].items():
        frozen = store.root / 'objects' / shared['source_id'] / 'files' / name
        if sha((HERE / name).read_bytes()) != digest or (HERE / name).read_bytes() != frozen.read_bytes():
            raise ValueError('CPU source changed after shared preparation')
    for problem, row in shared['published'].items():
        if row['case_id'] != p['shared_published_cases'][problem] or sha(row['text'].encode()) != row['ptx_sha256']:
            raise ValueError('shared published program identity differs')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    jobs = [{'kind': 'published', 'problem': problem, **row}
            for problem, row in shared['published'].items()]
    jobs += [{'kind': 'cpu', **row} for row in p['additional_cpu_checks']]
    rows = []
    deadline = time.monotonic() + 240

    def limits():
        resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))

    for i, job in enumerate(jobs):
        if time.monotonic() + 35 > deadline:
            raise TimeoutError('CPU gate deadline; retain partial results, do not claim a pass')
        with tempfile.TemporaryDirectory(prefix='emmy-gate-input-') as temporary:
            folder = Path(temporary)
            folder.chmod(0o755)
            input_path = folder / 'input.json'
            input_path.write_text(json.dumps(job))
            input_path.chmod(0o644)
            env = {**os.environ, 'PYTHONPATH': str(HERE), 'OPENBLAS_NUM_THREADS': '1',
                   'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
            child = subprocess.Popen([sys.executable, '-B', '-m', 'tools.experiment_gate', '_worker', str(input_path)],
                                     cwd=HERE, env=env, user=uid, group=uid, extra_groups=[],
                                     preexec_fn=limits, start_new_session=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                stdout, stderr = child.communicate(timeout=35)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                stdout, stderr = child.communicate()
                (out / f'{i}.stdout').write_text(stdout)
                (out / f'{i}.stderr').write_text(stderr)
                raise TimeoutError(f'CPU check {i} timed out; owned process group killed')
        (out / f'{i}.stdout').write_text(stdout)
        (out / f'{i}.stderr').write_text(stderr)
        if child.returncode:
            raise ValueError(f'CPU check {i} failed; inspect retained stdout/stderr')
        row = json.loads(stdout)
        if not row['ok'] or row['uid'] != uid:
            raise ValueError(f'CPU check {i} failed')
        rows.append(row)
        (out / 'progress.json').write_text(json.dumps(rows, indent=2) + '\n')
    leaves = sum(row['leaf_checks'] for row in rows if 'method' in row)
    if leaves != p['additional_cpu_leaf_checks'] or len(shared['published']) != 15:
        raise ValueError('executed CPU inventory differs from plan')
    result = {'ok': True, 'plan_id': p['plan_id'], 'source_id': shared['source_id'],
              'shared_checks_id': shared_id, 'gpu_evidence_id': shared['gpu_evidence_id'],
              'gpu_cases': p['gpu_cases'], 'additional_cpu_leaf_checks': leaves,
              'integrated_published_cases': 15, 'total_checks': p['total_checks'],
              'gpu_execution_reused': True, 'new_gpu_executions': 0, 'cpu_rows': rows,
              'note': 'New CPU results plus unchanged source-bound GPU evidence; not a fresh GPU execution receipt.'}
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('prepare-shared')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--gpu-evidence', required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('check-cpu')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--shared-id', required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('_worker', help=argparse.SUPPRESS)
    p.add_argument('input')
    args = parser.parse_args()
    if args.command == '_worker':
        return worker(args.input)
    if args.command == 'plan':
        value = plan()
        if args.out.exists():
            raise ValueError('use a fresh plan output')
        args.out.write_text(json.dumps(value, indent=2) + '\n')
        print(json.dumps({k: value[k] for k in ('plan_id', 'gpu_cases', 'additional_cpu_leaf_checks', 'total_checks')}))
    elif args.command == 'prepare-shared':
        if args.out.exists():
            raise ValueError('use a fresh shared-check output')
        print(json.dumps({'shared_checks_id': prepare_shared(args.root, args.gpu_evidence, args.out)}))
    else:
        result = run_cpu(args.root, args.shared_id, args.out)
        print(json.dumps({k: result[k] for k in ('ok', 'source_id', 'total_checks', 'new_gpu_executions')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
