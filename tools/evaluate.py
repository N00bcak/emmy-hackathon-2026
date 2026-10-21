#!/usr/bin/env python3
"""Bounded sequential upstream correctness sweep; no performance claims."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CHALLENGE = ROOT / 'emmy-hackathon-2026'
sys.path.insert(0, str(ROOT))
from tools.published import published_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', type=Path, default=ROOT / 'compile.py')
    parser.add_argument('--out', type=Path, required=True, help='New artifact directory')
    parser.add_argument('--problem', action='append', help='Repeat to select cases; default all')
    parser.add_argument('--timeout', type=float, default=180, help='Seconds per problem')
    parser.add_argument('--budget', type=float, default=900, help='Total wall seconds')
    parser.add_argument('--server', default='http://127.0.0.1:48702', help='Local runner by default; public runner requires an explicit URL')
    parser.add_argument('--arch')
    args = parser.parse_args()
    compiler = args.compiler.resolve()
    if not compiler.is_file():
        parser.error(f'compiler does not exist: {compiler}')
    if args.timeout <= 0 or args.budget <= 0:
        parser.error('timeout and budget must be positive')
    cases = {p.parent.name: p.parent for p in sorted((CHALLENGE / 'problems').glob('*/meta.json'))}
    selected = args.problem or list(cases)
    if not selected or any(p not in cases for p in selected):
        parser.error('select existing problem IDs')
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'compiler': str(compiler),
              'compiler_sha256': hashlib.sha256(compiler.read_bytes()).hexdigest(),
              'python': sys.version, 'cases': [], 'timing_is_smoke_only': True}
    report['source_hashes'] = {}
    import os
    report['source_revision_label'] = os.environ.get('EMMY_SOURCE_REVISION')
    for directory, folders, files in os.walk(ROOT):
        folders[:] = [f for f in folders if f not in {
            '.git', '.venv', '__pycache__', 'emmy-hackathon-2026', 'experiments'}]
        for filename in files:
            path = Path(directory) / filename
            if path.suffix == '.py' or filename == 'requirements.txt':
                report['source_hashes'][str(path.relative_to(ROOT))] = (
                    hashlib.sha256(path.read_bytes()).hexdigest())
    started = time.monotonic()
    for problem in selected:
        remaining = args.budget - (time.monotonic() - started)
        record = {'problem': problem, 'meta_sha256': hashlib.sha256(
            (cases[problem] / 'meta.json').read_bytes()).hexdigest()}
        if remaining <= 0:
            record['status'] = 'unverified_budget'
        else:
            command = [sys.executable, str(CHALLENGE / 'submit.py'), problem,
                       '--compiler', str(compiler)]
            for option in ['server', 'arch']:
                if getattr(args, option):
                    command += [f'--{option}', getattr(args, option)]
            record['command'] = command
            before = time.monotonic()
            log = args.out / f'{problem}.log'
            with log.open('wb') as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    code = process.wait(timeout=min(args.timeout, remaining))
                    record.update(status='pass' if code == 0 else 'failed', exit_code=code)
                except subprocess.TimeoutExpired:
                    import signal
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    record['status'] = 'unverified_timeout'
            record['wall_seconds'] = time.monotonic() - before
        report['cases'].append(record)
        (args.out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f"{problem}: {record['status']}", flush=True)
    report['published_correctness'] = published_status(report['cases'], source_unchanged=(
        hashlib.sha256(compiler.read_bytes()).hexdigest() == report['compiler_sha256']))
    report['valid_for_benchmarking'] = False
    report['qualification_note'] = 'Standalone correctness smoke only; research benchmarks require exact frozen-source/suite Store gates.'
    (args.out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if all(r['status'] == 'pass' for r in report['cases']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
