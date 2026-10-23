"""Select the compact GPU regression witnesses from an unchanged case archive.

This is a case selector, not an evaluator or a benchmark. Its output is a normal
case-list accepted by the existing isolated preparation/correctness adapters.
The archive catalog may be a case-list, a list of records with ``path``, or an
inventory mapping absolute case paths to records. Runtime paths stay outside
the checked-in policy. No case data, tolerance, or launch metadata is rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.published import PUBLISHED_PROBLEMS

DEFAULT_POLICY = Path(__file__).resolve().parents[1] / 'tests/gpu-regression.json'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def catalog_paths(catalog):
    value = read(catalog)
    if isinstance(value, dict) and 'cases' in value:
        value = value['cases']
    if isinstance(value, dict):
        paths = list(value)
    elif isinstance(value, list):
        paths = [v['path'] if isinstance(v, dict) else v for v in value]
    else:
        raise ValueError('catalog must contain case paths')
    if not paths or any(not isinstance(p, str) or not Path(p).is_absolute() for p in paths):
        raise ValueError('catalog case paths must be absolute')
    return paths


def verify_case(path, expected=None):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('case directory is a symlink')
    record = read(path / 'case.json')
    if record['id'] != sha(canonical(record['files'])):
        raise ValueError('case identity differs from its file manifest')
    if expected and sha(canonical(record)) != expected['record_sha256']:
        raise ValueError('case record changed, including isolation/sanitizer flags')
    for name, digest in record['files'].items():
        file = path / name
        if Path(name).name != name or file.is_symlink() or sha(file.read_bytes()) != digest:
            raise ValueError('case artifact changed: ' + name)
    return record


def select(catalogs, policy=DEFAULT_POLICY, *, full=False, expand_versions=()):
    contract = read(policy)
    required = contract['cases']
    ids = [r['id'] for r in required]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate policy case')
    index = {}
    for catalog in catalogs:
        for path in catalog_paths(catalog):
            record = read(Path(path) / 'case.json')
            # Repeated archive aliases are one case, not additional GPU work.
            index.setdefault(record['id'], (path, record))
    missing = set(ids) - index.keys()
    if missing:
        raise ValueError('required regression cases missing: ' + ', '.join(sorted(missing)))
    chosen = list(ids)
    expansions = set(expand_versions)
    known_versions = {r.get('spec', {}).get('version', r.get('version')) for _, r in index.values()}
    if expansions - known_versions:
        raise ValueError('unknown expansion version: ' + ', '.join(sorted(expansions - known_versions)))
    for case_id, (_, r) in sorted(index.items()):
        if case_id not in chosen and (full or r.get('spec', {}).get('version', r.get('version')) in expansions):
            chosen.append(case_id)
    policies = {r['id']: r for r in required}
    paths, records = [], []
    for case_id in chosen:
        path, _ = index[case_id]
        records.append(verify_case(path, policies.get(case_id)))
        paths.append(path)
    published = [r.get('spec', {}).get('problem') for r in records
                 if r.get('spec', {}).get('family') == 'published']
    if sorted(published) != sorted(PUBLISHED_PROBLEMS):
        raise ValueError('selection must contain exactly the 15 published contracts')
    policy_bytes = Path(policy).read_bytes()
    return {'version': contract['version'], 'policy_sha256': sha(policy_bytes),
            'selection': 'full' if full else 'compact', 'expanded_versions': sorted(expansions),
            'catalog_unique_cases': len(index), 'cases': paths, 'case_ids': chosen,
            'published_cases': len(published), 'generated_cases': len(chosen) - len(published),
            'kernel_count': sum(len(r['ir_paths']) for r in records),
            'requires_sanitizer': sum(bool(r.get('requires_sanitizer')) for r in records),
            'note': 'Correctness selection only; fresh confirmation and all15 source/suite gates still apply.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, action='append', required=True)
    parser.add_argument('--policy', type=Path, default=DEFAULT_POLICY)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--full', action='store_true', help='Replay the archive explicitly')
    parser.add_argument('--expand-version', action='append', default=[], help='Add a changed mechanism’s archived variants')
    args = parser.parse_args()
    result = select(args.catalog, args.policy, full=args.full, expand_versions=args.expand_version)
    # Reservations are immutable: interruption must not silently redraw/reselect.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('cases', 'case_ids')}, indent=2))


if __name__ == '__main__':
    main()
