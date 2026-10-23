"""Verify the portable fixed cases and optionally resolve a local catalog."""
import argparse
import json
from pathlib import Path

from tools.regression_suite import sha, verify_case

ROOT = Path(__file__).resolve().parents[1]


def resolve(suite):
    base = ROOT / 'tests/gpu-cases'
    manifest = json.loads((base / 'manifest.json').read_bytes())
    for relative, digest in manifest['files'].items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('nonportable artifact path')
        target = base / path
        if any(p.is_symlink() for p in (target, *target.parents)):
            raise ValueError('symlink in artifact path')
        if sha(target.read_bytes()) != digest:
            raise ValueError('artifact changed: ' + relative)
    rows = manifest['cases']
    ids = [r['id'] for r in rows]
    if len(ids) != manifest['case_count'] or len(set(ids)) != len(ids):
        raise ValueError('case count or identities changed')
    paths = {}
    for row in rows:
        if row['path'] != 'cases/' + row['id']:
            raise ValueError('nonportable case path')
        path = base / row['path']
        record = verify_case(path, row)
        if record['id'] != row['id']:
            raise ValueError('case identity changed')
        paths[row['id']] = str(path)
    policy_path = ROOT / 'tests/gpu-regression.json'
    policy_bytes = policy_path.read_bytes()
    if sha(policy_bytes) != manifest['default_policy_sha256']:
        raise ValueError('default policy changed')
    policy = json.loads(policy_bytes)
    if [r['id'] for r in policy['cases']] != manifest['suites']['default93']:
        raise ValueError('default selection changed')
    for row in policy['cases']:
        verify_case(paths[row['id']], row)
    selected = manifest['suites'][suite]
    if len(set(selected)) != len(selected) or any(i not in paths for i in selected):
        raise ValueError('invalid suite selection')
    return {'cases': [{'id': i, 'path': paths[i]} for i in selected]}, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('default93', 'campaign100'), default='default93')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    catalog, manifest = resolve(args.suite)
    if args.out:
        with args.out.open('x') as output:
            json.dump(catalog, output, indent=2)
            output.write('\n')
    print(f"Verified {manifest['case_count']} cases; selected {len(catalog['cases'])} "
          f"({args.suite}); {manifest['published_cases']} published contracts.")


if __name__ == '__main__':
    main()
