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
import os
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


def select(catalogs, policy=DEFAULT_POLICY, *, full=False, expand_versions=(), include_case_ids=()):
    contract = read(policy)
    required = contract['cases']
    ids = [r['id'] for r in required]
    scoped_records = {}
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate policy case')
    if scope_ref := contract.get('scope_manifest'):
        name = scope_ref['filename']
        if Path(name).name != name:
            raise ValueError('scope manifest must be beside its policy')
        scope_bytes = (Path(policy).parent / name).read_bytes()
        if sha(scope_bytes) != scope_ref['sha256']:
            raise ValueError('scope manifest changed')
        scope = json.loads(scope_bytes)
        previous = scope['previous_policy']
        if Path(previous['filename']).name != previous['filename']:
            raise ValueError('previous policy must be beside its policy')
        previous_bytes = (Path(policy).parent / previous['filename']).read_bytes()
        if sha(previous_bytes) != previous['sha256']:
            raise ValueError('previous policy changed')
        rows = scope['cases']
        if scope.get('version') == 'gpu-regression-scope-v6':
            previous = {r['id']: r for r in json.loads(previous_bytes)['cases']}
            inventory = {r['id']: r for r in rows}
            if (len(inventory) != len(rows) or len(rows) != scope['inventory_case_count']
                    or not previous.keys() <= inventory.keys()
                    or any(inventory[cid]['record_sha256'] != row['record_sha256']
                           for cid, row in previous.items())
                    or any(r['role'] not in ('mandatory', 'core', 'targeted', 'diagnostic', 'replace')
                           or r['disposition'] != ('retain' if r['role'] in ('mandatory', 'core') else r['role'])
                           or not r['reason'] or not r['run_when'] for r in rows)
                    or {r['id'] for r in rows if r['disposition'] == 'retain'} != set(ids)
                    or any(inventory[r['id']]['record_sha256'] != r['record_sha256'] for r in required)):
                raise ValueError('audited scope dispositions or record identities do not match policy')
            scoped_records = inventory
        elif (len({r['id'] for r in rows}) != len(rows)
                or {r['id'] for r in rows} != {r['id'] for r in json.loads(previous_bytes)['cases']}
                or any(r['disposition'] not in ('retain', 'archive') or not r['reason'] for r in rows)
                or {r['id'] for r in rows if r['disposition'] == 'retain'} != set(ids)):
            raise ValueError('scope dispositions do not match policies')
        if (scope.get('version') != 'gpu-regression-scope-v6'
                and any(r['marked_operators'] and r['disposition'] != 'retain' for r in rows)):
            raise ValueError('scope loses marked-operator coverage')
        if contract['assembly_targets'] != ['sm_89', 'sm_120'] or scope['assembly_targets'] != contract['assembly_targets']:
            raise ValueError('both assembly targets are required')
    if audit_ref := contract.get('coverage_manifest'):
        name = audit_ref['filename']
        if Path(name).name != name:
            raise ValueError('coverage manifest must be beside its policy')
        audit_bytes = (Path(policy).parent / name).read_bytes()
        if sha(audit_bytes) != audit_ref['sha256']:
            raise ValueError('coverage manifest changed')
        audit = json.loads(audit_bytes)
        selected = set(ids)
        rows = audit['cases']
        if set(audit['selected_ids']) != selected or len({r['case_id'] for r in rows}) != audit['campaign_cases'] or len(rows) != audit['campaign_cases']:
            raise ValueError('coverage manifest does not match selected cases or campaign inventory')
        compounds = {r['case_id']: r['checks'] for r in audit['compound_checks']}
        for row in rows:
            witnesses = set(row['witness_ids'])
            if not witnesses or not witnesses <= selected or row['selected'] != (row['case_id'] in selected):
                raise ValueError('unrepresented archived case in coverage manifest')
            if not row['selected'] and not row['same_sample_normalized_ir_topology'] and not any(c['ordered_subsequence'] and c['witness_id'] in witnesses for c in compounds.get(row['case_id'], [])):
                raise ValueError('archived case lacks a structural witness')
        if set(audit['surface_tokens']) != set(audit['retained_surface_tokens']):
            raise ValueError('coverage manifest loses IR surface features')
    index = {}
    for catalog in catalogs:
        for path in catalog_paths(catalog):
            record = read(Path(path) / 'case.json')
            # Repeated archive aliases are one case, not additional GPU work.
            index.setdefault(record['id'], (path, record))
    missing = set(ids) - index.keys()
    if missing:
        raise ValueError('required regression cases missing: ' + ', '.join(sorted(missing)))
    extras = set(include_case_ids)
    if extras - index.keys():
        raise ValueError('unknown additional case: ' + ', '.join(sorted(extras - index.keys())))
    chosen = list(ids)
    expansions = set(expand_versions)
    known_versions = {r.get('spec', {}).get('version', r.get('version')) for _, r in index.values()}
    if expansions - known_versions:
        raise ValueError('unknown expansion version: ' + ', '.join(sorted(expansions - known_versions)))
    for case_id, (_, r) in sorted(index.items()):
        if case_id not in chosen and (full or case_id in extras
                or r.get('spec', {}).get('version', r.get('version')) in expansions):
            chosen.append(case_id)
    policies = {**scoped_records, **{r['id']: r for r in required}}
    paths, records = [], []
    for case_id in chosen:
        path, _ = index[case_id]
        records.append(verify_case(path, policies.get(case_id)))
        paths.append(path)
    published = [r.get('spec', {}).get('problem') for r in records
                 if r.get('spec', {}).get('family') == 'published']
    if sorted(published) != sorted(PUBLISHED_PROBLEMS):
        raise ValueError('selection must contain exactly the 15 published contracts')
    if scoped_records and {cid for cid, row in scoped_records.items() if row['role'] == 'mandatory'} != {
            r['id'] for r in records if r.get('spec', {}).get('family') == 'published'}:
        raise ValueError('audited mandatory roles must match the 15 published contracts')
    policy_bytes = Path(policy).read_bytes()
    return {'version': contract['version'], 'policy_sha256': sha(policy_bytes),
            'assembly_targets': contract.get('assembly_targets', ['sm_89', 'sm_120']),
            'selection': 'full' if full else 'compact', 'expanded_versions': sorted(expansions),
            'included_case_ids': sorted(extras),
            'catalog_unique_cases': len(index), 'cases': paths, 'case_ids': chosen,
            'archive_only_cases': len(index) - len(chosen),
            'published_cases': len(published), 'generated_cases': len(chosen) - len(published),
            'kernel_count': sum(len(r['ir_paths']) for r in records),
            'requires_sanitizer': sum(bool(r.get('requires_sanitizer')) for r in records),
            'note': 'Correctness selection only; fresh confirmation and all15 source/suite gates still apply.'}


def current(root):
    """Refuse stale defaults instead of replaying an obsolete compiler's cases."""
    from tools.research_store import Store
    root = Path(root).resolve()
    store = Store(root)
    value = read(root / 'regression-suites/current.json')
    if value['source_id'] != store._state()['baseline_id']:
        raise ValueError('regression defaults name a stale retained compiler; synchronize first')
    compiler = root / 'objects' / value['source_id'] / 'files/compile.py'
    store.verify_object(value['source_id'], 'source')
    if sha(compiler.read_bytes()) != value['compiler_sha256']:
        raise ValueError('retained compiler bytes differ from synchronized defaults')
    if sha(Path(value['policy']).read_bytes()) != value['policy_sha256']:
        raise ValueError('regression policy changed; synchronize first')
    return value


def install(root, selection, case_list, catalogs, policy):
    """Install one source-bound default; aliases never copy compiler bytes."""
    from tools.research_store import Store, atomic_json
    root = Path(root).resolve()
    store = Store(root)
    pointer = root / 'regression-suites/current.json'
    prior = read(pointer) if pointer.exists() else {}
    with store._lock():
        state = store._state()
        if state.get('active_gpu') or state.get('preparation') or state.get('discovery_enabled'):
            raise ValueError('cannot synchronize defaults while research or owned work is active')
        source_id = state['baseline_id']
        store.verify_object(source_id, 'source')
        compiler = root / 'objects' / source_id / 'files/compile.py'
        value = {'version': selection['version'], 'source_id': source_id,
                 'compiler': str(compiler), 'compiler_sha256': sha(compiler.read_bytes()),
                 'case_list': str(Path(case_list).resolve()),
                 'catalog': str(Path(catalogs[0]).resolve()),
                 'catalogs': [str(Path(p).resolve()) for p in catalogs],
                 'policy': str(Path(policy).resolve()), 'policy_sha256': selection['policy_sha256'],
                 'case_count': len(selection['cases']), 'generated_cases': selection['generated_cases'],
                 'published_cases': selection['published_cases'],
                 'evaluator_root': prior.get('evaluator_root'), 'selection_only': True,
                 'use': 'Routine suite only; archive expansion is explicit. Existing evidence must match source/evaluator/case identities.'}
        if prior.get('source_id') == source_id:
            for key in ('official_validation', 'verified_source_evidence'):
                if key in prior:
                    value[key] = prior[key]
        retained = root / 'retained'
        retained.mkdir(exist_ok=True)
        for name, target in [('current.json', pointer), ('compile.py', compiler)]:
            destination = retained / name
            if destination.exists() and not destination.is_symlink():
                raise ValueError('refusing to overwrite a retained file: ' + str(destination))
            temporary = retained / ('.' + name + '.link-' + str(os.getpid()))
            temporary.symlink_to(target)
            os.replace(temporary, destination)
        # Historical source objects remain immutable; identical compiler bytes
        # share one inventory entry without merging distinct source identities.
        versions = {}
        for sid in dict.fromkeys([source_id] + [r['source_id'] for r in state['baseline_history']]):
            store.verify_object(sid, 'source')
            path = root / 'objects' / sid / 'files/compile.py'
            key = sha(path.read_bytes())
            row = versions.setdefault(key, {'compiler_sha256': key, 'compiler': str(path),
                                             'source_ids': [], 'active': False})
            row['source_ids'].append(sid)
            row['active'] |= sid == source_id
        atomic_json(retained / 'versions.json', {'compilers': list(versions.values())})
        atomic_json(pointer, value)
    return current(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, action='append')
    parser.add_argument('--root', type=Path, help='Use source-checked synchronized remote defaults')
    parser.add_argument('--install', action='store_true', help='Synchronize closed-run compiler and suite defaults')
    parser.add_argument('--policy', type=Path, default=DEFAULT_POLICY)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--full', action='store_true', help='Replay the archive explicitly')
    parser.add_argument('--expand-version', action='append', default=[], help='Add a changed mechanism’s archived variants')
    parser.add_argument('--include-case', action='append', default=[], help='Add an exact case ID for a changed mechanism')
    args = parser.parse_args()
    if args.install and not args.root:
        parser.error('--install requires --root')
    defaults = current(args.root) if args.root and not args.install else None
    catalogs = args.catalog or (defaults.get('catalogs', [defaults['catalog']]) if defaults else [])
    if not catalogs:
        parser.error('supply --catalog, or --root with synchronized defaults')
    policy = Path(defaults['policy']) if defaults and args.policy == DEFAULT_POLICY else args.policy
    result = select(catalogs, policy, full=args.full, expand_versions=args.expand_version,
                    include_case_ids=args.include_case)
    # Reservations are immutable: interruption must not silently redraw/reselect.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    if args.install:
        install(args.root, result, args.out, catalogs, policy)
    print(json.dumps({k: v for k, v in result.items() if k not in ('cases', 'case_ids')}, indent=2))


if __name__ == '__main__':
    main()
