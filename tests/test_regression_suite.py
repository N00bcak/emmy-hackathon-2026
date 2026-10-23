"""Selection integrity tests; these do not claim GPU execution evidence."""
import copy
import json
import shutil
from pathlib import Path
import tempfile
import unittest

from tools import regression_suite as suite
from tools.published import PUBLISHED_PROBLEMS


class RegressionSelection(unittest.TestCase):
    def test_legacy_v5_scope_and_receipts_remain_unchanged(self):
        policy_path = suite.DEFAULT_POLICY.parent / 'gpu-regression-compact-v5.json'
        policy = suite.read(policy_path)
        scope_path = suite.DEFAULT_POLICY.parent / policy['scope_manifest']['filename']
        self.assertEqual(suite.sha(scope_path.read_bytes()), policy['scope_manifest']['sha256'])
        scope = suite.read(scope_path)
        previous_path = scope_path.parent / scope['previous_policy']['filename']
        self.assertEqual(suite.sha(previous_path.read_bytes()), scope['previous_policy']['sha256'])
        previous = suite.read(previous_path)
        rows = scope['cases']
        self.assertEqual({r['id'] for r in rows}, {r['id'] for r in previous['cases']})
        self.assertEqual(len(rows), len({r['id'] for r in rows}))
        self.assertEqual({r['id'] for r in rows if r['disposition'] == 'retain'}, {r['id'] for r in policy['cases']})
        self.assertTrue(all(r['disposition'] == 'retain' for r in rows if r['marked_operators']))
        self.assertEqual(sum(r['group'] == 'required_gate' for r in policy['cases']), 15)
        self.assertEqual(policy['assembly_targets'], ['sm_89', 'sm_120'])
        receipts = suite.read(suite.DEFAULT_POLICY.parents[1] / 'docs/evidence/powenn-test-scope-v5.json')
        self.assertEqual(receipts['policy_sha256'], suite.sha(policy_path.read_bytes()))
        expected = {r['id']: r['record_sha256'] for r in policy['cases']}
        for arm in ('ours', 'theirs'):
            results = [r for r in receipts['results'] if r['arm'] == arm]
            self.assertEqual({r['case_id']: r['case_record_sha256'] for r in results}, expected)
            for row in results:
                self.assertEqual({r['arch'] for r in row['assembly']}, {'sm_89', 'sm_120'})

    def test_audited_default_selects_exact_65_with_unchanged_records(self):
        from tools.bundled_regressions import resolve
        catalog, manifest = resolve('campaign100')
        self.catalog.write_text(json.dumps(catalog))
        result = suite.select([self.catalog])
        self.assertEqual(result['case_ids'], manifest['suites']['core65'])
        self.assertEqual(len(result['cases']), 65)
        self.assertEqual(result['published_cases'], 15)
        self.assertEqual(result['generated_cases'], 50)
        self.assertEqual(result['archive_only_cases'], 35)
        self.assertEqual(result['assembly_targets'], ['sm_89', 'sm_120'])
        default, _ = resolve()
        self.assertEqual([r['id'] for r in default['cases']], result['case_ids'])
        legacy = suite.DEFAULT_POLICY.parent / 'gpu-regression-compact-v5.json'
        self.assertEqual(suite.select([self.catalog], legacy)['case_ids'], manifest['suites']['default93'])
        self.assertEqual(len(suite.select([self.catalog], full=True)['cases']), 100)

    def test_audited_scope_rejects_role_record_and_inventory_tampering(self):
        from tools.bundled_regressions import resolve
        catalog, _ = resolve('campaign100')
        self.catalog.write_text(json.dumps(catalog))
        policy = suite.read(suite.DEFAULT_POLICY)
        source_scope = suite.DEFAULT_POLICY.parent / policy['scope_manifest']['filename']
        original = suite.read(source_scope)
        legacy_name = original['previous_policy']['filename']
        (self.root / legacy_name).write_bytes((source_scope.parent / legacy_name).read_bytes())
        scope_path = self.root / 'scope.json'
        def check(scope):
            scope_path.write_text(json.dumps(scope))
            policy['scope_manifest'] = {'filename': scope_path.name, 'sha256': suite.sha(scope_path.read_bytes())}
            self.policy.write_text(json.dumps(policy))
            return suite.select([self.catalog], self.policy)
        self.assertEqual(len(check(original)['cases']), 65)
        for mutate in (
            lambda scope: scope['cases'].pop(),
            lambda scope: scope['cases'][0].update(disposition='targeted'),
            lambda scope: scope['cases'][0].update(record_sha256='0' * 64),
            lambda scope: scope['cases'][0].update(reason=''),
        ):
            scope = copy.deepcopy(original)
            mutate(scope)
            with self.assertRaisesRegex(ValueError, 'audited scope'):
                check(scope)
        scope = copy.deepcopy(original)
        scope['cases'][0]['role'] = 'core'
        with self.assertRaisesRegex(ValueError, 'mandatory roles'):
            check(scope)

    def test_targeted_expansion_is_explicit_and_keeps_record_binding(self):
        from tools.bundled_regressions import resolve
        catalog, _ = resolve('campaign100')
        policy = suite.read(suite.DEFAULT_POLICY)
        scope = suite.read(suite.DEFAULT_POLICY.parent / policy['scope_manifest']['filename'])
        extra = next(r['id'] for r in scope['cases'] if r['disposition'] == 'targeted')
        self.catalog.write_text(json.dumps(catalog))
        selection = suite.select([self.catalog], include_case_ids=[extra, extra])
        self.assertEqual(len(selection['cases']), 66)
        self.assertEqual(selection['case_ids'].count(extra), 1)
        with self.assertRaisesRegex(ValueError, 'unknown additional case'):
            suite.select([self.catalog], include_case_ids=['not-a-case'])
        row = next(r for r in catalog['cases'] if r['id'] == extra)
        copied = self.root / 'targeted-copy'
        shutil.copytree(row['path'], copied)
        row['path'] = str(copied)
        record = suite.read(copied / 'case.json')
        record['requires_sanitizer'] = not record.get('requires_sanitizer', False)
        (copied / 'case.json').write_text(json.dumps(record))
        self.catalog.write_text(json.dumps(catalog))
        with self.assertRaisesRegex(ValueError, 'record changed'):
            suite.select([self.catalog], include_case_ids=[extra])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.paths, self.records, rows = [], [], []
        for i, problem in enumerate([*PUBLISHED_PROBLEMS, None, None]):
            folder = self.root / str(i)
            folder.mkdir()
            (folder / 'meta.json').write_text('{}')
            files = {'meta.json': suite.sha(b'{}')}
            # Unique content identities, including among the synthetic anchors.
            (folder / 'kernel.ir.json').write_text(str(i))
            files['kernel.ir.json'] = suite.sha(str(i).encode())
            record = {'id': suite.sha(suite.canonical(files)), 'files': files,
                      'ir_paths': ['kernel.ir.json'], 'requires_sanitizer': i == 15,
                      'spec': {'family': 'published' if problem else 'pointwise',
                               'problem': problem, 'version': 'unit-case-v1'}}
            (folder / 'case.json').write_text(json.dumps(record))
            self.paths.append(str(folder))
            self.records.append(record)
            if i != 16:
                rows.append({'id': record['id'], 'record_sha256': suite.sha(suite.canonical(record))})
        self.policy = self.root / 'policy.json'
        self.policy.write_text(json.dumps({'version': 'unit', 'cases': rows}))
        self.catalog = self.root / 'catalog.json'
        self.catalog.write_text(json.dumps({'cases': self.paths + self.paths[:1]}))

    def test_default_deduplicates_aliases_and_keeps_all_anchors(self):
        result = suite.select([self.catalog], self.policy)
        self.assertEqual(len(result['cases']), 16)
        self.assertEqual(result['published_cases'], 15)
        self.assertEqual(result['requires_sanitizer'], 1)
        self.assertEqual(len(suite.select([self.catalog], self.policy, full=True)['cases']), 17)
        expanded = suite.select([self.catalog], self.policy, expand_versions=['unit-case-v1'])
        self.assertEqual(len(expanded['cases']), 17)

    def test_scope_dispositions_and_marked_coverage_fail_closed(self):
        original = self.policy.read_bytes()
        prior = self.root / 'previous.json'
        prior.write_bytes(original)
        policy = json.loads(original)
        policy['cases'] = policy['cases'][:-1]
        scope = {'previous_policy': {'filename': prior.name, 'sha256': suite.sha(original)},
                 'assembly_targets': ['sm_89', 'sm_120'],
                 'cases': [{'id': row['id'], 'disposition': 'retain' if i < 15 else 'archive',
                            'reason': 'unit scope', 'marked_operators': []}
                           for i, row in enumerate(json.loads(original)['cases'])]}
        scope_path = self.root / 'scope.json'
        def write_scope():
            scope_path.write_text(json.dumps(scope))
            policy['scope_manifest'] = {'filename': scope_path.name, 'sha256': suite.sha(scope_path.read_bytes())}
            policy['assembly_targets'] = scope['assembly_targets']
            self.policy.write_text(json.dumps(policy))
        write_scope()
        self.assertEqual(len(suite.select([self.catalog], self.policy)['cases']), 15)
        scope['cases'][-1]['marked_operators'] = ['Select']
        write_scope()
        with self.assertRaisesRegex(ValueError, 'marked-operator'):
            suite.select([self.catalog], self.policy)
        scope['cases'][-1]['marked_operators'] = []
        scope['cases'].pop()
        write_scope()
        with self.assertRaisesRegex(ValueError, 'dispositions'):
            suite.select([self.catalog], self.policy)

    def test_missing_witness_and_unknown_expansion_fail(self):
        self.catalog.write_text(json.dumps(self.paths[:-2]))
        with self.assertRaisesRegex(ValueError, 'missing'):
            suite.select([self.catalog], self.policy)
        self.catalog.write_text(json.dumps(self.paths))
        with self.assertRaisesRegex(ValueError, 'unknown expansion'):
            suite.select([self.catalog], self.policy, expand_versions=['typo'])

    def test_payload_and_sanitizer_flag_tampering_fail(self):
        path = Path(self.paths[15])
        (path / 'meta.json').write_text('{"atol":999}')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            suite.select([self.catalog], self.policy)
        (path / 'meta.json').write_text('{}')
        record = copy.deepcopy(self.records[15])
        record['requires_sanitizer'] = False
        (path / 'case.json').write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'record changed'):
            suite.select([self.catalog], self.policy)

    def test_omitted_published_case_cannot_be_hidden_by_policy(self):
        policy = suite.read(self.policy)
        policy['cases'].pop(0)
        self.policy.write_text(json.dumps(policy))
        with self.assertRaisesRegex(ValueError, '15 published'):
            suite.select([self.catalog], self.policy)

    def test_coverage_manifest_rejects_tampering_and_unrepresented_cases(self):
        policy = suite.read(self.policy)
        selected = [r['id'] for r in policy['cases']]
        audit = {'selected_ids': selected, 'campaign_cases': len(self.records),
                 'surface_tokens': ['node:Load'], 'retained_surface_tokens': ['node:Load'],
                 'compound_checks': [], 'cases': [
                     {'case_id': r['id'], 'selected': r['id'] in selected,
                      'same_sample_normalized_ir_topology': True,
                      'witness_ids': [r['id'] if r['id'] in selected else selected[-1]]}
                     for r in self.records]}
        path = self.root / 'coverage.json'
        def save():
            path.write_text(json.dumps(audit))
            policy['coverage_manifest'] = {'filename': path.name, 'sha256': suite.sha(path.read_bytes())}
            self.policy.write_text(json.dumps(policy))
        save()
        self.assertEqual(len(suite.select([self.catalog], self.policy)['cases']), 16)
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'coverage manifest changed'):
            suite.select([self.catalog], self.policy)
        audit['cases'][-1]['witness_ids'] = [self.records[-1]['id']]
        save()
        with self.assertRaisesRegex(ValueError, 'unrepresented archived case'):
            suite.select([self.catalog], self.policy)

    def test_synchronized_defaults_share_one_compiler_and_reject_stale_source(self):
        from tools.research_store import Store
        source = self.root / 'source'
        source.mkdir()
        (source / 'compile.py').write_text("print('unit source only')\n")
        store = Store(self.root / 'store', self.root / 'work')
        store.init(source)
        selection = suite.select([self.catalog], self.policy)
        case_list = self.root / 'selected.json'
        case_list.write_text(json.dumps(selection))
        value = suite.install(store.root, selection, case_list, [self.catalog], self.policy)
        self.assertEqual(value['source_id'], store._state()['baseline_id'])
        self.assertEqual(value['generated_cases'], 1)
        self.assertEqual(selection['archive_only_cases'], 1)
        retained = store.root / 'retained'
        self.assertEqual((retained / 'compile.py').resolve(), Path(value['compiler']))
        self.assertEqual((retained / 'current.json').resolve(), (store.root / 'regression-suites/current.json').resolve())
        self.assertEqual(len(suite.read(retained / 'versions.json')['compilers']), 1)
        value['source_id'] = 'obsolete'
        (store.root / 'regression-suites/current.json').write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'stale retained compiler'):
            suite.current(store.root)


if __name__ == '__main__':
    unittest.main()
