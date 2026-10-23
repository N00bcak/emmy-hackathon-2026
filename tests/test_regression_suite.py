"""Selection integrity tests; these do not claim GPU execution evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools import regression_suite as suite
from tools.published import PUBLISHED_PROBLEMS


class RegressionSelection(unittest.TestCase):
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
