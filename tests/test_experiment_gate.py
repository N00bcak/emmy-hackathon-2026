"""Budget and coverage checks; no compiler or GPU execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools import experiment_gate as gate


class ExperimentGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.policy = Path(self.temporary.name) / 'policy.json'
        self.original = gate.read(gate.POLICY)

    def changed_plan(self, mutate, cases=None):
        policy = copy.deepcopy(self.original)
        mutate(policy)
        self.policy.write_text(json.dumps(policy))
        return gate.plan(self.policy, cases)

    def test_full_coverage_is_exactly_one_hundred_actual_cases(self):
        p = gate.plan()
        self.assertEqual((p['gpu_cases'], p['additional_cpu_leaf_checks'], p['total_checks']), (66, 34, 100))
        self.assertEqual(len(p['shared_published_cases']), 15)
        original = gate.read(gate.HERE / self.original['gpu_policy'])
        self.assertLessEqual({r['id'] for r in original['cases']}, set(p['gpu_case_ids']))

    def test_six_distinct_address_inputs_are_not_one_named_method(self):
        p = gate.plan()
        row = next(r for r in p['additional_cpu_checks']
                   if r['method'] == 'test_cp_async_destination_address_arithmetic')
        self.assertEqual(row['cases'], [{'dynamic': d, 'base': b, 'offset': o}
                                       for d in (False, True)
                                       for b, o in ((0, 48), (1024, 48), (6144, 256))])

    def test_extra_existing_gpu_case_fails_before_execution(self):
        p = gate.plan()
        extra = next(folder for folder in (gate.HERE / 'tests/gpu-cases/cases').iterdir()
                     if folder.is_dir() and folder.name not in p['gpu_case_ids'])
        with self.assertRaisesRegex(ValueError, '101 actual cases'):
            gate.plan(case_dirs=[*p['cases'], extra])

    def test_lower_budget_is_not_silently_exceeded(self):
        with self.assertRaisesRegex(ValueError, '100 actual cases; limit is 99'):
            self.changed_plan(lambda p: p.update(max_checks=99))

    def test_duplicate_or_missing_gpu_case_is_not_consolidation(self):
        p = gate.plan()
        with self.assertRaisesRegex(ValueError, 'missing or duplicated'):
            gate.plan(case_dirs=[*p['cases'][:-1], p['cases'][0]])

    def test_unaccounted_cpu_method_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'unaccounted or duplicated'):
            self.changed_plan(lambda p: p['additional_cpu_checks'].pop())

    def test_unimplemented_shared_validator_cannot_hide_a_cpu_case(self):
        def mutate(p):
            p['shared_cpu_methods']['test_reject_invalid_abi'] = ['p00_scalar_add']
            p['additional_cpu_checks'] = [r for r in p['additional_cpu_checks']
                                           if r['method'] != 'test_reject_invalid_abi']
        with self.assertRaisesRegex(ValueError, 'three implemented shared validators'):
            self.changed_plan(mutate)

    def test_changed_unit_source_requires_new_inventory_review(self):
        with self.assertRaisesRegex(ValueError, 'compiler tests changed'):
            self.changed_plan(lambda p: p.update(compiler_tests_sha256='0' * 64))


if __name__ == '__main__':
    unittest.main()
