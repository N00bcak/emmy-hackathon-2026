"""A partial or stale correctness sweep must never authorize benchmarking."""
import unittest
from tools.published import PUBLISHED_PROBLEMS, published_status


class PublishedGateTests(unittest.TestCase):
    def records(self):
        return [{'problem': p, 'status': 'pass'} for p in PUBLISHED_PROBLEMS]

    def test_all_fifteen_distinct_passes_required(self):
        self.assertEqual(len(PUBLISHED_PROBLEMS), 15)
        self.assertTrue(published_status(self.records())['all15_passed'])
        self.assertFalse(published_status(self.records()[:-1])['all15_passed'])
        self.assertFalse(published_status(self.records(), source_unchanged=False)['all15_passed'])

    def test_failure_and_duplicate_cannot_be_hidden_by_pass(self):
        rows = self.records()
        rows[0]['status'] = 'failed'
        self.assertFalse(published_status(rows)['all15_passed'])
        rows.append({'problem': rows[0]['problem'], 'status': 'pass'})
        self.assertFalse(published_status(rows)['all15_passed'])

    def test_local_client_exit_status_accepted_but_missing_not_defaulted(self):
        rows = [{'problem': p, 'exit_code': 0} for p in PUBLISHED_PROBLEMS]
        self.assertTrue(published_status(rows)['all15_passed'])
        rows[-1].pop('exit_code')
        self.assertFalse(published_status(rows)['all15_passed'])


if __name__ == '__main__':
    unittest.main()
