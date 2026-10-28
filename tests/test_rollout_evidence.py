import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('rollout_evidence', Path(__file__).resolve().parents[1] / 'src/cape_tta/rollout_evidence.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class RolloutSummaryTests(unittest.TestCase):
    def branches(self):
        return {'base': {'soft_gain': .2, 'sparse_gain': 0., 'success': False},
                'worse': {'soft_gain': -.1, 'sparse_gain': 0., 'success': False},
                'better': {'soft_gain': .3, 'sparse_gain': 0., 'success': False}}

    def test_baseline_inclusion_does_not_hide_bad_samples(self):
        row = module.summarize_group(['worse'] * 4, self.branches(), 'base')
        self.assertEqual(row['oracle_soft_margin'], 0.)
        self.assertLess(row['mean_sampled_soft_margin'], 0.)
        self.assertEqual(row['unique_sampled_actions'], 1)
        self.assertEqual(row['sampled_strictly_worse_count'], 4)

    def test_best_candidate_margin(self):
        row = module.summarize_group(['base', 'worse', 'better', 'base'], self.branches(), 'base')
        self.assertAlmostEqual(row['oracle_soft_margin'], .1)
        self.assertEqual(row['eligible_count_including_baseline'], 5)
        self.assertEqual(row['unique_eligible_actions'], 3)
        self.assertFalse(row['any_candidate_success_within_horizon'])

    def test_nonfinite_result_rejected(self):
        branches = self.branches(); branches['better']['soft_gain'] = float('nan')
        with self.assertRaises(ValueError):
            module.summarize_group(['better'], branches, 'base')

    def test_replay_mismatch_rejected(self):
        first = {'final_state': [0., 1.], 'soft_progress_end': .3, 'success': False}
        repeated = dict(first, final_state=[0., 2.])
        self.assertFalse(module.replay_summary(first, repeated)['passed'])

    def test_replay_zero_error_passes(self):
        first = {'final_state': [0., 1.], 'soft_progress_end': .3, 'success': False}
        self.assertTrue(module.replay_summary(first, first)['passed'])


if __name__ == '__main__':
    unittest.main()
