import importlib.util
import math
from pathlib import Path
import unittest
import io

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sampling_diagnostics', ROOT / 'src/cape_tta/sampling_diagnostics.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DistributionTests(unittest.TestCase):
    def test_single_token_support_is_detected(self):
        result = module.probability_summary([0, 1, 0])
        self.assertEqual(result['support_size'], 1)
        self.assertEqual(result['entropy_nats'], 0)
        self.assertEqual(result['effective_support'], 1)

    def test_uniform_distribution_entropy(self):
        result = module.probability_summary([0.25] * 4)
        self.assertAlmostEqual(result['entropy_nats'], math.log(4))
        self.assertAlmostEqual(result['effective_support'], 4)

    def test_invalid_probabilities_rejected(self):
        for values in ([float('nan'), 0.5], [-0.5, 1.5], [0.1, 0.1], []):
            with self.subTest(values=values), self.assertRaises(ValueError):
                module.probability_summary(values)

    def test_token_diversity_can_collapse_in_executed_actions(self):
        candidates = [{'tokens': [10] * 7, 'env_action': [0.] * 7},
                      {'tokens': [11] * 7, 'env_action': [0.] * 7}]
        result = module.candidate_summary(candidates, [10] * 7, [0.] * 7)
        self.assertEqual(result['unique_token_candidates'], 2)
        self.assertEqual(result['unique_executed_actions'], 1)
        self.assertEqual(result['baseline_token_matches'], 1)


class TeeTests(unittest.TestCase):
    def test_shutdown_proxy_preserves_live_console_and_ignores_closed_log(self):
        spec = importlib.util.spec_from_file_location('probe', ROOT / 'scripts/probe_openvla_diversity.py')
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        console, log = io.StringIO(), io.StringIO()
        tee = probe.Tee(console, log)
        tee.write('before\n')
        log.close()
        tee.close()
        tee.write('after\n')
        self.assertFalse(console.closed)
        self.assertEqual(console.getvalue(), 'before\nafter\n')

    def test_repeated_candidates_are_not_independent_actions(self):
        candidate = {'tokens': [10] * 7, 'env_action': [0.] * 7}
        result = module.candidate_summary([candidate] * 4, [10] * 7, [0.] * 7)
        self.assertEqual(result['candidate_count'], 4)
        self.assertEqual(result['unique_executed_actions'], 1)


if __name__ == '__main__':
    unittest.main()
