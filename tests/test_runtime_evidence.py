import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'src/cape_tta' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConfigTests(unittest.TestCase):
    def test_real_line_breaks_and_path_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'repo with spaces'
            for part in ('bddl_files', 'init_files', 'assets'):
                (root / 'libero/libero' / part).mkdir(parents=True)
            target = Path(tmp) / 'config/config.yaml'
            paths = load('runtime_config').write_libero_config(root, target)
            lines = target.read_text().splitlines()
            self.assertEqual(len(lines), 5)
            parsed = dict((line.split(': ', 1)[0], json.loads(line.split(': ', 1)[1])) for line in lines)
            self.assertEqual(parsed, paths)
            self.assertTrue(Path(parsed['datasets']).is_dir())

    def test_missing_assets_stop_before_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load('runtime_config').write_libero_config(tmp, Path(tmp) / 'config.yaml')


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        candidate = {'kind': 'greedy_base', 'raw_action': [0.] * 7,
                     'env_action': [0.] * 7, 'success': False,
                     'tokens': [31999] * 7, 'logprob': -10., 'steps_executed': 12, 'soft_gain': 0.1}
        second = dict(candidate, kind='sampled', soft_gain=0.2, tokens=[31998] * 7)
        self.data = {'kind': 'OpenVLA Oracle-CAPE pilot', 'scientific_result': False,
                     'cape_commit': 'a' * 40, 'checkpoint_revision': 'b' * 40,
                     'wrapper_parity_max_abs': 0., 'horizon': 12, 'candidate_count': 2,
                     'fresh_replay_determinism': {'passed': True, 'final_state_linf': 0.,
                                                'soft_progress_abs': 0., 'success_consistent': True},
                     'candidates': [candidate, second], 'base_soft_gain': 0.1,
                     'oracle_best_index': 1, 'oracle_best_soft_gain': 0.2,
                     'oracle_margin_over_base': 0.1}

    def validate(self):
        return load('pilot_validation').validate_pilot(self.data)

    def test_consistent_pilot_passes_but_remains_pilot(self):
        result = self.validate()
        self.assertTrue(result['passed'])
        self.assertEqual(result['unique_token_candidates'], 2)
        self.assertFalse(self.data['scientific_result'])

    def test_incorrect_oracle_margin_rejected(self):
        self.data['oracle_margin_over_base'] = 1.0
        self.assertFalse(self.validate()['passed'])

    def test_nonfinite_action_rejected(self):
        self.data['candidates'][0]['raw_action'][0] = float('nan')
        self.assertFalse(self.validate()['passed'])

    def test_missing_replay_evidence_rejected(self):
        self.data['fresh_replay_determinism'] = {}
        self.assertFalse(self.validate()['passed'])

    def test_empty_candidates_rejected(self):
        self.data['candidates'] = []
        self.assertFalse(self.validate()['passed'])

    def test_bad_parity_rejected(self):
        self.data['wrapper_parity_max_abs'] = 0.1
        self.assertFalse(self.validate()['passed'])


class FailureBundleTests(unittest.TestCase):
    def test_missing_gpu_still_produces_failure_bundle(self):
        # Avoid installing or running anything: hide executables from the child.
        import os
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / 'run'
            proc = subprocess.run([sys.executable, str(ROOT / 'scripts/colab_bootstrap.py'),
                                   '--run-dir', str(run_dir)],
                                  env=dict(os.environ, PATH=''), capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1)
            manifest = json.loads((run_dir / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(manifest['stage'], 'GPU check')
            with zipfile.ZipFile(str(run_dir) + '.zip') as bundle:
                self.assertIn('run.log', bundle.namelist())
                self.assertIn('manifest.json', bundle.namelist())
                self.assertIn('sha256.json', bundle.namelist())


if __name__ == '__main__':
    unittest.main()
