import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('full_episode', Path(__file__).resolve().parents[1] / 'src/cape_tta/full_episode.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class Environment:
    def __init__(self, done_at=None, error_at=None):
        self.done_at = done_at; self.error_at = error_at; self.calls = []; self.closed = False

    def reset(self): pass
    def set_init_state(self, state): return {'state': state}
    def step(self, action):
        self.calls.append(action)
        if len(self.calls) == self.error_at: raise RuntimeError('injected env error')
        return {'count': len(self.calls)}, 0., len(self.calls) == self.done_at, {}
    def close(self): self.closed = True


class EpisodeTests(unittest.TestCase):
    def test_first_intervention_only_and_exact_budget(self):
        env = Environment(); policy_calls = []; rows = []
        def policy(obs): policy_calls.append(obs); return [2] * 7
        result = module.roll_episode(lambda: env, [0], [1] * 7, policy, max_steps=5, settle_steps=2, on_step=rows.append)
        self.assertEqual(len(env.calls), 7)
        self.assertEqual(env.calls[2], [1] * 7)
        self.assertEqual(len(policy_calls), 4)
        self.assertEqual(len(rows), 5)
        self.assertFalse(result['success']); self.assertTrue(env.closed)

    def test_done_stops_early(self):
        env = Environment(done_at=3)
        result = module.roll_episode(lambda: env, [0], [1] * 7, lambda obs: [2] * 7, max_steps=5, settle_steps=0)
        self.assertTrue(result['success']); self.assertEqual(result['steps_executed'], 3)

    def test_exception_closes_env(self):
        env = Environment(error_at=1)
        with self.assertRaises(RuntimeError):
            module.roll_episode(lambda: env, [0], [1] * 7, lambda obs: [2] * 7, settle_steps=0)
        self.assertTrue(env.closed)

    def test_settling_termination_is_invalid_comparison(self):
        env = Environment(done_at=1)
        with self.assertRaises(RuntimeError):
            module.roll_episode(lambda: env, [0], [1] * 7, lambda obs: [2] * 7)
        self.assertTrue(env.closed)


if __name__ == '__main__': unittest.main()
