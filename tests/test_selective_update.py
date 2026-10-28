import importlib.util
from pathlib import Path
import unittest, math
spec=importlib.util.spec_from_file_location('gate',Path(__file__).resolve().parents[1]/'src/cape_tta/selective_update.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class GateTests(unittest.TestCase):
 def test_small_sample_abstains(self):
  q=m.correction([1]*3,[0]*3);self.assertTrue(math.isinf(q));self.assertFalse(m.accept(100,q))
 def test_exact_rank_and_strict_gate(self):
  q=m.correction([0,1,2,3],[0]*4,alpha=.4);self.assertEqual(q,2);self.assertFalse(m.accept(2,q));self.assertTrue(m.accept(3,q))
 def test_invalid(self):
  for a,b in [([float('nan')],[0]),([1],[])]:
   with self.assertRaises(ValueError):m.correction(a,b)
 def test_success_dominates_speed(self):
  self.assertGreater(m.episode_utility(True,220),m.episode_utility(False,1))
 def test_censored_failure(self):
  self.assertAlmostEqual(m.episode_utility(False,220),-.1)
if __name__=='__main__': unittest.main()
