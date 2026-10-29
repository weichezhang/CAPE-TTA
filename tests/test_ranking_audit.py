import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('ranking',Path(__file__).resolve().parents[1]/'src/cape_tta/ranking_audit.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def b(success,steps,score):return dict(success=success,steps_executed=steps,horizon_potential={'12':score},trajectory=[dict(action=[0]*7,done=success)])
class RankingTests(unittest.TestCase):
 def test_duplicates_keep_random_selection_mass(self):
  rows={'base':b(False,220,0),'good':b(True,100,1)}
  r=m.pool_summary(['base','base','base','good'],rows,'base')
  self.assertEqual(r['selected_success'],1);self.assertEqual(r['uniform_pool_expected_success'],.2);self.assertEqual(r['uniform_draw_expected_success'],.25)
 def test_ties_favor_baseline(self):
  r=m.pool_summary(['other'],{'base':b(False,220,1),'other':b(True,100,1)},'base');self.assertEqual(r['selected_action_id'],'base')
 def test_proxy_winner_can_have_positive_regret(self):
  r=m.pool_summary(['bad','good'],{'base':b(False,220,0),'bad':b(False,220,2),'good':b(True,80,1)},'base')
  self.assertEqual(r['selected_success'],0);self.assertGreater(r['utility_regret'],1)
 def test_incomplete_pool_not_silently_excluded(self):
  with self.assertRaises(ValueError):m.pool_summary(['missing'],{'base':b(True,20,0)},'base')
 def test_nan_score_rejected(self):
  with self.assertRaises(ValueError):m.pool_summary(['bad'],{'base':b(True,20,0),'bad':b(True,10,float('nan'))},'base')
 def test_censored_failure_utility_is_resource_penalty_not_completion(self):
  self.assertAlmostEqual(m.utility(b(False,220,0)),-.1)
 def test_replay_checks_actions_and_horizon_potential(self):
  x=b(False,220,0);y=b(False,220,0);self.assertTrue(m.replay_equal(x,y));y['trajectory'][0]['action'][0]=1;self.assertFalse(m.replay_equal(x,y))
if __name__=='__main__':unittest.main()
