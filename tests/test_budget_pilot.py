"""Synthetic fixtures here test software only; never emitted as experiment data."""
import copy
import math
import tempfile
import unittest
from pathlib import Path
from cape_tta.budget_pilot import atomic_json, seal, load_sealed, conformal_q, episode_split, evaluate_benefits


def fixture():
    rows=[]
    for i in range(12):
        m=(-1 if i%2 else 1)*.01
        rows.append(dict(episode_id=f'unit-test-{i:02}', oracle_kind='parameter_adaptation',
            features=[i/12, i%2],realized_benefit=m,
            frozen_outcome=dict(success=False,soft_gain=.01),adapted_outcome=dict(success=False,soft_gain=.01+m)))
    return rows


class BudgetPilotTests(unittest.TestCase):
    def test_quantile_sign(self):
        self.assertEqual(conformal_q([.4,.1,.2],[.1,.2,.3],.25),.4-.1)
    def test_finite_sample_infinity(self):
        self.assertTrue(math.isinf(conformal_q([0,0],[0,0],.1)))
        self.assertEqual(conformal_q([0]*9,[0]*9,.1),0)
    def test_no_episode_leakage(self):
        rows=fixture(); split=episode_split(rows)
        sets=[set(v) for v in split.values()]
        self.assertEqual(len(set.union(*sets)),len(rows))
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertEqual(split,episode_split(list(reversed(rows))))
        with self.assertRaises(ValueError): episode_split(rows+[rows[0]])
    def test_reject_action_oracle(self):
        rows=fixture();rows[0]['oracle_kind']='action_consequence'
        with self.assertRaises(ValueError): evaluate_benefits(rows)
    def test_no_acceptance_does_not_mean_zero_HAR(self):
        r=evaluate_benefits(fixture(),alpha=.1)
        self.assertTrue(r['q_is_infinite'])
        self.assertEqual(r['CAPE']['adaptation_acceptance_rate'],0)
        self.assertIsNone(r['CAPE']['HAR'])
        self.assertEqual(r['status'],'NO_GO_FOR_EXPANSION')
    def test_test_labels_cannot_change_predictor_or_quantile(self):
        rows=fixture();r=evaluate_benefits(rows,alpha=.25)
        changed=copy.deepcopy(rows)
        for row in changed:
            if row['episode_id'] in r['split']['test']: row['realized_benefit']+=.3
        other=evaluate_benefits(changed,alpha=.25)
        self.assertEqual(r['predictor'],other['predictor'])
        self.assertEqual(r['q'],other['q'])
    def test_sealed_cache_detects_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bank.json';atomic_json(p,{'real':False});seal(p)
            self.assertEqual(load_sealed(p),{'real':False})
            atomic_json(p,{'real':True})
            with self.assertRaises(ValueError): load_sealed(p)


if __name__=='__main__': unittest.main()
