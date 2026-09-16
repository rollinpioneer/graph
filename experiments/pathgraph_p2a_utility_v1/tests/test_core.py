from __future__ import annotations
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from p2a.env import SkillEnv,Spec,ACTIONS,CONDITIONS,collect_episode,expert_action
from p2a.weights import make_weights,shuffled_weights,METHODS
from p2a.stats import paired_interval
from p2a.experiment import SPLITS

class EnvironmentTests(unittest.TestCase):
    def env(self,c='FREE_ORDER',horizon=64):return SkillEnv(Spec(2099000,c,101,'test',horizon))
    def test_action_count(self):self.assertEqual(len(ACTIONS),13)
    def test_observation_shape(self):self.assertEqual(self.env().observe().shape,(31,))
    def test_wait_cannot_auto_succeed(self):
        e=self.env()
        while not e.done:e.step(0)
        self.assertFalse(e.success);self.assertTrue(e.truncated)
    def test_action_changes_state(self):
        a=self.env();b=self.env(); a.step(0);b.step(ACTIONS.index('ACQUIRE_A'))
        self.assertNotEqual(a.snapshot()['objects'],b.snapshot()['objects'])
    def test_seed_reproducible(self):
        s=Spec(2099000,'TWO_LOSSES',101,'same')
        self.assertEqual(collect_episode(s,'noisy'),collect_episode(s,'noisy'))
    def test_wrong_action_no_progress(self):
        e=self.env();p=copy.deepcopy(e.objects);e.step(ACTIONS.index('ADVANCE_A'))
        self.assertEqual(p,e.objects);self.assertEqual(e.invalid_actions,1)
    def test_done_rejected(self):
        e=self.env(horizon=1);e.step(0)
        with self.assertRaises(RuntimeError):e.step(0)
    def test_invalid_action_rejected(self):
        for v in (-1,13,True,.1):
            with self.subTest(v=v):
                with self.assertRaises(ValueError):self.env().step(v)
    def test_no_scenario_in_snapshot(self):
        s=self.env('LATE_LOSS').snapshot()
        for k in ('condition','family','reward','psi','teacher_id'):self.assertNotIn(k,s)
    def test_current_obs_not_future_condition(self):
        np.testing.assert_array_equal(self.env('FREE_ORDER').observe(),self.env('LATE_LOSS').observe())
    def test_snapshot_not_aliased(self):
        e=self.env();s=e.snapshot(); e.step(ACTIONS.index('ACQUIRE_A'))
        self.assertFalse(s['objects']['A']['held'])
    def test_truncation_not_failure_label(self):
        e=self.env(horizon=1);e.step(0)
        self.assertFalse(e.snapshot()['terminal_failure'])
    def test_unilateral_valid_not_success(self):
        e=self.env('A_VALID_B_LOSS');self.assertTrue(e.objects['A']['valid']);self.assertFalse(e.success)
    def test_explicit_release_not_loss(self):
        e=self.env('A_STARTED');e.step(ACTIONS.index('RELEASE_A'))
        self.assertEqual(e.loss_total,0);self.assertFalse(e.objects['A']['held'])
    def test_expert_action_trajectories(self):
        for c in CONDITIONS:
            with self.subTest(c=c):
                r=collect_episode(Spec(2099000,c,101,'expert_'+c),'expert')
                self.assertTrue(r['info']['success'])
    def test_second_loss_observed(self):
        r=collect_episode(Spec(2099000,'TWO_LOSSES',101,'two'),'expert')
        self.assertEqual(r['info']['observed_losses'],2);self.assertEqual(r['info']['restored_losses'],2)
    def test_event_causality_and_pairing(self):
        r=collect_episode(Spec(2099000,'TWO_LOSSES',101,'two'),'expert');open_ids=set()
        for s in r['states']:
            for e in s['events']:
                self.assertLessEqual(e['known_at_ns'],s['available_at_ns'])
                k=(e['object_id'],e['loss_id'])
                if e['kind']=='LOSS':self.assertNotIn(k,open_ids);open_ids.add(k)
                if e['kind']=='HOLD_REESTABLISHED':self.assertIn(k,open_ids);open_ids.remove(k)
        self.assertEqual(open_ids,set())
    def test_alignment(self):
        r=collect_episode(Spec(2099000,'FREE_ORDER',101,'data'),'pause')
        self.assertEqual(len(r['states']),len(r['actions'])+1);self.assertEqual(len(r['obs']),len(r['actions']))
    def test_replay_applied_actions(self):
        s=Spec(2099000,'INVALIDATE_FIRST',101,'replay');r=collect_episode(s,'noisy');e=SkillEnv(s)
        for i,a in enumerate(r['actions']):e.step(a);self.assertEqual(e.snapshot(),r['states'][i+1])
    def test_split_disjoint(self):
        names=list(SPLITS)
        for i,a in enumerate(names):
            for b in names[i+1:]:self.assertFalse(set(SPLITS[a])&set(SPLITS[b]))

class WeightTests(unittest.TestCase):
    def test_uniform(self):np.testing.assert_array_equal(make_weights([-1,0,1],uniform=True)[0],np.ones(3))
    def test_mean_one(self):self.assertAlmostEqual(float(make_weights([-1,0,.1,1])[0].mean()),1.,places=6)
    def test_nonpositive_not_deleted(self):self.assertGreater(float(make_weights([-1,0,1])[0].min()),0)
    def test_zero_degenerate(self):
        w,m=make_weights([0,0,0]);np.testing.assert_array_equal(w,np.ones(3));self.assertTrue(m['all_zero_signal'])
    def test_rescale_invariance(self):
        np.testing.assert_allclose(make_weights([-.1,0,.2,.3])[0],make_weights([-10,0,20,30])[0])
    def test_invalid_nan(self):
        with self.assertRaises(ValueError):make_weights([0,float('nan')])
    def test_empty(self):
        with self.assertRaises(ValueError):make_weights([])
    def test_floor_invalid(self):
        with self.assertRaises(ValueError):make_weights([1],floor=0)
    def test_shuffle_multiset(self):
        w,_=make_weights(np.linspace(-1,1,100));s=shuffled_weights(w)
        np.testing.assert_array_equal(np.sort(w),np.sort(s));self.assertFalse(np.array_equal(w,s))
    def test_signed_vs_positive(self):
        _,r=make_weights([-1,1]);self.assertEqual(r['signed_return'],0);self.assertEqual(r['positive_mass'],1)
    def test_ess_bounds(self):
        _,r=make_weights([-1,0,.1,1]);self.assertGreaterEqual(r['ess'],1);self.assertLessEqual(r['ess'],4)
    def test_nine_methods(self):self.assertEqual(len(METHODS),9)

class StatisticsTests(unittest.TestCase):
    def test_equal_paired(self):
        r=paired_interval(np.zeros((5,16)));self.assertEqual(r['low'],0);self.assertEqual(r['high'],0)
    def test_constant_effect(self):
        r=paired_interval(np.full((5,16),.1));self.assertAlmostEqual(r['mean'],.1);self.assertAlmostEqual(r['low'],.1)
    def test_sign_change(self):
        d=np.random.default_rng(1).normal(size=(5,16));a=paired_interval(d);b=paired_interval(-d)
        self.assertAlmostEqual(a['low'],-b['high'])
    def test_nonscalar_units(self):
        with self.assertRaises(ValueError):paired_interval([1,2,3])
    def test_repeatable(self):
        d=np.arange(80).reshape(5,16)/80;self.assertEqual(paired_interval(d),paired_interval(d))
if __name__=='__main__':unittest.main()
