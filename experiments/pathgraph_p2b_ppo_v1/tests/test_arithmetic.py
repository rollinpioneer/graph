import math,unittest
import numpy as np
from p2b.shaping import shaped_transition,DiscountAudit
from p2b.statistics import normalized_auc,first_threshold,crossed_bootstrap
from p2b.io_utils import protocol,seed_for
from p2b.specs import eval_specs,TrainStream

class ArithmeticTests(unittest.TestCase):
    def test_discount_formula(self):
        r=shaped_transition(1,-.5,-.2,gamma=.9)
        self.assertAlmostEqual(r['training_reward'],1.32)
    def test_terminal_zero(self):
        r=shaped_transition(0,-.5,-.9,terminated=True)
        self.assertEqual(r['phi_after_effective'],0.);self.assertEqual(r['training_reward'],.5)
    def test_success_base_plus_shaping(self):
        self.assertEqual(shaped_transition(1,-.5,0.,terminated=True)['training_reward'],1.5)
    def test_truncation_keeps_potential(self):
        r=shaped_transition(0,-.5,-.9,truncated=True)
        self.assertEqual(r['phi_after_effective'],-.9)
    def test_rollout_cut_is_neither_terminal_nor_truncated(self):
        r=shaped_transition(0,-.5,-.2)
        self.assertFalse(r['terminated'] or r['truncated'])
    def test_two_flags_rejected(self):
        with self.assertRaises(ValueError):shaped_transition(0,0,0,terminated=True,truncated=True)
    def test_boolean_flags(self):
        with self.assertRaises(TypeError):shaped_transition(0,0,0,terminated=1)
    def test_bad_gamma(self):
        for g in (0,-1,1.2):
            with self.assertRaises(ValueError):shaped_transition(0,0,0,gamma=g)
    def test_nonfinite(self):
        for x in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):shaped_transition(0,0,x)
    def test_gamma_one_closed_loop_zero(self):
        p=[-.5,-.2,-.9,-.5];a=DiscountAudit(gamma=1.)
        for x,y in zip(p,p[1:]):a.add(shaped_transition(0,x,y,gamma=1.))
        self.assertTrue(a.result()['passed']);self.assertAlmostEqual(a.result()['discounted_shaping'],0.)
    def test_discount_closed_loop_not_forced_zero(self):
        p=[-.5,-.2,-.9,-.5];a=DiscountAudit(gamma=.99)
        for x,y in zip(p,p[1:]):a.add(shaped_transition(0,x,y,gamma=.99))
        self.assertTrue(a.result()['passed']);self.assertNotEqual(a.result()['discounted_shaping'],0.)
        self.assertAlmostEqual(a.result()['discounted_shaping'],(.99**3-1)*p[0])
    def test_wait_discount_term_not_semantic_progress(self):
        r=shaped_transition(0,-.5,-.5)
        self.assertEqual(r['undiscounted_delta_raw'],0.);self.assertAlmostEqual(r['shaping_reward'],.005)
    def test_complete_episode_constant_shift(self):
        for path in ([-.8,-.5,-.3,0.],[-.8,-.8,-.8,-.8]):
            a=DiscountAudit()
            for i,(x,y) in enumerate(zip(path,path[1:])):a.add(shaped_transition(0,x,y,terminated=i==len(path)-2))
            self.assertAlmostEqual(a.result()['discounted_shaping'],.8)
    def test_bootstrap_consistency_trunc(self):
        g=.99;p=-.4;q=-.2;v=2.
        r=shaped_transition(.3,p,q,gamma=g,truncated=True)
        # Shaped value V'=V-Phi; one-step target differs only by -Phi(current).
        self.assertAlmostEqual(r['training_reward']+g*(v-q),.3+g*v-p)
    def test_discontinuous_bank_rejected(self):
        a=DiscountAudit();a.add(shaped_transition(0,0,-.1))
        with self.assertRaises(ValueError):a.add(shaped_transition(0,-.2,0))
    def test_boundary_reuse_rejected(self):
        a=DiscountAudit();a.add(shaped_transition(0,-.5,0,terminated=True))
        with self.assertRaises(RuntimeError):a.add(shaped_transition(0,0,0))
    def test_no_reward_clip(self):
        self.assertEqual(shaped_transition(1,-2,0,terminated=True)['training_reward'],3.)
    def test_tampered_shaping_fails(self):
        r=shaped_transition(0,-.2,-.1);r['shaping_reward']+=1
        a=DiscountAudit();a.add(r);self.assertFalse(a.result()['passed'])
    def test_random_long_identity(self):
        rng=np.random.default_rng(7)
        for _ in range(20):
            p=rng.uniform(-1.5,0,129);a=DiscountAudit()
            for i in range(128):a.add(shaped_transition(0,p[i],p[i+1],terminated=i==127))
            self.assertTrue(a.result()['passed'])
    def test_auc(self):self.assertAlmostEqual(normalized_auc([0,5,10],[0,.5,1]),.5)
    def test_threshold_censor(self):
        self.assertTrue(first_threshold([0,1,2],[0,.7,.9])['right_censored'])
    def test_threshold_pair(self):self.assertEqual(first_threshold([0,1,2],[0,.8,.9])['first_observed_step'],1)
    def test_bootstrap_constant(self):
        r=crossed_bootstrap(np.full((8,32),.1),resamples=50)
        self.assertAlmostEqual(r['low'],.1);self.assertAlmostEqual(r['high'],.1)
    def test_bootstrap_shape_guard(self):
        with self.assertRaises(ValueError):crossed_bootstrap(np.ones((1,3)))
    def test_splits(self):
        c=protocol();a=[set(v) for v in c['families'].values()]
        for i in range(len(a)):
            for j in range(i):self.assertFalse(a[i]&a[j])
    def test_eval_counts(self):
        c=protocol();self.assertEqual(len(eval_specs(c,'test')),2048);self.assertEqual(len(eval_specs(c,'validation')),512)
    def test_eval_seed_unique(self):
        rows=eval_specs(protocol(),'test');self.assertEqual(len({r['seed'] for r in rows}),len(rows))
    def test_train_stream_method_free(self):
        a=TrainStream(protocol(),211,0);b=TrainStream(protocol(),211,0)
        for _ in range(100):self.assertEqual(a.next(),b.next())
    def test_rank_streams_differ(self):self.assertNotEqual(TrainStream(protocol(),211,0).next(),TrainStream(protocol(),211,1).next())
    def test_stable_seed(self):self.assertEqual(seed_for('x',1),seed_for('x',1))
    def test_training_counts(self):
        c=protocol();t=c['training'];self.assertEqual(len(c['methods'])*len(t['policy_seeds']),48)
        self.assertEqual(t['total_timesteps']%(t['n_steps']*t['n_envs']),0)
    def test_no_weighting_or_masks(self):
        c=protocol()['training'];self.assertFalse(c['bc_initialization']);self.assertFalse(c['action_mask']);self.assertFalse(c['clip_reward']);self.assertFalse(c['normalize_reward'])

if __name__=='__main__':unittest.main()
