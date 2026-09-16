"""Requires real byte-matched sources, not mocked V6 formulas."""
import copy,os,unittest
import numpy as np
from p2b.io_utils import protocol
from p2b.load_sources import load_sources
from p2b.task_core import TaskCore
from p2b.potentials import PotentialBank,METHODS

class FrozenAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo=os.environ.get('P2B_SOURCE_REPO');tools=os.environ.get('P2B_V6_TOOLS')
        if not repo or not tools:
            raise RuntimeError('Provide P2B_SOURCE_REPO and P2B_V6_TOOLS; source integration is not silently skipped')
        cls.env,cls.Cap,cls.sources=load_sources(repo,tools);cls.c=protocol()
    def spec(self,condition='FREE_ORDER',profile='BASE64',seed=42):
        return {'family':1200000,'profile':profile,'condition':condition,'repeat':0,'seed':seed,'episode_id':f'test_{condition}_{profile}_{seed}'}
    def make(self,method='GRAPH_FULL_PBRS',condition='FREE_ORDER',profile='BASE64'):
        c=TaskCore(self.env,self.Cap,self.c,method);c.reset(self.spec(condition,profile));return c
    def test_sources_are_real(self):self.assertEqual(len(self.sources),4)
    def test_obs31(self):self.assertEqual(self.make().env.observe().shape,(31,))
    def test_actions13(self):self.assertEqual(len(self.env.ACTIONS),13)
    def test_goal_snapshot_no_fake_terminal(self):self.assertFalse(self.make().env.snapshot()['terminal_failure'])
    def test_expert_has_actual_effects(self):
        c=self.make()
        while not c.done:c.step(self.env.expert_action(c.env))
        self.assertTrue(c.env.success);self.assertTrue(c.summary()['audit']['passed'])
    def test_wait_hits_finite_deadline(self):
        c=self.make()
        while not c.done:_,_,term,trunc,info,_=c.step(0)
        self.assertTrue(term);self.assertFalse(trunc);self.assertTrue(info['legacy_truncated'])
        self.assertEqual(info['termination_reason'],'task_deadline');self.assertEqual(info['task_reward'],0.)
        self.assertFalse(c.last_state['terminal_failure'])
    def test_long_chain_modifier(self):
        a=self.make();b=self.make(profile='LONG_CHAIN128')
        self.assertAlmostEqual(b.env.step_size,a.env.step_size*.5);self.assertEqual(b.env.spec.horizon,128)
    def test_same_actions_same_dynamics(self):
        a=self.make('TASK_ONLY');b=self.make('GRAPH_FULL_PBRS')
        for _ in range(20):
            action=self.env.expert_action(a.env);x=a.step(action);y=b.step(action)
            self.assertEqual(a.last_state,b.last_state);np.testing.assert_equal(x[0],y[0])
            if a.done:break
    def test_no_extra_policy_fields(self):
        a=self.make('GEOM_COUNT_EVENTS_PBRS');b=self.make('GRAPH_FULL_PBRS')
        np.testing.assert_equal(a.env.observe(),b.env.observe())
    def test_same_initial_weights_not_trajectories_contract(self):self.assertFalse(self.c['training']['bc_initialization'])
    def test_invalid_action_held_not_autocorrected(self):
        c=self.make();c.step(self.env.ACTIONS.index('ADVANCE_A'))
        self.assertEqual(c.env.invalid_actions,1);self.assertEqual(c.env.objects['A']['phase'],'WAIT')
    def test_bad_action(self):
        for a in (True,1.5,99):
            with self.assertRaises(ValueError):self.make().step(a)
    def test_both_orders(self):
        for cond in ('A_STARTED','B_STARTED'):
            c=self.make(condition=cond)
            while not c.done:c.step(self.env.expert_action(c.env))
            self.assertTrue(c.env.success)
    def test_two_loss_pairing(self):
        c=self.make('COUNT_EVENTS_PBRS',condition='TWO_LOSSES')
        while not c.done:c.step(self.env.expert_action(c.env))
        self.assertEqual(c.env.loss_total,2);self.assertEqual(c.env.restore_total,2);self.assertFalse(c.bank.open)
    def test_release_does_not_create_loss(self):
        c=self.make(condition='A_STARTED');c.step(self.env.ACTIONS.index('RELEASE_A'))
        self.assertEqual(c.env.loss_total,0);self.assertFalse(c.bank.open)
    def test_v6_raw_label_only_zero_but_discount_not_zero(self):
        c=self.make(condition='A_VALID_B_LOSS')
        while not c.env.open_loss and not c.done:c.step(self.env.expert_action(c.env))
        oid=next(iter(c.env.open_loss));before=c.values['GRAPH_FULL_PBRS']
        *_,info,trace=c.step(self.env.ACTIONS.index('START_RECOVERY_'+oid))
        self.assertAlmostEqual(c.values['GRAPH_FULL_PBRS'],before)
        self.assertAlmostEqual(info['reward_components']['shaping_reward'],(.99-1)*before)
    def test_frozen_source_goal_guard(self):
        c=self.make();s=copy.deepcopy(c.last_state);s['state_index']+=1;s['capture_order']+=1;s['available_at_ns']+=1
        s['objects']['A']['target_xy'][0]+=.1
        with self.assertRaises(ValueError):c.bank.observe(s)
    def test_bank_cross_episode(self):
        c=self.make();s=copy.deepcopy(c.last_state);s['episode_id']='other'
        with self.assertRaises(ValueError):c.bank.observe(s)
    def test_prefix_and_future_independence(self):
        a=self.make();states=[copy.deepcopy(a.last_state)]
        for _ in range(10):a.step(self.env.expert_action(a.env));states.append(copy.deepcopy(a.last_state))
        def run(ss):
            b=PotentialBank(ss[0]['episode_id'],self.Cap);return [b.observe(x)[0] for x in ss]
        full=run(states);short=run(states[:5]);self.assertEqual(full[:5],short)
        modified=copy.deepcopy(states);modified[-1]['objects']['A']['pos'][0]+=.01
        self.assertEqual(run(modified)[:5],short)
    def test_all_conditions_methods_identity(self):
        for cond in self.c['conditions']:
            for m in METHODS:
                c=self.make(m,condition=cond)
                while not c.done:c.step(self.env.expert_action(c.env))
                self.assertTrue(c.summary()['audit']['passed'])

if __name__=='__main__':unittest.main()
