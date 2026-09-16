"""No Gym/SB3 dependency here: auditable adapter to frozen action dynamics."""
from __future__ import annotations
import copy,math
import numpy as np
from .potentials import PotentialBank,METHODS
from .shaping import shaped_transition,DiscountAudit

class TaskCore:
    def __init__(self,env_module,cap_class,config,method):
        if method not in METHODS:raise ValueError(method)
        self.envmod=env_module;self.cap_class=cap_class;self.c=config;self.method=method
        self.env=None;self.last_state=None;self.values=None;self.done=True
    def reset(self,spec):
        self.spec=dict(spec);p=self.c['profiles'][spec['profile']]
        e=self.envmod.SkillEnv(self.envmod.Spec(family=spec['family'],condition=spec['condition'],
            seed=spec['seed'],episode_id=spec['episode_id'],horizon=p['horizon']))
        # Explicit P2B profile modifier, same for every reward arm. Never edits the frozen class.
        e.step_size*=p['step_size_multiplier'];self.env=e
        self.bank=PotentialBank(spec['episode_id'],self.cap_class)
        self.last_state=e.snapshot();self.values,self.vmeta=self.bank.observe(self.last_state)
        self.audit=DiscountAudit(self.c['gamma'],self.c['beta']);self.done=False
        self.discounted_task=0.;self.training_return=0.;self.task_return=0.;self.hist=np.zeros(e.action_dim,dtype=int)
        return e.observe().copy(),{'spec':dict(spec)}
    def step(self,action):
        if self.done:raise RuntimeError('step after task end')
        a=np.asarray(action)
        if a.shape!=() or a.dtype.kind=='b':raise ValueError('scalar Discrete action required')
        ai=int(a)
        if float(a)!=ai:raise ValueError('integer action required')
        before=copy.deepcopy(self.last_state)
        obs,legacy_done,info=self.env.step(ai)
        nxt=self.env.snapshot();values,vmeta=self.bank.observe(nxt)
        # Task has a declared deadline and remaining time in observation.
        # Preserve old raw truncated flag; expose NEW finite-horizon API termination.
        terminated=bool(legacy_done);truncated=False
        task_reward=1. if info['success'] else 0.
        row=shaped_transition(task_reward,self.values[self.method],values[self.method],
            gamma=self.c['gamma'],beta=self.c['beta'],terminated=terminated,truncated=truncated)
        self.audit.add(row);self.hist[ai]+=1
        self.discounted_task+=self.c['gamma']**(self.env.t-1)*task_reward
        self.task_return+=task_reward;self.training_return+=row['training_reward']
        self.last_state=nxt;self.values=values;self.vmeta=vmeta;self.done=terminated
        reason='success' if info['success'] else ('task_deadline' if terminated else 'ongoing')
        merged=dict(info,termination_reason=reason,legacy_truncated=bool(info['truncated']),
                    task_reward=task_reward,discounted_task_return=self.discounted_task,
                    reward_components=row,method=self.method,profile=self.spec['profile'])
        if terminated:merged['episode_summary']=self.summary()
        trace={'before':before,'after':copy.deepcopy(nxt),'action_applied':ai,
               'components':row,'v6':vmeta['v6'],'all_potentials_after':values}
        return obs.copy(),row['training_reward'],terminated,truncated,merged,trace
    def summary(self):
        if not self.done:raise RuntimeError('summary before episode boundary')
        ar=self.audit.result()
        if not ar['passed']:raise RuntimeError('discounted shaping ledger failed')
        return dict(self.spec,method=self.method,success=bool(self.env.success),
            steps=int(self.env.t),task_deadline=not bool(self.env.success),
            invalid_actions=int(self.env.invalid_actions),waits=int(self.hist[0]),
            observed_losses=int(self.env.loss_total),restored_losses=int(self.env.restore_total),
            task_return=float(self.task_return),discounted_task_return=float(self.discounted_task),
            training_return=float(self.training_return),action_histogram=self.hist.tolist(),
            audit=ar,teacher_takeover=0,policy_controlled=True)
