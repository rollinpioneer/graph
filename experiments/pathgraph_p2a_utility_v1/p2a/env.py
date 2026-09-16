"""Action-driven abstract skill MDP; no reward-model or graph imports.
All skill effects below are declared benchmark assumptions, not physics results.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import numpy as np

PHASES = ('WAIT','HELD','TRANSPORT','PLACED','VALID','LOST','RECOVERING')
OPERATIONS = ('ACQUIRE','ADVANCE','PLACE','START_RECOVERY','REGRASP','RELEASE')
ACTIONS = ('WAIT',) + tuple(f'{op}_{oid}' for oid in ('A','B') for op in OPERATIONS)
CONDITIONS = ('FREE_ORDER','A_STARTED','B_STARTED','A_VALID_B_LOSS',
              'B_VALID_A_LOSS','INVALIDATE_FIRST','LATE_LOSS','TWO_LOSSES')
EVIDENCE = 'ACTION_CONDITIONED_ABSTRACT_SKILL_MDP_NOT_PHYSICS'

@dataclass(frozen=True)
class Spec:
    family: int
    condition: str
    seed: int
    episode_id: str
    horizon: int = 64

class SkillEnv:
    action_dim = len(ACTIONS)
    def __init__(self, spec: Spec):
        if spec.condition not in CONDITIONS or spec.horizon < 1:
            raise ValueError('unsupported spec')
        self.spec = spec
        self.reset()

    def reset(self):
        f = np.random.default_rng(self.spec.family)
        self.step_size = float(f.uniform(.045,.075))
        self.skill_prob = float(f.uniform(.94,.99))
        self.rng = np.random.default_rng(self.spec.seed)
        # Disturbance random draws are indexed by time, not conditional RNG calls.
        self.noise = self.rng.random((self.spec.horizon + 1, 4))
        self.starts = {}; self.targets = {}; self.objects = {}
        for j, oid in enumerate(('A','B')):
            start = np.array([f.uniform(-.18,.06), (-1 if j == 0 else 1)*f.uniform(.10,.24), .03])
            goal = np.array([f.uniform(.30,.54), (-1 if j == 0 else 1)*f.uniform(.10,.24)])
            self.starts[oid] = start
            self.targets[oid] = goal
            self.objects[oid] = dict(pos=start.tolist(),vel=[0.,0.,0.],quat=[1.,0.,0.,0.],
                angular_vel=[0.,0.,0.],target_xy=goal.tolist(),held=False,valid=False,phase='WAIT')
        self.t=0; self.done=False; self.success=False; self.truncated=False
        self.events=[]; self.loss_counter=0; self.open_loss={}; self.loss_steps={}
        self.loss_total=0; self.restore_total=0; self.invalidate_done=False
        self.first_valid_order=[]; self.invalid_actions=0
        c=self.spec.condition
        if c == 'A_STARTED': self.objects['A'].update(held=True,phase='HELD')
        if c == 'B_STARTED': self.objects['B'].update(held=True,phase='HELD')
        if c in ('A_VALID_B_LOSS','B_VALID_A_LOSS'):
            first,other = ('A','B') if c.startswith('A_') else ('B','A')
            self.objects[first]['pos'][:2] = self.targets[first].tolist()
            self.objects[first].update(valid=True,held=False,phase='VALID')
            self.objects[other].update(held=True,phase='HELD')
            self.first_valid_order=[first]
        self.last_action=0
        return self.observe()

    def snapshot(self):
        held=[o for o in self.objects.values() if o['held']]
        eef=(np.asarray(held[0]['pos'])+[0,0,.13]).tolist() if held else [0.,0.,.30]
        return dict(episode_id=self.spec.episode_id,state_index=self.t,task='dual_order',
            available_at_ns=self.t*100_000_000,physical_time_ns=self.t*100_000_000,
            capture_order=self.t,success=self.success,terminal_failure=False,
            gripper_closed=bool(held),eef=eef,objects=copy.deepcopy(self.objects),
            events=copy.deepcopy(self.events),evidence_tier=EVIDENCE)

    def observe(self):
        out=[]
        # Same current facts for all learners. No case, graph cost, reward or expert ID.
        for oid in ('A','B'):
            o=self.objects[oid]
            out += o['pos'] + o['target_xy']
            out += [float(o['held']),float(o['valid']),float(oid in self.open_loss)]
            out += [float(o['phase']==p) for p in PHASES]
        out += [max(0.,1-self.t/self.spec.horizon)]
        return np.asarray(out,dtype=np.float32)

    def _event(self,kind,oid,lid):
        self.events.append(dict(kind=kind,object_id=oid,loss_id=lid,known_at_ns=self.t*100_000_000))

    def _disturb(self,oid):
        o=self.objects[oid]; self.loss_counter+=1
        lid=f'L{self.loss_counter}'; self.open_loss[oid]=lid; self.loss_steps[lid]=self.t
        p=np.asarray(o['pos']); p[:2]=.65*p[:2]+.35*self.starts[oid][:2]
        o.update(pos=p.tolist(),held=False,valid=False,phase='LOST')
        self.loss_total+=1; self._event('LOSS',oid,lid)

    def step(self, action):
        if self.done: raise RuntimeError('step after done')
        if isinstance(action, bool) or int(action)!=action or not 0<=int(action)<len(ACTIONS):
            raise ValueError('invalid action index')
        action=int(action); name=ACTIONS[action]
        self.events=[]; invalid=False
        if action:
            j=(action-1)//len(OPERATIONS); oid=('A','B')[j]; op=OPERATIONS[(action-1)%len(OPERATIONS)]
            o=self.objects[oid]; held_any=any(x['held'] for x in self.objects.values())
            if op=='ACQUIRE' and not held_any and oid not in self.open_loss:
                if self.noise[self.t,0] < self.skill_prob:
                    o.update(held=True,valid=False,phase='HELD')
            elif op=='ADVANCE' and o['held']:
                p=np.asarray(o['pos']); d=self.targets[oid]-p[:2]; n=float(np.linalg.norm(d))
                if n>0: p[:2]+=d*min(1.,self.step_size/n)
                o.update(pos=p.tolist(),phase='TRANSPORT')
            elif op=='PLACE' and o['held'] and np.linalg.norm(np.asarray(o['pos'])[:2]-self.targets[oid])<=.02:
                o.update(held=False,phase='PLACED')
            elif op=='START_RECOVERY' and oid in self.open_loss and not held_any:
                o['phase']='RECOVERING'
            elif op=='REGRASP' and o['phase']=='RECOVERING' and oid in self.open_loss and not held_any:
                if self.noise[self.t,1] < self.skill_prob:
                    o.update(held=True,phase='HELD')
                    # Event is stamped at the resulting-state time below.
                    lid=self.open_loss.pop(oid); self.restore_total+=1
                    self.events.append(dict(kind='HOLD_REESTABLISHED',object_id=oid,loss_id=lid))
            elif op=='RELEASE' and o['held']:
                # Explicit release is NOT an accidental loss event.
                o.update(held=False,phase='WAIT')
            else: invalid=True
        self.t+=1; self.last_action=action; self.invalid_actions+=int(invalid)
        # Current goal validity: geometry and holding, not scenario/outcome labels.
        for oid,o in self.objects.items():
            old=o['valid']
            o['valid']=bool(not o['held'] and oid not in self.open_loss and
                np.linalg.norm(np.asarray(o['pos'])[:2]-self.targets[oid])<=.02)
            if o['valid']:
                o['phase']='VALID'
                if not old: self.first_valid_order.append(oid)
        c=self.spec.condition
        active=[oid for oid,o in self.objects.items() if o['held'] and o['phase']=='TRANSPORT']
        if active:
            oid=active[0]; o=self.objects[oid]
            initial=np.linalg.norm(self.targets[oid]-self.starts[oid][:2])
            progress=1-np.linalg.norm(self.targets[oid]-np.asarray(o['pos'])[:2])/initial
            maxloss=2 if c=='TWO_LOSSES' else 1
            eligible=(c in ('A_VALID_B_LOSS','B_VALID_A_LOSS','LATE_LOSS','TWO_LOSSES'))
            threshold=.75 if c=='LATE_LOSS' else .40
            if eligible and self.loss_total<maxloss and progress>=threshold:
                self._disturb(oid)
            if c=='INVALIDATE_FIRST' and not self.invalidate_done and progress>=.40:
                others=[k for k,v in self.objects.items() if k!=oid and v['valid']]
                if others:
                    k=others[0]; self.objects[k].update(pos=self.starts[k].tolist(),valid=False,phase='WAIT')
                    self.invalidate_done=True
        for e in self.events: e['known_at_ns']=self.t*100_000_000
        self.success=all(o['valid'] for o in self.objects.values())
        self.truncated=bool(self.t>=self.spec.horizon and not self.success)
        self.done=bool(self.success or self.truncated)
        info=dict(action_applied=action,success=self.success,truncated=self.truncated,
            observed_losses=self.loss_total,restored_losses=self.restore_total,
            invalid_action=invalid,task_steps=self.t,order=list(self.first_valid_order))
        return self.observe(),self.done,info

def expert_action(env: SkillEnv):
    """Uses current object facts only; no reward or future disturbance schedule."""
    objs=env.objects
    for oid,o in objs.items():
        if o['held']:
            d=np.linalg.norm(np.asarray(o['pos'])[:2]-np.asarray(o['target_xy']))
            return ACTIONS.index(('PLACE_' if d<=.02 else 'ADVANCE_')+oid)
    for oid,o in objs.items():
        if oid in env.open_loss:
            return ACTIONS.index(('REGRASP_' if o['phase']=='RECOVERING' else 'START_RECOVERY_')+oid)
    todo=[oid for oid,o in objs.items() if not o['valid']]
    if not todo: return 0
    oid=min(todo,key=lambda k:np.linalg.norm(np.asarray(objs[k]['pos'])[:2]-objs[k]['target_xy']))
    return ACTIONS.index('ACQUIRE_'+oid)

def collect_episode(spec: Spec, quality: str):
    if quality not in ('expert','pause','noisy','recovery_blind'): raise ValueError(quality)
    env=SkillEnv(spec); rng=np.random.default_rng(spec.seed+717)
    states=[env.snapshot()]; observations=[]; actions=[]; info={}
    while not env.done:
        a=expert_action(env)
        if quality=='pause' and rng.random()<.35: a=0
        if quality=='noisy' and rng.random()<.35: a=int(rng.integers(len(ACTIONS)))
        if quality=='recovery_blind' and env.open_loss and rng.random()<.75:
            a=ACTIONS.index('ACQUIRE_'+sorted(env.open_loss)[0])
        observations.append(env.observe().tolist()); actions.append(a)
        _,_,info=env.step(a); states.append(env.snapshot())
    return dict(spec=vars(spec),quality=quality,states=states,obs=observations,actions=actions,info=info)
