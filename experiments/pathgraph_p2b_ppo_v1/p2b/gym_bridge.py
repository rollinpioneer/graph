from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .task_core import TaskCore
from .specs import TrainStream
from .io_utils import canonical,seed_for

class SkillGymEnv(gym.Env):
    """P2B finite-deadline wrapper. Never adds an expert or action mask."""
    metadata={'render_modes':[]}
    def __init__(self,sources,config,method,*,policy_seed=211,rank=0,fixed_spec=None,log_root=None):
        super().__init__()
        envmod,cap,_=sources
        self.core=TaskCore(envmod,cap,config,method);self.c=config
        self.fixed_spec=fixed_spec;self.stream=TrainStream(config,policy_seed,rank)
        self.action_space=spaces.Discrete(len(envmod.ACTIONS))
        self.observation_space=spaces.Box(low=-np.inf,high=np.inf,shape=(31,),dtype=np.float32)
        self.handles={};self.rank=rank;self.audit_selected=False
        if log_root is not None:
            p=Path(log_root);p.mkdir(parents=True,exist_ok=True)
            for n in ('episodes','audit_traces'):
                self.handles[n]=(p/f'{n}_worker_{rank}.jsonl').open('x',encoding='utf-8')
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        spec=(options or {}).get('spec',self.fixed_spec)
        if spec is None:spec=self.stream.next()
        self.audit_selected=seed_for('audit',spec['episode_id'])%self.c['audit']['train_trace_sample_modulus']==0
        return self.core.reset(spec)
    def step(self,action):
        obs,reward,terminated,truncated,info,trace=self.core.step(action)
        if 'audit_traces' in self.handles and self.audit_selected:
            self.handles['audit_traces'].write(canonical(trace)+'\n')
        if terminated or truncated:
            if 'episodes' in self.handles:
                self.handles['episodes'].write(canonical(info['episode_summary'])+'\n')
        return obs,float(reward),terminated,truncated,info
    def close(self):
        for f in self.handles.values():f.close()
        self.handles={}
