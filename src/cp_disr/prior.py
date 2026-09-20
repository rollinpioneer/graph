"""The sole training prior sampler; no API, corruption, or network-visible mode."""
import random
from dataclasses import dataclass
from .common import digest,seed32,DataIntegrityError

ORIGINAL_PROBABILITY=0.8
@dataclass(frozen=True)
class EpisodePrior:
    env_id:str
    episode_id:str
    edges:tuple
    original_hash:str
    hash:str
    audit_mode:str

class PriorSampler:
    def __init__(self,training_seed):self.seed=training_seed;self.streams={};self.active={};self.draw_count=0
    def start(self,env_id,episode_id,original_edges):
        if env_id in self.active and self.active[env_id].episode_id==episode_id:raise DataIntegrityError('Episode already started; reuse saved prior')
        rng=self.streams.setdefault(env_id,random.Random(seed32('prior_dropout_train',env_id,0,self.seed)))
        original=tuple(sorted(tuple(e) for e in original_edges)); keep=rng.random()<ORIGINAL_PROBABILITY;self.draw_count+=1
        edges=original if keep else ()
        value=EpisodePrior(env_id,episode_id,edges,digest(original),digest(edges),'original' if keep else 'absent')
        self.active[env_id]=value;return value
    def get(self,env_id,episode_id):
        value=self.active[env_id]
        if value.episode_id!=episode_id:raise DataIntegrityError('Prior episode mismatch')
        return value
    def state_dict(self):return {'rng':{k:v.getstate() for k,v in self.streams.items()},'active':dict(self.active),'draw_count':self.draw_count}
    def load_state_dict(self,state):
        self.streams={}
        for k,v in state['rng'].items():r=random.Random();r.setstate(v);self.streams[k]=r
        self.active=dict(state['active']);self.draw_count=state['draw_count']
