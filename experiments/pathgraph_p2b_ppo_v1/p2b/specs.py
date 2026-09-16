from __future__ import annotations
from .io_utils import seed_for

def eval_specs(config,split):
    if split not in ('validation','test','smoke'):raise ValueError(split)
    repeats=config['evaluation_repeats'] if split!='smoke' else 1
    rows=[]
    for family in config['families'][split]:
        for profile in config['profiles']:
            for condition in config['conditions']:
                for repeat in range(repeats):
                    rows.append({'family':family,'condition':condition,'profile':profile,'repeat':repeat,
                                 'seed':seed_for('P2B_V1',split,family,profile,condition,repeat),
                                 'episode_id':f'P2B_{split}_{family}_{profile}_{condition}_{repeat}'})
    return rows

class TrainStream:
    def __init__(self,config,policy_seed,rank):
        self.c=config;self.policy_seed=policy_seed;self.rank=rank;self.i=0
    def next(self):
        i=self.i;self.i+=1
        # Family / profile / condition selected independently of method and returns.
        fams=self.c['families']['train'];profiles=list(self.c['profiles']);conds=self.c['conditions']
        family=fams[seed_for('family',self.policy_seed,self.rank,i)%len(fams)]
        profile=profiles[(i+self.rank)%len(profiles)]
        condition=conds[(i//len(profiles)+self.rank)%len(conds)]
        return {'family':family,'profile':profile,'condition':condition,'repeat':i,
                'seed':seed_for('P2B_train',self.policy_seed,self.rank,i),
                'episode_id':f'P2B_train_{self.policy_seed}_{self.rank}_{i}'}
