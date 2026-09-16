from __future__ import annotations
import random
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from .io_utils import canonical,dump
from .task_core import TaskCore

@contextmanager
def preserve_global_rng():
    import torch
    p=random.getstate();n=np.random.get_state();t=torch.get_rng_state()
    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:yield
    finally:
        random.setstate(p);np.random.set_state(n);torch.set_rng_state(t)
        if cuda is not None:torch.cuda.set_rng_state_all(cuda)

def evaluate_policy(model,sources,config,method,specs,out,*,policy_seed,checkpoint_step,save_traces=False):
    """All episodes stay in denominator. Success is never judged from shaped reward."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    envmod,cap,_=sources
    core=TaskCore(envmod,cap,config,method)
    records=[]
    with preserve_global_rng(), (out/'episodes.jsonl').open('x',encoding='utf-8') as f:
        tf=(out/'fixed_subset_traces.jsonl').open('x',encoding='utf-8') if save_traces else None
        try:
            for spec in specs:
                obs,_=core.reset(spec)
                while not core.done:
                    action,_=model.predict(obs,deterministic=True)
                    obs,_,_,_,_,trace=core.step(int(action))
                    if tf is not None and spec['repeat']==0:
                        tf.write(canonical(dict(trace,policy_seed=policy_seed,checkpoint_step=checkpoint_step))+'\n')
                row=dict(core.summary(),policy_seed=policy_seed,checkpoint_step=checkpoint_step)
                f.write(canonical(row)+'\n');records.append(row)
        finally:
            if tf is not None:tf.close()
    summary={'n_episodes':len(records),'success':sum(x['success'] for x in records)/len(records),
             'checkpoint_step':checkpoint_step,'teacher_takeover':0,
             'evidence_tier':config['evidence_tier'],'test_or_validation_used_for_gradient':False,
             'max_audit_error':max(abs(x['audit']['identity_error']) for x in records)}
    dump(out/'summary.json',summary)
    return summary
