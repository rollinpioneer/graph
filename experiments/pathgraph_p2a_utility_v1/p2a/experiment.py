from __future__ import annotations
import csv,json
from pathlib import Path
from dataclasses import asdict
import numpy as np
from .env import CONDITIONS,Spec,SkillEnv,collect_episode,ACTIONS
from .io import new_dir,dump,sha,read_jsonl

SPLITS={'train':list(range(2011000,2011024)),
        'validation':list(range(2012000,2012008)),
        'test':list(range(2013000,2013016))}

def make_specs(split,families,reps):
    return [Spec(f,c,f*10000+j*100+r,f'P2A_{split}_{f}_{j}_{r}')
            for f in families for j,c in enumerate(CONDITIONS) for r in range(reps)]

def prepare(out,smoke=False):
    out=new_dir(out)
    split=({'train':[2099100,2099101],'validation':[2099200],'test':[2099300]} if smoke else SPLITS)
    reps={'train':(2 if smoke else 10),'validation':(1 if smoke else 4),'test':(1 if smoke else 8)}
    qualities=['expert']*5+['pause']*2+['noisy']*2+['recovery_blind']
    summaries=[]
    with (out/'train_episodes.jsonl').open('w',encoding='utf-8') as f:
        for i,s in enumerate(make_specs('train',split['train'],reps['train'])):
            q=qualities[i%len(qualities)]
            ep=collect_episode(s,q); f.write(json.dumps(ep,allow_nan=False)+'\n')
            summaries.append(dict(episode_id=s.episode_id,family=s.family,condition=s.condition,
                quality=q,steps=ep['info']['task_steps'],success=ep['info']['success'],
                loss=ep['info']['observed_losses'],restored=ep['info']['restored_losses']))
    for part in ('validation','test'):
        specs=[asdict(s) for s in make_specs(part,split[part],reps[part])]
        dump(out/(part+'_specs.json'),specs)
    dump(out/'collection_summary.json',summaries)
    dump(out/'manifest.json',dict(evidence_tier='ACTION_CONDITIONED_ABSTRACT_SKILL_MDP_NOT_PHYSICS',
        smoke_only=smoke,split_families=split,replicates=reps,train_episodes=len(summaries),
        train_sha256=sha(out/'train_episodes.jsonl'),validation_specs_sha256=sha(out/'validation_specs.json'),
        test_specs_sha256=sha(out/'test_specs.json'),generator_reads_reward=False,
        action_count=len(ACTIONS),p1_holdout_reused=False))
    return out

def evaluate(checkpoint,data_root,split,out):
    import torch
    from .train import Policy
    if split not in ('validation','test'): raise ValueError('evaluation split')
    data_root=Path(data_root); m=json.loads((data_root/'manifest.json').read_text())
    sp=data_root/(split+'_specs.json')
    if sha(sp)!=m[split+'_specs_sha256']: raise ValueError('evaluation specs changed')
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if ck['actions']!=list(ACTIONS): raise ValueError('action space mismatch')
    model=Policy(ck['obs_dim']); model.load_state_dict(ck['model']); model.eval()
    mean=ck['mean'].numpy(); std=ck['std'].numpy(); torch.set_num_threads(1)
    rows=[]; loss_rows=[]
    for d in json.loads(sp.read_text()):
        env=SkillEnv(Spec(**d)); obs=env.observe(); decisions=[]; input_rows=[]; losses={}
        while not env.done:
            with torch.no_grad():
                a=int(model(torch.as_tensor(((obs-mean)/std)[None],dtype=torch.float32)).argmax(-1).item())
            input_rows.append(obs.tolist()); decisions.append(a)
            obs,_,info=env.step(a)
            for event in env.events:
                key=(event['object_id'],event['loss_id'])
                if event['kind']=='LOSS':
                    losses[key]=dict(object_id=key[0],loss_id=key[1],loss_step=env.t,
                        restore_step=None,valid_count_at_loss=sum(int(o['valid']) for o in env.objects.values()))
                elif event['kind']=='HOLD_REESTABLISHED':
                    if key not in losses: raise ValueError('unmatched restore in evaluation')
                    losses[key]['restore_step']=env.t
        for rec in losses.values():
            loss_rows.append(dict(method=ck['method'],policy_seed=ck['seed'],family=d['family'],
                condition=d['condition'],episode_id=d['episode_id'],**rec,
                restored=rec['restore_step'] is not None,
                restore_delay_steps=None if rec['restore_step'] is None else rec['restore_step']-rec['loss_step'],
                episode_success=info['success']))
        rows.append(dict(method=ck['method'],policy_seed=ck['seed'],family=d['family'],
            condition=d['condition'],episode_id=d['episode_id'],eval_seed=d['seed'],
            success=int(info['success']),truncated=int(info['truncated']),steps=info['task_steps'],
            training_steps=ck['steps'],smoke_only=bool(m['smoke_only']),
            observed_losses=info['observed_losses'],restored_losses=info['restored_losses'],
            invalid_actions=env.invalid_actions,first_valid_order='>'.join(info['order']),
            wait_fraction=decisions.count(0)/len(decisions),
            checkpoint_sha256=sha(checkpoint),action_sequence=decisions,
            # Sensor/current-state inputs support later attribution; no reward needed in eval.
            observation_sequence=input_rows))
    out=new_dir(out)
    scalar_keys=[k for k in rows[0] if k not in ('action_sequence','observation_sequence')]
    with (out/'episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=scalar_keys); w.writeheader()
        w.writerows({k:r[k] for k in scalar_keys} for r in rows)
    loss_fields=['method','policy_seed','family','condition','episode_id','object_id','loss_id','loss_step',
        'restore_step','valid_count_at_loss','restored','restore_delay_steps','episode_success']
    with (out/'per_loss_recovery.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=loss_fields);w.writeheader();w.writerows(loss_rows)
    with (out/'rollout_actions.jsonl').open('w') as f:
        for r in rows: f.write(json.dumps(r,allow_nan=False)+'\n')
    dump(out/'summary.json',dict(split=split,method=ck['method'],policy_seed=ck['seed'],
        episodes=len(rows),success_rate=float(np.mean([r['success'] for r in rows])),
        learned_policy_executed=True,teacher_override_calls=0,physical_simulation=False,
        manifest_sha256=sha(data_root/'manifest.json'),confirmation_passed=False))
    return out
