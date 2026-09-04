#!/usr/bin/env python3
"""Collect real-action graph demonstrations and a content-group-disjoint split."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import yaml
from tools.stage6.policy_env import EpisodeSpec, rollout_demo

def stable_split(group: str, split_seed: int) -> str:
    value=int(hashlib.sha256(f'{split_seed}:{group}'.encode()).hexdigest()[:8],16)%10
    return 'val' if value < 3 else 'train'

def write_jsonl(path: Path, rows):
    with path.open('w',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,sort_keys=True)+'\n')

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    cfg=yaml.safe_load(a.config.read_text()); c=cfg['collection']; out=a.out.resolve(); eps=out/'episodes'; eps.mkdir(parents=True,exist_ok=True)
    jobs=[]
    jobs += [('transport_dual_order','A_first','A_first',c['dual_order_a_first'])]
    jobs += [('transport_dual_order','B_first','B_first',c['dual_order_b_first'])]
    jobs += [('transport_recovery','natural','',c['recovery_natural'])]
    jobs += [('transport_recovery','drop_regrasp','',c['recovery_drop_regrasp'])]
    jobs += [('transport_recovery','gripper_reopen','',c['recovery_gripper_reopen'])]
    jobs += [('transport_recovery','controlled_failure','',c['recovery_controlled_failure'])]
    rows=[]; all_nonzero=[]; action_std=[]
    idx=0
    for task,scenario,order,count in jobs:
        for n in range(int(count)):
            eid=f'{task}__{scenario}__{n:03d}'; group=f'{task}__{scenario}_group_{n:03d}'
            demo=rollout_demo(EpisodeSpec(task,scenario,order,int(c['seed'])+idx)); idx+=1
            ep_path=eps/f'{eid}.npz'; np.savez_compressed(ep_path,observations=demo['observations'],action_applied=demo['action_applied'],action_passed=demo['action_passed'],env_rewards=demo['env_rewards'])
            equal=bool(np.array_equal(demo['action_applied'],demo['action_passed']))
            if not equal: raise RuntimeError(f'action logging mismatch in {eid}')
            acts=demo['action_applied']; all_nonzero.extend(np.any(np.abs(acts)>1e-6,axis=1).tolist()); action_std.append(float(acts.std()))
            rows.append({'episode_id':eid,'content_group_id':group,'task_id':task,'scenario':scenario,'order':order,'split':stable_split(group,int(c['split_seed'])),'source_format':'env_step_action_rollout_npz','controller_source':'scripted_env_controller','action_key':'action_applied','action_passed_key':'action_passed','action_logged_equals_passed':equal,'episode_path':str(ep_path),'num_steps':int(len(acts)),'success':bool(demo['success']),'failed':bool(demo['failed']),'recovery_count':int(demo['recovery_count']),'path_signature':demo['path_signature']})
    # guarantee each requested stratum has validation examples without group leakage.
    by_key={}
    for r in rows: by_key.setdefault((r['task_id'],r['scenario']),[]).append(r)
    for rs in by_key.values():
        if not any(r['split']=='val' for r in rs): rs[0]['split']='val'
        if not any(r['split']=='train' for r in rs): rs[-1]['split']='train'
    write_jsonl(out/'policy_episode_manifest.jsonl',rows)
    write_jsonl(out/'train_manifest.jsonl',[r for r in rows if r['split']=='train'])
    write_jsonl(out/'val_manifest.jsonl',[r for r in rows if r['split']=='val'])
    report={'episodes':len(rows),'train_episodes':sum(r['split']=='train' for r in rows),'val_episodes':sum(r['split']=='val' for r in rows),'action_nonzero_ratio':float(np.mean(all_nonzero)),'per_episode_action_std_min':float(min(action_std)),'logging_equality_rate':float(np.mean([r['action_logged_equals_passed'] for r in rows])),'task_counts':{t:sum(r['task_id']==t for r in rows) for t in sorted({r['task_id'] for r in rows})}}
    (out/'collection_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
