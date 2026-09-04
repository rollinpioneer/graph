#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def rows(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args(); data=a.data.resolve(); allrows=rows(data/'policy_episode_manifest.jsonl'); train=rows(data/'train_manifest.jsonl'); val=rows(data/'val_manifest.jsonl')
 by=lambda rr,t:[x for x in rr if x['task_id']==t]
 checks={};
 for t in ('transport_recovery','transport_dual_order'):
  checks[f'{t}_train_min60']=len(by(train,t))>=60; checks[f'{t}_val_min20']=len(by(val,t))>=20
 for o in ('A_first','B_first'): checks[f'dual_{o}_train_min20']=sum(x['task_id']=='transport_dual_order' and x['order']==o for x in train)>=20
 checks['recovery_success_train_min20']=sum(x['task_id']=='transport_recovery' and x['success'] for x in train)>=20
 checks['all_env_step_source']=all(x['source_format']=='env_step_action_rollout_npz' and x['action_logged_equals_passed'] for x in allrows)
 group_overlap=bool(set(x['content_group_id'] for x in train)&set(x['content_group_id'] for x in val));checks['content_group_disjoint']=not group_overlap
 nonzero=[]; std=[]
 for x in allrows:
  d=np.load(x['episode_path']);act=d['action_applied'];passed=d['action_passed']; nonzero.extend(np.any(np.abs(act)>1e-6,axis=1));std.append(float(act.std()));checks.setdefault('action_arrays_equal',True);checks['action_arrays_equal'] &= bool(np.array_equal(act,passed))
 checks['action_nonzero_ratio_ge_0_5']=float(np.mean(nonzero))>=.5;checks['action_std_ge_1e_4']=min(std)>=1e-4
 decision='POLICY_DATA_REAL_ACTION_READY' if all(checks.values()) else 'POLICY_DATA_BLOCKED'
 result={'decision':decision,'checks':checks,'counts':{'all':len(allrows),'train':len(train),'val':len(val),'action_nonzero_ratio':float(np.mean(nonzero)),'min_episode_action_std':min(std)}}
 a.out.mkdir(parents=True,exist_ok=True);(a.out/'policy_data_gate.json').write_text(json.dumps(result,indent=2)+'\n');(a.out/'policy_data_decision.md').write_text('# Stage 6.1 policy-data gate\n\nDecision: `'+decision+'`\n\nAll BC actions are `action_applied` values recorded from the actual `env.step(action)` call; Stage 2 synthetic zero-action traces were not read.\n')
 print(json.dumps(result,indent=2));return 0 if decision.endswith('READY') else 2
if __name__=='__main__':raise SystemExit(main())
