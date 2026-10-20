import os,sys,json,time
from pathlib import Path
R=Path(os.environ['R']);sys.path.insert(0,str(R/'code'))
from common import *
from collections import Counter

D=R/'diagnosis_s_vs_g';L=G/'eg_lg1_local_geometry_v1_20261020';check=json.load(open(D/'results/checkpoint_diagnostics.json'));sampling=json.load(open(D/'results/sampling_information.json'));c3=json.load(open(R/'results/C3_summary.json'));cfg=json.load(open(R/'frozen_config.json'))
trajectories=[];curves=[];identity=[]
for family in ['G','C','S']:
 for seed in SEEDS:
  root=L if family=='G' else R;old=json.load(open(root/f'results/train_{family}_{seed}.json'));loss=np.array(old['losses']);curves.append(dict(family=family,seed=seed,blocks100=[float(loss[i:i+100].mean()) for i in range(0,1000,100)],mean501_750=float(loss[500:750].mean()),mean751_1000=float(loss[750:].mean()),late_slope=float(np.polyfit(np.arange(500),loss[500:],1)[0])))
  if family!='G':
   v=next(x for x in check['tasks'] if x['family']==family and x['seed']==seed and x['step']==1000)
   for role in ['model_val','gate_cal']:assert abs(v['roles'][role]['MSE']-old['diagnostics'][role]['MSE'])<1e-6
  for step in ([1000] if family=='G' else [250,500,750,1000]):
   v=next(x for x in check['tasks'] if x['family']==family and x['seed']==seed and x['step']==step);assert sha(v['checkpoint'])==v['checkpoint_sha256'];identity.append(dict(family=family,seed=seed,step=step,path=v['checkpoint'],sha256=v['checkpoint_sha256']))
 for step in ([1000] if family=='G' else [250,500,750,1000]):
  rows=[x for x in check['tasks'] if x['family']==family and x['step']==step]
  for role in ['fit','model_val','gate_cal']:
   trajectories.append(dict(family=family,step=step,role=role,MSE=float(np.mean([v['roles'][role]['MSE'] for v in rows])),original_MSE=float(np.mean([v['roles'][role]['original_pose_MSE'] for v in rows])),Top64_AUROC=float(np.mean([v['roles'][role]['Top64_original']['success_AUROC'] for v in rows])),Top64_safe_AUROC=float(np.mean([v['roles'][role]['Top64_original']['safe_strong_weak_AUROC'] for v in rows])),Top64_within_frame_inversion=float(np.mean([v['roles'][role]['Top64_original']['within_frame_success_pair_inversion'] for v in rows]))))
transitions=Counter();bad=Counter();by_scene=[]
for scene in sum(cfg['evaluation']['C3'].values(),[]):
 tr=Counter();categories=Counter()
 for frame in range(256):
  rows={x['configuration']:x for x in json.load(open(R/f'evaluation/development/{scene:04d}/{frame:04d}.json'))}
  for seed in SEEDS:
   a=rows[f'G_{seed}_CD'];b=rows[f'S_{seed}_CD'];key=('success' if a['top1_valid'] else 'failure')+'->'+('success' if b['top1_valid'] else 'failure');tr[key]+=1
   if a['top1_valid'] and not b['top1_valid']:categories[b['top1_category']]+=1
 transitions.update(tr);bad.update(categories);by_scene.append(dict(scene=scene,pairs=768,transitions=dict(tr),regression_categories=dict(categories)))
write(D/'results/diagnosis_summary.json',dict(status='PASS',trajectories=trajectories,training_curves=curves,sampling=sampling['summary'],C3_paired_transitions=dict(transitions),C3_regression_categories=dict(bad),C3_per_scene=by_scene,checkpoint_identity=identity,fixed1000_CPU_matches_archived_GPU_MSE=True,training_updates=0,new_GT=0,EG_forward=0,C4_read=False,GPU_used=False))
csv_write(D/'results/checkpoint_trajectory.csv',trajectories);csv_write(D/'results/training_curve_summary.csv',curves)
print('DIAG_SUMMARY_COMPLETE',dict(transitions),dict(bad),flush=True)
