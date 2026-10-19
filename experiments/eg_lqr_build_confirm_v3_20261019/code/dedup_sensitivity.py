"""C1 sensitivity: raw RGB/depth pixels plus intrinsic K; never prediction-selected."""
import os,json,csv
from pathlib import Path
import numpy as np

R=Path(os.environ['LQR_ROOT']);cfg=json.loads((R/'frozen_config.json').read_text());family=json.loads((R/'method_lock.json').read_text())['family'];seeds=[h['seed'] for h in cfg['heads']];splits=['seen','similar','novel']
with open(R/'results/C1_per_frame.csv',newline='') as f:records=list(csv.DictReader(f))
baseline={(int(x['scene']),int(x['frame'])):x for x in records if x['configuration']=='R0_CD'};assert len(baseline)==3072;assert all(x['raw_observation_sha256'] for x in baseline.values())
lookup={(int(x['scene']),int(x['frame']),x['configuration']):x for x in records};modes={}
for scope in ['global','within_scene']:
 seen=set();retained=[]
 for key in sorted(baseline):
  obs=baseline[key]['raw_observation_sha256'];identity=obs if scope=='global' else (key[0],obs)
  if identity not in seen:retained.append(key);seen.add(identity)
 counts={scene:sum(s==scene for s,f in retained) for scene in sum([cfg['C1'][x] for x in splits],[])};empty=[s for s,n in counts.items() if not n];result=dict(retained_frames=len(retained),removed_frames=3072-len(retained),per_scene_retained=counts,empty_scenes=empty,selection='first lexicographic scene/frame per raw observation identity; no performance selection')
 if empty:
  result['status']='NOT_COMPUTABLE_FOR_ALL12_SCENE_ESTIMAND';result['reason']='Global dedup removed complete scenes; do not silently change split/scene weights'
 else:
  rng=np.random.default_rng(20261019);ix={split:rng.integers(0,4,(10000,4)) for split in splits};summary={}
  for metric in ['ap','ap04','ap08','top1_valid','top1_collision']:
   values={}
   for split in splits:
    matrix=[]
    for seed in seeds:
     by_scene=[]
     for scene in cfg['C1'][split]:
      pairs=[key for key in retained if key[0]==scene];d=[]
      for s,f in pairs:
       a=lookup[s,f,f'{family}_{seed}_CD'][metric];b=baseline[s,f][metric]
       convert=(lambda x:1. if x=='True' else 0.) if metric.startswith('top1') else float
       d.append(convert(a)-convert(b))
      by_scene.append(float(np.mean(d)))
     matrix.append(by_scene)
    values[split]=np.array(matrix)
   factor=1 if metric.startswith('ap') else 100;distribution=np.mean([v.mean(0)[ix[s]].mean(1) for s,v in values.items()],axis=0)*factor
   summary[metric]=dict(delta=float(np.mean([v.mean() for v in values.values()])*factor),ci95=np.quantile(distribution,[.025,.975]).tolist(),seed_delta=(np.mean([v.mean(1) for v in values.values()],axis=0)*factor).tolist())
  result.update(status='COMPLETE_SENSITIVITY_NOT_OFFICIAL_FULL_FRAME_AP',metrics=summary)
 modes[scope]=result
result=dict(status='COMPLETE',family=family,identity='decoded raw RGB pixels + depth pixels + intrinsic matrix K; excludes predictions/quality and GT object poses',full_protocol_preserved=True,modes=modes)
(R/'results/C1_duplicate_sensitivity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:dict(status=v['status'],retained=v['retained_frames']) for k,v in modes.items()}),flush=True)
import subprocess,sys
subprocess.run([sys.executable,str(R/'code/build_final_report.py')],check=True)
