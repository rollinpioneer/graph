from pathlib import Path
import json,csv,numpy as np,hashlib
R=Path(__file__).resolve().parents[1];rows=[];identity={};spatial=[]
for alg in ['GFP','ReBRAC']:
 for seed in [0,1]:
  d=R/'runs'/f'{alg}_seed{seed}';rt=json.loads((d/'runtime.json').read_text());output=Path(rt['output_dir']);native=list(csv.DictReader((output/'success.csv').open()))
  for step in [800000,900000,1000000]:
   p=d/'matched'/f'{step}.json'
   if not p.exists():continue
   v=json.loads(p.read_text());assert v['status']=='PASS' and v['optimizer_updates']==0
   for ep in v['episodes']:
    old=identity.setdefault(ep['episode'],ep['initial_observation_sha256']);assert old==ep['initial_observation_sha256'],'Initial state mismatch'
    with np.load(ep['trajectory']) as z:obs=z['observations']
    c=((obs[:,:2]+6)/4).astype(int)[:,::-1];cells,counts=np.unique(c,axis=0,return_counts=True)
    spatial.append(dict(algorithm=alg,seed=seed,checkpoint=step,episode=ep['episode'],success=ep['success'],length=ep['length'],terminal_cell=','.join(map(str,c[-1])),occupancy={','.join(map(str,k)):int(n) for k,n in zip(cells,counts)}))
   rows.append(dict(algorithm=alg,seed=seed,checkpoint=step,success_rate=v['success_rate'],return_=v['mean_return'],evaluation_seconds=v['elapsed_seconds'],episodes=len(v['episodes'])))
summary=[]
for alg in ['GFP','ReBRAC']:
 for seed in [0,1]:
  v=[x for x in rows if x['algorithm']==alg and x['seed']==seed]
  if len(v)==3:summary.append(dict(algorithm=alg,seed=seed,last_three_success=float(np.mean([x['success_rate'] for x in v])),last_three_return=float(np.mean([x['return_'] for x in v]))))
(R/'reports/matched_results.json').write_text(json.dumps(dict(checkpoints=rows,seeds=summary,initial_hashes_matched=len(identity),complete=len(rows)==12),indent=2))
(R/'reports/evaluation_spatial.json').write_text(json.dumps(spatial,indent=2))
print(json.dumps(summary))
