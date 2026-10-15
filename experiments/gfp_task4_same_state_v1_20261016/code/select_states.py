from pathlib import Path
import numpy as np,json,hashlib,time
R=Path(__file__).resolve().parents[1]
P=R.parent/'gfp_task4_baseline_pilot_v1_20261015'
states=[];eligible=[]
for seed in [0,1]:
 meta=json.loads((P/'runs'/f'GFP_seed{seed}'/'matched/1000000.json').read_text())
 chosen=0
 for ep in meta['episodes']:
  p=Path(ep['trajectory']);assert hashlib.sha256(p.read_bytes()).hexdigest()==ep['trajectory_sha256']
  with np.load(p) as f:o=f['observations'];a=f['actions']
  cell=((o[:,:2]+6)/4).astype(int)[:,::-1];inside=np.all(cell==[6,5],axis=1)
  stable=(o[:,23]>=1.2)&(o[:,38]>=.7)
  for t in range(20,min(401,len(a)-200)):
   if inside[:t+1].all() and (o[:t+1,0]>14.75).all() and 15<=o[t,0]<=17.5 and 18.5<=o[t,1]<=21.5 and stable[t-19:t+1].all():
    row=dict(seed=seed,episode=ep['episode'],step=t,observation=o[t].tolist(),source=ep)
    eligible.append(dict(seed=seed,episode=ep['episode'],step=t))
    if chosen<4:
     row.update(id=f's{seed}_e{ep["episode"]:03d}_t{t:04d}',index=len(states));states.append(row);chosen+=1
    break
out=dict(status='FROZEN_BEFORE_INTERVENTION',time=time.time(),max_states=8,states=states,all_eligible=eligible,selection_uses_outcome=False)
(R/'configs/states.json').write_text(json.dumps(out,indent=2))
print(json.dumps(dict(selected=[(s['id'],s['observation'][0:2]) for s in states],eligible=len(eligible))))
