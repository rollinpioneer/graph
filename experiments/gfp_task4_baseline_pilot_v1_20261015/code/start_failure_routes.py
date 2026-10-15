"""Count existing complete task4 routes within original dataset episode boundaries."""
from pathlib import Path
import numpy as np,json
R=Path(__file__).resolve().parents[1];out=[]
for split,suffix in [('train',''),('validation','-val')]:
 with np.load(R/'data'/f'humanoidmaze-medium-navigate-v0{suffix}.npz') as z:o=z['observations'];tm=z['terminals'].astype(bool)
 c=((o[:,:2]+6)/4).astype(int)[:,::-1];st=np.all(c==[6,5],axis=1);goal=np.linalg.norm(o[:,:2]-[0,20],axis=1)<=.5
 ends=np.flatnonzero(tm);starts=np.r_[0,ends[:-1]+1];found=[]
 for tid,(a,b) in enumerate(zip(starts,ends)):
  ix=np.flatnonzero(st[a:b+1]);gs=np.flatnonzero(goal[a:b+1])
  if len(ix) and len(gs) and gs[-1]>ix[0]:
   first=int(ix[0]);hit=int(gs[gs>first][0]);found.append(dict(trajectory=tid,raw_start=int(a+first),raw_goal=int(a+hit),steps=hit-first))
 out.append(dict(split=split,task4_goal_radius=.5,within_original_trajectory=True,trajectories_start_then_goal=len(found),examples=found))
(R/'reports/start_failure/data_full_route.json').write_text(json.dumps(out,indent=2))
print([(x['split'],x['trajectories_start_then_goal']) for x in out])
