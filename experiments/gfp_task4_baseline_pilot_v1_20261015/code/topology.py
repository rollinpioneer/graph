"""Descriptive topology stratification of existing inference arrays; no new model calls."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];Z=Path('/home/xushijie3/work/gfp_guidance_distance_z01_20261015');geom=json.loads((R/'reports/geometry.json').read_text());res=[]
def ci(y,tid):
 groups,inv=np.unique(tid,return_inverse=True);n=len(groups)
 if n<2:return None
 num=np.bincount(inv,weights=y,minlength=n);den=np.bincount(inv,minlength=n);w=np.random.default_rng(2026101516).multinomial(n,np.full(n,1/n),size=200)
 return np.quantile((w@num)/(w@den),[.025,.975]).tolist()
for task,root,goal,idx in [('task1',Z,np.array([20,20]),0),('task4',R,np.array([0,20]),3)]:
 grid=np.array(geom['tasks'][idx]['bfs_to_goal'])
 for split,si in [('train',0),('validation',1)]:
  f=dict(np.load(root/'arrays'/f'{split}_samples.npz'));xy=f['observations'][:,:2];cross=np.zeros(len(xy),bool)
  for t in np.linspace(0,1,161):
   p=xy*(1-t)+goal*t;ji=((p+6)/4).astype(int);j,i=ji.T;ok=(i>=0)&(i<8)&(j>=0)&(j<8);cross|=~ok;cross[ok]|=grid[i[ok],j[ok]]<0
  groups={'line_crosses_wall':cross,'clear_line':~cross,'far':f['bfs']>=8,'near':(f['bfs']>=0)&(f['bfs']<=3)}
  for seed in [0,1]:
   for cp in [200000,600000,1000000]:
    a=np.load(root/'arrays'/f's{seed}_{cp}.npz');m=(a['repeat']==0)&(a['split']==si);g=a['g'][m];delta=a['delta'][m];md=a['member_delta'][m]
    for label,mask in groups.items():
     if not mask.any():continue
     neutral=((g[mask]>=.4)&(g[mask]<=.6));agreement=np.sign(md[mask,0])==np.sign(md[mask,1])
     res.append(dict(task=task,split=split,seed=seed,checkpoint=cp,group=label,n=int(mask.sum()),trajectories=int(len(np.unique(f['trajectory'][mask]))),neutral=float(neutral.mean()),neutral_ci95=ci(neutral.astype(float),f['trajectory'][mask]),ranking_agreement=float(agreement.mean()),mean_abs_delta=float(abs(delta[mask]).mean()),mean_bfs=float(f['bfs'][mask].mean()),mean_euclidean=float(f['euclidean'][mask].mean())))
(R/'reports/topology_comparison.json').write_text(json.dumps(res,indent=2))
