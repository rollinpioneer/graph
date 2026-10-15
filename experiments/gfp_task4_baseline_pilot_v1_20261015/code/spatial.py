from pathlib import Path
import json,collections,numpy as np
R=Path(__file__).resolve().parents[1]
geom=json.loads((R/'reports/geometry.json').read_text());bfs=np.array(geom['tasks'][3]['bfs_to_goal']);out={}
def cell(xy):return ((xy+6)/4).astype(int)[:,::-1]
def key(c):return ','.join(map(str,c))
for split,suffix in [('train',''),('validation','-val')]:
 with np.load(R/'data'/f'humanoidmaze-medium-navigate-v0{suffix}.npz') as z:o=z['observations'];t=z['terminals'].astype(bool)
 c=cell(o[:,:2]);ends=np.flatnonzero(t);starts=np.r_[0,ends[:-1]+1];occupancy=collections.Counter();trajs=collections.Counter();edges=collections.Counter();turns=collections.Counter()
 for a,b in zip(starts,ends):
  cc=c[a:b+1];ks=[key(v) for v in cc[:-1]];occupancy.update(ks);trajs.update(set(ks));changed=np.r_[True,np.any(cc[1:]!=cc[:-1],axis=1)];seq=cc[changed]
  for x,y in zip(seq[:-1],seq[1:]):
   if np.abs(x-y).sum()==1:edges[key(x)+'>'+key(y)]+=1
  for x,y,z in zip(seq[:-2],seq[1:-1],seq[2:]):
   d=y-x;e=z-y
   if np.abs(d).sum()==1 and np.abs(e).sum()==1 and np.dot(d,e)==0:turns[key(y)]+=1
 out[split]=dict(cell_transitions=dict(occupancy),cell_unique_trajectories=dict(trajs),directed_edges=dict(edges),right_angle_turns=dict(turns),trajectories=len(ends))
(R/'reports/data_coverage.json').write_text(json.dumps(out,indent=2))
rows=[]
for seed in [0,1]:
 for cp in [200000,600000,1000000]:
  z=np.load(R/'arrays'/f's{seed}_{cp}.npz');sel=(z['repeat']==0)&(z['split']==1);g=z['g'][sel];delta=z['delta'][sel];members=z['member_delta'][sel];f=np.load(R/'arrays/validation_samples.npz');c=cell(f['observations'][:,:2])
  for cc in np.unique(c,axis=0):
   m=np.all(c==cc,axis=1);i,j=cc;rows.append(dict(seed=seed,checkpoint=cp,cell=key(cc),bfs=int(bfs[i,j]),count=int(m.sum()),trajectories=int(len(np.unique(f['trajectory'][m]))),neutral=float(((g[m]>=.4)&(g[m]<=.6)).mean()),mean_abs_delta=float(abs(delta[m]).mean()),member_ranking_agreement=float((np.sign(members[m,0])==np.sign(members[m,1])).mean()),proxy_positive_rate=float((f['progress100'][m]>=1).mean())))
(R/'reports/spatial_guidance.json').write_text(json.dumps(rows,indent=2))
