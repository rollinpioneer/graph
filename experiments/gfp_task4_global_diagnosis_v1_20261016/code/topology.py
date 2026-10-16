"""Observed-state value topology; no constructed states, no action ground truth."""
from pathlib import Path
from collections import deque
import numpy as np,json
R=Path(__file__).resolve().parents[1];P=Path('/home/xushijie3/work/gfp_task4_baseline_pilot_v1_20261015');Z=Path('/home/xushijie3/work/gfp_guidance_distance_z01_20261015')
geo=json.loads((P/'reports/geometry.json').read_text());maze=np.array(geo['maze_map']);support=json.loads((R/'reports/data_support.json').read_text())
def bfs(start):
 d=np.full((8,8),-1);d[start]=0;q=deque([start])
 while q:
  u=q.popleft()
  for v in [(u[0]+1,u[1]),(u[0]-1,u[1]),(u[0],u[1]+1),(u[0],u[1]-1)]:
   if 0<=v[0]<8 and 0<=v[1]<8 and maze[v]==0 and d[v]<0:d[v]=d[u]+1;q.append(v)
 return d
floors=list(map(tuple,np.argwhere(maze==0)));walls=[]
for i,u in enumerate(floors):
 for v in floors[i+1:]:
  d=np.abs(np.array(u)-v)
  if sorted(d.tolist())==[0,2] and maze[tuple((np.array(u)+v)//2)]==1 and bfs(u)[v]>=4:walls.append((u,v))
def compare(o,tid,cell,q,far,near):
 a=far[0]*8+far[1];b=near[0]*8+near[1];m=(cell==a)|(cell==b)
 if min(np.sum(cell==a),np.sum(cell==b))<30:return dict(status='LOW_SUPPORT')
 x=o[m][:,[23,38,39,40,41]];x=(x-x.mean(0))/np.maximum(x.std(0),1e-6);xx=np.c_[np.ones(len(x)),cell[m]==a,x];y=q[m];tt=tid[m];eps=np.unique(tt)
 beta=np.linalg.lstsq(xx,y,rcond=None)[0];res=y-xx@beta;bread=np.linalg.pinv(xx.T@xx);sc=np.array([xx[tt==e].T@res[tt==e] for e in eps]);cov=bread@(sc.T@sc)@bread
 cov*=len(eps)/max(1,len(eps)-1)*(len(y)-1)/max(1,len(y)-np.linalg.matrix_rank(xx));se=float(np.sqrt(max(0,cov[1,1])));lo=float(beta[1]-1.96*se);hi=float(beta[1]+1.96*se)
 return dict(status='ASSOCIATION_ONLY',far_n=int(np.sum(cell==a)),near_n=int(np.sum(cell==b)),episodes=len(eps),far_minus_near=float(beta[1]),ci95_cluster_normal=[lo,hi],consistent_sign=bool(beta[1]<0),interval_reversed=bool(lo>0),interval_consistent=bool(hi<0))
out=[];maps=[]
for task in [1,4]:
 src=Z if task==1 else P;t=next(t for t in geo['tasks'] if t['task']==task);dg=np.array(t['bfs_to_goal']);edges=next(x for x in support if x['task']==task and x['split']=='train')['shortest_edges']
 for split in ['train','validation']:
  s=dict(np.load(src/'arrays'/f'{split}_samples.npz'));o=s['observations'];cell=((o[:,:2]+6)/4).astype(int)[:,::-1];cid=cell[:,0]*8+cell[:,1]
  for seed in [0,1]:
   for cp in [200000,600000,1000000]:
    d=dict(np.load(src/'arrays'/f's{seed}_{cp}.npz'));m=(d['repeat']==0)&(d['split']==(split=='validation'));q=d['qdata'][m].mean(1)
    base=dict(task=task,split=split,seed=seed,checkpoint=cp)
    maps.append(dict(**base,cells=[dict(cell=list(map(int,u)),bfs=int(dg[u]),n=int((cid==u[0]*8+u[1]).sum()),mean_q=float(q[cid==u[0]*8+u[1]].mean())) for u in floors]))
    for u,v in walls:
     if dg[u]==dg[v]:continue
     far,near=(u,v) if dg[u]>dg[v] else (v,u)
     out.append(dict(**base,kind='across_wall',far=list(map(int,far)),near=list(map(int,near)),bfs_gap=int(dg[far]-dg[near]),**compare(o,s['trajectory'],cid,q,far,near)))
    for e in edges:
     far,near=e['source'],e['destination']
     out.append(dict(**base,kind='shortest_edge',far=far,near=near,bfs_gap=1,**compare(o,s['trajectory'],cid,q,far,near)))
(R/'reports/value_topology.json').write_text(json.dumps(out,indent=2));(R/'reports/value_cell_maps.json').write_text(json.dumps(maps,indent=2))
print('TOPOLOGY_DONE',len(out),len(maps))
