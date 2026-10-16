"""Read-only global support analysis and prespecified state selection; no env imports."""
from pathlib import Path
from collections import deque, Counter, defaultdict
import numpy as np, json, hashlib, time, resource
R=Path(__file__).resolve().parents[1]
P=Path('/home/xushijie3/work/gfp_task4_baseline_pilot_v1_20261015')
C=Path('/home/xushijie3/work/flow_critic_comparative_train_v1_20261014')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(name,obj):(R/'reports'/name).write_text(json.dumps(obj,indent=2,allow_nan=False))
def summary(a):
 a=np.asarray(a);a=a[np.isfinite(a)]
 return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9))) if len(a) else dict(n=0)
geo=json.loads((P/'reports/geometry.json').read_text());maze=np.array(geo['maze_map'])
def bfs(start):
 d=np.full(maze.shape,-1);d[tuple(start)]=0;q=deque([tuple(start)])
 while q:
  u=q.popleft()
  for v in [(u[0]+1,u[1]),(u[0]-1,u[1]),(u[0],u[1]+1),(u[0],u[1]-1)]:
   if 0<=v[0]<8 and 0<=v[1]<8 and maze[v]==0 and d[v]<0:d[v]=d[u]+1;q.append(v)
 return d
started=time.time();records=[];identities=[]
for split,suffix in [('train',''),('validation','-val')]:
 path=C/'data'/f'humanoidmaze-medium-navigate-v0{suffix}.npz'
 with np.load(path) as f:o=f['observations'];a=f['actions'];terminal=f['terminals'].astype(bool);fields={k:dict(shape=list(f[k].shape),dtype=str(f[k].dtype)) for k in f.files}
 identities.append(dict(split=split,path=str(path),sha256=sha(path),bytes=path.stat().st_size,fields=fields))
 ends=np.flatnonzero(terminal);starts=np.r_[0,ends[:-1]+1];assert ends[-1]==len(o)-1
 tid=np.searchsorted(ends,np.arange(len(o)),side='left');valid=~terminal
 cells=((o[:,:2]+6)/4).astype(int)[:,::-1];assert np.all((cells>=0)&(cells<8));assert np.all(maze[cells[:,0],cells[:,1]]==0)
 cid=cells[:,0]*8+cells[:,1];cross=np.flatnonzero(valid[:-1]&(cid[:-1]!=cid[1:]));edge_ids=cid[cross]*64+cid[cross+1]
 edge_count=Counter(edge_ids.tolist());edge_eps={int(e):int(len(np.unique(tid[cross[edge_ids==e]]))) for e in np.unique(edge_ids)}
 triple_n=Counter();triple_ep=defaultdict(set)
 for ep,(lo,hi) in enumerate(zip(starts,ends)):
  z=cid[lo:hi+1];z=z[np.r_[True,z[1:]!=z[:-1]]]
  for u,v,w in zip(z[:-2],z[1:-1],z[2:]):triple_n[(int(u),int(v),int(w))]+=1;triple_ep[(int(u),int(v),int(w))].add(ep)
 for task in [1,4]:
  t=next(x for x in geo['tasks'] if x['task']==task);dg=np.array(t['bfs_to_goal']);ds=bfs(t['start_ij']);L=t['bfs_cells'];corr=(ds>=0)&(dg>=0)&(ds+dg==L)
  edges=[]
  for u in map(tuple,np.argwhere(corr)):
   for v in [(u[0]+1,u[1]),(u[0]-1,u[1]),(u[0],u[1]+1),(u[0],u[1]-1)]:
    if 0<=v[0]<8 and 0<=v[1]<8 and corr[v] and ds[v]==ds[u]+1:
     eid=(u[0]*8+u[1])*64+v[0]*8+v[1];edges.append(dict(source=list(u),destination=list(v),crossings=edge_count[eid],episodes=edge_eps.get(eid,0)))
  turns=[]
  for e in edges:
   for f in edges:
    if e['destination']==f['source']:
     u,v,w=map(np.array,[e['source'],e['destination'],f['destination']])
     if np.dot(v-u,w-v)==0:
      key=tuple(int(x[0]*8+x[1]) for x in [u,v,w]);turns.append(dict(cells=[u.tolist(),v.tolist(),w.tolist()],occurrences=triple_n[key],episodes=len(triple_ep[key])))
  at_start=np.all(cells==t['start_ij'],axis=1);near_start=np.linalg.norm(o[:,:2]-t['start_xy'],axis=1)<=1
  at_goal=np.linalg.norm(o[:,:2]-t['goal_xy'],axis=1)<=.5;routes=[];near_routes=[]
  for ep,(lo,hi) in enumerate(zip(starts,ends)):
   gs=np.flatnonzero(at_goal[lo:hi+1])
   for mask,dest in [(at_start,routes),(near_start,near_routes)]:
    ix=np.flatnonzero(mask[lo:hi+1])
    if len(ix) and len(gs) and gs[-1]>ix[0]:
     first=int(ix[0]);hit=int(gs[gs>first][0]);dest.append(dict(episode=ep,start=int(lo+first),goal=int(lo+hit),steps=hit-first))
  route_mask=np.zeros(len(o),bool)
  for rr in routes:route_mask[rr['start']:rr['goal']]=True
  dist=dg[cells[:,0],cells[:,1]];eu=np.linalg.norm(o[:,:2]-t['goal_xy'],axis=1)
  ix=np.arange(len(o));future=np.minimum(ix+100,len(o)-1);window=(ix+100<=ends[tid])&valid
  progress=dist-dist[future];edelta=eu[future]-eu;disp=np.linalg.norm(o[future,:2]-o[:,:2],axis=1)
  region=corr[cells[:,0],cells[:,1]]&window
  masks=[region&(progress>=1)&(edelta>.5),region&(progress>=1)&(edelta<-.5),region&(progress<=-1),region&(progress==0)&(disp<.5)]
  labels=['detour_progress','aligned_progress','regress','stationary'];indices=[];groups=[];group_stats=[]
  for group,(label,mask) in enumerate(zip(labels,masks)):
   candidates=np.flatnonzero(mask);rng=np.random.default_rng(20261016+task*100+group+(1000 if split=='validation' else 0));rng.shuffle(candidates)
   chosen=[];counts=Counter()
   for i in candidates:
    ep=int(tid[i])
    if counts[ep]>=8:continue
    chosen.append(int(i));counts[ep]+=1
    if len(chosen)==256:break
   indices.extend(chosen);groups.extend([group]*len(chosen))
   group_stats.append(dict(group=label,states=int(mask.sum()),episodes=int(len(np.unique(tid[mask]))),selected=len(chosen),selected_episodes=len(counts),head_height=summary(o[mask,23]),upright=summary(o[mask,38]),speed=summary(np.linalg.norm(o[mask,39:42],axis=1))))
  indices=np.asarray(indices,dtype=int);groups=np.asarray(groups,dtype=int)
  np.savez_compressed(R/'arrays'/f't{task}_{split}_selected.npz',indices=indices,group=groups,trajectory=tid[indices],observations=o[indices],actions=np.clip(a[indices],-1+1e-5,1-1e-5),bfs=dist[indices],euclidean=eu[indices],progress100=progress[indices],euclidean_change100=edelta[indices],route_member=route_mask[indices],cell=cid[indices])
  np.savez_compressed(R/'arrays'/f't{task}_{split}_labels.npz',bfs=dist,progress100=progress,euclidean_change100=edelta,valid_window=window,corridor=corr[cells[:,0],cells[:,1]],cell=cid,trajectory=tid,route_member=route_mask)
  records.append(dict(task=task,split=split,episodes=len(ends),valid_transitions=int(valid.sum()),start_cell_states=int((at_start&valid).sum()),start_cell_episodes=int(len(np.unique(tid[at_start]))),start_radius1_episodes=int(len(np.unique(tid[near_start]))),goal_hit_episodes=int(len(np.unique(tid[at_goal]))),complete_routes=len(routes),complete_routes_radius1=len(near_routes),routes=routes,route_duration=summary([x['steps'] for x in routes]),route_states=int(route_mask.sum()),route_posture=dict(head=summary(o[route_mask,23]),upright=summary(o[route_mask,38]),speed=summary(np.linalg.norm(o[route_mask,39:42],axis=1))),shortest_edges=edges,shortest_turns=turns,group_stats=group_stats))
  print('DATA',task,split,'complete',len(routes),'start_episodes',len(np.unique(tid[at_start])),'selected',len(indices),flush=True)
write('data_support.json',records);write('data_identity.json',identities)
(R/'receipts/data.json').write_text(json.dumps(dict(status='PASS',seconds=time.time()-started,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,optimizer_updates=0,environment_steps=0),indent=2))
