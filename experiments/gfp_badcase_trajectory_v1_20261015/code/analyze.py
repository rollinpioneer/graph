"""Descriptive trace metrics; thresholds are proxies, never causal diagnoses."""
import json, pathlib,collections,hashlib
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1]
meta=json.loads((R/'reports/environment.json').read_text())
maze=np.array(meta['maze_map']);unit=meta['maze_unit'];off=np.array(meta['offset'])
def spans(mask):
 a=np.r_[False,mask,False].astype(int);starts=np.where(np.diff(a)==1)[0];ends=np.where(np.diff(a)==-1)[0]
 return [(int(s),int(e),int(e-s)) for s,e in zip(starts,ends)]
def longest(mask):return max([n for s,e,n in spans(mask)],default=0)
def first(mask,n=40,begin=0):
 return next((s for s,e,l in spans(np.asarray(mask)&(np.arange(len(mask))>=begin)) if l>=n),None)
def cells(xy):return ((xy+off+unit/2)/unit).astype(int)[:,::-1]
def bfs(goal):
 ij=cells(np.asarray(goal)[None])[0];d=np.full(maze.shape,-1);d[tuple(ij)]=0;q=collections.deque([tuple(ij)])
 while q:
  i,j=q.popleft()
  for di,dj in ((-1,0),(1,0),(0,-1),(0,1)):
   x,y=i+di,j+dj
   if 0<=x<maze.shape[0] and 0<=y<maze.shape[1] and maze[x,y]==0 and d[x,y]<0:d[x,y]=d[i,j]+1;q.append((x,y))
 return d
rows=[];allcases={}
for jp in sorted((R/'trajectories').glob('*.json')):
 result=json.loads(jp.read_text());c=result['case'];a=np.load(jp.with_suffix('.npz'))
 xy=a['xy'];distance=np.linalg.norm(xy-a['goal_xy'],axis=1);cell=cells(xy);d=bfs(a['goal_xy'][0])
 valid=(cell[:,0]>=0)&(cell[:,0]<maze.shape[0])&(cell[:,1]>=0)&(cell[:,1]<maze.shape[1])
 geo=np.array([d[i,j] if v else -1 for (i,j),v in zip(cell,valid)])
 low=a['head_height']<.8;upright=(a['head_height']>1.2)&(a['torso_up']>.6)
 stand=first(upright,20) # Needed before describing a later fall-like posture.
 fall_onset=first(low,40,(stand+20) if stand is not None else len(low))
 stalled=np.zeros(len(xy),bool)
 for t in range(200,len(xy)):
  stalled[t]=np.linalg.norm(np.ptp(xy[t-200:t+1],axis=0))<1.0
 seq=[tuple(cell[0])]
 for ij in cell[1:]:
  if tuple(ij)!=seq[-1]:seq.append(tuple(ij))
 backs=sum(seq[k]==seq[k-2] for k in range(2,len(seq)))
 wall=a['wall_contact_count']>0
 tailwall=float(wall[-400:].mean());taillow=float(low[-400:].mean())
 obs=result['observed'];flags=[]
 if not obs['success']:
  if distance.min()<=2.0:flags.append('near_goal_unfinished')
  if longest(stalled)>=100:flags.append('sustained_spatial_stall')
  if tailwall>=.25:flags.append('wall_contact_frequent_final_400')
  if taillow>=.6:flags.append('low_head_frequent_final_400')
  if backs>=4:flags.append('repeated_cell_reversal')
 row=dict(id=c['id'],model_seed=c['model_seed'],episode=c['episode'],checkpoint=c['checkpoint'],kind=c['kind'],**obs,
 min_goal_distance=float(distance.min()),final_goal_distance=float(distance[-1]),min_goal_step=int(distance.argmin()),
 path_length=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),unique_cells=len(set(seq)),cell_transitions=len(seq)-1,
 immediate_cell_reversals=backs,initial_bfs_cells=int(geo[0]),minimum_bfs_cells=int(geo[geo>=0].min()),
 final_bfs_cells=int(geo[-1]),head_low_fraction=float(low.mean()),head_low_final400=taillow,
 longest_low_head_steps=longest(low),first_standing_step=stand,first_low_after_standing=fall_onset,
 wall_contact_fraction=float(wall.mean()),wall_contact_final400=tailwall,first_wall_step=int(np.flatnonzero(wall)[0])+1 if wall.any() else None,
 first_stall_window_end=first(stalled,100),longest_stall_window_end_run=longest(stalled),flags=flags,
 final_xy=xy[-1].tolist(),min_goal_xy=xy[distance.argmin()].tolist())
 rows.append(row);allcases[c['id']]=(c,a,result)
pairs=[]
for c,a,res in allcases.values():
 if c['kind'] not in ('earlier','rng'):continue
 fid=f"s{c['model_seed']}_k1000000_e{c['episode']:03d}_failure"
 f,fa,fr=allcases[fid];n=min(len(a['xy']),len(fa['xy']))
 delta=np.linalg.norm(a['xy'][:n]-fa['xy'][:n],axis=1);onset=first(delta>2,40)
 pairs.append(dict(reference_1m=fid,control=c['id'],kind=c['kind'],same_initial_observation=
 res['observed']['initial_observation_sha256']==fr['observed']['initial_observation_sha256'],
 control_success=res['observed']['success'],control_length=res['observed']['length'],
 first_sustained_2m_path_divergence_step=onset,reference_xy_at_divergence=fa['xy'][onset].tolist() if onset is not None else None,
 control_xy_at_divergence=a['xy'][onset].tolist() if onset is not None else None))
failed=[r for r in rows if r['kind']=='failure'];succ=[r for r in rows if r['kind']=='success']
summary=dict(n=len(rows),primary_failed=len(failed),primary_success=len(succ),
 failure_labels=dict(collections.Counter(f for r in failed for f in r['flags'])),
 primary_failure_all_timeout=all(r['truncated'] and not r['terminated'] for r in failed),
 low_pose_episode_count={'failure':sum(r['first_low_after_standing'] is not None for r in failed),
 'success':sum(r['first_low_after_standing'] is not None for r in succ)},
 rng_control_success=sum(x['control_success'] for x in pairs if x['kind']=='rng'),rng_control_n=sum(x['kind']=='rng' for x in pairs),
 failed_final_cell_counts=dict(collections.Counter(str(tuple(cells(np.array(r['final_xy'])[None])[0])) for r in failed)),
 interpretation='Counts are within deliberately enriched selected cases. Flags overlap; low pose and wall contact also occur in successes.')
(R/'reports/case_metrics.json').write_text(json.dumps(rows,indent=2))
(R/'reports/paired_comparisons.json').write_text(json.dumps(pairs,indent=2))
(R/'reports/behavior_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
for r in failed:print(r['id'],r['flags'],'minD',round(r['min_goal_distance'],2),'low',round(r['head_low_final400'],2),'wall',round(r['wall_contact_final400'],2))

