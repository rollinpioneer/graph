import pathlib,json,hashlib
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];P=pathlib.Path('/home/xushijie3/work/gfp_badcase_trajectory_v1_20261015')
rows=json.loads((P/'reports/case_metrics.json').read_text());index={x['case']['id']:x for x in json.loads((P/'reports/trajectory_index.json').read_text())}
states=[]
def add(row,t,category,matched=None):
 p=P/'trajectories'/f"{row['id']}.npz";a=np.load(p)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==index[row['id']]['trajectory_sha256']
 d=dict(id=f"{category}_s{row['model_seed']}_e{row['episode']:03d}_t{t}",category=category,trajectory=row['id'],step=int(t),
 seed=row['model_seed'],source_npz=str(p),trajectory_sha256=index[row['id']]['trajectory_sha256'],
 checkpoint=1000000,reference=index[row['id']],matched_to=matched,
 observation_sha256=hashlib.sha256(a['observations'][t].tobytes()).hexdigest(),xy=a['xy'][t].tolist(),
 goal_distance=float(np.linalg.norm(a['xy'][t]-a['goal_xy'][t])),head_height=float(a['head_height'][t]),
 alternate_policy_seed=930000+1000*row['model_seed']+row['episode'])
 states.append(d);return a,t,d
for seed in (0,1):
 failures=sorted([r for r in rows if r['model_seed']==seed and r['kind']=='failure'],key=lambda r:r['episode'])
 near=[];stall=[]
 for row in failures:
  a=np.load(P/'trajectories'/f"{row['id']}.npz");dist=np.linalg.norm(a['xy']-a['goal_xy'],axis=1)
  ts=np.flatnonzero((dist<=1.2)&(np.arange(len(dist))>=100)&(np.arange(len(dist))<=1700))
  if len(ts) and len(near)<2:near.append(add(row,int(ts[0]),'near_goal'))
 for row in failures:
  if row['first_stall_window_end'] is None:continue
  t=max(100,row['first_stall_window_end']-250);a=np.load(P/'trajectories'/f"{row['id']}.npz")
  if np.linalg.norm(a['xy'][t]-a['goal_xy'][t])>3 and len(stall)<2:stall.append(add(row,t,'pre_stall'))
 assert len(near)==len(stall)==2
 for target,cat in [(near[0],'success_near'),(stall[0],'success_stall_region')]:
  a,t,td=target;desc=np.r_[a['xy'][t]/4,a['head_height'][t]/.5,a['torso_up'][t]/.5,a['qvel'][t,:3]/3]
  best=None
  for row in rows:
   if row['model_seed']!=seed or row['kind']!='success':continue
   b=np.load(P/'trajectories'/f"{row['id']}.npz");ts=np.arange(100,min(len(b['actions']),1701),10)
   features=np.column_stack((b['xy'][ts]/4,b['head_height'][ts]/.5,b['torso_up'][ts]/.5,b['qvel'][ts,:3]/3))
   ds=np.linalg.norm(features-desc,axis=1);i=int(ds.argmin());cand=(float(ds[i]),row['episode'],int(ts[i]),row)
   if best is None or cand[:3]<best[:3]:best=cand
  add(best[3],best[2],cat,dict(failure_state=td['id'],descriptor_distance=best[0],exact_same_state=False))
assert len(states)==12
out=dict(states=states,max_environment_steps=10800,max_states=12,branches_per_state=3,max_branch_steps=300,
 optimizer_updates=0,distribution_samples_per_source_per_state=32,selection='First eligible episode IDs; near first<=1.2m at100..1700; pre-stall first detected window minus250; success nearest position/posture/root-velocity descriptor (not identical states).')
(R/'configs/states.json').write_text(json.dumps(out,indent=2))
for s in states:print(s['id'],[round(x,2) for x in s['xy']],'D',round(s['goal_distance'],2))

