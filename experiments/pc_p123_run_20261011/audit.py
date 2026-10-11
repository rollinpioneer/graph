import json,pathlib,numpy as np,hashlib
from prepare import events
R=pathlib.Path(__file__).resolve().parent;P=pathlib.Path('/home/xushijie3/work/pc_scale02_20261011/experiments/pc_scale02_20261011');O=pathlib.Path('/home/xushijie3/work/cpdisr_arm_20261011/stage5_pushcube_20261011')
def dump(n,x):(R/'results'/n).write_text(json.dumps(x,indent=2))
def scalar(x):return float(x)
def chain(z,i,t):
 lo=max(0,t-20);ee=z['eef'][lo:t+1,i];st=z['states'][lo:t+1,i];dd=z['distances'][lo:t+1,i];ac=z['actions'][lo:t,i];q=z['q'][lo:t+1,i]
 r={'start_state_step':lo,'end_state_step':t,'eef_max_displacement_m':scalar(np.linalg.norm(ee-ee[0],axis=-1).max()),'eef_path_length_m':scalar(np.linalg.norm(np.diff(ee,axis=0),axis=-1).sum()),'eef_z_range_m':[scalar(ee[:,2].min()),scalar(ee[:,2].max())],'cube_max_displacement_m':scalar(np.linalg.norm(st[...,:3]-st[0,:,:3],axis=-1).max()),'goal_mean_improvement_m':scalar(dd[0].mean()-dd.mean(-1).min()),'action_abs_mean':np.abs(ac).mean(0).tolist(),'action_clip_fraction':scalar((np.abs(ac)>=1).mean()),'q_max_abs_change_rad':scalar(np.abs(q-q[0]).max()),'min_eef_cube_center_distance_m':scalar(np.linalg.norm(ee[:,None,:]-st[...,:3],axis=-1).min()),'contact_counterpart':'unknown: original trajectory has no physical substep contacts','causal_class':'unknown; kinematic association only'}
 if 'controller_goal' in z:
  cg=z['controller_goal'][lo:t,i];r['controller_goal_max_change_m']=scalar(np.linalg.norm(cg-cg[0],axis=-1).max());r['controller_goal_eef_offset_mean_m']=scalar(np.linalg.norm(cg-ee[:-1],axis=-1).mean());r['controller_goal_ranges']=np.ptp(cg,axis=0).tolist()
 if 'controller_goal' in z:
  import yaml
  cfg=yaml.safe_load((O/'EC-Diffuser/env_config/generalization_num_cubes/IsaacPandaPushConfig.yaml').read_text())['env'];table=cfg['tableDims'];center=cfg['tablePos'];cube=cfg['cubeSize'];top=center[2]+table[2]/2
  low=np.array([center[0]-table[0]/2-cube/2,center[1]-table[1]/2-cube/2,top+.02]);high=np.array([center[0]+table[0]/2+cube/2,center[1]+table[1]/2+cube/2,top+cube])
  expected=np.clip(ee[:-1]+np.clip(ac,-1,1)*.125,low,high)
  r['expected_scaled_clamped_goal_max_error_m']=float(np.abs(expected-z['controller_goal'][lo:t,i]).max())
  r['action_stored_stage']='official policy output after normalizer.unnormalize, before wrapper adds rotation/gripper components'
  r['command_transform']='xyz clipped +/-1, multiplied .125 in OSC, added to current world eef position, clamped by actual table-dependent limits'
 return r
rows=[]
for cond,i,t in [('6_crowded',0,42),('6_crowded',2,30),('6_sparse',0,89),('6_sparse',1,166)]:
 f=P/'runs'/('A_'+cond)/'data/trajectory.npz';z=np.load(f)
 rows.append(dict(condition=cond,group=i,detected_step=t,source=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),chain=chain(z,i,t)))
dump('P2_original_four.json',rows)
# Complete first-fall list, old and new populations kept separate.
falls=[];episodes=[]
files=[('PC_SCALE02',f) for f in sorted((P/'runs').glob('A_*/data/trajectory.npz'))]+[('historical',f) for n in [3,6] for f in sorted((O/('runs/core%d/data'%n)).glob('trajectory_*.npz'))]
for cohort,f in files:
 z=np.load(f);states=z['states'];dist=z['distances'];eef=z['eef']
 for i in range(states.shape[1]):
  ev=events(states[:,i],dist[:,i],eef[:,i]);episodes.append(dict(cohort=cohort,path=str(f),slot=i,n=states.shape[2],available_steps=len(states)-1,events=ev))
  t=ev['first_fall']
  if t is None:continue
  j=int(np.argmin(states[t,i,:,2]));lo=max(0,t-25);part=states[lo:t+1,i,j];qu=part[:,3:7];x,y,zq,w=qu.T
  rot=np.stack([1-2*(y*y+zq*zq),2*(x*y-zq*w),2*(x*zq+y*w),2*(x*y+zq*w),1-2*(x*x+zq*zq),2*(y*zq-x*w),2*(x*zq-y*w),2*(y*zq+x*w),1-2*(x*x+y*y)],-1).reshape(-1,3,3)
  half=np.abs(rot).sum(-1)*.0175
  margins=np.minimum(.25-np.abs(part[:,0])-half[:,0],.30-np.abs(part[:,1])-half[:,1]);cross=np.flatnonzero(margins<0)
  falls.append(dict(cohort=cohort,source=str(f),slot=i,n=states.shape[2],available_steps=len(states)-1,first_fall_detection=t,object_identity=j,window_start=lo,window_end=t,first_bbox_cross_in_window=int(lo+cross[0]) if len(cross) else None,pre_fall_edge_signed_distance_m=margins.tolist(),object_velocity_xyz=part[:,7:10].tolist(),eef_xyz=eef[lo:t+1,i].tolist(),object_xyz=part[:,:3].tolist(),actions=z['actions'][lo:t,i].tolist(),mechanism_labels=['insufficient_contact_pair_evidence'],unknown=True))
dump('P3_first_falls.json',falls);dump('original_episode_index.json',episodes)
print('AUDIT_OK',len(rows),len(falls),len(episodes))
print(json.dumps([dict(condition=x['condition'],group=x['group'],chain=x['chain']) for x in rows]))
