"""Supplemental CPU-only process sensitivity and control-chain descriptions."""
import pathlib,json,numpy as np,datetime
from prepare import events,first_run
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'results'
OLD=pathlib.Path('/home/xushijie3/work/cpdisr_arm_20261011/stage5_pushcube_20261011')
definitions={'computed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'role':'Supplemental exploratory sensitivity, computed after A. Main registered endpoints and event definitions are unchanged.',
 'stall_windows':[10,20,30],'stall_displacement_and_progress_tolerances_m':[.0001,.001,.005],
 'fall_center_z_m':[.970,.975,.980],'fall_consecutive_steps':[3,5,7],
 'departure_hit_distances_m':[.035,.040,.045],'departure_off_minus_hit_m':.01,'departure_consecutive_steps':[3,5,7],
 'causal_limit':'Joint/end-effector response and target motion do not establish contact force, perception error, or goal-association error.'}
(OUT/'process_sensitivity_definition.json').write_text(json.dumps(definitions,indent=2))
rows=[]
for directory in sorted((ROOT/'runs').glob('A_*')):
 p=directory/'data';z=np.load(p/'trajectory.npz');init=np.load(p/'initial.npz')
 states,dd,eef=z['states'],z['distances'],z['eef'];q=z['q'];actions=z['actions'];targets=z['controller_goal']
 assert init['initial_latent'].shape==(4,2,24,10) and init['goal_latent'].shape==(4,2,24,10)
 assert np.isfinite(init['initial_latent']).all() and np.isfinite(init['goal_latent']).all()
 assert all(np.std(x)>0 for x in init['initial_image'].reshape(-1,3,128,128))
 assert all(np.std(x)>0 for x in init['goal_image'].reshape(-1,3,128,128))
 for i in range(4):
  for end in [100,200]:
   st,d,e=states[:end+1,i],dd[:end+1,i],eef[:end+1,i]
   main=events(st,d,e);sensitivity={}
   for w in [10,20,30]:
    for tol in [.0001,.001,.005]:sensitivity['stall_w%d_tol%.4f'%(w,tol)]=events(st,d,e,w,tol)['first_no_progress']
   fall={}
   for threshold in [.970,.975,.980]:
    for length in [3,5,7]:fall['z%.3f_k%d'%(threshold,length)]=first_run((st[...,2]<threshold).any(axis=-1),length)
   departure={}
   for hit in [.035,.04,.045]:
    for length in [3,5,7]:
     departures=[]
     for j in range(d.shape[-1]):
      h=first_run(d[:,j]<hit,length);leave=first_run(d[h+1:,j]>hit+.01,length) if h is not None else None
      departures.append(leave+h+1 if leave is not None else None)
     departure['hit%.3f_off%.3f_k%d'%(hit,hit+.01,length)]=departures
   chain=None;t=main['first_no_progress']
   if t is not None:
    lo=t-20;ac=actions[lo:t,i];controller=targets[lo:t,i]
    chain={'window':[lo,t],'mean_policy_action_norm':float(np.linalg.norm(ac,axis=-1).mean()),
     'policy_action_fraction_components_outside_clip_1':float((np.abs(ac)>1).mean()),
     'mean_controller_target_delta_from_pre_step_eef_m':float(np.linalg.norm(controller-eef[lo:t,i],axis=-1).mean()),
     'max_joint_motion_from_window_start_rad':float(np.abs(q[lo:t+1,i]-q[lo,i]).max()),
     'max_eef_motion_m':float(np.linalg.norm(eef[lo:t+1,i]-eef[lo,i],axis=-1).max()),
     'max_cube_motion_m':float(np.linalg.norm(states[lo:t+1,i,:,:3]-states[lo,i,:,:3],axis=-1).max()),
     'mean_distance_improvement_m':float(dd[lo,i].mean()-dd[lo:t+1,i].mean(axis=-1).min())}
   rows.append({'condition':directory.name[2:],'group':i,'window_end':end,'main_events':main,'stall_sensitivity':sensitivity,
       'fall_sensitivity':fall,'departure_sensitivity':departure,'control_chain_at_first_no_progress':chain,
       'endpoint_any_z_below_.975':bool((st[-1,:,2]<.975).any()),'official_endpoint_fraction':float((d[-1]<.04).mean())})
summary={}
for c in ['3_sparse','3_crowded','6_sparse','6_crowded']:
 summary[c]={}
 for end in [100,200]:
  rr=[r for r in rows if r['condition']==c and r['window_end']==end]
  summary[c][str(end)]={'episodes':len(rr),'fall_episodes':sum(r['main_events']['first_fall'] is not None for r in rr),
   'no_progress_episodes':sum(r['main_events']['first_no_progress'] is not None for r in rr),
   'departure_episodes':sum(any(x is not None for x in r['main_events']['departure_steps']) for r in rr),
   'recovery_episodes':sum(any(x is not None for x in r['main_events']['recovery_steps']) for r in rr)}
(OUT/'process_audit.json').write_text(json.dumps({'definition':definitions,'by_condition':summary,'episodes':rows},indent=2))
old_rows=[]
for n in [3,6]:
 for file in sorted((OLD/('runs/core%d/data'%n)).glob('trajectory_*.npz')):
  z=np.load(file);st,dist,e=z['states'],z['distances'],z['eef'];seed=int(file.stem.split('_')[-1])
  for i in range(st.shape[1]):
   for end in ([100] if n==3 else [100,200,400]):
    cube,d,ee=st[:end+1,i],dist[:end+1,i],e[:end+1,i]
    fall={};stall={};departure={}
    for threshold in [.970,.975,.980]:
     for length in [3,5,7]:fall['z%.3f_k%d'%(threshold,length)]=first_run((cube[...,2]<threshold).any(axis=-1),length)
    for w in [10,20,30]:
     for tol in [.0001,.001,.005]:stall['w%d_tol%.4f'%(w,tol)]=events(cube,d,ee,w,tol)['first_no_progress']
    for hit in [.035,.04,.045]:
     for length in [3,5,7]:
      leave=[]
      for j in range(n):
       h=first_run(d[:,j]<hit,length);off=first_run(d[h+1:,j]>hit+.01,length) if h is not None else None
       leave.append(off+h+1 if off is not None else None)
      departure['hit%.3f_off%.3f_k%d'%(hit,hit+.01,length)]=leave
    old_rows.append({'n':n,'seed':seed,'slot':i,'window_end':end,'main_events':events(cube,d,ee),
      'fall_sensitivity':fall,'stall_sensitivity':stall,'departure_sensitivity':departure})
(OUT/'historical_process_sensitivity.json').write_text(json.dumps({'role':'Old data exploration and calibration only; no 3C steps after100 are synthesized.','episodes':old_rows},indent=2))
# State/goal/endpoint camera panels for every formal episode, not selected failures.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
case=OUT/'A_cases';case.mkdir(exist_ok=True);index=[]
for directory in sorted((ROOT/'runs').glob('A_*')):
 z=np.load(directory/'data/trajectory.npz');init=np.load(directory/'data/initial.npz')
 frames=z['images'];ts=z['image_steps'];assert ts[-1]==200
 for i in range(4):
  fig,axes=plt.subplots(2,3,figsize=(8,5))
  for view in range(2):
   images=[init['initial_image'][i,view],init['goal_image'][i,view],frames[-1,i,view]]
   for k,image in enumerate(images):
    axes[view,k].imshow(image.transpose(1,2,0));axes[view,k].axis('off');axes[view,k].set_title(['Initial','Goal','Endpoint 200'][k])
  name=directory.name[2:]+'_group%d.png'%i;fig.suptitle(directory.name[2:]+' group %d'%i);fig.tight_layout();fig.savefig(case/name,dpi=100);plt.close(fig)
  index.append({'condition':directory.name[2:],'group':i,'image':name,'raw_initial':str(directory/'data/initial.npz'),'raw_trajectory':str(directory/'data/trajectory.npz')})
(case/'index.json').write_text(json.dumps(index,indent=2))
print(json.dumps(summary),flush=True)
