"""Post hoc analysis of saved task4 trajectories/data only; no simulator/model calls."""
from pathlib import Path
import collections, hashlib, json, time
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'reports/start_failure';O.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def save(n,v):(O/n).write_text(json.dumps(v,indent=2,allow_nan=False))
def avg(v):return float(np.mean(v)) if len(v) else None
def first(v,k=1):
 z=np.flatnonzero(np.convolve(np.asarray(v,int),np.ones(k,int),'valid')>=k)
 return int(z[0]) if len(z) else None
def quant(v):return [float(x) for x in np.quantile(v,[.05,.25,.5,.75,.95])] if len(v) else []
def cell(o):return ((o[:,:2]+6)/4).astype(int)[:,::-1]
start=time.time()
plan=dict(stage='Existing-asset diagnostic follow-up',source_commit='6038532cef2178d007550d0b4abbc4b4eb59bd7c',
 trajectories='All existing matched trajectories: 4 models x3 checkpoints x100 episodes; no new episodes',
 stable_posture='head_height>=1.2 and torso_zz>=0.7; 20 consecutive frames for onset; descriptive, not environment success/health',
 low_posture='head_height<0.6 or torso_zz<0.3; initial random posture is expected',
 boundary='initial cell (row6,col5), west wall x=14, east open exit x>=18; wall-adjacent proxy x<=14.75',
 directed_motion='Measured COM vx or displacement, not torque/intended direction; +x starts detour, -x points toward goal',
 thresholds_sensitivity={'height':[1.0,1.2,1.4],'torso_zz':[0.5,0.7]},
 dataset='Full original train/validation; same-trajectory future progress and observed departures are descriptive collection-policy proxies',
 optimizer_updates=0,new_environment_steps=0,scope='No training, tuning, new environments, or simulation branches')
save('protocol.json',plan)
episodes=[]
for alg in ['GFP','ReBRAC']:
 for seed in [0,1]:
  for cp in [800000,900000,1000000]:
   meta=read(R/'runs'/f'{alg}_seed{seed}'/'matched'/f'{cp}.json')
   for ep in meta['episodes']:
    p=Path(ep['trajectory']);assert hashlib.sha256(p.read_bytes()).hexdigest()==ep['trajectory_sha256']
    with np.load(p) as z:o=z['observations'];a=z['actions']
    assert o.shape==(len(a)+1,69) and np.isfinite(o).all() and np.isfinite(a).all()
    cc=cell(o);init=np.all(cc==[6,5],axis=1);h=o[:,23];up=o[:,38];vx=o[:,39];stable=(h>=1.2)&(up>=.7)
    x=o[:,0];y=o[:,1];outside=np.flatnonzero(~init);end=int(outside[0]) if len(outside) else len(o)-1
    pre=np.arange(len(o))<=end;st=stable&pre;moving=st&(abs(vx)>.1)
    wrong=(x<=(x[0]-.75))&pre;right=(x>=(x[0]+.75))&pre;wall=(x<=14.75)&init
    row=dict(algorithm=alg,seed=seed,checkpoint=cp,episode=ep['episode'],success=ep['success'],length=len(a),
      initial_height=float(h[0]),initial_upright=float(up[0]),left_initial_cell=bool(len(outside)),exit_index=int(outside[0]) if len(outside) else None,
      first_stable20=first(stable,20),first_west_075=first(wrong),first_east_075=first(right),first_wall_near20=first(wall,20),
      stable_fraction=float(stable.mean()),low_fraction=float(((h<.6)|(up<.3)).mean()),initial_cell_fraction=float(init.mean()),
      pre_exit_stable_frames=int(st.sum()),pre_exit_stable_vx=avg(vx[st]),pre_exit_stable_east_fraction=avg((vx[moving]>.1).astype(float)),
      wall_near_fraction=float(wall.mean()),stable_wall_fraction=avg(wall[stable]),height_quantiles=quant(h),upright_quantiles=quant(up),
      x_range=[float(x.min()),float(x.max())],y_range=[float(y.min()),float(y.max())],action_rms=float(np.sqrt(np.mean(a*a))),
      action_delta_rms=float(np.sqrt(np.mean(np.diff(a,axis=0)**2))),action_clipped_fraction=float((abs(a)>=.999).mean()),
      stable_thresholds={f'{hh}_{uu}':float(((h>=hh)&(up>=uu)).mean()) for hh in [1.,1.2,1.4] for uu in [.5,.7]},
      windows={str(k):dict(dx=float(x[min(k,len(x)-1)]-x[0]),dy=float(y[min(k,len(y)-1)]-y[0]),stable=float(stable[:k].mean()),low=float(((h[:k]<.6)|(up[:k]<.3)).mean()),vx=float(vx[:k].mean())) for k in [50,100,200,500]})
    episodes.append(row)
   print('TRAJ_DONE',alg,seed,cp,flush=True)
save('episodes.json',episodes)
groups=[]
for alg in ['GFP','ReBRAC']:
 for seed in [0,1]:
  for cp in [800000,900000,1000000]:
   for outcome in ['all','failure','success']:
    rows=[x for x in episodes if x['algorithm']==alg and x['seed']==seed and x['checkpoint']==cp and (outcome=='all' or bool(x['success'])==(outcome=='success'))]
    if not rows:continue
    d=dict(algorithm=alg,seed=seed,checkpoint=cp,outcome=outcome,n=len(rows),never_stable20=sum(x['first_stable20'] is None for x in rows),
      never_left=sum(not x['left_initial_cell'] for x in rows),reached_west075=sum(x['first_west_075'] is not None for x in rows),reached_east075=sum(x['first_east_075'] is not None for x in rows),
      stable_onset_median=avg([]),reached_wall20=sum(x['first_wall_near20'] is not None for x in rows))
    on=[x['first_stable20'] for x in rows if x['first_stable20'] is not None];d['stable_onset_quantiles']=quant(on)
    for k in ['stable_fraction','low_fraction','initial_cell_fraction','pre_exit_stable_vx','pre_exit_stable_east_fraction','wall_near_fraction','stable_wall_fraction','action_rms','action_delta_rms','action_clipped_fraction']:
     d[k]=avg([x[k] for x in rows if x[k] is not None])
    d['threshold_sensitivity']={k:avg([x['stable_thresholds'][k] for x in rows]) for k in rows[0]['stable_thresholds']}
    d['windows']={w:{k:avg([x['windows'][w][k] for x in rows]) for k in rows[0]['windows'][w]} for w in ['50','100','200','500']}
    groups.append(d)
save('trajectory_summary.json',groups)

coverage=[];probe_groups=[]
for split,suffix in [('train',''),('validation','-val')]:
 with np.load(R/'data'/f'humanoidmaze-medium-navigate-v0{suffix}.npz') as z:o=z['observations'];terminal=z['terminals'].astype(bool)
 cc=cell(o);isstart=np.all(cc==[6,5],axis=1);ends=np.flatnonzero(terminal);begins=np.r_[0,ends[:-1]+1]
 future100=np.full(len(o),np.nan);future25=np.full(len(o),np.nan);departure=np.full(len(o),-1,np.int8);transits=[]
 for tid,(lo,hi) in enumerate(zip(begins,ends)):
  ids=np.arange(lo,hi+1);valid=ids+100<=hi;future100[ids[valid]]=o[ids[valid]+100,0]-o[ids[valid],0]
  valid=ids+25<=hi;future25[ids[valid]]=o[ids[valid]+25,0]-o[ids[valid],0]
  mask=isstart[lo:hi+1];diff=np.diff(np.r_[False,mask,False].astype(int));a=np.flatnonzero(diff==1)+lo;b=np.flatnonzero(diff==-1)+lo
  for v,w in zip(a,b):
   if w<=hi:
    east=bool(np.array_equal(cc[w],[6,6]));departure[v:w]=int(east)
    transits.append(dict(trajectory=tid,start=int(v),stop=int(w),frames=int(w-v),east=east,
     stable_fraction=float(((o[v:w,23]>=1.2)&(o[v:w,38]>=.7)).mean()),dx=float(o[w,0]-o[v,0])))
 sel=isstart&~terminal;stable=(o[:,23]>=1.2)&(o[:,38]>=.7)
 coverage.append(dict(split=split,start_states=int(sel.sum()),start_trajectories=int(sum(isstart[lo:hi+1].any() for lo,hi in zip(begins,ends))),
   stable_fraction=avg(stable[sel]),height_quantiles=quant(o[sel,23]),upright_quantiles=quant(o[sel,38]),
   stable_vx_quantiles=quant(o[sel&stable,39]),east_departures=sum(x['east'] for x in transits),all_departures=len(transits),
   departures=transits,proxy_future100_positive=int((future100[sel]>.75).sum()),proxy_future100_negative=int((future100[sel]<-.75).sum())))
 f=dict(np.load(R/'arrays'/f'{split}_samples.npz'));idx=f['indices'];sm=isstart[idx];dx=future100[idx];elig=stable[idx]
 for seed in [0,1]:
  for cp in [200000,600000,1000000]:
   with np.load(R/'arrays'/f's{seed}_{cp}.npz') as z:
    mask=(z['repeat']==0)&(z['split']==int(split=='validation'));v={k:z[k][mask] for k in ['g','delta','qdata','qactor','qvabc','member_delta','actor_data_l2','vabc_data_l2']}
   for pose in ['all','stable']:
    for label,extra in [('east_progress',dx>.75),('west_progress',dx<-.75),('flat',abs(dx)<=.75),('observed_east_departure',departure[idx]==1)]:
     mask=sm&extra&(elig if pose=='stable' else True)
     if not mask.any():continue
     probe_groups.append(dict(split=split,seed=seed,checkpoint=cp,pose=pose,group=label,n=int(mask.sum()),trajectories=int(len(np.unique(f['trajectory'][mask]))),
      guidance=avg(v['g'][mask]),neutral=avg((abs(v['g'][mask]-.5)<=.1).astype(float)),delta=avg(v['delta'][mask]),qdata=avg(v['qdata'][mask].mean(1)),
      qactor=avg(v['qactor'][mask].mean(1)),qvabc=avg(v['qvabc'][mask].mean(1)),actor_data_l2=avg(v['actor_data_l2'][mask]),vabc_data_l2=avg(v['vabc_data_l2'][mask]),
      mean_height=avg(f['head_height'][mask]),mean_upright=avg(f['upright'][mask]),mean_vx=avg(f['observations'][mask,39])))
 print('DATA_DONE',split,flush=True)
save('data_start_coverage.json',coverage);save('existing_probe_start_groups.json',probe_groups)
save('receipt.json',dict(status='PASS',existing_episodes=len(episodes),verified_trajectory_hashes=len(episodes),new_environment_steps=0,optimizer_updates=0,new_model_evaluations=0,wall_seconds=time.time()-start))
print('DONE',time.time()-start)
