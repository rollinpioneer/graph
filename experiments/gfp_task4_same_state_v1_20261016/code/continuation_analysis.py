"""Read existing trajectories only; no models, simulator imports, or environment steps."""
from pathlib import Path
import json,hashlib,time,resource,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=R/'reports/continuation';O.mkdir(exist_ok=True)
STATES=['s1_e010_t0061','s1_e003_t0056'];start=time.time()
def read(p):return json.loads(Path(p).read_text())
def save(n,x):(O/n).write_text(json.dumps(x,indent=2,allow_nan=False))
def run_start(mask,length):
 v=np.flatnonzero(np.convolve(np.asarray(mask,dtype=int),np.ones(length,dtype=int),'valid')==length)
 return int(v[0]) if len(v) else None
protocol=dict(source_commit='0fad5452a634a602b6c2fe7510ec5d612fb5ddfe',states=STATES,scope='All 96 existing branches for the two requested states; 0 new simulation/model calls/training',time_axis='h=0 is captured state; action[h] produces observation[h+1]; original episode index=h+61 or h+56',stable='head>=1.2 and torso_zz>=0.7',low='head<0.6 or torso_zz<0.3',events='first leaving initial cell and first return; cardinal x thresholds do not encode torque direction',sustained_west='10 consecutive observations with COM vx<-0.1m/s',sustained_low='10 consecutive observations with head<0.6 or torso_zz<0.3',sustained_pair_position='first 20 consecutive observations with |paired root x difference|>=0.5m, signed difference also retained',sustained_pair_action='first 20 consecutive actions with between-policy RMS difference>=0.2 per actuator; descriptive only',comparison='common saved follow-up RNG; no causal attribution from differing actions at already divergent states',new_environment_steps=0,optimizer_updates=0)
save('protocol.json',protocol)
allrows=sum([read(R/'reports'/f'branches_s{s}.json') for s in [0,1]],[]);rows=[x for x in allrows if x['state'] in STATES];assert len(rows)==96
arrays={};metrics=[];identities=[]
for x in rows:
 p=Path(x['path']);sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha==x['sha256'];z=dict(np.load(p));assert all(np.isfinite(v).all() for v in z.values());key=(x['state'],x['mode'],x['candidate'],x['repeat']);arrays[key]=z
 o=z['observations'];a=z['actions'];n=len(a);assert o.shape==(201,69) and n==200;dt=np.diff(z['time']);assert np.allclose(dt,dt[0],atol=1e-12,rtol=0)
 cell=((o[:,:2]+6)/4).astype(int)[:,::-1];inside=np.all(cell==[6,5],axis=1);outside=~inside;stable=(o[:,23]>=1.2)&(o[:,38]>=.7);low=(o[:,23]<.6)|(o[:,38]<.3);exit_ix=np.flatnonzero(outside);leave=int(exit_ix[0]) if len(exit_ix) else None
 ret_ix=np.flatnonzero(inside[leave+1:])+leave+1 if leave is not None else np.array([],int);back=int(ret_ix[0]) if len(ret_ix) else None
 jerk=np.r_[np.nan,np.sqrt(np.mean(np.diff(a,axis=0)**2,axis=1))]
 def window(lo,hi):
  lo=max(0,int(lo));hi=min(n,int(hi))
  if hi<=lo:return None
  obs=o[lo:hi+1];act=a[lo:hi];j=jerk[max(lo,1):hi]
  return dict(start=lo,end=hi,dx=float(obs[-1,0]-obs[0,0]),dy=float(obs[-1,1]-obs[0,1]),mean_vx=float(obs[:,39].mean()),mean_vy=float(obs[:,40].mean()),east_velocity_fraction=float((obs[:,39]>.1).mean()),west_velocity_fraction=float((obs[:,39]<-.1).mean()),head_mean=float(obs[:,23].mean()),head_min=float(obs[:,23].min()),upright_mean=float(obs[:,38].mean()),stable_fraction=float(stable[lo:hi+1].mean()),low_fraction=float(low[lo:hi+1].mean()),wall_contact_fraction=float((z['wall_contacts'][lo:hi+1]>0).mean()),action_rms=float(np.sqrt(np.mean(act*act))),action_change_rms=float(j.mean()) if len(j) else None,clipped_action_fraction=float((abs(act)>=.999).mean()))
 def after(mask,index):
  if index is None:return None
  v=run_start(mask[index:],10);return None if v is None else v+index
 events={}
 for name,ix in [('exit',leave),('return',back)]:
  if ix is not None:events[name]=dict(index=ix,observation=dict(x=float(o[ix,0]),y=float(o[ix,1]),vx=float(o[ix,39]),vy=float(o[ix,40]),head=float(o[ix,23]),upright=float(o[ix,38])),before20=window(ix-20,ix),after20=window(ix,ix+20))
 m=dict(state=x['state'],mode=x['mode'],candidate=x['candidate'],repeat=x['repeat'],dt=float(dt[0]),first_exit=leave,first_return=back,outside_final=bool(outside[-1]),outside_frames=int(outside[1:].sum()),first_west10=after(o[:,39]<-.1,0),first_west10_after_exit=after(o[:,39]<-.1,leave),first_low10=after(low,0),first_low10_after_exit=after(low,leave),stable_fraction=float(stable[1:].mean()),low_fraction=float(low[1:].mean()),maximum_x=float(o[:,0].max()),final_x=float(o[-1,0]),final_y=float(o[-1,1]),windows={f'{lo}_{hi}':window(lo,hi) for lo,hi in [(0,20),(20,50),(50,100),(100,150),(150,200)]},events=events)
 metrics.append(m);identities.append(dict(path=str(p),sha256=sha,bytes=p.stat().st_size))
save('branch_events.json',metrics)

pairs=[]
for state,mode,label in [(STATES[0],'first_action','actor_2'),(STATES[0],'first_action','vabc_2'),(STATES[1],'first_action','rebrac'),(STATES[1],'short_policy','rebrac'),(STATES[1],'short_policy','vabc')]:
 for rep in range(4):
  z=arrays[(state,mode,label,rep)];ref=arrays[(state,'first_action','original_actor',rep)];assert np.array_equal(z['policy_keys'][1:],ref['policy_keys'][1:]);assert np.array_equal(z['observations'][0],ref['observations'][0]);delta=z['observations']-ref['observations'];ad=np.sqrt(np.mean((z['actions']-ref['actions'])**2,axis=1));pd=delta[:,0]
  pairs.append(dict(state=state,mode=mode,candidate=label,repeat=rep,position_divergence20=run_start(abs(pd)>=.5,20),east_advantage20=run_start(pd>=.5,20),west_disadvantage20=run_start(pd<=-.5,20),action_divergence20=run_start(ad>=.2,20),first_action_rms_difference=float(ad[0]),windows={f'{lo}_{hi}':dict(mean_x_difference=float(pd[lo:hi+1].mean()),end_x_difference=float(pd[hi]),mean_vx_difference=float(delta[lo:hi+1,39].mean()),mean_head_difference=float(delta[lo:hi+1,23].mean()),mean_upright_difference=float(delta[lo:hi+1,38].mean()),action_rms_difference=float(ad[lo:hi].mean())) for lo,hi in [(0,20),(20,50),(50,100),(100,150),(150,200)]}))
save('paired_divergence.json',pairs)

# ReBRAC's deterministic repeats must not masquerade as independent successes.
repcheck={}
for state in STATES:
 z0=arrays[(state,'short_policy','rebrac',0)]
 repcheck[state]={str(rep):{n:float(np.max(np.abs(arrays[(state,'short_policy','rebrac',rep)][n]-z0[n]))) for n in ['observations','actions','qpos','qvel']} for rep in [1,2,3]}
save('deterministic_repeat_check.json',repcheck)

selected=[]
for state in STATES:
 for mode,label in ([('first_action','original_actor'),('first_action','actor_2'),('first_action','vabc_2')] if state==STATES[0] else [('first_action','original_actor'),('first_action','rebrac'),('short_policy','rebrac'),('short_policy','vabc')]):
  for rep in range(4):selected.append((state,mode,label,rep))
with (O/'focal_timeseries.csv').open('w',newline='') as f:
 fields=['state','mode','candidate','repeat','h','elapsed_seconds','x','y','head','upright','vx','vy','outside','stable','wall_contacts','action_rms','action_change_rms'];w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
 for key in selected:
  z=arrays[key];o=z['observations'];a=z['actions'];d=np.r_[np.nan,np.sqrt(np.mean(np.diff(a,axis=0)**2,axis=1))]
  for h in range(201):
   w.writerow(dict(state=key[0],mode=key[1],candidate=key[2],repeat=key[3],h=h,elapsed_seconds=float(z['time'][h]-z['time'][0]),x=o[h,0],y=o[h,1],head=o[h,23],upright=o[h,38],vx=o[h,39],vy=o[h,40],outside=not np.array_equal(((o[h,:2]+6)/4).astype(int)[::-1],[6,5]),stable=o[h,23]>=1.2 and o[h,38]>=.7,wall_contacts=z['wall_contacts'][h],action_rms=float(np.sqrt(np.mean(a[h]**2))) if h<200 else '',action_change_rms=float(d[h]) if 0<h<200 else ''))

# Time-ordered comparisons; traces, not action-sign interpretations.
for state in STATES:
 fig,axes=plt.subplots(4,4,figsize=(16,10),sharex=True,layout='constrained')
 lines=[('first_action','original_actor','GFP original','#777777'),('first_action','actor_2','Actor alternative','#0072B2'),('first_action','vabc_2','VaBC first action','#D55E00')] if state==STATES[0] else [('first_action','original_actor','GFP original','#777777'),('first_action','rebrac','ReBRAC first action','#56B4E9'),('short_policy','rebrac','ReBRAC continuous','#009E73')]
 for rep in range(4):
  for mode,label,title,color in lines:
   z=arrays[(state,mode,label,rep)];o=z['observations']
   vals=[o[:,0],np.convolve(o[:,39],np.ones(10)/10,'valid'),o[:,23],o[:,38]]
   times=[np.arange(201),np.arange(9,201),np.arange(201),np.arange(201)]
   for j in range(4):axes[j,rep].plot(times[j],vals[j],label=title,color=color,lw=1.25,alpha=.9)
  axes[0,rep].set_title(f'{state} / continuation {rep}');axes[0,rep].axhline(18,color='black',ls='--',lw=.8);axes[0,rep].axhline(14.75,color='gray',ls=':',lw=.6)
  axes[1,rep].axhline(0,color='black',lw=.6);axes[2,rep].axhline(1.2,color='black',ls='--',lw=.6);axes[3,rep].axhline(.7,color='black',ls='--',lw=.6);axes[3,rep].set_xlabel('Steps since shared state')
 for j,n in enumerate(['Root x (m)','COM vx (m/s), trailing10','Head height (m)','Torso upright']):axes[j,0].set_ylabel(n)
 axes[0,0].legend(fontsize=8,loc='best');fig.savefig(O/f'{state}_timeline.png',dpi=125);plt.close(fig)

receipt=dict(status='PASS',source_branches_total=len(allrows),branches_read=len(rows),all_selected_hashes_verified=True,source_models_loaded=0,new_environment_steps=0,optimizer_updates=0,seconds=time.time()-start,cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,inputs=identities)
save('receipt.json',receipt)
print('DONE',len(rows),time.time()-start)
for m in metrics:
 if (m['state']==STATES[0] and m['candidate'] in ['original_actor','actor_2','vabc_2'] and m['mode']=='first_action') or (m['state']==STATES[1] and m['candidate'] in ['original_actor','rebrac']):
  print(m['state'],m['mode'],m['candidate'],m['repeat'],'exit',m['first_exit'],'return',m['first_return'],'west_after_exit',m['first_west10_after_exit'],'low_after_exit',m['first_low10_after_exit'],'outside_final',m['outside_final'],'x_final',round(m['final_x'],3))
