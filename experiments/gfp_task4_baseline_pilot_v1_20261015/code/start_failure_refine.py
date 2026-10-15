from pathlib import Path
import json,time
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'reports/start_failure'
def load(p):return json.loads(Path(p).read_text())
def mean(a):return float(np.mean(a)) if len(a) else None
def save(n,a):(O/n).write_text(json.dumps(a,indent=2,allow_nan=False))
t=time.time();episodes=load(O/'episodes.json');directions=[]
for row in episodes:
 if row['checkpoint']!=1000000:continue
 alg=row['algorithm'];s=row['seed'];ep=row['episode']
 with np.load(R/'runs'/f'{alg}_seed{s}'/'matched/1000000'/f'episode_{ep:03d}.npz') as z:o=z['observations']
 onset=row['first_stable20'];end=row['exit_index'] if row['exit_index'] is not None else len(o)-1
 stable=(o[:,23]>=1.2)&(o[:,38]>=.7);delta=[];offwall=[]
 for k in range(0,max(0,end-20),20):
  if stable[k:k+21].all():
   dx=float(o[k+20,0]-o[k,0]);delta.append(dx)
   if o[k,0]>14.75:offwall.append(dx)
 directions.append(dict(algorithm=alg,seed=s,episode=ep,success=row['success'],first_stable20=onset,
  dx50_after_stable=None if onset is None else float(o[min(onset+50,len(o)-1),0]-o[onset,0]),
  stable20_block_dx_mean=mean(delta),offwall_stable20_block_dx_mean=mean(offwall),
  stable20_blocks=len(delta),offwall_stable20_blocks=len(offwall),
  west_before_east=row['first_west_075'] is not None and (row['first_east_075'] is None or row['first_west_075']<row['first_east_075']),
  west_before_stable=row['first_west_075'] is not None and (onset is None or row['first_west_075']<onset)))
summary=[]
for alg in ['GFP','ReBRAC']:
 for s in [0,1]:
  a=[x for x in directions if x['algorithm']==alg and x['seed']==s]
  d=dict(algorithm=alg,seed=s,n=len(a),west_before_east=sum(x['west_before_east'] for x in a),west_before_stable=sum(x['west_before_stable'] for x in a))
  for k in ['dx50_after_stable','stable20_block_dx_mean','offwall_stable20_block_dx_mean']:
   v=[x[k] for x in a if x[k] is not None];d[k]=dict(mean=mean(v),median=float(np.median(v)) if v else None,n=len(v),negative=sum(vv<0 for vv in v))
  summary.append(d)
save('posture_conditioned_motion.json',dict(summary=summary,episodes=directions,block_length=20,offwall_x_gt=14.75))

def cluster_fit(X,y,tid,within=True):
 unique=np.unique(tid);Xs=[];ys=[]
 for u in unique:
  x=X[tid==u].astype(float);v=y[tid==u].astype(float)
  if within:x-=x.mean(0);v-=v.mean()
  else:x=np.c_[np.ones(len(x)),x]
  Xs.append(x);ys.append(v)
 xx=np.array([x.T@x for x in Xs]);xy=np.array([x.T@y for x,y in zip(Xs,ys)])
 beta=np.linalg.lstsq(xx.sum(0),xy.sum(0),rcond=None)[0];rng=np.random.default_rng(2026101522);bs=[]
 for _ in range(200):
  k=rng.integers(0,len(unique),len(unique));bs.append(np.linalg.lstsq(xx[k].sum(0),xy[k].sum(0),rcond=None)[0])
 return dict(coef=beta.tolist(),ci95=np.quantile(bs,[.025,.975],axis=0).T.tolist(),n=len(y),trajectories=len(unique),within_trajectory=within)
out=[]
for split,suffix in [('train',''),('validation','-val')]:
 with np.load(R/'arrays'/f'{split}_samples.npz') as z:f=dict(z)
 with np.load(R/'data'/f'humanoidmaze-medium-navigate-v0{suffix}.npz') as z:raw=z['observations'];term=z['terminals'].astype(bool)
 idx=f['indices'];ends=np.flatnonzero(term);ok=idx+100<=ends[f['trajectory']];future=raw[np.minimum(idx+100,len(raw)-1),0]-raw[idx,0]
 assert np.allclose(raw[idx],f['observations'])
 o=f['observations'];cell=((o[:,:2]+6)/4).astype(int)[:,::-1];initial=np.all(cell==[6,5],axis=1)
 cov=np.c_[o[:,0],o[:,1],o[:,23],o[:,38],o[:,39],o[:,40],np.linalg.norm(o[:,39:42],axis=1)]
 for s in [0,1]:
  with np.load(R/'arrays'/f's{s}_1000000.npz') as z:
   mask=(z['repeat']==0)&(z['split']==int(split=='validation'));g=z['g'][mask];qd=z['qdata'][mask].mean(1)
  m=initial&ok&(abs(future)>.75);X=np.c_[(future>.75).astype(float),cov]
  a=cluster_fit(X[m],g[m],f['trajectory'][m]);b=cluster_fit(cov[initial],qd[initial],f['trajectory'][initial])
  out.append(dict(split=split,seed=s,weight_east_vs_west=a,qdata_covariates=b,
   weight_features=['east100_vs_west100','x','y','head_height','torso_zz','vx','vy','speed'],
   q_features=['x','y','head_height','torso_zz','vx','vy','speed'],interpretation='Conditional observational associations, not true advantage or causal effects; collection goal unobserved'))
save('controlled_data_associations.json',out)
save('refinement_receipt.json',dict(status='PASS',new_environment_steps=0,new_model_evaluations=0,optimizer_updates=0,wall_seconds=time.time()-t))
print('MOTION',summary)
for a in out:print('ASSOC',a['split'],a['seed'],'east_guidance',a['weight_east_vs_west']['coef'][0],a['weight_east_vs_west']['ci95'][0],'x_Q',a['qdata_covariates']['coef'][0],a['qdata_covariates']['ci95'][0])
