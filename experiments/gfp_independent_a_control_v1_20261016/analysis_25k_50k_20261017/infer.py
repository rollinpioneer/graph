import os,sys,time,json,pickle,resource
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp,flax
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
sys.path.insert(0,str(ROOT/'code'))
from common import make_agent,digest,sha,SUPPORT
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False))
t0=time.time();template,_,_=make_agent();agents={};before={}
for it in (25000,50000):
 p=ROOT/f'runs/A/checkpoint_{it}.pkl';before[str(p)]=sha(p)
 assert before[str(p)]==json.loads(p.with_suffix('.json').read_text())['sha256']
 with p.open('rb') as f:payload=pickle.load(f)
 a=flax.serialization.from_state_dict(template,payload['agent']);assert int(a.network.step)==350001+it
 agents[it]=a
states_before={str(it):digest(a) for it,a in agents.items()}
query=np.load(SUPPORT/'arrays/queries.npz');noise=np.load(SUPPORT/'arrays/noise.npz');obs=query['obs'];n=len(obs)
@jax.jit
def actions(a,o,z):
 raw=a.network.select('actor')(o,z)
 params=dict(a.network.params);params['modules_actor']=a.reference
 return raw,jnp.clip(raw,-1,1),a.compute_flow_actions(o,z),jnp.clip(a.network.select('actor')(o,z,params=params),-1,1)
@jax.jit
def score(a,o,x):return a.network.select('critic')(o,actions=x)
@jax.jit
def grad(a,o,x):return jax.grad(lambda y:a.network.select('critic')(o,actions=y).mean(0).sum())(x)
arrays={};batch=256
for it,a in agents.items():
 for zi,key in enumerate(('z0','z1')):
  parts=[[] for _ in range(4)]
  for start in range(0,n,batch):
   o=jnp.asarray(obs[start:start+batch]);z=jnp.asarray(noise[key][start:start+batch]);v=actions(a,o,z)
   for dest,x in zip(parts,v):dest.append(np.asarray(x))
  for name,vals in zip(('raw','actor','vabc','reference'),parts):arrays[f'{it}_{zi}_{name}']=np.concatenate(vals)
for cit,a in agents.items():
 for ait in agents:
  for zi in range(2):
   for name in ('actor','vabc','reference'):
    x=arrays[f'{ait}_{zi}_{name}'];parts=[np.asarray(score(a,jnp.asarray(obs[k:k+batch]),jnp.asarray(x[k:k+batch]))) for k in range(0,n,batch)]
    arrays[f'q{cit}_{ait}_{zi}_{name}']=np.concatenate(parts,axis=1)
 for zi in range(2):
  x=arrays[f'25000_{zi}_actor'];arrays[f'grad{cit}_{zi}']=np.concatenate([np.asarray(grad(a,jnp.asarray(obs[k:k+batch]),jnp.asarray(x[k:k+batch]))) for k in range(0,n,batch)])
assert all(np.isfinite(x).all() for x in arrays.values())
np.savez_compressed(OUT/'frozen_outputs.npz',**arrays)
rows=[]
for split in np.unique(query['split']):
 for region in (0,1):
  mask=(query['split']==split)&(query['region']==region);row=dict(split=str(split),region='start' if region else 'outside',states=int(mask.sum()),noise_repeats=2)
  for name in ('actor','vabc','reference'):
   delta=np.stack([arrays[f'50000_{z}_{name}']-arrays[f'25000_{z}_{name}'] for z in range(2)])[:,mask]
   row[name+'_change_rms']=float(np.sqrt(np.mean(delta**2)))
  for it in agents:
   aa=np.stack([arrays[f'{it}_{z}_actor'] for z in range(2)])[:,mask];vv=np.stack([arrays[f'{it}_{z}_vabc'] for z in range(2)])[:,mask];raw=np.stack([arrays[f'{it}_{z}_raw'] for z in range(2)])[:,mask]
   row[f'{it}_actor_vabc_rms']=float(np.sqrt(np.mean((aa-vv)**2)))
   row[f'{it}_raw_distill_mse']=float(np.mean((raw-vv)**2))
   row[f'{it}_actor_noise_rms']=float(np.sqrt(np.mean((aa[0]-aa[1])**2)))
   row[f'{it}_actor_clip_fraction']=float(np.mean(np.abs(aa)>=1))
   qnew=np.stack([arrays[f'q{it}_50000_{z}_actor'] for z in range(2)])[:,:,mask]
   qold=np.stack([arrays[f'q{it}_25000_{z}_actor'] for z in range(2)])[:,:,mask]
   delta=qnew-qold
   row[f'critic{it}_new_minus_old_actor_Q_mean']=float(delta.mean())
   row[f'critic{it}_new_actor_preferred_fraction']=float((delta.mean(1)>0).mean())
   row[f'critic{it}_ensemble_order_agreement']=float(((delta[:,0]>0)==(delta[:,1]>0)).mean())
  g0=np.stack([arrays[f'grad25000_{z}'] for z in range(2)])[:,mask];g1=np.stack([arrays[f'grad50000_{z}'] for z in range(2)])[:,mask]
  norm0=np.linalg.norm(g0,axis=-1);norm1=np.linalg.norm(g1,axis=-1);cos=(g0*g1).sum(-1)/(norm0*norm1+1e-12)
  row['fixed_25k_action_gradient_cos_mean']=float(cos.mean());row['gradient_cos_negative_fraction']=float((cos<0).mean());row['gradient_norm_mean_ratio']=float(norm1.mean()/norm0.mean())
  qd=np.stack([arrays[f'q50000_25000_{z}_actor']-arrays[f'q25000_25000_{z}_actor'] for z in range(2)])[:,:,mask]
  row['fixed_action_Q_drift_mean']=float(qd.mean());row['fixed_action_Q_drift_abs_p95']=float(np.quantile(np.abs(qd.mean(1)),.95))
  rows.append(row)
save('frozen_comparison.json',rows)
assert all(digest(a)==states_before[str(it)] for it,a in agents.items())
assert all(sha(p)==h for p,h in before.items())
save('frozen_receipt.json',dict(parameter_updates=0,new_environment_steps=0,checkpoints_unchanged=True,model_optimizer_rng_trees_unchanged=True,checkpoints=before,query_sha256=sha(SUPPORT/'arrays/queries.npz'),noise_sha256=sha(SUPPORT/'arrays/noise.npz'),states=n,noise_repeats=2,device=str(jax.devices()),seconds=time.time()-t0,cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(json.dumps(rows))
