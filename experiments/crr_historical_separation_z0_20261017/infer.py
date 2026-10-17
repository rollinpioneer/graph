"""Read checkpoints, apply saved weights and differentiate actions only. No updates."""
from pathlib import Path
import sys,os,json,pickle,hashlib,copy,time,resource,importlib.metadata as md
import numpy as np
R=Path(__file__).resolve().parent
sys.path.insert(0,'/home/xushijie3/work/gfp_eta0_ablation_20261015/source/gfp')
calls=dict(optimizer_updates=0,environment_constructions=0,environment_resets=0,environment_steps=0,new_model_initializations=0,forbidden_attempts=0)
def forbidden(*a,**kw):
 calls['forbidden_attempts']+=1;raise RuntimeError('CRR-Z0 forbidden operation')
import gymnasium as gym,mujoco
gym.make=forbidden;gym.Env.reset=forbidden;gym.Env.step=forbidden;mujoco.mj_step=forbidden
import jax,jax.numpy as jnp,flax
from flax import linen as nn
from agents.gfp import GFPAgent
from utils.flax_utils import ModuleDict,TrainState
from utils.networks import Value,ActorVectorField
for cls,names in [(GFPAgent,['create','update','batch_update','multi_sample_and_update','target_update']),(TrainState,['apply_loss_fn','apply_gradients'])]:
 for name in names:
  if hasattr(cls,name):setattr(cls,name,forbidden)
nn.Module.init=forbidden
P=json.loads((R/'protocol.json').read_text());rows=json.loads((R/'checkpoint_manifest.json').read_text());cfgs=json.loads((R/'configs.json').read_text());D=dict(np.load(R/'arrays/samples.npz'));N=4096
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):(R/n).write_text(json.dumps(x,indent=2,allow_nan=False))
def digest(x):
 h=hashlib.sha256()
 for path,v in jax.tree_util.tree_flatten_with_path(x)[0]:
  v=np.asarray(v);h.update(str(path).encode());h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(v.tobytes())
 return h.hexdigest()
def read(row):
 assert sha(row['path'])==row['sha256']
 with open(row['path'],'rb') as f:payload=pickle.load(f)
 p=payload['agent'];assert int(p['network']['step'])==row['step']+1 and int(p['network']['opt_state']['0']['count'])==row['step']
 if row['delta']:
  assert payload['branch']==row['arm'] and payload['start']==row['fork']
 return p,digest(p)
C=Value((512,)*4,layer_norm=True,num_ensembles=2)
A=ActorVectorField((512,)*4,action_dim=21,layer_norm=False,need_time=False)
V=ActorVectorField((512,)*4,action_dim=21,layer_norm=False,need_time=True,encode_time_dim=64)
@jax.jit
def actions(params,obs,nobs,z):
 aa=jnp.clip(A.apply({'params':params['modules_actor']},nobs,z),-1,1)
 va=z
 for i in range(10):
  tt=jnp.full((*obs.shape[:-1],1),i/10)
  va=va+V.apply({'params':params['modules_vabc']},obs,va,tt,is_encoded=True)/10
 return aa,jnp.clip(va,-1,1)
def q(p,obs,a):return C.apply({'params':p},obs,a).mean(0)
@jax.jit
def score(tp,op,obs,aa,va):
 qa=q(tp,obs,aa);qv=q(tp,obs,va)
 g=jax.grad(lambda acts:jnp.sum(q(op,obs,acts)))(aa)
 return (qa+qv)/2,qa,qv,jnp.linalg.norm(g,axis=-1)
def native(p,seed):
 cfg=dict(cfgs[str(seed)]['agent']);assert cfg['q_agg']=='mean' and cfg['target_agg']=='mean' and cfg['guidance_agg']=='min' and not cfg['target_use_next']
 assert cfg['encoder'] is None and cfg['flow_steps']==10 and tuple(cfg['actor_hidden_dims'])==(512,)*4 and tuple(cfg['value_hidden_dims'])==(512,)*4
 cfg.update(ob_dims=(69,),action_dim=21,log_metrics=True)
 if cfg['eta_temperature_online'] is None:cfg['eta_temperature_online']=cfg['eta_temperature']
 definition=ModuleDict(dict(critic=copy.deepcopy(C),target_critic=copy.deepcopy(C),actor=copy.deepcopy(A),vabc=copy.deepcopy(V)))
 net=TrainState(step=p['network']['step'],apply_fn=definition.apply,model_def=definition,params=p['network']['params'],tx=None,opt_state=None)
 return GFPAgent(rng=p['rng'],network=net,config=flax.core.FrozenDict(cfg))
@jax.jit
def native_actions(g,obs,nobs,z):
 return jnp.clip(g.network.select('actor')(nobs,z),-1,1),g.compute_flow_actions(obs,z)
@jax.jit
def native_score(g,nobs,aa,va):
 t=g.value_with_all_options('mean',nobs,aa,va)
 grad=jax.grad(lambda acts:jnp.sum(g.network.select('critic')(nobs,acts).mean(0)))(aa)
 return t,jnp.linalg.norm(grad,axis=-1),g.network.select('target_critic')(nobs,aa).mean(0)
start=time.monotonic();valid=[];integrity=[];fixed={}
for seed in [0,3]:
 row=next(r for r in rows if r['seed']==seed and r['delta']==0);p,old=read(row);par=p['network']['params'];g=native(p,seed);out=[]
 for j in range(2):
  aa=[];va=[]
  for k in range(0,N,256):
   ar,vr=actions(par,D['observations'][k:k+256],D['next_observations'][k:k+256],D[f'z{j}'][k:k+256]);aa.append(np.asarray(ar));va.append(np.asarray(vr))
   if k==0:
    na,nv=native_actions(g,D['observations'][:256],D['next_observations'][:256],D[f'z{j}'][:256]);err=max(float(np.max(np.abs(na-ar))),float(np.max(np.abs(nv-vr))));assert err<=3e-5
    valid.append(dict(seed=seed,latent=j,actor_vabc_native_max_abs_error=err))
  out.append(np.stack([np.concatenate(aa),np.concatenate(va)]))
 fixed[seed]=np.stack(out);np.save(R/f'arrays/s{seed}_fixed_actions.npy',fixed[seed]);assert digest(p)==old
 save(f'receipts/s{seed}_fixed_action_identity.json',dict(shape=list(fixed[seed].shape),sha256=sha(R/f'arrays/s{seed}_fixed_actions.npy'),fork_checkpoint_sha256=row['sha256'],sample_sha256=sha(R/'arrays/samples.npz'),actions_generated_once=True,clip_end_only=True))
 del p,g
for n,row in enumerate(rows):
 p,old=read(row);par=p['network']['params'];out=[]
 for j in range(2):
  parts=[]
  for k in range(0,N,256):
   aa,va=fixed[row['seed']][j,:,k:k+256];o=D['next_observations'][k:k+256]
   vals=score(par['modules_target_critic'],par['modules_critic'],o,aa,va);vals=np.stack([np.asarray(v) for v in vals]);assert vals.shape==(4,256) and np.isfinite(vals).all();parts.append(vals)
   if k==0 and j==0 and row['delta'] in [0,100000]:
    g=native(p,row['seed']);nt,ng,na=native_score(g,o,aa,va)
    errs=[float(np.max(np.abs(nt-vals[0]))),float(np.max(np.abs(ng-vals[3]))),float(np.max(np.abs(na-vals[1])))];assert max(errs)<=3e-5,errs
    valid.append(dict(id=row['id'],native_target_gradient_mapping_max_abs_errors=errs,tolerance=3e-5));del g
  out.append(np.concatenate(parts,axis=1))
 arr=np.stack(out);np.save(R/f'arrays/{row["id"]}.npy',arr)
 assert digest(p)==old and sha(row['path'])==row['sha256'];integrity.append(dict(id=row['id'],step=row['step'],Adam_count=row['step'],checkpoint_sha256=row['sha256'],payload_digest=old,model_optimizer_rng_unchanged=True))
 save('receipts/progress.json',dict(completed=n+1,total=len(rows),last=row['id'],seconds=time.monotonic()-start,**calls));print('SCORED',row['id'],round(time.monotonic()-start,2),flush=True);del p,par
for seed in [0,3]:
 x=np.load(R/f'arrays/s{seed}_B0_000000.npy');y=np.load(R/f'arrays/s{seed}_B1a_000000.npy');assert np.array_equal(x,y)
assert calls['forbidden_attempts']==0
save('receipts/validation.json',dict(status='PASS',native_checks=valid,fork_B0_B1a_bitwise_equal=True,online_gradient_not_target_guidance_min=True,**calls));save('receipts/integrity.json',integrity)
save('receipts/inference.json',dict(status='COMPLETE',models=len(rows),distinct_checkpoint_files=42,fixed_states=N,latent_repeats=2,model_state_latent_evaluations=len(rows)*N*2,seconds=time.monotonic()-start,cpu_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,cpu_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,devices=[str(x) for x in jax.devices()],versions={k:md.version(k) for k in ['ogbench','numpy','jax','jaxlib','flax','optax']},**calls))
print('COMPLETE',flush=True)
