from pathlib import Path
import os,sys,json,time,hashlib,resource
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
R=Path(__file__).resolve().parents[1]; P=Path('/home/xushijie3/work/flow_critic_comparative_train_v1_20261014')
os.environ.update(COMPARATIVE_ROOT=str(P),COMPARATIVE_JOB='G0_seed0')
sys.path.insert(0,str(P/'workspaces/gfp_floq'))
import numpy as np,jax,jax.numpy as jnp,gymnasium as gym,mujoco
calls={'environment_steps':0,'optimizer_updates':0,'forbidden_attempts':0}
def forbidden(*a,**k):
 calls['forbidden_attempts']+=1
 raise RuntimeError('Z01 forbids environment construction/interaction and optimizer updates')
gym.make=forbidden;gym.Env.step=forbidden;gym.Env.reset=forbidden;mujoco.mj_step=forbidden
from agents.gfp import GFPAgent
from training_wiring import restore_state,state_digest,digest
for name in ['update','batch_update','multi_sample_and_update','target_update']:
 if hasattr(GFPAgent,name):setattr(GFPAgent,name,forbidden)
from utils.flax_utils import TrainState
for name in ['apply_loss_fn','apply_gradients']:
 if hasattr(TrainState,name):setattr(TrainState,name,forbidden)
assert digest(P/'workspaces/gfp_floq/agents/gfp.py')==digest(R/'sources/gfp/agents/gfp.py')
protocol=json.loads((R/'configs/protocol.json').read_text()); samples={s:dict(np.load(R/'arrays'/f'{s}_samples.npz')) for s in ['train','validation']}
@jax.jit
def measure(agent,obs,dataact,key):
 _,distill_rng,_,_=jax.random.split(key,4)
 noise=jax.random.normal(distill_rng,dataact.shape)
 aa=jnp.clip(agent.network.select('actor')(obs,noise),-1,1)
 va=agent.compute_flow_actions(obs,noises=noise)
 qo=agent.network.select('critic')(obs,actions=aa)
 norm=jnp.abs(qo.mean(axis=0)).mean()
 qd=agent.network.select('target_critic')(obs,dataact)
 qa=agent.network.select('target_critic')(obs,aa)
 qv=agent.network.select('target_critic')(obs,va)
 choose_a=qa.mean(0)<=qv.mean(0)
 qr=jnp.where(choose_a[None,:],qa,qv)
 delta=qd.mean(0)-qr.mean(0); denom=norm*agent.config['eta_temperature']
 g,_=agent.compute_guidance({'observations':obs,'actions':dataact},aa,va,1/norm)
 stable=jax.nn.sigmoid(delta/denom)
 return dict(g=g,delta=delta,logit=delta/denom,denom=jnp.full_like(g,denom),qdata=qd.T,qactor=qa.T,qvabc=qv.T,qonline=qo.T,reference_actor=choose_a,member_delta=(qd-qr).T,actor_data_l2=jnp.linalg.norm(aa-dataact,axis=1),vabc_data_l2=jnp.linalg.norm(va-dataact,axis=1),formula_error=jnp.abs(g-stable))
start=time.time();records=[]; count=0; maincount=0
for seed in [0,1]:
 rt=json.loads((R/'configs'/f'G0_seed{seed}_runtime.json').read_text()); cfg=rt['agent_config']
 template=GFPAgent.create(seed,samples['train']['observations'][:1],samples['train']['actions'][:1],dict(cfg))
 for cp in [200000,600000,1000000]:
  agent=restore_state(template,Path(rt['output_dir']),cp)
  old=state_digest(agent);oldstep=int(agent.network.step); adam=int(agent.network.opt_state[0].count); ck=Path(rt['output_dir'])/f'params_{cp}.pkl'; ckhash=digest(ck)
  assert oldstep==cp+1 and adam==cp
  then=time.time();err=0.; entries=[]
  jobs=[(0,s,i,protocol['action_rng_base']) for i in range(100) for s in ['train','validation']]
  jobs += [(rep+1,s,i,base) for rep,base in enumerate(protocol['extra_rng_repeats']['base_seeds']) for i in range(10) for s in ['train','validation']]
  for k,(rep,split,i,base) in enumerate(jobs):
   sl=slice(i*256,(i+1)*256);d=samples[split]
   key=jax.random.fold_in(jax.random.PRNGKey(base),i+(100 if split=='validation' else 0))
   result={n:np.asarray(v) for n,v in measure(agent,d['observations'][sl],d['actions'][sl],key).items()}
   if not all(np.all(np.isfinite(v)) for v in result.values()):raise RuntimeError('Nonfinite')
   err=max(err,float(result['formula_error'].max()))
   if err>.001:raise RuntimeError('Official formula mismatch')
   result.update(split=np.full(256,int(split=='validation')),repeat=np.full(256,rep),batch=np.full(256,i),sample_index=np.arange(i*256,(i+1)*256))
   entries.append(result);count+=256;maincount+=256*(rep==0)
   assert maincount<=307200 and count<=368640
   if k==19:
    assert state_digest(agent)==old
    print('PILOT_PASS',seed,cp,'20 batches',err,flush=True)
   if k in [19,99,199,239]:
    (R/'receipts/progress.json').write_text(json.dumps(dict(seed=seed,checkpoint=cp,batches_done=k+1,state_evaluations=count,primary_evaluations=maincount,elapsed_seconds=time.time()-start,**calls)))
  np.savez_compressed(R/'arrays'/f's{seed}_{cp}.npz',**{n:np.concatenate([x[n] for x in entries],axis=0) for n in entries[0]})
  assert state_digest(agent)==old and int(agent.network.step)==oldstep and int(agent.network.opt_state[0].count)==adam and digest(ck)==ckhash
  records.append(dict(seed=seed,checkpoint=cp,state_sha256=old,checkpoint_sha256=ckhash,model_adam_rng_unchanged=True,optimizer_steps_before=adam,optimizer_steps_after=adam,seconds=time.time()-then,formula_max_error=err))
  print('MODEL_DONE',seed,cp,records[-1]['seconds'],flush=True)
(R/'receipts/probe_result.json').write_text(json.dumps(dict(status='PASS',records=records,primary_evaluations=maincount,total_evaluations=count,seconds=time.time()-start,cpu_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,cpu_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,devices=[str(x) for x in jax.devices()],jax_enable_x64=jax.config.jax_enable_x64,matmul_precision=str(jax.config.jax_default_matmul_precision),**calls),indent=2))
