"""Frozen, bounded proposal/fit inference. No environment or optimizer calls."""
from pathlib import Path
import os,sys,json,time,hashlib,resource
R=Path(__file__).resolve().parents[1]
P=Path('/home/xushijie3/work/gfp_task4_baseline_pilot_v1_20261015')
Z=Path('/home/xushijie3/work/gfp_guidance_distance_z01_20261015')
C=Path('/home/xushijie3/work/flow_critic_comparative_train_v1_20261014')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.update(COMPARATIVE_ROOT=str(R),COMPARATIVE_JOB='frozen_global_probe')
sys.path.insert(0,str(P/'workspaces/gfp'))
import numpy as np,jax,jax.numpy as jnp,gymnasium as gym,mujoco
calls=dict(environment_steps=0,optimizer_updates=0,forbidden_attempts=0)
def forbidden(*a,**k):
 calls['forbidden_attempts']+=1
 raise RuntimeError('Global diagnosis forbids environment construction/steps and updates')
gym.make=forbidden;gym.Env.step=forbidden;gym.Env.reset=forbidden;mujoco.mj_step=forbidden
from agents.gfp import GFPAgent
from training_wiring import restore_state,state_digest,digest
from utils.flax_utils import TrainState
for n in ['update','batch_update','multi_sample_and_update','target_update']:
 if hasattr(GFPAgent,n):setattr(GFPAgent,n,forbidden)
for n in ['apply_loss_fn','apply_gradients']:
 if hasattr(TrainState,n):setattr(TrainState,n,forbidden)
assert digest(P/'workspaces/gfp/agents/gfp.py')==digest(C/'workspaces/gfp_floq/agents/gfp.py')
@jax.jit
def measure(agent,obs,act,key):
 k1,k2,k3=jax.random.split(key,3)
 noise=jax.random.normal(k1,act.shape)
 raw=agent.network.select('actor')(obs,noise);aa=jnp.clip(raw,-1,1)
 va=agent.compute_flow_actions(obs,noises=noise)
 x0=jax.random.normal(k2,act.shape);t=jax.random.uniform(k3,(len(obs),1));xt=(1-t)*x0+t*act
 v=agent.network.select('vabc')(obs,xt,t)
 qd=agent.network.select('target_critic')(obs,act).T
 qa=agent.network.select('target_critic')(obs,aa).T
 qv=agent.network.select('target_critic')(obs,va).T
 return dict(actor_mse=jnp.mean((aa-act)**2,axis=-1),vabc_mse=jnp.mean((va-act)**2,axis=-1),actor_vabc_mse=jnp.mean((raw-va)**2,axis=-1),vabc_flow_mse=jnp.mean((v-(act-x0))**2,axis=-1),qdata=qd,qactor=qa,qvabc=qv)
start=time.time();records=[];trials=0
for task in [1,4]:
 samples={s:dict(np.load(R/'arrays'/f't{task}_{s}_selected.npz')) for s in ['train','validation']}
 for seed in [0,1]:
  runtime=P/'runs'/f'GFP_seed{seed}'/'runtime.json' if task==4 else Z/'configs'/f'G0_seed{seed}_runtime.json'
  rt=json.loads(runtime.read_text());cfg=rt['agent_config']
  template=GFPAgent.create(seed,samples['train']['observations'][:1],samples['train']['actions'][:1],dict(cfg))
  for cp in [200000,600000,1000000]:
   then=time.time();ck=Path(rt['output_dir'])/f'params_{cp}.pkl';ckhash=digest(ck)
   agent=restore_state(template,Path(rt['output_dir']),cp);before=state_digest(agent);adam=int(agent.network.opt_state[0].count);step=int(agent.network.step)
   assert adam==cp and step==cp+1
   for split,dat in samples.items():
    results=[]
    for lo in range(0,len(dat['indices']),256):
     obs=dat['observations'][lo:lo+256];act=dat['actions'][lo:lo+256];n=len(obs)
     # Fixed-size JIT, padded duplicates discarded; separately account for their inference.
     pad=256-n;ob=np.pad(obs,((0,pad),(0,0)),mode='edge');ac=np.pad(act,((0,pad),(0,0)),mode='edge')
     rr=[]
     for trial in range(8):
      key=jax.random.fold_in(jax.random.PRNGKey(20261016),trial+8*(lo//256)+100*(split=='validation'))
      out={k:np.asarray(v)[:n] for k,v in measure(agent,ob,ac,key).items()}
      assert all(np.isfinite(v).all() for v in out.values());rr.append(out);trials+=n
     results.append({k:np.stack([x[k] for x in rr],axis=1) for k in rr[0]})
    out={k:np.concatenate([x[k] for x in results]) for k in results[0]}
    np.savez_compressed(R/'arrays'/f't{task}_s{seed}_{cp}_{split}_probe.npz',**out)
   assert state_digest(agent)==before and int(agent.network.opt_state[0].count)==adam and int(agent.network.step)==step and digest(ck)==ckhash
   records.append(dict(task=task,seed=seed,checkpoint=cp,path=str(ck),checkpoint_sha256=ckhash,state_sha256=before,parameter_optimizer_rng_unchanged=True,seconds=time.time()-then,selected_states=sum(len(x['indices']) for x in samples.values())))
   assert trials<=196608
   (R/'receipts/probe_progress.json').write_text(json.dumps(dict(records=records,state_noise_trials=trials,seconds=time.time()-start,**calls),indent=2))
   print('MODEL_DONE',task,seed,cp,round(time.time()-then,2),flush=True)
receipt=dict(status='PASS',records=records,state_noise_trials=trials,padded_state_noise_trials=sum(8*256*((len(x['indices'])+255)//256)-8*len(x['indices']) for task in [1,4] for x in [dict(np.load(R/'arrays'/f't{task}_{s}_selected.npz')) for s in ['train','validation']])*6,seconds=time.time()-start,cpu_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,cpu_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,devices=[str(x) for x in jax.devices()],jax_enable_x64=jax.config.jax_enable_x64,matmul_precision=str(jax.config.jax_default_matmul_precision),source_sha256=digest(P/'workspaces/gfp/agents/gfp.py'),**calls)
(R/'receipts/probe.json').write_text(json.dumps(receipt,indent=2))
