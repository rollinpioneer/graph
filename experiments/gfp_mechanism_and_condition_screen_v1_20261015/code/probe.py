"""Frozen inference; 12 states x 3 single-action interventions; no updates."""
import os,sys,json,time,random,hashlib,traceback,resource
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=Path('/home/xushijie3/work/flow_critic_comparative_train_v1_20261014')
os.environ.update(COMPARATIVE_ROOT=str(P),COMPARATIVE_JOB='G0_seed0')
sys.path.insert(0,str(P/'workspaces/gfp_floq'))
import numpy as np,jax,jax.numpy as jnp,mujoco,ogbench
from agents.floq_adapter import create_agent
from training_wiring import restore_state,state_digest,digest
S=json.loads((R/'configs/states.json').read_text());task='humanoidmaze-medium-navigate-singletask-task1-v0'
def save(p,o):p.write_text(json.dumps(o,indent=2))
def sha(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
assert not (R/'receipts/probe_result.json').exists()
env=ogbench.make_env_and_datasets(task,env_only=True);u=env.unwrapped;spec=mujoco.mjtState.mjSTATE_INTEGRATION
assert u._teleport_info is None
agents={};assets=[];start=time.time();envsteps=0;results=[];distributions=[]
for seed in (0,1):
 rt=json.loads((P/f'runs/G0_seed{seed}/runtime.json').read_text());cfg=rt['agent_config']
 f=np.load(P/'data'/f'{task}.batch.npz');template=create_agent('gfp',seed,f['observations'][:1],f['actions'][:1],cfg)
 a=restore_state(template,Path(rt['output_dir']),1000000);ref=next(s for s in S['states'] if s['seed']==seed)['reference']['case']
 cp=Path(rt['output_dir'])/'params_1000000.pkl';sd=state_digest(a);ch=digest(cp)
 assert sd==ref['state_sha256'] and ch==ref['checkpoint_sha256']
 agents[seed]=a;assets.append(dict(seed=seed,path=str(cp),checkpoint_sha256=ch,state_sha256=sd,config=cfg))
save(R/'receipts/assets_before.json',assets)
save(R/'receipts/device.json',dict(devices=[str(d) for d in jax.devices()],gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),pid=os.getpid(),jax=jax.__version__,mujoco=mujoco.__version__,precision=str(jax.config.jax_default_matmul_precision),x64=bool(jax.config.jax_enable_x64)))
geom=[mujoco.mj_id2name(u.model,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(u.model.ngeom)]
def snapshot(ob):
 state=np.empty(mujoco.mj_stateSize(u.model,spec));mujoco.mj_getState(u.model,u.data,state,spec)
 return dict(observations=np.array(ob),integration=state,qpos=u.data.qpos.copy(),qvel=u.data.qvel.copy(),xy=u.get_xy().copy(),head_height=float(u.data.xpos[2,2]),torso_up=float(u.data.xmat[1,8]))
def restore(c,f):
 ref=c['reference']['case']['reference'];np.random.seed(ref['numpy_seed']);random.seed(ref['python_seed'])
 ob,_=env.reset(seed=ref['reset_seed']);assert sha(ob)==ref['initial_observation_sha256']
 u.set_goal(goal_xy=f['goal_xy'][c['step']].copy());u.np_random.bit_generator.state=c['reference']['initial_env_rng']
 mujoco.mj_setState(u.model,u.data,f['integration'][c['step']],spec)
 mujoco.mj_step1(u.model,u.data)
 e=env
 while True:
  if '_elapsed_steps' in vars(e):e._elapsed_steps=c['step']
  if not hasattr(e,'env'):break
  e=e.env
 ob=u.get_ob();err=float(np.max(np.abs(ob-f['observations'][c['step']])))
 if err>1e-8:raise AssertionError(f'RESTORE_OBSERVATION {c["id"]}: {err}')
 return ob,err
@jax.jit
def score(a,ob,act):
 return a.network.select('critic')(ob,act),a.network.select('target_critic')(ob,act)
@jax.jit
def preclip(a,ob,key):return a.network.select('actor')(ob,a.sample_noises(ob,key))
def check_assets():
 for d in assets:
  assert state_digest(agents[d['seed']])==d['state_sha256'];assert digest(Path(d['path']))==d['checkpoint_sha256']
def run_state(c):
 global envsteps
 f=np.load(c['source_npz']);assert digest(Path(c['source_npz']))==c['trajectory_sha256'];a=agents[c['seed']];t=c['step']
 ref=c['reference']['case'];key=jax.random.PRNGKey(ref['policy_seed']);keys=[]
 for i in range(2000):key,sub=jax.random.split(key);keys.append(sub)
 assert np.array_equal(keys[t],f['policy_keys'][t])
 ob=f['observations'][t];alt=jax.random.PRNGKey(c['alternate_policy_seed'])
 cand=[np.asarray(a.sample_actions(ob,seed=keys[t],temperature=0)),np.asarray(a.sample_actions(ob,seed=alt,temperature=0)),np.asarray(a.sample_flow_actions(ob,seed=alt,temperature=0))]
 assert np.max(np.abs(cand[0]-f['actions'][t]))<=1e-7
 for label,first in zip(['original_actor','resampled_actor','vabc'],cand):
  ob,initial_err=restore(c,f);states=[snapshot(ob)];acts=[];rewards=[];success=[];terms=[];truncs=[];walls=[];maxerrs={'observation':initial_err,'action':0.,'qpos':0.,'qvel':0.}
  q,qt=score(a,ob,first);q=np.asarray(q);qt=np.asarray(qt)
  for h in range(min(300,2000-t)):
   if envsteps>=10800:raise RuntimeError('ENVIRONMENT_STEP_CAP')
   act=first if h==0 else np.asarray(a.sample_actions(ob,seed=keys[t+h],temperature=0))
   assert np.isfinite(act).all()
   ob,r,te,tr,info=env.step(act);envsteps+=1
   states.append(snapshot(ob));acts.append(act);rewards.append(float(r));success.append(float(info.get('success',0)));terms.append(te);truncs.append(tr)
   walls.append(sum(geom[int(x.geom1)].startswith('block_') or geom[int(x.geom2)].startswith('block_') for x in u.data.contact))
   if label=='original_actor':
    idx=t+h
    errors={'observation':np.max(np.abs(ob-f['observations'][idx+1])),'action':np.max(np.abs(act-f['actions'][idx])),'qpos':np.max(np.abs(u.data.qpos-f['qpos'][idx+1])),'qvel':np.max(np.abs(u.data.qvel-f['qvel'][idx+1]))}
    maxerrs={k:max(maxerrs[k],float(v)) for k,v in errors.items()}
    if any(v>(1e-7 if k=='action' else 1e-8) for k,v in errors.items()) or r!=f['rewards'][idx] or te!=f['terminated'][idx] or tr!=f['truncated'][idx]:
     save(R/'receipts/restore_failure.json',dict(state=c['id'],step=h,errors=maxerrs,environment_steps=envsteps));raise AssertionError('ORIGINAL_REPLAY_DIVERGED')
   if te or tr:break
  arr={k:np.asarray([v[k] for v in states]) for k in states[0]};arr.update(actions=np.asarray(acts),rewards=np.asarray(rewards),success=np.asarray(success),terminated=np.asarray(terms),truncated=np.asarray(truncs),wall_contacts=np.asarray(walls))
  assert all(np.isfinite(v).all() for v in arr.values())
  dest=R/'branches'/f'{c["id"]}_{label}.npz';assert not dest.exists();np.savez_compressed(dest,**arr)
  distances=np.linalg.norm(arr['xy']-f['goal_xy'][t],axis=-1);gamma=float(a.config['discount'])
  row=dict(state=c['id'],category=c['category'],seed=c['seed'],candidate=label,steps=len(acts),success=max(success),return_=sum(rewards),discounted_partial_return=float(np.dot(np.power(gamma,np.arange(len(rewards))),rewards)),initial_distance=float(distances[0]),minimum_distance=float(min(distances)),final_distance=float(distances[-1]),displacement=float(np.linalg.norm(arr['xy'][-1]-arr['xy'][0])),low_head_fraction=float(np.mean(arr['head_height']<1.0)),wall_contact_steps=int(np.count_nonzero(walls)),q=q.tolist(),q_mean=float(q.mean()),target_q=qt.tolist(),target_q_mean=float(qt.mean()),action_l2_from_original=float(np.linalg.norm(first-cand[0])),restore_max_errors=maxerrs if label=='original_actor' else {'initial':initial_err},path=str(dest),sha256=digest(dest))
  results.append(row);save(R/'reports/branch_results.json',results);save(R/'receipts/progress.json',dict(branches=len(results),environment_steps=envsteps,optimizer_updates=0));print('BRANCH',c['id'],label,len(acts),max(success),round(float(distances[-1]),3),flush=True)
 ob=f['observations'][t];draws={}
 for source in ['actor','vabc']:
  aa=[];qq=[];tq=[];raw=[]
  for i in range(32):
   k=jax.random.fold_in(alt,i+1)
   x=np.asarray(a.sample_actions(ob,seed=k,temperature=0) if source=='actor' else a.sample_flow_actions(ob,seed=k,temperature=0));v,tv=score(a,ob,x)
   aa.append(x);qq.append(np.asarray(v));tq.append(np.asarray(tv))
   if source=='actor':raw.append(np.asarray(preclip(a,ob,k)))
  aa=np.asarray(aa);qq=np.asarray(qq);tq=np.asarray(tq)
  draws[source]=dict(actions=aa.tolist(),q=qq.tolist(),target_q=tq.tolist(),action_std_mean=float(aa.std(0).mean()),clipped_coordinate_fraction=float(np.mean(np.abs(aa)>=1)),q_mean=float(qq.mean()),q_std=float(qq.mean(-1).std()),preclip_coordinate_fraction=float(np.mean(np.abs(raw)>1)) if raw else None)
 distributions.append(dict(state=c['id'],draws=draws));save(R/'reports/proposal_distributions.json',distributions)
try:
 for c in S['states']:run_state(c)
 check_assets();save(R/'receipts/probe_result.json',dict(status='COMPLETE',branches=len(results),states=len(S['states']),environment_steps=envsteps,optimizer_updates=0,assets_unchanged=True,elapsed_seconds=time.time()-start,cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
except Exception as exc:
 unchanged=False
 try:check_assets();unchanged=True
 except Exception:pass
 save(R/'receipts/probe_result.json',dict(status='BLOCKED',error=repr(exc),traceback=traceback.format_exc(),branches=len(results),environment_steps=envsteps,optimizer_updates=0,assets_unchanged=unchanged,elapsed_seconds=time.time()-start));raise
finally:env.close()
