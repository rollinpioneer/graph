"""Frozen inference; 12 states x 3 saved actions x 7 new suffixes; no updates."""
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
agents={};assets=[];start=time.time();envsteps=0;results=[]
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
F=json.loads((R/'configs/frozen_protocol.json').read_text());B=Path(F['parent_root'])
assert json.loads((R/'receipts/protocol_push.json').read_text())['verified']
assert F['optimizer_updates']==0 and len(F['new_sequence_seeds'])==7
for p,h in F['source_hashes'].items():assert digest(Path(p))==h
firsts={(d['state'],d['candidate']):d for d in json.loads((R/'configs/first_actions.json').read_text())}
prior={(d['state'],d['candidate']):d for d in json.loads((B/'reports/branch_results.json').read_text())}
keys={}
for seed in F['new_sequence_seeds']:
 k=jax.random.PRNGKey(seed);kk=[]
 for h in range(300):k,sub=jax.random.split(k);kk.append(np.asarray(sub))
 keys[seed]=np.asarray(kk)
save(R/'receipts/suffix_key_hashes.json',{str(k):sha(v) for k,v in keys.items()})
def check_assets():
 for d in assets:
  assert state_digest(agents[d['seed']])==d['state_sha256'];assert digest(Path(d['path']))==d['checkpoint_sha256']
 for p,h in F['source_hashes'].items():assert digest(Path(p))==h
 for d in firsts.values():assert digest(Path(d['path']))==d['sha256']
def run_state(c):
 global envsteps
 with np.load(c['source_npz']) as f:
  assert digest(Path(c['source_npz']))==c['trajectory_sha256'];a=agents[c['seed']];t=c['step']
  for seed in F['new_sequence_seeds']:
   for label in ['original_actor','resampled_actor','vabc']:
    d=firsts[c['id'],label];assert digest(Path(d['path']))==d['sha256']
    with np.load(d['path']) as old:
     first=old['actions'][0].copy();assert sha(first)==d['action_sha256']
     ob,initial_err=restore(c,f);states=[snapshot(ob)];acts=[];rewards=[];success=[];terms=[];truncs=[];walls=[];errors={}
     for h in range(min(300,2000-t)):
      if envsteps>=75600:raise RuntimeError('ENVIRONMENT_STEP_CAP')
      act=first if h==0 else np.asarray(a.sample_actions(ob,seed=jnp.asarray(keys[seed][h]),temperature=0))
      assert np.isfinite(act).all()
      ob,r,te,tr,info=env.step(act);envsteps+=1;snap=snapshot(ob)
      assert all(np.isfinite(v).all() for v in snap.values()) and np.isfinite(r)
      if h==0:
       errors={k:float(np.max(np.abs(snap[k]-old[k][1]))) for k in ['observations','qpos','qvel','integration']}
       assert max(errors.values())<=1e-8,('FIRST_TRANSITION',c['id'],label,errors)
       assert r==old['rewards'][0] and te==old['terminated'][0] and tr==old['truncated'][0]
      states.append(snap);acts.append(act);rewards.append(float(r));success.append(float(info.get('success',0)));terms.append(te);truncs.append(tr)
      walls.append(sum(geom[int(x.geom1)].startswith('block_') or geom[int(x.geom2)].startswith('block_') for x in u.data.contact))
      if te or tr:break
     arr={k:np.asarray([v[k] for v in states]) for k in states[0]};arr.update(actions=np.asarray(acts),rewards=np.asarray(rewards),success=np.asarray(success),terminated=np.asarray(terms),truncated=np.asarray(truncs),wall_contacts=np.asarray(walls),policy_keys=keys[seed][:len(acts)])
     dest=R/'branches'/f'{c["id"]}_{label}_r{seed}.npz';assert not dest.exists();np.savez_compressed(dest,**arr)
     distances=np.linalg.norm(arr['xy']-f['goal_xy'][t],axis=-1)
     windows=[np.linalg.norm(arr['xy'][i:i+101]-arr['xy'][i],axis=-1).max()<0.5 for i in range(len(acts)-99)]
     p=prior[c['id'],label]
     row=dict(state=c['id'],category=c['category'],seed=c['seed'],candidate=label,sequence_seed=seed,steps=len(acts),success=max(success),return_=sum(rewards),discounted_partial_return=float(np.dot(np.power(float(a.config['discount']),np.arange(len(rewards))),rewards)),initial_distance=float(distances[0]),minimum_distance=float(min(distances)),final_distance=float(distances[-1]),progress_final=float(distances[0]-distances[-1]),progress_min=float(distances[0]-min(distances)),displacement=float(np.linalg.norm(arr['xy'][-1]-arr['xy'][0])),path_length=float(np.linalg.norm(np.diff(arr['xy'],axis=0),axis=-1).sum()),stagnation_windows=sum(bool(x) for x in windows),window_count=len(windows),stagnation_fraction=float(np.mean(windows)) if windows else None,low_head_fraction=float(np.mean(arr['head_height']<1)),wall_contact_steps=int(np.count_nonzero(walls)),wall_contact_fraction=float(np.mean(np.asarray(walls)>0)),q_mean=p['q_mean'],target_q_mean=p['target_q_mean'],initial_restore_error=initial_err,first_transition_errors=errors,first_action_sha256=sha(first),path=str(dest),sha256=digest(dest))
     results.append(row);save(R/'reports/branch_results.json',results);save(R/'receipts/progress.json',dict(branches=len(results),environment_steps=envsteps,optimizer_updates=0));print('BRANCH',len(results),c['id'],label,seed,len(acts),max(success),flush=True)
try:
 for c in S['states']:run_state(c)
 assert len(results)==252 and envsteps<=75600
 check_assets();save(R/'receipts/probe_result.json',dict(status='COMPLETE',branches=len(results),states=len(S['states']),environment_steps=envsteps,optimizer_updates=0,assets_unchanged=True,elapsed_seconds=time.time()-start,cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
except Exception as exc:
 unchanged=False
 try:check_assets();unchanged=True
 except Exception:pass
 save(R/'receipts/probe_result.json',dict(status='BLOCKED',error=repr(exc),traceback=traceback.format_exc(),branches=len(results),environment_steps=envsteps,optimizer_updates=0,assets_unchanged=unchanged,elapsed_seconds=time.time()-start));raise
finally:env.close()
