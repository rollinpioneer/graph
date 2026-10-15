"""Frozen GFP replay only. No optimizer/update entry point is called."""
import os,sys,json,time,random,hashlib,traceback,resource
from pathlib import Path
R=Path(__file__).resolve().parents[1]
S=json.loads((R/'configs/selection.json').read_text());P=Path(S['parent_root'])
os.environ.update(COMPARATIVE_ROOT=str(P),COMPARATIVE_JOB='G0_seed0')
sys.path.insert(0,str(P/'workspaces/gfp_floq'))
import numpy as np,jax,mujoco,ogbench
from agents.floq_adapter import create_agent
from training_wiring import restore_state,state_digest,digest
def save(path,obj):
 path.write_text(json.dumps(obj,indent=2))
def sha(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
spec=mujoco.mjtState.mjSTATE_INTEGRATION
env=ogbench.make_env_and_datasets(S['task'],env_only=True);u=env.unwrapped
geom_names=[mujoco.mj_id2name(u.model,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(u.model.ngeom)]
body_names=[mujoco.mj_id2name(u.model,mujoco.mjtObj.mjOBJ_BODY,i) or '' for i in range(u.model.nbody)]
meta=dict(task=S['task'],ogbench_version=ogbench.__version__ if hasattr(ogbench,'__version__') else '1.1.0',
 mujoco_version=mujoco.__version__,jax_version=jax.__version__,jax_x64=bool(jax.config.jax_enable_x64),
 matmul_precision=str(jax.config.jax_default_matmul_precision),devices=[str(x) for x in jax.devices()],
 state_spec=int(spec),state_size=mujoco.mj_stateSize(u.model,spec),geom_names=geom_names,body_names=body_names,
 maze_map=u.maze_map.tolist(),maze_unit=u._maze_unit,offset=[u._offset_x,u._offset_y],goal_tolerance=u._goal_tol,
 dt=u.dt,horizon=env.spec.max_episode_steps,environment_source='fixed installed OGBench; no modifications')
save(R/'reports/environment.json',meta)
assets=[];agents={}
for seed in (0,1):
 runtime=json.loads((P/f'runs/G0_seed{seed}/runtime.json').read_text());cfg=runtime['agent_config']
 with np.load(P/'data'/f"{S['task']}.batch.npz") as f:
  template=create_agent('gfp',seed,f['observations'][:1],f['actions'][:1],cfg)
 for step in sorted({c['checkpoint'] for c in S['cases'] if c['model_seed']==seed}):
  directory=Path(runtime['output_dir']);cp=directory/f'params_{step}.pkl'
  agent=restore_state(template,directory,step);sd=state_digest(agent);ch=digest(cp)
  reference=next(c for c in S['cases'] if c['model_seed']==seed and c['checkpoint']==step)
  assert sd==reference['state_sha256'] and ch==reference['checkpoint_sha256']
  agents[seed,step]=agent
  assets.append(dict(seed=seed,checkpoint=step,path=str(cp),checkpoint_sha256=ch,state_sha256=sd,
    agent_class=str(type(agent)),config=cfg))
save(R/'reports/asset_identity.json',assets)
start=time.time();results=[];envsteps=0
def capture_state(ob):
 a=np.empty(meta['state_size'],np.float64);mujoco.mj_getState(u.model,u.data,a,spec)
 return dict(observations=np.asarray(ob).copy(),integration=a,qpos=u.data.qpos.copy(),qvel=u.data.qvel.copy(),
 xy=u.get_xy().copy(),goal_xy=np.asarray(u.cur_goal_xy).copy(),head_height=np.array(u.data.xpos[2,2]),
 torso_up=np.array(u.data.xmat[1,8]),torso_matrix=u.data.xmat[1].copy(),body_positions=u.data.xpos.copy(),
 sim_time=np.array(u.data.time))
def run_case(c):
 global envsteps
 agent=agents[c['model_seed'],c['checkpoint']];ref=c['reference'];dest=R/'trajectories'/c['id']
 if dest.with_suffix('.npz').exists():raise FileExistsError(str(dest))
 np.random.seed(ref['numpy_seed']);random.seed(ref['python_seed'])
 ob,_=env.reset(seed=ref['reset_seed']);initial_sha=sha(ob)
 if initial_sha!=ref['initial_observation_sha256']:raise AssertionError('INITIAL_STATE_MISMATCH '+c['id'])
 initial_env_rng=u.np_random.bit_generator.state
 states=[capture_state(ob)];actions=[];raws=[];keys=[];rewards=[];succ=[];terms=[];truncs=[];contacts=[];counts=[];wallcounts=[]
 key=jax.random.PRNGKey(c['policy_seed']);t0=time.time();done=False
 while not done:
  if envsteps>=96000:raise RuntimeError('ENVIRONMENT_STEP_CAP')
  key,sub=jax.random.split(key)
  raw=np.asarray(agent.sample_actions(observations=ob,seed=sub,temperature=0));a=np.clip(raw,-1,1)
  if not np.isfinite(a).all():raise FloatingPointError(c['id'])
  ob,r,te,tr,info=env.step(a);done=te or tr;envsteps+=1
  states.append(capture_state(ob));raws.append(raw);actions.append(a);keys.append(np.asarray(sub))
  rewards.append(r);succ.append(float(info.get('success',0)));terms.append(te);truncs.append(tr)
  wall=0
  for j in range(u.data.ncon):
   con=u.data.contact[j];force=np.zeros(6);mujoco.mj_contactForce(u.model,u.data,j,force)
   g1,g2=int(con.geom1),int(con.geom2);wall+=int(geom_names[g1].startswith('block_') or geom_names[g2].startswith('block_'))
   contacts.append([len(actions),g1,g2,float(con.dist),*con.pos.tolist(),*force.tolist()])
  counts.append(u.data.ncon);wallcounts.append(wall)
  if len(actions)>2000:raise RuntimeError('HORIZON')
 arrays={k:np.stack([x[k] for x in states]) for k in states[0]}
 arrays.update(actions=np.asarray(actions),raw_actions=np.asarray(raws),policy_keys=np.asarray(keys),
 rewards=np.asarray(rewards),success=np.asarray(succ),terminated=np.asarray(terms),truncated=np.asarray(truncs),
 contacts=np.asarray(contacts).reshape(-1,13),contact_count=np.asarray(counts),wall_contact_count=np.asarray(wallcounts))
 if not all(np.isfinite(v).all() for v in arrays.values()):raise FloatingPointError('NONFINITE_TRACE '+c['id'])
 np.savez_compressed(dest.with_suffix('.npz'),**arrays)
 observed=dict(success=max(succ),return_=float(sum(rewards)),length=len(actions),terminated=bool(terms[-1]),
 truncated=bool(truncs[-1]),initial_observation_sha256=initial_sha)
 comparisons={k:observed[k]==ref[k] for k in observed}
 match=all(comparisons.values()) if c['kind']!='rng' else comparisons['initial_observation_sha256']
 result=dict(case=c,observed=observed,reference_comparisons=comparisons,reference_match=match,
 initial_env_rng=initial_env_rng,elapsed_seconds=time.time()-t0,optimizer_updates=0,path=str(dest.with_suffix('.npz')),
 trajectory_sha256=digest(dest.with_suffix('.npz')),bytes=dest.with_suffix('.npz').stat().st_size)
 save(dest.with_suffix('.json'),result);results.append(result)
 save(R/'receipts/progress.json',dict(complete=len(results),total=len(S['cases']),environment_steps=envsteps,
 last=c['id'],elapsed_seconds=time.time()-start,optimizer_updates=0))
 print('REPLAY',len(results),c['id'],observed['success'],observed['length'],'MATCH',match,flush=True)
 if not match:raise AssertionError('REPLAY_OUTCOME_MISMATCH '+c['id'])

def check_unchanged():
 for a in assets:
  assert state_digest(agents[a['seed'],a['checkpoint']])==a['state_sha256']
  assert digest(Path(a['path']))==a['checkpoint_sha256']
try:
 for cid in S['gate_ids']:run_case(next(c for c in S['cases'] if c['id']==cid))
 check_unchanged()
 save(R/'receipts/gate.json',dict(status='PASS',cases=S['gate_ids'],exact_replay=True,optimizer_updates=0,
 checkpoint_state_unchanged=True,epoch=time.time()))
 for c in S['cases']:
  if c['id'] not in S['gate_ids']:run_case(c)
 check_unchanged()
 usage=resource.getrusage(resource.RUSAGE_SELF)
 save(R/'reports/trajectory_index.json',results)
 save(R/'receipts/record_result.json',dict(status='PASS',episodes=len(results),environment_steps=envsteps,
 elapsed_seconds=time.time()-start,cpu_user_seconds=usage.ru_utime,cpu_system_seconds=usage.ru_stime,
 max_rss_kib=usage.ru_maxrss,optimizer_updates=0,source_checkpoint_unchanged=True,
 model_optimizer_rng_unchanged=True,gate_pass=True))
except BaseException as exc:
 check_error=None
 try:check_unchanged()
 except Exception as e:check_error=repr(e)
 save(R/'receipts/record_result.json',dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),
 complete=len(results),environment_steps=envsteps,optimizer_updates=0,state_check_error=check_error))
 raise
finally:env.close()

