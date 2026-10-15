"""Read-only primary evaluation with fixed global/reset/policy RNG and saved numeric trajectories."""
import os,sys,json,time,random,hashlib,pickle
from pathlib import Path
R=Path(__file__).resolve().parents[1];job=sys.argv[1];d=R/'runs'/job
os.environ.update(COMPARATIVE_ROOT=str(R),COMPARATIVE_JOB=job)
sys.path.insert(0,str(R/'workspaces/gfp'))
import numpy as np,jax,ogbench
from agents import agents
from utils.flax_utils import restore_agent
from training_wiring import restore_state,state_digest,digest,finite,canonical_state_digest
runtime=json.loads((d/'runtime.json').read_text());directory=Path(runtime['output_dir']);host='gfp' if job.startswith('GFP') else 'rebrac';seed=int(job[-1]);adopted=runtime.get('adopted',False)
cfg=runtime['cfg']['agent'] if adopted else runtime['agent_config'];cfg=dict(cfg,log_metrics=True)
s=np.load('/home/xushijie3/work/gfp_guidance_distance_z01_20261015/arrays/train_samples.npz')
agent=agents[host].create(seed,s['observations'][:1],s['actions'][:1],cfg)
out=d/'matched';out.mkdir(exist_ok=True);lock=out/'started.json'
with lock.open('x') as f:json.dump(dict(pid=os.getpid(),time=time.time(),optimizer_updates=0),f)
env=ogbench.make_env_and_datasets('humanoidmaze-medium-navigate-singletask-task4-v0',env_only=True)
for step in [800000,900000,1000000]:
 agent=restore_agent(agent,str(directory),step) if adopted else restore_state(agent,directory,step)
 finite((agent.network.params,agent.network.opt_state));assert int(agent.network.step)==step+1
 before=state_digest(agent);fh=digest(directory/f'params_{step}.pkl');eps=[];t=time.time();td=out/str(step);td.mkdir()
 for ep in range(100):
  rs=20261015+ep;np.random.seed(rs);random.seed(rs);ob,_=env.reset(seed=rs);ih=hashlib.sha256(np.asarray(ob).tobytes()).hexdigest();key=jax.random.PRNGKey(420000+ep)
  obs=[np.asarray(ob)];acts=[];rew=[];succ=[];done=False
  while not done:
   key,sub=jax.random.split(key);a=np.clip(np.asarray(agent.sample_actions(observations=ob,seed=sub,temperature=0)),-1,1);assert np.isfinite(a).all()
   ob,r,term,trunc,info=env.step(a);done=term or trunc;obs.append(np.asarray(ob));acts.append(a);rew.append(r);succ.append(float(info.get('success',0)));assert len(acts)<=2000
  trajectory=td/f'episode_{ep:03d}.npz';np.savez_compressed(trajectory,observations=np.asarray(obs),actions=np.asarray(acts),rewards=np.asarray(rew),success=np.asarray(succ))
  eps.append(dict(episode=ep,success=max(succ),return_=float(sum(rew)),length=len(rew),reset_seed=rs,global_seed=rs,policy_seed=420000+ep,initial_observation_sha256=ih,terminated=bool(term),truncated=bool(trunc),trajectory=str(trajectory),trajectory_sha256=digest(trajectory)))
  (out/'progress.json').write_text(json.dumps(dict(step=step,episodes_completed=len(eps),time=time.time(),optimizer_updates=0)))
 assert state_digest(agent)==before and digest(directory/f'params_{step}.pkl')==fh
 result=dict(status='PASS',step=step,episodes=eps,optimizer_updates=0,success_rate=float(np.mean([x['success'] for x in eps])),mean_return=float(np.mean([x['return_'] for x in eps])),elapsed_seconds=time.time()-t,checkpoint_sha256=fh,state_sha256=before,global_rng_control_verified=True)
 (out/f'{step}.json').write_text(json.dumps(result,indent=2));print(job,step,result['success_rate'],flush=True)
env.close();(out/'completed.json').write_text(json.dumps(dict(status='PASS',optimizer_updates=0,time=time.time())))
