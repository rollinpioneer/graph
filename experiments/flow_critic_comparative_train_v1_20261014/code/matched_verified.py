"""Read-only matched evaluation; controls OGBench's global RNG as well as reset(seed)."""
import os,sys,json,time,random,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
job=json.loads((ROOT/'configs'/f'{sys.argv[1]}.job.json').read_text())
name=job['job'];d=ROOT/'runs'/name
os.environ.update(COMPARATIVE_ROOT=str(ROOT),COMPARATIVE_JOB=name)
sys.path.insert(0,str(ROOT/f"workspaces/{job['host']}_floq"))
import numpy as np,jax,ogbench
from agents.floq_adapter import create_agent
from training_wiring import restore_state,state_digest,digest
runtime=json.loads((d/'runtime.json').read_text());cfg=runtime['agent_config'];directory=Path(runtime['output_dir'])
with np.load(ROOT/'data'/f"{job['task']}.batch.npz") as f:
    agent=create_agent(job['host'],job['seed'],f['observations'][:1],f['actions'][:1],cfg)
env=ogbench.make_env_and_datasets(job['task'],env_only=True)
for step in [800000,900000,1000000]:
    dest=d/'matched_verified'/f'{step}.json';dest.parent.mkdir(exist_ok=True)
    if dest.exists():raise FileExistsError('No overwrite/repeat evaluation')
    agent=restore_state(agent,directory,step);before=state_digest(agent)
    file_hash=digest(directory/f'params_{step}.pkl');np_state=np.random.get_state();py_state=random.getstate()
    episodes=[];start=time.time()
    try:
        for ep in range(100):
            seed=20261014+ep;np.random.seed(seed);random.seed(seed)
            ob,_=env.reset(seed=seed);initial=np.asarray(ob)
            initial_sha=hashlib.sha256(initial.tobytes()).hexdigest();key=jax.random.PRNGKey(420000+ep)
            done=False;ret=0.;length=0;success=0.
            while not done:
                key,sub=jax.random.split(key)
                action=np.clip(np.asarray(agent.sample_actions(observations=ob,seed=sub,temperature=0)),-1,1)
                if not np.isfinite(action).all():raise FloatingPointError('Nonfinite evaluation action')
                ob,reward,terminated,truncated,info=env.step(action)
                done=terminated or truncated;ret+=float(reward);length+=1
                success=max(success,float(info.get('success',0)))
            episodes.append(dict(episode=ep,success=success,return_=ret,length=length,
                reset_seed=seed,numpy_seed=seed,python_seed=seed,policy_seed=420000+ep,
                initial_observation_sha256=initial_sha,terminated=bool(terminated),truncated=bool(truncated)))
        assert before==state_digest(agent) and file_hash==digest(directory/f'params_{step}.pkl')
        result=dict(status='PASS',step=step,episodes=episodes,optimizer_updates=0,
            success_rate=float(np.mean([x['success'] for x in episodes])),
            mean_return=float(np.mean([x['return_'] for x in episodes])),elapsed_seconds=time.time()-start,
            checkpoint_sha256=file_hash,state_sha256=before,global_rng_control_verified=True,
            policy_randomness='Gaussian keys still sampled when temperature=0')
        dest.write_text(json.dumps(result,indent=2))
        print('MATCHED_VERIFIED',name,step,result['success_rate'],flush=True)
    finally:np.random.set_state(np_state);random.setstate(py_state)
env.close()
(d/'matched_verified/completed.json').write_text(json.dumps(dict(status='PASS',optimizer_updates=0,epoch=time.time()),indent=2))
