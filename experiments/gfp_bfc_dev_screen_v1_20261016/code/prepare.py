import os,json,shutil,inspect,hashlib
import numpy as np
from common import ROOT,FORMAL,write,sha,load_base

def main():
    import ogbench,jax,flax,optax,mujoco
    from utils.datasets import ReplayBuffer
    (ROOT/'data').mkdir(exist_ok=True)
    # Same official loader/relabeling and action clipping, without an unnecessary reset.
    env,train,val=ogbench.make_env_and_datasets('humanoidmaze-medium-navigate-singletask-task1-v0')
    train['actions']=np.clip(train['actions'],-1+1e-5,1-1e-5)
    size=len(train['observations']);rms=float(np.sqrt(np.mean(np.square(train['rewards'].astype(np.float64)))))
    assert rms>0 and np.isfinite(rms)
    buf=ReplayBuffer.create_from_initial_dataset(train,size=max(2000000,size+1))
    files={}
    for k,v in buf.items():
        p=ROOT/'data'/f'{k}.npy';np.save(p,v);files[k]=dict(shape=list(v.shape),dtype=str(v.dtype),sha256=sha(p))
    env.close()
    sources={}
    for seed in (0,3):
        ag,cfg,p=load_base(seed)
        sources[str(seed)]=dict(path=str(p),sha256=sha(p),config=cfg,step=int(ag.network.step))
    arms={'R0':dict(tau_pi=None,epsilon=None)}
    for i,tau in enumerate((.005,.0005,.00005),1):arms[f'R{i}']=dict(tau_pi=tau,epsilon=0.)
    for i,c in ((4,.1),(5,.5)):arms[f'R{i}']=dict(tau_pi=.0005,epsilon=c*rms,reward_rms_multiplier=c)
    write(ROOT/'protocol.json',dict(arms=arms,train_reward_rms=rms,seeds=[0,3],updates_per_arm=100000,
        total_updates_cap=1200000,eval_updates=[50000,100000],eval_episodes=100,eval_seed=4242,
        max_eval_control_calls=5000000,per_evaluation_control_call_cap=208333,online_updates=0,
        probe_every=5000,anomaly_gap_increment=15,anomaly_gradient_ratio=2,
        anomaly_failure_consecutive=2,no_hours_cap=True))
    write(ROOT/'receipts/assets.json',dict(dataset_size=size,reward_rms=rms,data=files,sources=sources,
      probe=dict(path=str(FORMAL/'analysis_p0p1/out/probe/states.npz'),sha256=sha(FORMAL/'analysis_p0p1/out/probe/states.npz')),
      versions={k:str(v.__version__) for k,v in dict(jax=jax,flax=flax,optax=optax,numpy=np,mujoco=mujoco).items()},
      ogbench_version=__import__('importlib.metadata',fromlist=['version']).version('ogbench'),
      source_manifest=(FORMAL/'receipts/source_manifest.txt').read_text(),
      source_files={str(p.relative_to(ROOT/'source/gfp')):sha(p) for p in (ROOT/'source/gfp').rglob('*.py')},
      data_preparation_environment_resets=0,data_preparation_external_steps=0))
    print('PREPARED',size,rms,flush=True)

if __name__=='__main__':main()
