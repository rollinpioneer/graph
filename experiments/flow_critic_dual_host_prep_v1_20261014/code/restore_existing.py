"""Read-only restore in ORIGINAL environment. No parameters or optimizer update."""
import argparse, copy, hashlib, json, os, pickle, sys
from pathlib import Path
import flax
import jax
import jax.numpy as jnp
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/home/xushijie3/work/flow_baselines_preflight_20261013')
FLOQ=Path('/home/xushijie3/work/floq_min_repro_v2_20261013')
parser=argparse.ArgumentParser();parser.add_argument('--method',choices=['gfp','floq','fql'],required=True)
a=parser.parse_args();name=a.method
source=FLOQ/'source' if name=='floq' else OLD/'sources'/name
sys.path.insert(0,str(source))
from utils.flax_utils import TrainState,restore_agent
def forbidden(*a,**kw):raise RuntimeError('NOT_AUTHORIZED')
TrainState.apply_loss_fn=forbidden;TrainState.apply_gradients=forbidden
if name=='floq':from agents.floq import FloQAgent as Agent
elif name=='gfp':from agents.gfp import GFPAgent as Agent
else:from agents.fql import FQLAgent as Agent
for attr in ['update','batch_update','multi_sample_and_update','target_update']:
    if hasattr(Agent,attr):setattr(Agent,attr,forbidden)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def state_hash(agent):return hashlib.sha256(flax.serialization.to_bytes(agent)).hexdigest()

results=[]
for seed in [0,1]:
    run=(FLOQ if name=='floq' else OLD)/'runs'/(f'formal_seed{seed}' if name=='floq' else f'formal_{name}_seed{seed}')
    if name=='floq':cfg=json.loads((run/'actual_config.json').read_text())
    else:cfg=json.loads((run/'actual_runtime.json').read_text())['expanded_config']['agent']
    cfg=copy.deepcopy(cfg)
    for key in ['actor_hidden_dims','value_hidden_dims','ob_dims']:
        if key in cfg:cfg[key]=tuple(cfg[key])
    batchfile=ROOT/'data/antmaze-large-navigate-singletask-task1-v0.batch.npz'
    if batchfile.exists():
        with np.load(batchfile) as z:obs=jnp.asarray(z['observations'][:8]);actions=jnp.asarray(z['actions'][:8])
    else:
        # Existing exact AntMaze data through the original OGBench official loader.
        import ogbench
        env,data,val=ogbench.make_env_and_datasets('antmaze-large-navigate-singletask-task1-v0',
              dataset_dir=str(OLD/'home/.ogbench/data'))
        obs=jnp.asarray(data['observations'][:8]);actions=jnp.asarray(np.clip(data['actions'][:8],-1+1e-5,1-1e-5));env.close()
    epochs=[1000000,2000000] if name=='floq' else [1000000]
    for epoch in epochs:
        matches=list(run.rglob(f'params_{epoch}.pkl'));assert len(matches)==1
        path=matches[0];before=sha(path)
        agent=Agent.create(seed,obs,actions,cfg.copy())
        agent=restore_agent(agent,str(path.parent),epoch)
        state_before=state_hash(agent);step=int(agent.network.step)
        q=agent.network.select('critic')(obs,actions)
        grad=jax.grad(lambda x:agent.network.select('critic')(obs,x).mean(axis=0).sum())(actions)
        assert np.isfinite(np.asarray(q)).all() and np.isfinite(np.asarray(grad)).all()
        entry=dict(method=name,seed=seed,epoch=epoch,path=str(path),checkpoint_sha256_before=before,
                   checkpoint_sha256_after=sha(path),state_sha256_before=state_before,state_sha256_after=state_hash(agent),
                   restored_state_step=step,optimizer_steps=0,q_shape=list(q.shape),
                   q_mean=float(q.mean()),action_gradient_norm=float(jnp.linalg.norm(grad)),device=str(q.device))
        if name=='floq':
            ratios=jnp.ones((8,1))*.5
            flow=agent.compute_current_flow_returns(obs,actions,ratios)
            target=agent.compute_target_flow_returns(obs,actions,ratios)
            assert np.isfinite(np.asarray(flow)).all() and np.isfinite(np.asarray(target)).all()
            entry['flow_shape']=list(flow.shape)
            # Export source value parameters for read-only mapping/array comparison in the new host environment.
            params={k:v for k,v in flax.serialization.to_state_dict(agent)['network']['params'].items()
                    if k in ['modules_floq','modules_target_floq','modules_critic']}
            export=ROOT/'tests_zero_train'/f'floq_source_value_seed{seed}_epoch{epoch}.pkl'
            with export.open('wb') as f:pickle.dump(params,f)
            np.savez(ROOT/'tests_zero_train'/f'floq_source_value_seed{seed}_epoch{epoch}.npz',
                     obs=np.asarray(obs),actions=np.asarray(actions),ratios=np.asarray(ratios),q=np.asarray(q),
                     grad=np.asarray(grad),flow=np.asarray(flow),target=np.asarray(target))
        assert entry['checkpoint_sha256_before']==entry['checkpoint_sha256_after']
        assert state_before==state_hash(agent) and step==int(agent.network.step)
        results.append(entry)
        print('RESTORED_READ_ONLY',name,seed,epoch,flush=True)
        del agent; jax.clear_caches()
(ROOT/'reports'/f'existing_restore_{name}.json').write_text(json.dumps(dict(status='PASS',optimizer_steps=0,results=results),indent=2))
