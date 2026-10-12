"""Read-only checkpoint loading and full-state replay gates."""
from preflight import *
import fcntl
assert __import__('os').environ.get('CUDA_VISIBLE_DEVICES')==''
assert all(d.platform=='cpu' for d in jax.devices())
def load_agent(kind,variant,step,seed=0):
    source=(OLD if variant=='A' else PREV) if seed==0 else ROOT
    name=f'{kind}_checkpoints.json' if seed==0 else f'C_{kind}_{variant}_s{seed}_checkpoints.json'
    entries=json.loads((source/'results'/name).read_text());options=[x for x in entries if x['step']==step]
    if not options:raise FileNotFoundError(f'MISSING_REGISTERED:{kind}:{variant}:{seed}:{step}')
    item=next((x for x in options if Path(x['path']).name==f'params_{step}.pkl'),options[0]);assert sha(item['path'])==item['sha256']
    env,data,val,config=load(f'pointmaze-large-{kind}-v0',variant);agent=init_agent(data,config,seed)
    with open(item['path'],'rb') as f:s=pickle.load(f)
    assert s['step']==step;agent=flax.serialization.from_state_dict(agent,s['agent']);agent=jax.device_put(agent,jax.devices('cpu')[0]);env.close()
    assert int(agent.network.step)==step+1
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(agent.network.params))
    return agent,dict(**item,kind=kind,variant=variant,seed=seed,native_counter=step+1)
def restore_gate(env,scene):
    s,info=restore(env,scene);actual=snapshot(env,info,scene['reset_seed'])
    errors={k:float(np.max(np.abs(np.asarray(actual[k])-np.asarray(scene[k])))) for k in ('integration','qpos','qvel','goal','goal_xy')}
    assert max(errors.values())<=1e-10,errors
    for k in ('env_rng','action_rng','numpy_rng','python_rng','wrappers'):assert digest(actual[k])==digest(scene[k]),k
    return s,max(errors.values())
def control_count(n=1):
    p=ROOT/'receipts/R_CONTROL_BUDGET.json'
    with open(str(p)+'.lock','a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);x=json.loads(p.read_text()) if p.exists() else dict(used=0,limit=120000)
        assert x['used']+n<=x['limit'],'R_CONTROL_CAP';x['used']+=n;write(p,x)
def first_divergence(a,b):
    delta=np.linalg.norm(a-b,axis=-1);counts=np.convolve((delta>=.30).astype(int),np.ones(5,dtype=int),'valid') if len(delta)>=5 else np.array([])
    idx=np.flatnonzero(counts>=3);return (int(idx[0]) if len(idx) else None),delta
