import os,json,time,pickle,random,hashlib
from pathlib import Path
import flax,jax,numpy as np
from utils.flax_utils import save_agent as upstream_save,restore_agent as upstream_restore
ROOT=Path(os.environ["COMPARATIVE_ROOT"])
def json_write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,sort_keys=True,default=lambda x:x.tolist()))

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical_state_digest(state):
    def ordered(x):
        if isinstance(x,dict):return {k:ordered(x[k]) for k in sorted(x)}
        if isinstance(x,list):return [ordered(v) for v in x]
        if isinstance(x,tuple):return tuple(ordered(v) for v in x)
        if isinstance(x,(np.ndarray,np.generic,jax.Array,int,float,bool)):return np.asarray(x)
        return x
    return hashlib.sha256(flax.serialization.msgpack_serialize(ordered(state))).hexdigest()

def state_digest(agent):return canonical_state_digest(flax.serialization.to_state_dict(agent))

def finite(tree):
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(tree)):
        raise FloatingPointError('Non-finite loss, gradient, parameter or optimizer state')

def read_execution_gate(path,host,task,critic_type,steps,online,seed,eval_episodes):
    if not path:raise RuntimeError('NOT_AUTHORIZED: --wiring_gate is required before native setup')
    gate=json.loads(Path(path).read_text())
    if gate.get('training_authorized') is not True:raise RuntimeError('NOT_AUTHORIZED: execution gate is closed')
    if (host,task,critic_type)!=(gate['host'],gate['task'],gate['critic_type']):
        raise RuntimeError('EXECUTION_IDENTITY_MISMATCH')
    if online!=0 or gate.get('online_learning_allowed',False):raise RuntimeError('Online learning not authorized')
    if steps<1 or steps>int(gate['optimizer_steps_allowed']):raise RuntimeError('UPDATE_BUDGET_EXHAUSTED')
    if int(seed)!=int(gate['seed']):raise RuntimeError('Unapproved seed')
    if gate['stage']=='T0':
        if steps!=2 or eval_episodes!=0 or critic_type!='floq':
            raise RuntimeError('T0 requires exactly 2 updates, no episodes, combined path')
    elif gate['stage'] not in ['T1','T2','T3']:
        raise RuntimeError('Unknown execution stage')
    # This new batch approves only the eight immutable identity-specific gates.
    return gate

def save_state(agent,directory,epoch):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'params_{epoch}.pkl'
    if path.exists():raise FileExistsError(f'Will not overwrite checkpoint {path}')
    upstream_save(agent,str(directory),epoch)
    rng_path=directory/f'params_{epoch}.rng.pkl'
    if rng_path.exists():raise FileExistsError(rng_path)
    with rng_path.open('wb') as f:pickle.dump({'numpy':np.random.get_state(),'python':random.getstate()},f)
    meta=dict(epoch=int(epoch),state_step=int(agent.network.step),completed_updates=int(agent.network.step)-1,
              state_sha256=state_digest(agent),state_fingerprint_version='canonical-v1',
              checkpoint_sha256=digest(path),config=dict(agent.config),
              config_sha256=hashlib.sha256(json.dumps(dict(agent.config),sort_keys=True).encode()).hexdigest(),
              data_rng_sha256=digest(rng_path))
    json_write(directory/f'params_{epoch}.identity.json',meta)
    with (directory/'checkpoint_index.jsonl').open('a') as f:f.write(json.dumps(meta)+'\n')
    return path

def restore_state(agent,directory,epoch):
    directory=Path(directory);meta=json.loads((directory/f'params_{epoch}.identity.json').read_text())
    path=directory/f'params_{epoch}.pkl';before=digest(path)
    if before!=meta['checkpoint_sha256']:raise RuntimeError('Checkpoint hash mismatch')
    if meta['config_sha256']!=hashlib.sha256(json.dumps(dict(agent.config),sort_keys=True).encode()).hexdigest():
        raise RuntimeError('Restore configuration mismatch')
    restored=upstream_restore(agent,str(directory),epoch)
    # Fresh templates can preserve a different dict insertion order. Verify actual
    # state content canonically, retaining strict file-byte identity for all versions.
    with path.open('rb') as f:saved_state=pickle.load(f)['agent']
    expected_state_hash=canonical_state_digest(saved_state)
    if meta.get('state_fingerprint_version')=='canonical-v1':
        if expected_state_hash!=meta['state_sha256']:raise RuntimeError('State identity mismatch')
    # Legacy raw serialization hashes are retained as historical identifiers.
    # Their strict file SHA plus canonical saved-vs-restored content verifies them.
    if 'data_rng_sha256' in meta:
        rng_path=directory/f'params_{epoch}.rng.pkl'
        if digest(rng_path)!=meta['data_rng_sha256']:raise RuntimeError('Dataset/Python RNG sidecar hash mismatch')
        with rng_path.open('rb') as f:saved_rng=pickle.load(f)
        np.random.set_state(saved_rng['numpy']);random.setstate(saved_rng['python'])
    if state_digest(restored)!=expected_state_hash or digest(path)!=before:
        raise RuntimeError('Checkpoint state or file changed on restore')
    finite((restored.network.params,restored.network.opt_state))
    return restored

def resume_index(agent):return int(agent.network.step)-1

def construct_agent(host,seed,obs,actions,config,dataset,gate,output_dir):
    from agents import agents
    import batch_hooks
    assert host in ['gfp','rebrac'] and config['agent_name']==host
    agent=agents[host].create(seed,obs,actions,dict(config))
    out=Path(output_dir).resolve()
    assert out.is_relative_to(ROOT.resolve())
    json_write(out/'actual_agent_config.json',dict(agent.config))
    batch_hooks.initial(agent,out,gate)
    return agent

def dispatch_fused(agent,n_steps,dataset,dataset_size,batch_size,gate):
    from batch_hooks import checked_dispatch
    return checked_dispatch(agent,gate,os.environ['COMPARATIVE_OUTPUT_DIR'],n=n_steps,dataset=dataset,size=dataset_size,batch_size=batch_size)

def dispatch_update(agent,batch,gate,output_dir):
    from batch_hooks import checked_dispatch
    return checked_dispatch(agent,gate,output_dir,batch=batch)
