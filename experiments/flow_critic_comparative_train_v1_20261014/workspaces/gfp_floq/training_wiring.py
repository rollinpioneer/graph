"""Shared native-main dispatcher, execution gate, state IO and T0 admission probes."""
import copy,hashlib,json,os,time,pickle,random
from pathlib import Path
import flax,jax,jax.numpy as jnp,numpy as np
from agents.floq_adapter import create_agent
from floq_value_core.core import TARGET_KEYS,ema_specification
from utils.flax_utils import save_agent as upstream_save,restore_agent as upstream_restore

ROOT=Path(__file__).resolve().parents[2]

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

def norms(tree):
    return {k:float(np.sqrt(sum(np.square(np.asarray(x,dtype=np.float64)).sum()
                     for x in jax.tree_util.tree_leaves(v)))) for k,v in tree.items()}

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


def construct_agent(host,seed,obs,actions,config,dataset,gate,output_dir):
    c=copy.deepcopy(dict(config))
    if c.get('critic_type','scalar')=='floq':
        defaults=json.loads((ROOT/'configs/floq_inherited_defaults.json').read_text())
        for key,value in defaults.items():c.setdefault(key,value)
        c.update(r_min=float(np.min(dataset['rewards'])),r_max=float(np.max(dataset['rewards'])))
        c.update(training_authorized=bool(gate['training_authorized']),
                 optimizer_steps_allowed=int(gate['optimizer_steps_allowed']),online_learning_allowed=False,
                 initial_optimizer_step=1)
    agent=create_agent(host,seed,obs,actions,c)
    out=Path(output_dir).resolve()
    if not out.is_relative_to(ROOT.resolve()):raise RuntimeError('Output must stay in this isolated work root')
    json_write(out/'actual_agent_config.json',dict(agent.config))
    json_write(ROOT/'receipts'/f'{host}_{os.environ["COMPARATIVE_JOB"]}_runtime.json',dict(gate=gate,output_dir=str(out),config=dict(agent.config),
        initial_state_step=int(agent.network.step),initial_optimizer_steps=0,created_epoch=time.time()))
    import batch_hooks
    batch_hooks.initial(agent,out,gate)
    return agent


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


@jax.jit
def probe_gradients(agent,batch,loss_rng):
    return jax.value_and_grad(lambda p:agent.total_loss(batch,p,rng=loss_rng),has_aux=True)(agent.network.params)


def dispatch_update(agent,batch,gate,output_dir):
    """Native CPU-dataset route. T0 probes real state before/after each update."""
    if gate['stage']!='T0':
        from batch_hooks import checked_dispatch
        return checked_dispatch(agent,gate,output_dir,batch=batch)
    host=gate['host'];out=Path(output_dir);used=int(agent.network.step)-1
    ledger=ROOT/'receipts'/f'{host}_optimizer_ledger.json'
    if ledger.exists():
        recorded=json.loads(ledger.read_text())
        if int(recorded['completed_updates'])!=used:
            raise RuntimeError('Restart would reset update budget; resume saved state instead')
    if used>=2:raise RuntimeError('T0 cap exhausted')
    old_hash=state_digest(agent);old_params=agent.network.params
    next_rng,loss_rng=jax.random.split(agent.rng)
    (loss,loss_info),grads=probe_gradients(agent,batch,loss_rng)
    finite((loss,loss_info,grads,old_params))
    grad_norms=norms(grads)
    target_names=['modules_'+v for v in TARGET_KEYS[host].values()]
    for name in target_names:
        if grad_norms[name]!=0:raise RuntimeError('Unexpected target gradient')
    new,info=agent.update(batch)
    # Persist actual update count immediately; failed probes cannot silently restart from zero.
    json_write(ledger,dict(completed_updates=int(new.network.step)-1,stage='T0',pid=os.getpid()))
    try:
        finite((new.network.params,new.network.opt_state,info))
        if state_digest(agent)!=old_hash:raise RuntimeError('Input Agent mutated')
        if int(new.network.step)!=int(agent.network.step)+1:raise RuntimeError('Step counter mismatch')
        np.testing.assert_array_equal(np.asarray(new.rng),np.asarray(next_rng))
        np.testing.assert_allclose(float(info['critic/floq_loss']),float(loss_info['critic/floq_loss']),rtol=2e-5,atol=2e-5)
        deltas={k:float(np.sqrt(sum(np.square(np.asarray(x,dtype=np.float64)-np.asarray(y,dtype=np.float64)).sum()
                     for x,y in zip(jax.tree_util.tree_leaves(v),jax.tree_util.tree_leaves(old_params[k])))))
                     for k,v in new.network.params.items()}
        expected_trainable={'modules_floq','modules_critic','modules_actor','modules_vabc'} if host=='gfp' else {
                           'modules_floq','modules_critic','modules_actor_bc_flow'}
        for name in expected_trainable:
            if deltas[name]<=0:raise RuntimeError('Trainable module did not change: '+name)
        targets={}
        for online,target in TARGET_KEYS[host].items():
            src='modules_'+online;dst='modules_'+target
            expected=ema_specification(old_params[src],old_params[dst],agent.config['tau'])
            post=ema_specification(new.network.params[src],old_params[dst],agent.config['tau'])
            err=max(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) for x,y in
                    zip(jax.tree_util.tree_leaves(new.network.params[dst]),jax.tree_util.tree_leaves(expected)))
            post_err=max(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) for x,y in
                    zip(jax.tree_util.tree_leaves(new.network.params[dst]),jax.tree_util.tree_leaves(post)))
            if err>2e-7:raise RuntimeError('EMA pre-optimizer timing mismatch')
            if post_err<=2e-7:raise RuntimeError('Probe cannot distinguish pre/post optimizer EMA')
            max_move=max(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) for x,y in
                    zip(jax.tree_util.tree_leaves(new.network.params[dst]),jax.tree_util.tree_leaves(old_params[dst])))
            if used==0 and max_move>2e-7:raise RuntimeError('First target should remain equal to initialization')
            if used==1 and max_move<=2e-7:raise RuntimeError('Second target did not move beyond rounding tolerance')
            targets[dst]=dict(max_abs_error_pre=err,max_abs_error_post=post_err,delta_norm=deltas[dst],max_abs_movement=max_move)
        adam_old=agent.network.opt_state[0];adam_new=new.network.opt_state[0]
        if int(adam_new.count)!=int(adam_old.count)+1:raise RuntimeError('Adam count mismatch')
        for name in target_names:
            for state in [adam_new.mu,adam_new.nu]:
                if any(np.any(np.asarray(x)!=0) for x in jax.tree_util.tree_leaves(state[name])):
                    raise RuntimeError('Target optimizer moments changed')
        # Official full-state serialization and restore, between the two real updates.
        resume_dir=out/'admission_resume';path=save_state(new,resume_dir,used+1)
        restored=restore_state(new,resume_dir,used+1)
        if state_digest(restored)!=state_digest(new):raise RuntimeError('State round-trip mismatch')
        forward_before=new.sample_actions(batch['observations'],jax.random.PRNGKey(1810))
        forward_after=restored.sample_actions(batch['observations'],jax.random.PRNGKey(1810))
        np.testing.assert_array_equal(np.asarray(forward_before),np.asarray(forward_after))
        row=dict(status='PASS',update=used+1,loss=float(loss),metrics={k:float(v) for k,v in info.items()},
                 gradient_norms=grad_norms,parameter_delta_norms=deltas,targets=targets,
                 state_step_before=int(agent.network.step),state_step_after=int(new.network.step),
                 adam_count_before=int(adam_old.count),adam_count_after=int(adam_new.count),
                 input_state_unchanged=True,restored_state_sha256=state_digest(restored),
                 checkpoint_path=str(path),checkpoint_sha256=digest(path),
                 batch_sha256={k:hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k,v in batch.items()},
                 native_main=True,timestamp=time.time())
        with (ROOT/'reports'/f'{host}_t0_steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('T0_REAL_UPDATE_PASS',host,used+1,'EMA pre-optimizer; saved/restored',flush=True)
        return restored,info
    except Exception:
        # Preserve the changed state on failure; no automatic restart or hyperparameter change.
        fail=out/'admission_failure';fail.mkdir(parents=True,exist_ok=True)
        if not (fail/f'params_{used+1}.pkl').exists():upstream_save(new,str(fail),used+1)
        json_write(ROOT/'reports'/f'{host}_t0_failure.json',dict(status='FAIL_STOPPED',completed_updates=int(new.network.step)-1))
        raise


def dispatch_fused(agent,n_steps,dataset,dataset_size,batch_size,gate):
    if gate['stage']=='T0':raise RuntimeError('T0 native protocol fixes CPU-dataset single-step dispatch')
    from batch_hooks import checked_dispatch
    return checked_dispatch(agent,gate,os.environ["COMPARATIVE_OUTPUT_DIR"],n=n_steps,
        dataset=dataset,size=dataset_size,batch_size=batch_size)


def resume_index(agent):return int(agent.network.step)-1
