"""Instrumentation only; no alternative policy, critic loss, or optimizer."""
import os,json,time,random,hashlib
from pathlib import Path
from functools import partial
import numpy as np
import jax,jax.numpy as jnp

ROOT=Path(os.environ['COMPARATIVE_ROOT'])
JOB=os.environ['COMPARATIVE_JOB']
OUT=ROOT/'runs'/JOB
START=time.time()
LAST_PROGRESS=0
INITIAL_TARGETS={}

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,default=lambda x:np.asarray(x).tolist()))
    tmp.replace(path)

def progress(stage,step,**extra):
    write(OUT/'training_progress.json',dict(job=JOB,stage=stage,updates=int(step),pid=os.getpid(),
        epoch=time.time(),elapsed=time.time()-START,**extra))

def initial(agent,output_dir,gate):
    OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/'runtime.json',dict(gate=gate,output_dir=str(output_dir),agent_config=dict(agent.config),
        jax_version=jax.__version__,devices=[str(x) for x in jax.devices()],
        matmul_precision=str(jax.config.jax_default_matmul_precision),jax_enable_x64=jax.config.jax_enable_x64,
        initial_step=int(agent.network.step),pid=os.getpid()))
    for key,value in agent.network.params.items():
        if 'target_' in key:INITIAL_TARGETS[key]=jax.tree.map(lambda x:np.asarray(x).copy(),value)
    progress('INITIALIZED',int(agent.network.step)-1)

def health(new,info):
    leaves=jax.tree.leaves((new.network.params,new.network.opt_state,info))
    checks=[jnp.all(jnp.isfinite(x)) for x in leaves]
    param_max=jnp.max(jnp.stack([jnp.max(jnp.abs(x)) for x in jax.tree.leaves(new.network.params)]))
    metric_max=jnp.max(jnp.stack([jnp.max(jnp.abs(x)) for x in jax.tree.leaves(info)]))
    return dict(finite=jnp.all(jnp.stack(checks)),parameter_max=param_max,metric_max=metric_max,
        state_step=new.network.step,adam_count=new.network.opt_state[0].count)

@jax.jit
def checked_single(agent,batch):
    new,info=(agent._update_impl(batch) if hasattr(agent,'_update_impl') else agent.update(batch))
    return new,info,health(new,info)

@partial(jax.jit,static_argnames=('n','size','batch_size'))
def checked_fused(agent,n,dataset,size,batch_size):
    if hasattr(agent,'_multi_sample_impl'):
        new,info=agent._multi_sample_impl(n,dataset,size,batch_size)
    else:new,info=agent.multi_sample_and_update(n,dataset,size,batch_size)
    return new,info,health(new,info)

def checked_dispatch(agent,gate,output_dir,batch=None,n=1,dataset=None,size=None,batch_size=None):
    import training_wiring as wiring
    global LAST_PROGRESS
    used=int(agent.network.step)-1
    if used+n>gate['optimizer_steps_allowed']:raise RuntimeError('UPDATE_BUDGET_EXHAUSTED')
    if hasattr(agent,'_check_update_permission'):agent._check_update_permission(n)
    if used==0:
        progress('COMPILING_FIRST_REAL_UPDATE',0,path='fused' if dataset is not None else 'single')
        write(OUT/'initial_state.json',dict(state_sha256=wiring.state_digest(agent),step=int(agent.network.step)))
    new,info,h=(checked_single(agent,batch) if dataset is None else checked_fused(agent,n,dataset,size,batch_size))
    h={k:float(np.asarray(v)) for k,v in h.items()}
    completed=used+n
    reason=None
    if not h['finite']:reason='NaN/Inf in metrics, parameters or optimizer'
    elif h['state_step']!=completed+1 or h['adam_count']!=completed:reason='Update/Adam count mismatch'
    elif h['parameter_max']>1e6 or h['metric_max']>1e12:reason='Frozen numerical magnitude threshold exceeded'
    if reason:
        failure=Path(output_dir)/'failure_state';wiring.save_state(new,failure,completed)
        progress('FAIL_STOPPED',completed,reason=reason,health=h)
        raise FloatingPointError(reason)
    # Actual native dispatch admission counts toward this job's 1M, never a warmup run.
    if used==0:
        admission=Path(output_dir)/'dispatch_admission'
        wiring.save_state(new,admission,completed)
        restored=wiring.restore_state(new,admission,completed)
        if wiring.state_digest(new)!=wiring.state_digest(restored):raise RuntimeError('Dispatch restore mismatch')
        new=restored
        write(OUT/'dispatch_admission.json',dict(status='PASS',actual_updates=n,completed_updates=completed,
            path='official_gpu_sampler_fused' if dataset is not None else 'official_cpu_sampler_single',
            health=h,checkpoint=str(admission),restored_state_sha256=wiring.state_digest(new)))
        progress('TRAINING',completed,health=h)
    if completed%5000==0 or time.time()-LAST_PROGRESS>30:
        progress('TRAINING',completed,health=h)
        LAST_PROGRESS=time.time()
    return new,info

def set_eval_context(step,component):
    progress('NATIVE_EVALUATION',step,component=component)
    os.environ['COMPARATIVE_EVAL_STEP']=str(step)
    os.environ['COMPARATIVE_EVAL_COMPONENT']=component

def native_receipt(result):
    stats,trajs,renders=result
    step=int(os.environ['COMPARATIVE_EVAL_STEP']);comp=os.environ['COMPARATIVE_EVAL_COMPONENT']
    episodes=[]
    for idx,t in enumerate(trajs):
        info=t['info'][-1] if t.get('info') else {}
        episodes.append(dict(episode=idx,return_=float(np.sum(t.get('reward',[]))),length=len(t.get('reward',[])),
            success=float(info.get('success',0)),terminal_info=info))
    write(OUT/'native'/f'{step}_{comp}.json',dict(step=step,component=comp,stats=stats,episodes=episodes,
        protocol='unchanged official evaluation function; frozen episode count/cadence'))
    return result

def checkpoint_hook(agent,output_dir,step):
    import training_wiring as wiring
    if step%100000==0:
        progress('CHECKPOINT_VALIDATION',step)
        restored=wiring.restore_state(agent,output_dir,step)
        if wiring.state_digest(agent)!=wiring.state_digest(restored):raise RuntimeError('Checkpoint restore mismatch')
    if step==50000 and os.environ['COMPARATIVE_PHASE']=='T2':
        movements={k:float(np.sqrt(sum(np.square(np.asarray(x,dtype=np.float64)-np.asarray(y,dtype=np.float64)).sum()
            for x,y in zip(jax.tree.leaves(v),jax.tree.leaves(INITIAL_TARGETS[k])))))
            for k,v in agent.network.params.items() if k in INITIAL_TARGETS}
        if not movements or any(v<=0 or not np.isfinite(v) for v in movements.values()):
            raise RuntimeError('Target did not move or became nonfinite')
        # Pure gradients; diagnostics cannot advance the Agent RNG or optimizer.
        batch=np.load(ROOT/'data'/f"{os.environ['COMPARATIVE_TASK']}.batch.npz")
        batch={k:batch[k] for k in batch.files}
        (loss,aux),grads=wiring.probe_gradients(agent,batch,jax.random.PRNGKey(14014))
        wiring.finite((loss,aux,grads))
        norms=wiring.norms(grads)
        if any(v!=0 for k,v in norms.items() if 'target_' in k):raise RuntimeError('Unexpected target gradient')
        write(OUT/'t2_admission.json',dict(status='PASS',updates=step,target_movements=movements,
            gradient_norms=norms,loss=float(loss),state_sha256=wiring.state_digest(agent)))
        progress('T2_PASS_SAVED',step,checkpoint_directory=str(output_dir))
    # Matched runs use the read-only evaluation manager with global OGBench RNG control.
    # Native evaluator statistics remain untouched.

def matched(agent,step):
    import ogbench
    progress('MATCHED_EVALUATION',step)
    env=ogbench.make_env_and_datasets(os.environ['COMPARATIVE_TASK'],env_only=True)
    np_state=np.random.get_state();py_state=random.getstate()
    episodes=[]
    try:
        for ep in range(100):
            reset_seed=20261014+ep;policy_seed=420000+ep
            ob,info=env.reset(seed=reset_seed);key=jax.random.PRNGKey(policy_seed)
            done=False;ret=0.;length=0;success=0.
            while not done:
                key,sub=jax.random.split(key)
                action=np.clip(np.asarray(agent.sample_actions(observations=ob,seed=sub,temperature=0)),-1,1)
                if not np.isfinite(action).all():raise FloatingPointError('Nonfinite matched action')
                ob,reward,term,trunc,info=env.step(action);done=term or trunc
                ret+=float(reward);length+=1;success=max(success,float(info.get('success',0)))
            episodes.append(dict(episode=ep,return_=ret,length=length,success=success,reset_seed=reset_seed,
                policy_seed=policy_seed,terminated=bool(term),truncated=bool(trunc)))
        write(OUT/'matched'/f'{step}.json',dict(step=step,episodes=episodes,
            success_rate=float(np.mean([e['success'] for e in episodes])),
            mean_return=float(np.mean([e['return_'] for e in episodes])),
            policy_randomness='explicit fresh Gaussian-policy key every action; temperature=0 remains stochastic'))
    finally:
        env.close();np.random.set_state(np_state);random.setstate(py_state)
    progress('TRAINING',step)

def finished(agent):
    step=int(agent.network.step)-1
    progress('PHASE_COMPLETED',step,phase=os.environ['COMPARATIVE_PHASE'])
