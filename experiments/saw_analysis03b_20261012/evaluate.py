"""CPU-only paired evaluation, complete state restore, immutable checkpoints."""
import argparse,fcntl
from state_core import *
assert __import__('os').environ.get('CUDA_VISIBLE_DEVICES')==''
assert all(d.platform=='cpu' for d in jax.devices())

def consume(pool,n=1):
    path=ROOT/'receipts'/f'{pool}_budget.json'
    with open(str(path)+'.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        x=json.loads(path.read_text()) if path.exists() else dict(used=0,limit={'E1':1600,'E2':10000,'final':2000}[pool])
        assert x['used']+n<=x['limit'],f'{pool}_COUNT_BUDGET'
        x['used']+=n;write_json(path,x)

def check_time(phase='analysis'):
    x=json.loads((ROOT/'receipts/CLOCKS.json').read_text())
    if time.time()>x[phase+'_deadline']-30:raise RuntimeError('CPU_WALL_BUDGET_STOP')

def load_checkpoint(kind,variant,step):
    base=OLD if variant=='A' else ROOT
    ident=json.loads((base/'results'/f'{kind}_checkpoints.json').read_text())
    options=[x for x in ident if x['step']==step]
    if not options:raise FileNotFoundError(f'MISSING_REGISTERED_CHECKPOINT:{kind}:{variant}:{step}')
    item=next((x for x in options if Path(x['path']).name==f'params_{step}.pkl'),options[0])
    path=Path(item['path']);expected=item['sha256']
    actual=sha(path);assert actual==expected
    env,data,val,config=load(f'pointmaze-large-{kind}-v0',variant)
    agent=init_agent(data,config)
    with open(path,'rb') as f:state=pickle.load(f)
    assert state['step']==step
    agent=flax.serialization.from_state_dict(agent,state['agent']);agent=jax.device_put(agent,jax.devices('cpu')[0])
    assert int(agent.network.step)==step+1,'OPTIMIZER_COUNTER_MISMATCH: official TrainState starts at 1'
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(agent.network.params))
    env.close()
    return agent,dict(path=str(path),sha256=actual,step=step,config=config)

def restore_gate(env,scene):
    s,info=restore(env,scene);now=snapshot(env,info,scene['reset_seed'])
    errors={k:float(np.max(np.abs(np.asarray(now[k])-np.asarray(scene[k])))) for k in ('integration','qpos','qvel','goal','goal_xy')}
    assert max(errors.values())<=1e-10,errors
    for k in ('env_rng','action_rng','numpy_rng','python_rng','wrappers'):assert digest(now[k])==digest(scene[k]),k
    return s,errors

def episode(agent,env,scene,pool):
    consume(pool);s,errors=restore_gate(env,scene);g=scene['goal'];u=env.unwrapped
    states=[s.copy()];qvel=[u.data.qvel.copy()];ints=[integration(u)];actions=[];flags=[]
    term=trunc=False
    while not(term or trunc):
        check_time('final' if pool=='final' else 'analysis')
        a=np.asarray(mode(agent,s,g));s,r,term,trunc,info=env.step(a)
        states.append(s.copy());qvel.append(u.data.qvel.copy());ints.append(integration(u));actions.append(a);flags.append((float(r),bool(term),bool(trunc),float(info['success'])))
        assert len(actions)<=1000
    xy=np.asarray(states);act=np.asarray(actions);move=np.linalg.norm(np.diff(xy,axis=0),axis=1)
    request=.2*np.linalg.norm(act,axis=1);ineff=(request>.05)&(move/(request+1e-12)<=.02)
    longest=run=0
    for flag in ineff:run=run+1 if flag else 0;longest=max(longest,run)
    row=dict(scene_id=scene['scene_id'],task=scene['task'],group=scene['group'],episode=scene['episode'],stream=scene['stream'],
        success=float(info['success']),steps=len(actions),initial_distance=float(np.linalg.norm(xy[0]-g)),final_distance=float(np.linalg.norm(xy[-1]-g)),
        path_length=float(move.sum()),ineffective_steps=int(ineff.sum()),longest_ineffective_run=longest,
        small_request_steps=int((request<=.05).sum()),terminated=bool(term),truncated=bool(trunc),restore_max_error=max(errors.values()),full_scene_sha256=digest(scene))
    return row,dict(initial=scene,states=xy,qvel=np.asarray(qvel),integration=np.asarray(ints),actions=act,flags=flags)

def main(kind,variant,phase):
    start=time.time();env=env_new(kind)
    with open(ROOT/'runs'/f'{phase}_{kind}_SCENES.pkl','rb') as f:scenes=pickle.load(f)
    scenes_id=json.loads((ROOT/'results'/f'{phase}_{kind}_SCENES_IDENTITY.json').read_text())
    assert sha(scenes_id['path'])==scenes_id['sha256']
    prefix=f'{phase}_{kind}_{variant}';all_rows=[];identities=[]
    planned=(100000,300000,500000,1000000) if phase=='E1' else (1000000,)
    steps=list(planned);missing=[]
    if variant=='B':
        registered=json.loads((ROOT/'results'/f'{kind}_checkpoints.json').read_text())
        available={x['step'] for x in registered};missing=[s for s in planned if s not in available]
        steps=[s for s in planned if s in available]
        if phase=='final' and missing:raise RuntimeError('NO_COMPLETE_1M_ENDPOINT')
    for step in steps:
        check_time('final' if phase=='final' else 'analysis');agent,identity=load_checkpoint(kind,variant,step);identities.append(identity);raw=[]
        for scene in scenes:
            row,trace=episode(agent,env,scene,phase);row.update(kind=kind,variant=variant,checkpoint_step=step);all_rows.append(row);raw.append(dict(**row,**trace))
            write_json(ROOT/'receipts'/f'{prefix}_progress.json',dict(rows=len(all_rows),step=step,seconds=time.time()-start,scene=row['scene_id']))
        path=ROOT/'runs'/f'{prefix}_{step}_trajectories.pkl';dump(path,raw)
        csv_rows(ROOT/'results'/f'{prefix}.csv',all_rows)
        print(prefix,step,'DONE',len(all_rows),flush=True)
    env.close();write_json(ROOT/'receipts'/f'{prefix}_exit.json',dict(status='PARTIAL_MISSING_CHECKPOINTS' if missing else 'COMPLETED',exit_code=0,missing_steps=missing,episodes=len(all_rows),seconds=time.time()-start,checkpoints=identities))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',required=True);p.add_argument('--variant',required=True,choices=['A','B']);p.add_argument('--phase',required=True,choices=['E1','final']);a=p.parse_args();main(a.kind,a.variant,a.phase)
