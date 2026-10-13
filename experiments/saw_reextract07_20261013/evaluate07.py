"""CPU-only frozen complete-state evaluation and exact consumed SAW06 probes."""
from common07 import *
import argparse, importlib.util, csv, random, traceback, signal
stopped=[False];signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
assert all(d.platform=='cpu' for d in jax.devices())
assert os.sched_getaffinity(0)=={75}
spec=importlib.util.spec_from_file_location('saw06env',D06/'core.py');ec=importlib.util.module_from_spec(spec);spec.loader.exec_module(ec)
bankmeta=json.loads((ROOT/'EVALUATION_BANK.json').read_text());assert sha(D06/'core.py')==bankmeta['d06_core_sha256']
assert sha(bankmeta['path'])==bankmeta['sha256'] and sha(bankmeta['distance_bank_path'])==bankmeta['distance_bank_sha256']
with open(bankmeta['path'],'rb') as f:scenes=pickle.load(f)
with open(bankmeta['distance_bank_path'],'rb') as f:localbank=pickle.load(f)
assert len(scenes)==250 and len(localbank)==93
ident=json.loads((ROOT/'REFERENCE_AND_INIT_IDENTITY.json').read_text())
counterpath=ROOT/'receipts/EVALUATION_COUNTER.json'
counter=json.loads(counterpath.read_text()) if counterpath.exists() else dict(complete_episodes=0,complete_steps=0,local_control_steps=0,local_trajectories=0,technical_episodes=0)
def localstep(env,action,phase):
    if stopped[0]:raise RuntimeError('OWNED_MONITOR_STOP')
    assert counter['local_control_steps']<32000;counter['local_control_steps']+=1
    return env.step(action)
ec.guard=lambda:None;ec.step=localstep
def rows(path,records):
    if not records:return
    with open(path,'x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
def morphology(xy,actions):
    v=np.diff(xy,axis=0)
    def reverse(v):
        if len(v)<2:return 0
        n=np.linalg.norm(v,axis=1);den=n[:-1]*n[1:]
        return int(np.sum((n[:-1]>.01)&(n[1:]>.01)&(np.sum(v[:-1]*v[1:],axis=1)/np.maximum(den,1e-30)<-.5)))
    repeat=sum(bool(np.any(np.linalg.norm(xy[:t-2]-xy[t],axis=1)<=.1)) for t in range(3,len(xy)))
    return reverse(v),reverse(actions),repeat
def complete(agent,label,r,arm,u,cp):
    env=ec.env_new();out=ROOT/'runs'/f'EVAL_{label}';out.mkdir(exist_ok=False);rec=[]
    chosen=[x for x in scenes if u in (0,300000,400000,500000) or x['episode']<10]
    action_fn=ec.raw_mode
    for x in chosen:
        assert counter['complete_episodes']<12200
        s,err=ec.restore(env,x['scene']);g=np.asarray(x['scene']['goal'],dtype=np.float32);actual_goal=env.unwrapped.cur_goal_xy.copy()
        full=[ec.integration(env.unwrapped)];xy=[s.copy()];vel=[env.unwrapped.data.qvel.copy()];raw=[];actions=[];reward=[];ts=[]
        term=trunc=False
        while not (term or trunc):
            if stopped[0]:raise RuntimeError('OWNED_MONITOR_STOP')
            a=np.asarray(action_fn(agent,np.asarray(s,dtype=np.float32),g));assert a.shape==(2,) and np.isfinite(a).all();c=np.clip(a,-1,1)
            s,rew,term,trunc,info=env.step(c)
            counter['complete_steps']+=1;raw.append(a);actions.append(c);reward.append(rew);ts.append((bool(term),bool(trunc),float(info['success'])))
            xy.append(s.copy());vel.append(env.unwrapped.data.qvel.copy());full.append(ec.integration(env.unwrapped))
            assert len(actions)<=1000
        states=np.asarray(xy);distance=float(np.linalg.norm(s-actual_goal));success=float(info['success'])
        # Independent original threshold/termination check from saved positions.
        inferred=(np.linalg.norm(states[1:]-actual_goal,axis=1)<=1.0).astype(float)
        assert np.array_equal(inferred,np.asarray(ts)[:,2]),'SUCCESS_RECOMPUTATION_MISMATCH'
        assert bool(term)==bool(inferred[-1]) and bool(trunc)==(len(actions)==1000)
        row=dict(block=r,arm=arm,student_step=u,model_label=label,stream=x['stream'],task=x['task'],episode=x['episode'],scene_id=x['scene_id'],success=success,steps=len(actions),terminated=bool(term),truncated=bool(trunc),initial_x=float(states[0,0]),initial_y=float(states[0,1]),goal_x=float(actual_goal[0]),goal_y=float(actual_goal[1]),final_distance=distance,path_length=float(np.linalg.norm(np.diff(states,axis=0),axis=1).sum()),restore_error=err,checkpoint_sha256=cp['sha256'],evaluation_bank_sha256=bankmeta['sha256'])
        path=out/f'{x["scene_id"]}.npz';np.savez_compressed(path,states=states,qvel=np.asarray(vel),integration=np.asarray(full),raw_actions=np.asarray(raw),actions=np.asarray(actions),rewards=np.asarray(reward),termination_success=np.asarray(ts),goal=actual_goal,model_goal=g)
        row.update(trace_path=str(path),trace_sha256=sha(path));rec.append(row);counter['complete_episodes']+=1;write(counterpath,counter)
        if len(rec)%25==0:print('EVAL',label,len(rec),flush=True)
    env.close();rows(ROOT/'results'/f'{label}_episodes.csv',rec)
    write(ROOT/'results'/f'{label}_evaluation.json',dict(status='COMPLETE',block=r,arm=arm,student_step=u,model_label=label,episodes=len(rec),per_task={str(t):float(np.mean([x['success'] for x in rec if x['task']==t])) for t in range(1,6)},overall=float(np.mean([x['success'] for x in rec])),checkpoint=cp,evaluation_bank_sha256=bankmeta['sha256'],independent_success_termination_recomputation=True,clip=True,temperature=0,max_steps=1000,success_tol=1.0))
def local(agent,label,r,arm,cp):
    env=ec.env_new();geom=ec.Geometry(env);out=ROOT/'runs'/f'DISTANCE_{label}';out.mkdir(exist_ok=False);records=[]
    fn=lambda s,g:ec.raw_mode(agent,np.asarray(s,dtype=np.float32),np.asarray(g,dtype=np.float32))
    for x in localbank:
        assert x['admitted']
        item=dict(state_id=x['condition_id'],group=x['bank'],cluster=str(x['start_cell']),source=x['source'],cell=x['start_cell'],scene=x['scene'],goals=[x['goal_xy']],goal_cells=[x['goal_cell']],firsts=[x['firsts']],routes=[geom.route(x['start_cell'],x['goal_cell'])])
        metric,trace=ec.rollout(env,geom,item,0,fn,'saw07_formal_local');xy=trace['states'];act=trace['actions'];norms=np.linalg.norm(act,axis=1);dr,ar,rv=morphology(xy,act)
        end=geom.cell(xy[-1]);progress=x['distance']-geom.distances[x['goal_cell']][end] if end in geom.free else None
        metric.update(block=r,arm=arm,condition_id=x['condition_id'],bank=x['bank'],role=x['role'],start_cell=json.dumps(x['start_cell']),goal_cell=json.dumps(x['goal_cell']),region=x['region'],distance=x['distance'],firsts=json.dumps(x['firsts']),median_action_norm=float(np.median(norms)),graph_progress=progress,displacement_reversals=dr,action_reversals=ar,repeated_visit_steps=rv,small_command_candidate=bool(norms.mean()<=.25),near_stationary=bool(metric['maximum_displacement']<=.1),backtracking_evidence=bool(metric['path_length']>=1 and dr>=3 and rv>=2),checkpoint_sha256=cp['sha256'])
        trace.update(metric=metric,goal_xy=x['goal_xy']);path=out/(x['condition_id']+'.pkl');dump(path,trace);metric.update(trace_path=str(path),trace_sha256=sha(path));records.append(metric);counter['local_trajectories']+=1;write(counterpath,counter)
    assert len(records)==93;env.close();rows(ROOT/'results'/f'{label}_distance.csv',records)
    write(ROOT/'results'/f'{label}_distance_status.json',dict(status='COMPLETE',count=93,bank_sha256=bankmeta['distance_bank_sha256'],all_restores_audited=True,checkpoint=cp))
def main(phase,watch):
    started=time.time();arms=[0,1] if phase=='I' else [2,3];done=[];models={}
    try:
        if phase=='I':
            for r in range(3):
                label=f'source_A_b{r}'
                if (ROOT/'results'/f'{label}_evaluation.json').exists():continue
                ref=load_ref(ident[r]['reference']);complete(ref,label,r,'SOURCE_A',0,ident[r]['reference']);done.append(label)
        while True:
            pending=False;executed=False
            for r in range(3):
                for a in arms:
                    label=f'b{r}_a{a}';ix=ROOT/'results'/f'{label}_checkpoints.json'
                    cps=json.loads(ix.read_text()) if ix.exists() else []
                    for u in (100000,200000,300000,400000,500000):
                        tag=f'{label}_u{u}';result=ROOT/'results'/f'{tag}_evaluation.json'
                        cp=next((x for x in cps if x['step']==u and x['formal_savepoint']),None)
                        if cp is None:pending=True;continue
                        if result.exists() and (u!=500000 or (ROOT/'results'/f'{tag}_distance_status.json').exists()):continue
                        assert sha(cp['path'])==cp['sha256']
                        if r not in models:models[r]=load_ref(ident[r]['reference'])
                        with open(cp['path'],'rb') as f:saved=pickle.load(f)
                        assert saved['student_step']==u and saved['bank_sha256']==bankmeta['sha256'];agent=policy(models[r],jax.device_put(saved['params']))
                        if not result.exists():complete(agent,tag,r,a,u,cp)
                        if u==500000 and not (ROOT/'results'/f'{tag}_distance_status.json').exists():local(agent,tag,r,a,cp)
                        done.append(tag);executed=True
            if not pending:break
            for r in range(3):
                for a in arms:
                    st=ROOT/'results'/f'b{r}_a{a}_status.json'
                    if st.exists():assert json.loads(st.read_text())['status']=='COMPLETE_500K','FORMAL_TRAINING_FAILED'
            if not watch:raise RuntimeError('REQUESTED_CHECKPOINTS_NOT_AVAILABLE')
            if not executed:time.sleep(5)
        write(ROOT/'receipts'/f'{phase}_EVALUATION_EXIT.json',dict(status='COMPLETE',exit_code=0,start_epoch=started,elapsed_seconds=time.time()-started,models_evaluated_this_process=done,counters=counter))
    except Exception as exc:
        write(ROOT/'receipts'/f'{phase}_EVALUATION_FAILURE.json',dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),elapsed_seconds=time.time()-started,counters=counter));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['I','II'],required=True);p.add_argument('--watch',action='store_true');x=p.parse_args();main(x.phase,x.watch)
