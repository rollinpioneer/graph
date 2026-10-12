"""Predetermined real A snapshots; complete restore and privileged feasibility gates."""
from evaluate import *
from scipy.spatial import cKDTree

low_mode=jax.jit(lambda a,s,g:jax.numpy.clip(a.network.select('low_actor')(s,g,temperature=0.).mode(),-1,1))

def choose_snapshots():
    selected=[]
    for kind in ('navigate','stitch'):
        for task in (1,2,4,5):
            for ep in range(4):
                path=EV/'runs'/f'A_{kind}_task{task}_scene{ep}.pkl'
                with open(path,'rb') as f:original=pickle.load(f)
                trace=original['official'];steps=trace['steps'];n=len(steps)
                states=np.asarray([trace['initial']['qpos']]+[s['qpos'] for s in steps]);acts=np.asarray([s['action'] for s in steps])
                request=.2*np.linalg.norm(acts,axis=1);move=np.linalg.norm(np.diff(states,axis=0),axis=1)
                bad=(request>.05)&(move/(request+1e-12)<=.02);success=bool(steps[-1]['success'])
                k=min(300,n//2);rule='failure_midpoint'
                if success:k=n//2;rule='success_midpoint'
                else:
                    for j in range(100,min(801,n-10)):
                        if bad[j:j+10].sum()>=8:k=j;rule='first_registered_stall';break
                k=max(0,min(k,n-1));scene=copy.deepcopy(original['scene'])
                if k:
                    previous=steps[k-1]
                    for key in ('integration','qpos','qvel'):scene[key]=previous[key].copy()
                    for wrapper in scene['wrappers']:
                        if '_elapsed_steps' in wrapper['fields']:wrapper['fields']['_elapsed_steps']=k
                sid=f'{kind}_t{task}_e{ep}_s{k}'
                selected.append(dict(id=sid,kind=kind,task=task,episode=ep,source=str(path),source_sha256=sha(path),
                    source_success=success,source_steps=n,snapshot_step=k,selection_rule=rule,scene=scene,next_step=copy.deepcopy(steps[k])))
    dump(ROOT/'runs/E2_SNAPSHOTS.pkl',selected)
    write(ROOT/'results/E2_SELECTION.json',dict(count=len(selected),path=str(ROOT/'runs/E2_SNAPSHOTS.pkl'),sha256=sha(ROOT/'runs/E2_SNAPSHOTS.pkl'),
        rule='pre-registered fixed datasets/tasks/episodes; no replacement after gate or branch results',
        entries=[{k:v for k,v in x.items() if k not in ('scene','next_step')} for x in selected]))
    return selected

def candidates(raw,tree,valid,scene):
    s=scene['qpos'];v=scene['qvel'];terminal=np.flatnonzero(raw['terminals'])
    _,near=tree.query(s,k=100);ids=valid[np.atleast_1d(near)]
    eligible=[]
    for i in ids:
        pe=float(np.max(np.abs(raw['qpos'][i]-s)));ve=float(np.max(np.abs(raw['qvel'][i]-v)))
        end=terminal[np.searchsorted(terminal,i)];w=min(i+25,end)
        delta=raw['observations'][w]-s;distance=float(np.linalg.norm(delta))
        if pe<=.15 and ve<=.5 and 1.05<distance<=5:
            eligible.append(dict(state_idx=int(i),waypoint_idx=int(w),qpos_error=pe,qvel_error=ve,distance=distance,goal=raw['observations'][w].copy(),delta=delta))
    if not eligible:return None,None,'NO_FULL_STATE_SUPPORTED_NEAR_TARGET'
    chosen=eligible[0];control=None
    for item in sorted(eligible,key=lambda x:x['state_idx']):
        cos=float(np.dot(item['delta'],chosen['delta'])/(item['distance']*chosen['distance']))
        if .75*chosen['distance']<=item['distance']<=1.25*chosen['distance'] and cos<=.5:
            control=item;break
    return chosen,control,None if control else 'NO_COMPARABLE_CONTROL_TARGET'

def feasibility(env,scene,target):
    s,_=restore_gate(env,scene);env.unwrapped.set_goal(goal_xy=np.asarray(target,dtype=float))
    moves=[]
    for step in range(25):
        check_time();consume('E2');a=np.clip((target-s)/.2,-1,1);s,r,t,tr,info=env.step(a);moves.append(s.copy())
        if info['success']:return True,step+1,np.asarray(moves)
        if tr:break
    return False,len(moves),np.asarray(moves)

def matching(env,scene,source,agent):
    s,errors=restore_gate(env,scene);a=np.asarray(mode(agent,s,scene['goal']))
    ae=float(np.max(np.abs(a-source['action'])));assert ae<=1e-6
    consume('E2');out,r,t,tr,info=env.step(a);u=env.unwrapped
    pe=float(np.max(np.abs(out-source['qpos'])));ve=float(np.max(np.abs(u.data.qvel-source['qvel'])))
    ie=float(np.max(np.abs(integration(u)-source['integration'])))
    assert max(pe,ve,ie)<=1e-5
    assert (float(r),bool(t),bool(tr),float(info['success']))==(source['reward'],source['terminated'],source['truncated'],source['success'])
    return dict(initial_max_error=max(errors.values()),action_error=ae,next_qpos_error=pe,next_qvel_error=ve,next_integration_error=ie)

def branch(env,snap,agent,early,condition,near,control):
    scene=snap['scene'];s,errors=restore_gate(env,scene);far=scene['goal'];local=control if condition=='C3' else near
    states=[s.copy()];velocities=[env.unwrapped.data.qvel.copy()];integ=[integration(env.unwrapped)];actions=[];flags=[];goals=[]
    local_reached=False;local_step=None;local_attempted=condition in ('C1','C2','C3');phase_local=local_attempted
    for step in range(50):
        check_time()
        if phase_local and np.linalg.norm(s-local)<=1.0:local_reached=True;local_step=step;phase_local=False
        if step>=25:phase_local=False
        target=local if phase_local else far
        policy=low_mode if condition=='C2' and phase_local else mode
        used=early if condition=='C4' else agent
        a=np.asarray(policy(used,s,target));consume('E2');s,r,t,tr,info=env.step(a)
        states.append(s.copy());velocities.append(env.unwrapped.data.qvel.copy());integ.append(integration(env.unwrapped));actions.append(a);goals.append(target.copy());flags.append((float(r),bool(t),bool(tr),float(info['success'])))
        if phase_local and np.linalg.norm(s-local)<=1.0:local_reached=True;local_step=step+1;phase_local=False
        if t or tr:break
    xy=np.asarray(states);act=np.asarray(actions);move=np.linalg.norm(np.diff(xy,axis=0),axis=1);request=.2*np.linalg.norm(act,axis=1)
    row=dict(snapshot_id=snap['id'],kind=snap['kind'],task=snap['task'],condition=condition,source_success=snap['source_success'],
        source_snapshot_step=snap['snapshot_step'],steps=len(actions),local_attempted=local_attempted,local_reached=local_reached,local_first_step=local_step,
        original_far_success=bool(info['success']),terminated=bool(t),truncated=bool(tr),
        initial_far_distance=float(np.linalg.norm(xy[0]-far)),final_far_distance=float(np.linalg.norm(xy[-1]-far)),
        far_progress=float(np.linalg.norm(xy[0]-far)-np.linalg.norm(xy[-1]-far)),net_displacement=float(np.linalg.norm(xy[-1]-xy[0])),
        path_length=float(move.sum()),ineffective_steps=int(((request>.05)&(move/(request+1e-12)<=.02)).sum()),restore_max_error=max(errors.values()))
    return row,dict(**row,initial=scene,states=xy,qvel=np.asarray(velocities),integration=np.asarray(integ),actions=act,flags=flags,policy_goals=np.asarray(goals),original_goal=far)

def main():
    start=time.time();selected=choose_snapshots();gates=[];rows=[];raw_branches=[];candidate_map=[]
    for kind in ('navigate','stitch'):
        agent,identity=load_checkpoint(kind,'A',1000000);early,earlyid=load_checkpoint(kind,'A',300000)
        env=env_new(kind);raw=np.load(OLD/'vendor/data'/f'pointmaze-large-{kind}-v0.npz');valid=np.flatnonzero(1-raw['terminals']);tree=cKDTree(raw['qpos'][valid])
        for snap in [x for x in selected if x['kind']==kind]:
            check_time();gate=dict(snapshot_id=snap['id'],kind=kind,task=snap['task'],source_success=snap['source_success'],status='STOPPED')
            try:
                matches=[matching(env,snap['scene'],snap['next_step'],agent) for _ in range(2)]
                gate.update(match=matches)
                near,control,reason=candidates(raw,tree,valid,snap['scene'])
                if reason:gate['reason']=reason;gates.append(gate);continue
                candidate_map.append(dict(snapshot_id=snap['id'],near=near,control=control))
                valid_near,nsteps,ntrace=feasibility(env,snap['scene'],near['goal']);valid_control,csteps,ctrace=feasibility(env,snap['scene'],control['goal'])
                gate.update(near=near,control=control,feasibility_near=valid_near,feasibility_control=valid_control,selftest_steps=nsteps+csteps)
                if not(valid_near and valid_control):gate['reason']='PRIVILEGED_REACHABILITY_GATE_FAILED';gates.append(gate);continue
                gate.update(status='PASS',reason=None)
                for cond in ('C0','C1','C2','C3','C4'):
                    row,trace=branch(env,snap,agent,early,cond,near['goal'],control['goal']);rows.append(row);raw_branches.append(trace)
                gates.append(gate)
            except (AssertionError,RuntimeError) as e:
                gate['reason']='MATCH_OR_BUDGET_GATE:'+str(e);gates.append(gate)
                if 'BUDGET' in str(e):break
            write(ROOT/'results/E2_GATES.json',gates)
            if rows:csv_rows(ROOT/'results/E2_BRANCHES.csv',rows)
            write(ROOT/'receipts/E2_progress.json',dict(processed=len(gates),branches=len(rows),seconds=time.time()-start))
            print('E2',snap['id'],gate['status'],flush=True)
        env.close();raw.close()
    write(ROOT/'results/E2_GATES.json',gates)
    if rows:csv_rows(ROOT/'results/E2_BRANCHES.csv',rows)
    dump(ROOT/'runs/E2_BRANCH_TRAJECTORIES.pkl',raw_branches);dump(ROOT/'runs/E2_CANDIDATES.pkl',candidate_map)
    write(ROOT/'receipts/E2_exit.json',dict(status='COMPLETED',candidate_snapshots=len(selected),gates=len(gates),passed=sum(x['status']=='PASS' for x in gates),branches=len(rows),seconds=time.time()-start,
        selftest_semantics='privileged coordinate controller verifies feasible local targets; not an agent success or training intervention'))
    print('E2_DONE',len(rows),flush=True)

if __name__=='__main__':main()
