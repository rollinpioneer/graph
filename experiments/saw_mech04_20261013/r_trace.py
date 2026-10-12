"""All preregistered discovery cases, earliest same-state split, physical replay."""
import argparse
from frozen import *
def sources(kind):
    if kind=='navigate':
        p=PREV/'runs/E1_navigate_A_300000_trajectories.pkl'
        with p.open('rb') as f:traces=pickle.load(f)
        selected=[x for x in traces if x['task']==5 or (x['task'] in (1,2) and x['episode'] in (0,1))]
        selected=sorted(selected,key=lambda x:(x['task']!=5,x['scene_id']))
        return [(f"RA_{x['scene_id']}",x,'Task5_main' if x['task']==5 else 'Task1_2_control') for x in selected],[]
    with (PREV/'runs/final_stitch_A_1000000_trajectories.pkl').open('rb') as f:a=pickle.load(f)
    with (PREV/'runs/final_stitch_B_1000000_trajectories.pkl').open('rb') as f:b={x['scene_id']:x for x in pickle.load(f)}
    selected=[];negative=[]
    for task in range(1,6):
        specs=[('both_success',1,1,1),('both_failure',0,0,1)]
        if task in (1,3,4):specs.insert(0,('A_success_B_failure',1,0,3))
        for label,sa,sb,n in specs:
            eligible=sorted([x for x in a if x['task']==task and x['success']==sa and b[x['scene_id']]['success']==sb],key=lambda x:x['scene_id'])
            negative.append(dict(kind=kind,task=task,category=label,eligible=len(eligible),planned=n,selected=min(n,len(eligible)),reason='NO_CASE_AVAILABLE' if not eligible else 'PREREGISTERED_ORDER'))
            for x in eligible[:n]:
                assert x['full_scene_sha256']==b[x['scene_id']]['full_scene_sha256']
                selected.append((f"RB_{label}_{x['scene_id']}",x,label))
    return selected,negative
def branch(env,scene,agent,case_id,anchor,variant,step,reference):
    s,error=restore_gate(env,scene);states=[s.copy()];acts=[];ints=[integration(env.unwrapped)];flags=[]
    for _ in range(50):
        a=np.asarray(mode(agent,s,scene['goal']));control_count();s,reward,term,trunc,info=env.step(a);states.append(s.copy());acts.append(a);ints.append(integration(env.unwrapped));flags.append((float(reward),bool(term),bool(trunc),float(info['success'])))
        if term or trunc:break
    points=np.asarray(states);row=dict(case_id=case_id,anchor=anchor,kind=case_id[1:2],variant=variant,checkpoint_step=step,task=scene['task'],steps=len(acts),
        initial_restore_error=error,original_success=bool(flags[-1][3]),net_displacement=float(np.linalg.norm(points[-1]-points[0])),path_length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
        reference_50step_distance=float(np.linalg.norm(points[-1]-reference)),far_euclidean_progress=float(np.linalg.norm(points[0]-scene['goal'])-np.linalg.norm(points[-1]-scene['goal'])))
    return row,dict(**row,scene=scene,states=points,actions=np.asarray(acts),integration=np.asarray(ints),flags=flags)
def main(kind):
    start=time.time();cases,negative=sources(kind);roles=[(v,s) for v in ('A','B') for s in (300000,500000,1000000)] if kind=='navigate' else [('A',1000000),('B',1000000)]
    agents={};ids=[]
    for v,step in roles:agents[v,step],i=load_agent(kind,v,step);ids.append(i)
    source_role=('A',300000) if kind=='navigate' else ('A',1000000);compare_role=('A',1000000) if kind=='navigate' else ('B',1000000)
    out=[];index=[];div=[];snapshots=[];raw=[];gates=[];env=env_new(kind);main_control_time=None;main_control_case=None
    for case_id,trace,category in cases:
        obs=trace['states'][:-1];g=np.repeat(trace['initial']['goal'][None],len(obs),axis=0)
        actions={role:np.asarray(mode(agent,obs,g)) for role,agent in agents.items()}
        hit,delta=first_divergence(actions[source_role],actions[compare_role]);center=hit if hit is not None else min(50,len(obs)//2)
        matched_time=None
        if category=='Task5_main' and main_control_time is None:main_control_time=center;main_control_case=case_id
        if category=='Task1_2_control':
            matched_time=main_control_time<len(obs)
            if matched_time:center=main_control_time
        center=min(center,len(obs)-1);points=sorted(set(max(0,min(len(obs)-1,center+k)) for k in (-5,0,5)))
        index.append(dict(case_id=case_id,kind=kind,task=trace['task'],category=category,source_scene=trace['scene_id'],source_success=trace['success'],source_steps=trace['steps'],selected_anchor_count=len(points),primary_anchor=center,branch_triggered=hit is not None,matched_control_time=matched_time,control_time_reference=main_control_case if category=='Task1_2_control' else None))
        div.append(dict(case_id=case_id,kind=kind,task=trace['task'],category=category,triggered=hit is not None,first_window=hit,threshold=.30,window=5,required_hits=3,
            action_difference_median=float(np.median(delta)),p90=float(np.quantile(delta,.9)),maximum=float(delta.max()),above_threshold_fraction=float(np.mean(delta>=.30))))
        np.savez_compressed(ROOT/'runs'/f'{case_id}_action_timeline.npz',states=obs,goal=trace['initial']['goal'],difference=delta,**{f'{v}_{step}':a for (v,step),a in actions.items()})
        s,_=restore_gate(env,trace['initial']);prepared=[];prefix_error=0.
        for t in range(max(points)+1):
            prefix_error=max(prefix_error,float(np.max(np.abs(integration(env.unwrapped)-trace['integration'][t]))))
            if t in points:
                sc=snapshot(env,{'goal':trace['initial']['goal']},trace['initial']['reset_seed']);sc.update(case_id=case_id,anchor=t)
                a=np.asarray(mode(agents[source_role],s,sc['goal']));ae=float(np.max(np.abs(a-trace['actions'][t])))
                prepared.append((t,sc,ae,prefix_error))
            control_count();s,r,term,trunc,info=env.step(trace['actions'][t])
            if t in points:
                pe=float(np.max(np.abs(integration(env.unwrapped)-trace['integration'][t+1])));fe=tuple(trace['flags'][t]);flags=(float(r),bool(term),bool(trunc),float(info['success']))
                gate=dict(case_id=case_id,anchor=t,source_action_error=prepared[-1][2],prefix_state_error=prepared[-1][3],next_state_error=pe,flags_equal=flags==fe)
                gate['status']='PASS' if gate['source_action_error']<=1e-6 and gate['prefix_state_error']<=1e-5 and pe<=1e-5 and gate['flags_equal'] else 'REPLAY_GATE_FAILED';gates.append(gate)
        for t,sc,ae,pe in prepared:
            gate=next(x for x in gates if x['case_id']==case_id and x['anchor']==t)
            snapshots.append(dict(case_id=case_id,kind=kind,category=category,anchor=t,primary_anchor=t==center,scene=sc,status=gate['status']))
            if gate['status']!='PASS':continue
            for role,agent in agents.items():
                row,tr=branch(env,sc,agent,case_id,t,*role,trace['states'][min(t+50,len(trace['states'])-1)]);row['kind']=kind;tr['kind']=kind;out.append(row);raw.append(tr)
        write(ROOT/'receipts'/f'R_trace_{kind}_progress.json',dict(cases=len(index),last_case=case_id,branches=len(out),seconds=time.time()-start))
        print('R_TRACE',kind,len(index),'DONE',flush=True)
    env.close();csv_rows(ROOT/'results'/f'R_CASE_INDEX_{kind}.csv',index);csv_rows(ROOT/'results'/f'R_EARLIEST_DIVERGENCE_{kind}.csv',div)
    if negative:csv_rows(ROOT/'results'/f'R_NEGATIVE_SELECTION_{kind}.csv',negative)
    if out:csv_rows(ROOT/'results'/f'R_BRANCHES_{kind}.csv',out)
    write(ROOT/'results'/f'R_REPLAY_GATES_{kind}.json',gates);dump(ROOT/'runs'/f'R_SNAPSHOTS_{kind}.pkl',snapshots);dump(ROOT/'runs'/f'R_BRANCH_TRACES_{kind}.pkl',raw)
    write(ROOT/'receipts'/f'R_trace_{kind}_exit.json',dict(status='COMPLETED',cases=len(index),anchors=len(snapshots),replay_pass=sum(x['status']=='PASS' for x in gates),branches=len(out),seconds=time.time()-start,checkpoints=ids,training_updates=0))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['navigate','stitch'],required=True);main(p.parse_args().kind)
