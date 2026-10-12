"""Sealed full-state CPU evaluations; immutable checkpoints, no training."""
from frozen import *
import argparse, signal

def reserve(phase):
    p=ROOT/'receipts'/f'{phase}_EPISODE_BUDGET.json'
    with open(str(p)+'.lock','a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        x=json.loads(p.read_text()) if p.exists() else dict(used=0,limit=5340 if phase=='C' else 20)
        assert x['used']<x['limit'],'EPISODE_CAP';x['used']+=1;write(p,x)

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',required=True);p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--step',type=int,required=True);p.add_argument('--scope',choices=['full','monitor','Rconfirm'],required=True);a=p.parse_args()
    phase='R' if a.scope=='Rconfirm' else 'C'
    label=f'{phase}eval_{a.kind}_{a.variant}_s{a.seed}_{a.step}_{a.scope}'
    assert not (ROOT/'receipts'/f'{label}_exit.json').exists()
    agent,identity=load_agent(a.kind,a.variant,a.step,a.seed)
    with open(ROOT/'runs'/f'{"Rconfirm" if phase=="R" else "C"}_{a.kind}_SCENES.pkl','rb') as f:scenes=pickle.load(f)
    if a.scope=='monitor':scenes=[s for s in scenes if s['task']==5 or (s['task'] in (1,2) and s['episode']<10)]
    env=env_new(a.kind);rows=[];traces=[];stopped=[False];start=time.monotonic()
    signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
    try:
        for scene in scenes:
            if stopped[0]:break
            reserve(phase);s,err=restore_gate(env,scene);g=scene['goal'];states=[s.copy()];qvels=[env.unwrapped.data.qvel.copy()];actions=[];contacts=[];tt=tr=False;info={'success':False}
            while not(tt or tr):
                if stopped[0]:break
                if phase=='R':control_count()
                ac=np.asarray(mode(agent,s,g));assert np.isfinite(ac).all()
                s,reward,tt,tr,info=env.step(ac);states.append(s.copy());qvels.append(env.unwrapped.data.qvel.copy());actions.append(ac);contacts.append(env.unwrapped.data.ncon)
            complete=bool(tt or tr);xy=np.asarray(states);d=np.linalg.norm(xy-g,axis=1)
            row=dict(label=label,kind=a.kind,variant=a.variant,training_seed=a.seed,checkpoint_step=a.step,scope=a.scope,scene_id=scene['scene_id'],group=scene['group'],task=scene['task'],episode=scene['episode'],scene_stream=scene['stream'],scene_sha256=digest(scene),checkpoint_sha256=identity['sha256'],initial_restore_error=err,complete=complete,success=int(bool(info['success'])) if complete else '',steps=len(actions),terminated=bool(tt),truncated=bool(tr),initial_distance=float(d[0]),final_distance=float(d[-1]),minimum_distance=float(d.min()),path_length=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),net_displacement=float(np.linalg.norm(xy[-1]-xy[0])),contact_fraction=float(np.mean(np.asarray(contacts)>0)) if contacts else 0.)
            rows.append(row);traces.append(dict(**row,initial=scene,states=xy,qvel=np.asarray(qvels),actions=np.asarray(actions),contacts=np.asarray(contacts)))
            if len(rows)%25==0:print(label,len(rows),'/',len(scenes),flush=True)
        raw=ROOT/'runs'/f'{label}_trajectories.pkl';dump(raw,traces)
        csv_rows(ROOT/'results'/f'{label}_episodes.csv',rows)
        write(ROOT/'receipts'/f'{label}_exit.json',dict(status='COMPLETE' if len(rows)==len(scenes) and all(r['complete'] for r in rows) else 'STOPPED',exit_code=0,identity=identity,selected_scenes=len(scenes),completed_episodes=sum(r['complete'] for r in rows),seconds=time.monotonic()-start,trajectory_path=str(raw),trajectory_sha256=sha(raw),source_sha256=sha(__file__)))
    finally:env.close()
if __name__=='__main__':main()
