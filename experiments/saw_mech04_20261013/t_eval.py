"""CPU T evaluation on sealed scenes, no policy substitution at evaluation."""
from frozen import *
import argparse,signal

def main(arm,step):
    p=ROOT/'receipts/T_AUTHORIZATION.json';assert p.exists() and json.loads(p.read_text())['approved']
    label=f'Teval_T{arm}_{step}';agent,source=load_agent('stitch','A',500000)
    if arm>=0:
        entries=json.loads((ROOT/'results'/f'T{arm}_checkpoints.json').read_text());source=next(x for x in entries if x['step']==step);assert sha(source['path'])==source['sha256']
        with open(source['path'],'rb') as f:s=pickle.load(f)
        assert s['student_step']==step and s['stream_cursor']==step and s['reference_native_step']==500001
        agent=flax.serialization.from_state_dict(agent,s['agent']);assert int(agent.network.step)==500001
    with open(ROOT/'runs/T_SCENES.pkl','rb') as f:scenes=pickle.load(f)
    if arm>=0 and step!=100000:scenes=[s for s in scenes if s['episode']<10]
    env=env_new('stitch');rows=[];traces=[];stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));start=time.time()
    for scene in scenes:
        if stop[0]:break
        bp=ROOT/'receipts/T_EPISODE_BUDGET.json'
        with open(str(bp)+'.lock','a') as f:
            fcntl.flock(f,fcntl.LOCK_EX);x=json.loads(bp.read_text()) if bp.exists() else dict(used=0,limit=3300);assert x['used']<3300;x['used']+=1;write(bp,x)
        s,err=restore_gate(env,scene);goal=scene['goal'];states=[s.copy()];actions=[];velocities=[env.unwrapped.data.qvel.copy()];tt=tr=False;info={'success':False}
        while not(tt or tr or stop[0]):
            action=np.asarray(mode(agent,s,goal));assert np.isfinite(action).all();s,r,tt,tr,info=env.step(action);states.append(s.copy());actions.append(action);velocities.append(env.unwrapped.data.qvel.copy())
        xy=np.asarray(states);d=np.linalg.norm(xy-goal,axis=1);complete=bool(tt or tr)
        row=dict(label=label,arm=arm,student_step=step,scene_id=scene['scene_id'],group=scene['group'],task=scene['task'],episode=scene['episode'],stream=scene['stream'],scene_sha256=digest(scene),checkpoint_sha256=source['sha256'],initial_restore_error=err,complete=complete,success=int(info['success']) if complete else '',steps=len(actions),terminated=bool(tt),truncated=bool(tr),initial_distance=float(d[0]),final_distance=float(d[-1]),minimum_distance=float(d.min()),path_length=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()))
        rows.append(row);traces.append(dict(**row,initial=scene,states=xy,actions=np.asarray(actions),qvel=np.asarray(velocities)))
        if len(rows)%25==0:print(label,len(rows),flush=True)
    env.close();raw=ROOT/'runs'/f'{label}_trajectories.pkl';dump(raw,traces);csv_rows(ROOT/'results'/f'{label}_episodes.csv',rows)
    write(ROOT/'receipts'/f'{label}_exit.json',dict(status='COMPLETE' if len(rows)==len(scenes) and all(x['complete'] for x in rows) else 'STOPPED',identity=source,episodes=len(rows),seconds=time.time()-start,path=str(raw),sha256=sha(raw)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',type=int,required=True);p.add_argument('--step',type=int,required=True);a=p.parse_args();main(a.arm,a.step)
