"""Reuse every preselected E2 target for offline value queries, without new rollouts."""
import argparse
from support_value import *
def main(variant):
    with open(ROOT/'runs/E2_CANDIDATES.pkl','rb') as f:targets=pickle.load(f)
    with open(ROOT/'runs/E2_SNAPSHOTS.pkl','rb') as f:snaps={x['id']:x for x in pickle.load(f)}
    gates={x['snapshot_id']:x for x in json.loads((ROOT/'results/E2_GATES.json').read_text())}
    branches=list(csv.DictReader((ROOT/'results/E2_BRANCHES.csv').open()));rows=[];missing=[]
    for kind in ('navigate','stitch'):
        raw=np.load(OLD/'vendor/data'/f'pointmaze-large-{kind}-v0.npz');obs=raw['observations'];valid=np.flatnonzero(1-raw['terminals'])
        env,data,val,config=load(f'pointmaze-large-{kind}-v0','A')
        for step in (100000,300000,500000,1000000):
            try:agent,identity=load_checkpoint(kind,variant,step)
            except FileNotFoundError:
                missing.append(dict(kind=kind,step=step));continue
            for target in [x for x in targets if snaps[x['snapshot_id']]['kind']==kind]:
                check_time();snap=snaps[target['snapshot_id']];scene=snap['scene'];g=scene['goal'];s=scene['qpos'];cv=np.asarray([target['near']['goal'],target['control']['goal']],dtype=np.float32)
                idx=np.array([target['near']['state_idx'],target['control']['state_idx']]);nextobs=obs[idx+1];actions=raw['actions'][idx]
                info={k:np.asarray(v) for k,v in sync(query(agent,np.repeat(s[None],2,0).astype(np.float32),np.repeat(g[None],2,0).astype(np.float32),cv,nextobs,actions)).items()}
                for c,label in enumerate(('near','control')):
                    item=target[label];i=item['state_idx'];end=int(data.terminal_locs[np.searchsorted(data.terminal_locs,i)]);future=np.arange(min(i+1,end),end+1)
                    weights=np.ones(len(future));weights[[0,-1]]=.5 if len(future)>1 else 1.;weights/=weights.sum()
                    hit=np.linalg.norm(obs-g,axis=1)<=1.0;pa=float(weights@hit[future]);pr=float(hit[valid].mean())
                    sid=snap['id'];gate=gates[sid];br={x['condition']:x for x in branches if x['snapshot_id']==sid}
                    row=dict(snapshot_id=sid,kind=kind,variant=variant,checkpoint_step=step,target=label,state_idx=i,waypoint_idx=item['waypoint_idx'],
                        qpos_match_error=item['qpos_error'],qvel_match_error=item['qvel_error'],
                        privileged_feasibility=gate.get('feasibility_'+label),E2_gate=gate['status'],
                        source_A_actor_near_reached=br.get('C1' if c==0 else 'C3',{}).get('local_reached'),source_A_teacher_near_reached=br.get('C2',{}).get('local_reached') if c==0 else None,
                        A_goal_region_probability=pa,B_goal_region_probability=.5*(pa+pr),
                        head1_preferred=int(np.argmax(info['w1'])),head2_preferred=int(np.argmax(info['w2'])),rho_saturated=bool(info['log_rho'][c]>=np.log(100)),
                        rho_before=float(info['rho_before'][c]) if np.isfinite(info['rho_before'][c]) else None,rho_before_finite=bool(np.isfinite(info['rho_before'][c])))
                    row.update({k:float(v[c]) for k,v in info.items() if v.ndim==1 and k!='rho_before'})
                    for p in ('actor','teacher'):
                        row.update({f'{p}_{d}':float(info[p][c,d]) for d in range(2)})
                    rows.append(row)
            print('E3_VERIFIED',kind,variant,step,flush=True)
        raw.close();env.close()
    csv_rows(ROOT/'results'/f'E3_{variant}_VERIFIED_TARGETS.csv',rows)
    write(ROOT/'receipts'/f'E3_{variant}_verified_exit.json',dict(status='PARTIAL_MISSING_CHECKPOINTS' if missing else 'COMPLETED',missing_steps=missing,rows=len(rows),additional_control_steps=0,training_updates=0,
        interpretation='post-E2 explanatory measurement of every preregistered E2 candidate; initial 16-candidate preregistered queries retained unchanged'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['A','B'],required=True);main(p.parse_args().variant)
