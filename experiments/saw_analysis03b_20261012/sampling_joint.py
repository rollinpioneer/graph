"""Exact native goal-region/waypoint joint support; no rollout or model update."""
from state_core import *
def main():
    with open(ROOT/'runs/E3_QUERY_ENTRIES.pkl','rb') as f:entries=pickle.load(f)
    with open(ROOT/'runs/E2_CANDIDATES.pkl','rb') as f:verified=pickle.load(f)
    selected={x['snapshot_id']:x for x in entries};rows=[]
    for kind in ('navigate','stitch'):
        env,data,val,config=load(f'pointmaze-large-{kind}-v0','A');assert not config['actor_geom_sample']
        obs=data.dataset['observations'];valid=np.flatnonzero(data.dataset['valids']>0)
        for entry in [x for x in entries if x['kind']==kind]:
            sid=entry['snapshot_id'];g=entry['scene']['goal'];hit=np.linalg.norm(obs-g,axis=1)<=1.0;random_mass=float(hit[valid].mean())
            options=[('fixed16',c,entry['state_idx'],int(w)) for c,w in enumerate(entry['candidate_indices'])]
            target=next((x for x in verified if x['snapshot_id']==sid),None)
            if target:
                options += [('E2_verified_'+label,-1,target[label]['state_idx'],target[label]['waypoint_idx']) for label in ('near','control')]
            for group,c,i,w in options:
                end=int(data.terminal_locs[np.searchsorted(data.terminal_locs,i)]);future=np.arange(min(i+1,end),end+1)
                weights=np.ones(len(future));weights[[0,-1]]=.5 if len(future)>1 else 1.;weights/=weights.sum()
                native_wp=np.minimum(i+25,future);future_mass=float(weights@hit[future]);joint=float(weights@(hit[future]&(native_wp==w))) if w>=0 else 0.
                random_wp=min(i+25,end);rjoint=random_mass if random_wp==w else 0.
                rows.append(dict(snapshot_id=sid,kind=kind,group=group,candidate=c,state_idx=i,waypoint_idx=w,compact_terminal_idx=end,
                    native_random_waypoint_idx=random_wp,random_waypoint_matches=bool(w==random_wp),
                    goal_region_probability_A=future_mass,goal_region_probability_B=.5*(future_mass+random_mass),
                    joint_far_region_waypoint_probability_A=joint,joint_far_region_waypoint_probability_B=.5*(joint+rjoint),
                    scope='radius1 goal neighborhood at matched recorded state; exact continuous goal probability and strict raw integration matching not claimed'))
        env.close()
    csv_rows(ROOT/'results/E3_JOINT_SAMPLING_SUPPORT.csv',rows)
    write(ROOT/'receipts/E3_joint_sampling_exit.json',dict(status='COMPLETED',rows=len(rows),training_updates=0,control_steps=0,
        formula='official round(uniform future index): endpoints half mass; random goals uniform over native valid indexes; native future/random waypoint clamps retained'))
    print('JOINT_SAMPLING_SUPPORT_DONE',len(rows),flush=True)
if __name__=='__main__':main()
