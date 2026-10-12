"""Independent pilot calibration and complete final-evaluation failure mapping."""
from collections import Counter
from common import *
from data_analysis import geodesic


def calibrate():
    ratios=[]; counts={}
    for kind in ('navigate','stitch'):
        path=ROOT/'runs'/f'pilot_{kind}'/'eval_trajectories.pkl'
        trajectories=pickle.load(open(path,'rb')); n=0
        for t in trajectories:
            motion=np.linalg.norm(np.diff(t['states'],axis=0),axis=1)
            requested=.2*np.linalg.norm(t['actions'],axis=1)
            take=requested>.05
            ratios.extend(np.clip(motion[take]/requested[take],1e-8,1).tolist()); n+=int(take.sum())
        counts[kind]=dict(path=str(path),sha256=sha(path),steps=sum(len(t['actions']) for t in trajectories),qualified_steps=n)
    # Unsupervised log-motion Otsu split on independent untrained-pilot trajectories.
    hist,edges=np.histogram(np.log10(ratios),bins=32,range=(-8,0))
    p=hist/(hist.sum()+1e-30); centers=(edges[:-1]+edges[1:])/2
    c=np.cumsum(p); m=np.cumsum(p*centers); total=m[-1]
    variance=(total*c-m)**2/(c*(1-c)+1e-30)
    split=int(np.argmax(variance[:-1])); raw=10**edges[split+1]
    threshold=float(np.clip(raw,.02,.2))
    config=dict(calibration='independent seed-0 pilot, before trained-model discovery; untrained distributions are not policy success targets',
                sources=counts,raw_log_ratio_otsu_threshold=raw,no_effect_ratio_threshold=threshold,
                min_requested_displacement=.05,window_steps=10,wall_proximity=.71,
                no_effect_definition='at least 8/10 commanded moves have observed/requested displacement below calibrated cutoff; request magnitude > .05',
                wall_proxy='near a wall rectangle within .71 distance (official sphere radius .7 plus .01 tolerance) and ineffective command; physical contact count includes floor, not treated as wall collision',
                revisit_definition='repeated cells after consecutive equal cells collapsed; continuous counts, no abnormal-turn label')
    write_json(ROOT/'results/EVENT_CALIBRATION.json',config)
    print('EVENT_CALIBRATION',threshold,flush=True)


def wall_distance(u,states):
    centers=np.asarray([u.ij_to_xy(tuple(c)) for c in np.argwhere(u.maze_map==1)])
    delta=np.maximum(np.abs(states[:,None,:]-centers[None,:,:])-u._maze_unit/2,0)
    return np.linalg.norm(delta,axis=2).min(axis=1)


def build():
    config=json.load(open(ROOT/'results/EVENT_CALIBRATION.json')); results={}
    for kind in ('navigate','stitch'):
        env=ogbench.make_env_and_datasets(f'pointmaze-large-{kind}-v0',env_only=True); u=env.unwrapped
        path=ROOT/'runs'/f'final_{kind}'/'confirmation_trajectories.pkl'
        trajectories=pickle.load(open(path,'rb')); rows=[]
        for t in trajectories:
            states=t['states']; actions=t['actions']; motion=np.linalg.norm(np.diff(states,axis=0),axis=1)
            request=.2*np.linalg.norm(actions,axis=1); ratios=motion/(request+1e-12)
            no_effect=(request>config['min_requested_displacement'])&(ratios<config['no_effect_ratio_threshold'])
            window=np.convolve(no_effect.astype(int),np.ones(10,dtype=int),mode='valid') if len(no_effect)>=10 else np.array([])
            wall=wall_distance(u,states[:-1])<config['wall_proximity']
            cells=[u.xy_to_ij(s) for s in states]; compressed=[cells[0]]+[b for a,b in zip(cells,cells[1:]) if a!=b]
            visit=Counter(compressed); unique=len(visit)
            sg=geodesic(u,states[[0,-1]],t['goal'])
            row=dict(task_id=t['task_id'],episode=t['episode'],reset_seed=t['reset_seed'],success=t['success'],steps=t['steps'],
                     final_distance=t['final_distance'],path_length=t['path_length'],first_success_step=t['first_success_step'],
                     initial_cell_distance=float(sg[0]),final_cell_distance=float(sg[-1]),intercell_moves=len(compressed)-1,
                     repeated_cell_visits=len(compressed)-unique,max_cell_visits=max(visit.values()),
                     revisit_fraction=(len(compressed)-unique)/max(1,len(compressed)),no_effect_steps=int(no_effect.sum()),
                     persistent_no_effect=bool((window>=8).any()),wall_no_effect_proxy_steps=int((wall&no_effect).sum()),
                     wall_near_steps=int(wall.sum()),small_command_steps=int((request<=config['min_requested_displacement']).sum()))
            rows.append(row)
        from diagnose import rows_to_csv
        rows_to_csv(ROOT/'results'/f'{kind}_FAILURE_MAP.csv',rows)
        results[kind]=dict(episodes=len(rows),success_rate=float(np.mean([r['success'] for r in rows])),
                           by_task={str(task):dict(n=sum(r['task_id']==task for r in rows),success=float(np.mean([r['success'] for r in rows if r['task_id']==task]))) for task in range(1,6)},
                           failure_count=sum(r['success']==0 for r in rows),persistent_no_effect_count=sum(r['persistent_no_effect'] for r in rows),
                           wall_no_effect_episode_count=sum(r['wall_no_effect_proxy_steps']>0 for r in rows),
                           untriggered_no_effect_count=sum(not r['persistent_no_effect'] for r in rows),
                           final_distance_mean=float(np.mean([r['final_distance'] for r in rows])),raw_trajectories=dict(path=str(path),sha256=sha(path)))
        env.close()
    write_json(ROOT/'results/FAILURE_MAP_SUMMARY.json',results)
    print('FAILURE_MAP_DONE',flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['calibrate','build'],required=True)
    a=p.parse_args();calibrate() if a.mode=='calibrate' else build()
