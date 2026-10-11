"""CPU-only identity audit, historical event calibration, and geometry generation."""
import csv, hashlib, json, pathlib, datetime, argparse, itertools
import numpy as np

OLD = pathlib.Path('/home/xushijie3/work/cpdisr_arm_20261011/stage5_pushcube_20261011')
ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)

def dump(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False))

def first_run(mask, length=5):
    for t in range(length - 1, len(mask)):
        if np.all(mask[t-length+1:t+1]):
            return int(t)
    return None

def events(states, distances, eef, window=20, tol=.001):
    fallen = (states[..., 2] < .975).any(axis=-1)
    stall = []
    response = []
    for t in range(window, len(states)):
        part = states[t-window:t+1]
        motion = float(np.linalg.norm(part[..., :3] - part[0, :, :3], axis=-1).max())
        d = distances[t-window:t+1].mean(axis=-1)
        improvement = float(d[0] - d.min())
        if not fallen[t-window:t+1].any() and not (distances[t] < .04).all() and motion < tol and improvement < tol:
            stall.append(t)
            response.append(float(np.linalg.norm(eef[t-window:t+1]-eef[t-window], axis=-1).max()))
    departure = []
    recovered = []
    for j in range(distances.shape[-1]):
        hit = first_run(distances[:, j] < .04)
        leave = first_run(distances[hit+1:, j] > .05) if hit is not None else None
        leave = leave + hit + 1 if leave is not None else None
        recovery = first_run(distances[leave+1:, j] < .04) if leave is not None else None
        recovered.append(recovery + leave + 1 if recovery is not None else None)
        departure.append(leave)
    return {'first_fall': first_run(fallen), 'first_no_progress': stall[0] if stall else None,
            'no_progress_windows': len(stall), 'first_no_progress_eef_motion_m': response[0] if response else None,
            'departure_steps': departure, 'recovery_steps': recovered}

def audit():
    identity = []
    seen = set()
    for line in (OLD / 'download_manifest.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['path'] in seen or not any(x in r['path'] for x in ['state_1200000.pt', 'dlp_panda_push.pth', 'replay_buffer_dlp.pkl']):
            continue
        seen.add(r['path'])
        p = OLD / 'resources' / r['path']
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(8*1024*1024), b''):
                h.update(chunk)
        assert h.hexdigest() == r['sha256'], p
        identity.append({'path': str(p), 'bytes': p.stat().st_size, 'sha256': h.hexdigest(), 'verified': True})
    assert len(identity) == 3
    dump('identity.json', {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'files': identity})
    rows, timelines, noise, moving = [], [], [], []
    for n in [3, 6]:
        for p in sorted((OLD / ('runs/core%d/data' % n)).glob('trajectory_*.npz')):
            seed = int(p.stem.split('_')[-1])
            z = np.load(p)
            states, dist, eef = z['states'], z['distances'], z['eef']
            assert np.allclose(np.linalg.norm(states[..., :2]-z['goal'], axis=-1), dist, atol=1e-6)
            for i in range(states.shape[1]):
                movement = np.linalg.norm(np.diff(states[:, i, :, :3], axis=0), axis=-1).max(axis=-1)
                noise.extend(movement[movement < 1e-5].tolist())
                moving.extend(movement[movement > .001].tolist())
                st, dd, ee = states[:, i], dist[:, i], eef[:, i]
                common = events(st[:101], dd[:101], ee[:101])
                full = events(st, dd, ee)
                sensitivity = {}
                for w in [10, 20, 30]:
                    for tol in [.0001, .001, .005]:
                        sensitivity['w%d_tol%.4f' % (w, tol)] = events(st[:101], dd[:101], ee[:101], w, tol)['first_no_progress']
                rows.append({'n': n, 'seed': seed, 'slot': i, 'steps': len(st)-1,
                             'common_0_100': common, 'full_available': full, 'sensitivity': sensitivity,
                             'fraction100': float((dd[100]<.04).mean()),
                             'final_fraction': float((dd[-1]<.04).mean())})
                for t in range(len(st)):
                    timelines.append({'n': n, 'seed': seed, 'slot': i, 'step': t,
                                      'hit_count': int((dd[t]<.04).sum()), 'mean_distance_m': float(dd[t].mean()),
                                      'min_z_m': float(st[t,:,2].min()),
                                      'eef_motion_from_initial_m': float(np.linalg.norm(ee[t]-ee[0])),
                                      'max_cube_motion_from_initial_m': float(np.linalg.norm(st[t,:,:3]-st[0,:,:3],axis=-1).max())})
    assert len(rows) == 32
    noise95 = float(np.quantile(noise, .95))
    assert noise95 * 10 < .001, noise95
    calibration = {'old_data_role': 'exploration and calibration only', 'record_interval': 'one policy/control step (~1/3 simulation second)',
                   'stationary_step_noise_p95_m': noise95, 'moving_step_p25_m': float(np.quantile(moving, .25)),
                   'stall_window_steps': 20, 'cube_max_displacement_tolerance_m': .001,
                   'maximum_mean_goal_improvement_tolerance_m': .001, 'eef_response_threshold_m': .001,
                   'fall_rule': 'any object z<0.975 for 5 consecutive recorded steps',
                   'departure_rule': '5 steps d<0.04 followed by 5 steps d>0.05',
                   'selected_B_event': 'unfallen sustained no progress',
                   'selection_reason': 'online measurement available before irrecoverable fall; observable chain response can be probed; no contact-force causal label',
                   'common_window_counts': {str(n): {'episodes': 16,
                       'falls': sum(r['common_0_100']['first_fall'] is not None for r in rows if r['n']==n),
                       'no_progress': sum(r['common_0_100']['first_no_progress'] is not None for r in rows if r['n']==n)} for n in [3,6]}}
    dump('historical_events.json', rows)
    dump('measurement_calibration.json', calibration)
    with (OUT/'historical_timeline.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(timelines[0]));writer.writeheader();writer.writerows(timelines)
    print(json.dumps(calibration), flush=True)

def metrics(p, g):
    result = {}
    for label, x in [('initial',p),('goal',g)]:
        d=np.linalg.norm(x[:,None]-x[None,:],axis=-1);np.fill_diagonal(d,np.inf)
        nearest=d.min(axis=1)
        edge=np.minimum(.25-np.abs(x[:,0]),.30-np.abs(x[:,1]))-.0175
        reach=np.linalg.norm(x-np.array([-.45,0.]),axis=-1)
        result.update({label+'_nn_mean':float(nearest.mean()),label+'_nn_min':float(nearest.min()),
                       label+'_edge_mean':float(edge.mean()),label+'_edge_min':float(edge.min()),
                       label+'_reach_mean':float(reach.mean()),label+'_reach_max':float(reach.max())})
    result['transport_mean']=float(np.linalg.norm(g-p,axis=-1).mean())
    result['initial_hit_count']=int((np.linalg.norm(g-p,axis=-1)<.04).sum())
    return result

def geometry():
    rng=np.random.RandomState(2026101102)
    default_eef=np.load(OLD/'runs/core6/data/trajectory_93001.npz')['eef'][0,0,:2]
    scenes=[];counts={}
    for group in range(6):  # 4 validation groups + 2 excluded calibration groups
        angle=[0,np.pi/2,np.pi,3*np.pi/2,0,np.pi/2][group]
        v=.08*np.array([np.cos(angle),np.sin(angle)])
        pools={}
        for n in [3,6]:
            for density in ['sparse','crowded']:
                key='%d_%s'%(n,density);pool=[];reject={}
                for attempt in range(18000):
                    radius=({'sparse':.085,'crowded':.059}[density] / (np.sqrt(3) if n==3 else 1.))
                    rot=rng.uniform(0,2*np.pi)
                    angles=np.arange(n)*2*np.pi/n+rot
                    p=radius*np.stack([np.cos(angles),np.sin(angles)],axis=-1)
                    if attempt%2:
                        spacing={'sparse':.085,'crowded':.059}[density]
                        shape=np.array([[0.,-1.],[0.,0.],[0.,1.]]) if n==3 else np.array([[-.5,-1.],[-.5,0.],[-.5,1.],[.5,-1.],[.5,0.],[.5,1.]])
                        rotation=np.array([[np.cos(rot),-np.sin(rot)],[np.sin(rot),np.cos(rot)]])
                        p=spacing*shape.dot(rotation.T)
                    p+=rng.normal(0,.0015,(n,2))
                    p+=rng.uniform([-.11,-.16],[.11,.16])
                    g=p+v
                    if np.any(np.abs(p)>[.15,.20]) or np.any(np.abs(g)>[.15,.20]):
                        reject['official_sampling_workspace']=reject.get('official_sampling_workspace',0)+1;continue
                    retreat=np.stack([np.clip(g[:,0],-.22,default_eef[0]),np.full(n,default_eef[1])],axis=-1)
                    if np.linalg.norm(p-default_eef,axis=-1).min()<.055 or np.linalg.norm(g-retreat,axis=-1).min()<.055:
                        reject['robot_initial_or_goal_retreat_clearance']=reject.get('robot_initial_or_goal_retreat_clearance',0)+1;continue
                    m=metrics(p,g)
                    lo,hi=(.075,.100) if density=='sparse' else (.052,.068)
                    if not (lo<=m['initial_nn_min']<=hi and lo<=m['initial_nn_mean']<=hi):
                        reject['band_or_overlap']=reject.get('band_or_overlap',0)+1;continue
                    pool.append((p,g,m))
                    if len(pool)==1600:break
                pools[key]=pool;counts['g%d_%s'%(group,key)]={'attempts':attempt+1,'accepted':len(pool),'rejections':reject}
        keys=['3_sparse','3_crowded','6_sparse','6_crowded']
        if not all(pools[k] for k in keys):
            dump('scenes.json',scenes)
            dump('geometry_gate.json',{'passed':False,'failed_group':group,'validated_group_count':group,
                'candidate_counts':counts,'reason':'A required condition has no legal candidates after robot clearance; minimum 4 matched groups not obtained. A stopped.'})
            print('GEOMETRY_GATE_FAILED_NO_LEGAL_CANDIDATES',group,flush=True);return
        # Anchor six-object sparse layouts, matching all four conditions before policy execution.
        match=None;best=None
        matching=[(label+'_'+metric,tol) for label in ['initial','goal']
                  for metric,tol in [('edge_mean',.025),('edge_min',.025),('reach_mean',.025),('reach_max',.05)]]
        matrices={k:np.array([[x[2][q]/tol for q,tol in matching] for x in pools[k]]) for k in keys}
        for ai,anchor in enumerate(pools['6_sparse']):
            reference=matrices['6_sparse'][ai]
            candidates={}
            for k in ['3_sparse','3_crowded','6_crowded']:
                delta=np.abs(matrices[k]-reference)
                fits=np.flatnonzero((delta<=1).all(axis=1))
                if len(fits)==0:break
                candidates[k]=fits[np.argsort(delta[fits].sum(axis=1))[:12]]
            if len(candidates)!=3:continue
            for indexes in itertools.product(*(candidates[k] for k in ['3_sparse','3_crowded','6_crowded'])):
                values=np.stack([reference]+[matrices[k][i] for k,i in zip(['3_sparse','3_crowded','6_crowded'],indexes)])
                if (np.ptp(values,axis=0)<=1).all():
                    match={'6_sparse':anchor}
                    match.update({k:pools[k][i] for k,i in zip(['3_sparse','3_crowded','6_crowded'],indexes)})
                    break
            if match is not None:break
        if match is None:
            dump('scenes.json',scenes)
            dump('geometry_gate.json',{'passed':group>=4,'failed_group':group,'candidate_counts':counts,
                                      'validated_group_count':group,
                                      'reason':'No four-condition match within frozen edge/reach tolerances; cancel A causal interpretation.'})
            print('GEOMETRY_GATE_FAILED',group,flush=True);return
        colors=[0,2,4,1,3,5] if group%2==0 else [1,3,5,0,2,4]
        for key in keys:
            p,g,m=match[key];n=int(key.split('_')[0])
            scenes.append({'group':group,'role':'validation_A' if group<4 else 'calibration',
                           'condition':key,'n':n,'initial_xy':p.tolist(),'goal_xy':g.tolist(),
                           'colors':colors[:n],'transport_vector':v.tolist(),'metrics':m,
                           'batch_policy_seed':94701})
    dump('scenes.json',scenes)
    dump('geometry_gate.json',{'passed':True,'candidate_counts':counts,
        'common_bands_m':{'crowded':[.052,.068],'sparse':[.075,.100]},
        'bands_apply_to':'both mean and minimum nearest-neighbor spacing, initial and goal, both counts',
        'matching_tolerances_m':dict(matching),'all_transport_vectors_preserved':True,
        'zero_initial_hits':True,'workspace_xy_abs_max':[.15,.20], 'robot_clearance_m':.055,
        'caveat':'Four conditions match geometric summaries, not exact physical worlds. Transport is common translation per group. Reach is geometric proxy only.'})
    print('GEOMETRY_GATE_PASSED',len(scenes),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['audit','geometry','all']);args=parser.parse_args()
    if args.mode in ['audit','all']:audit()
    if args.mode in ['geometry','all']:geometry()
