"""Policy-independent scene construction, complete-state gates and admission."""
from core import *
import inspect, subprocess

PREREG = dict(
    version='SAW-GOALPROBE-05 supplement 1; frozen before any model query',
    seed=9051301, bootstrap_seed=9051302, bootstrap_draws=10000,
    max_states=64, max_models=6, max_steps_per_rollout=25, max_control_steps=25000,
    CPU_wall_seconds=14400, CPU_cores=[72,73,74,75], GPU_hours=0, GPU_MiB=0, training_updates=0,
    initial_region='official maze cell: nearest-center square, half-width 2; first cell exit determines channel; no post-exit correction changes first choice',
    P1='first six lexicographically ordered degree>=3 cells with unique-first, different-channel equal-BFS-length goal pairs (prefer 3 then 2 then 4 edges), four states per junction',
    P1_offsets=[[-.25,-.25],[-.25,.25],[.25,-.25],[.25,.25]],
    P2='same junctions, first lexicographically ordered shared unique initial channel; closest reachable goal at 1 edge and another at 2 edges',
    P2_offsets=[[-.15,-.15],[-.15,.15],[.15,-.15],[.15,.15]],
    offline_substitution='fourth state per group/junction: nearest exact recorded qpos within radius .45 of cell center, qvel norm <=.1, nonterminal, observations exactly qpos; unique index; otherwise keep geometry fourth state',
    offline_alignment='only substituted exact observed/qpos/qvel data record, index+action recorded; original dataset integration unknown and reconstructed physics marked; no nearest-neighbor alignment on other states',
    P3='archived C stitch full initial snapshots, Task1/3, first four episodes in each of two streams; original official goal plus geometry goal with a different unique first channel if available, else same; minimize BFS distance mismatch then goal cell',
    P3_ambiguity='all BFS optimal first channels are accepted; lexicographic canonical path used for controller and descriptive progress, never enters P1 mechanism decision',
    controller='clip((center of canonical first neighboring cell - observation)/.2, -1,1); independent, no model queried; PointEnv is direct .2 position increment and zeroes qvel before stepping',
    admission='both goals must physically enter a valid first channel within 25 steps; no A/B-dependent deletion/replacement; full restore <=1e-10 and deterministic replay <=1e-10',
    CPU_history_action_tolerance=1e-6, state_restore_tolerance=1e-10, no_motion_max_displacement=.1, small_clipped_action_norm=.1,
    support_gate_1='seed1 and seed2 P1 CBR A-B >= .15 each',
    support_gate_2='both seed1/2 P1 mean clipped goal-pair action L2 B < A AND among A-correct/B-incorrect paired goal tests, mean state L2 B < A; report raw sensitivity too; strict inequality has no invented minimum',
    support_gate_3='both seed1/2 P2 CBR A-B < .15, used as conservative operational definition of no comparable general deficit; also report projected progress and P1-P2 interaction, small-action/no-motion counterexamples',
    uncertainty='within training seed, paired cluster bootstrap resamples entire junction/source clusters and keeps all states, two goals, A/B together; percentile95%; three training seeds descriptive only',
    figures_selection='first lexicographic (seed,state_id,goal_index,variant) test in each P1 correct/wrong_channel/not_left_region/no_motion class; all missing classes marked; additional paired divergences first per seed',
    exclusions='physics/geometry only before policy; insufficient P1<16 or junctions<4 or P2<8 => descriptive only; all exclusions retained; no threshold/goal/window changes after freeze')

def goal_pairs(geom, c):
    options = []
    for g in geom.cells:
        if g == c: continue
        first = geom.firsts(c,g); d = geom.distances[g][c]
        if len(first)==1: options.append((g,first[0],d))
    pairs = []
    for k,a in enumerate(options):
        for b in options[k+1:]:
            if a[1]!=b[1] and a[2]==b[2] and a[2] in (2,3,4):
                pairs.append(((abs(a[2]-3),a[0],b[0]),[a[0],b[0]]))
    controls=[]
    for a in options:
        for b in options:
            if a[1]==b[1] and a[2]==1 and b[2]==2:controls.append(((a[1],a[0],b[0]),[a[0],b[0]]))
    return (min(pairs)[1] if pairs else None), (min(controls)[1] if controls else None)

def main():
    assert not (ROOT/'receipts/RESOURCE_START.json').exists(), 'NO_RERUN'
    write(ROOT/'receipts/RESOURCE_START.json',dict(epoch=time.time(),pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0)),devices=[str(d) for d in jax.devices()],GPU_hours=0,GPU_MiB=0,new_training_updates=0))
    write(ROOT/'receipts/CONTROL_BUDGET.json',dict(used=0,limit=25000,by_phase={}))
    write(ROOT/'PREREGISTRATION.json',PREREG)
    entries=model_entries();write(ROOT/'MODEL_IDENTITY.json',dict(source_commit='176ad8d17ec9d71eaca9ed55d87a8ee0bf837a28',models=entries,status='SIX_FILE_IDENTITIES_VERIFIED_NO_POLICY_LOADED'))
    official=OLD/'vendor/saw';assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=official,text=True).strip()=='3535a16c901e5de1233ac5a586401b3ddfa5e30d'
    assert not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=official,text=True).strip()
    data_path=OLD/'vendor/data/pointmaze-large-stitch-v0.npz';data=np.load(data_path);qpos=data['qpos'];qvel=data['qvel'];obs=data['observations'];terminal=data['terminals'];used=set()
    e=env_new();geo=Geometry(e);junctions=[]
    for c in geo.cells:
        p1,p2=goal_pairs(geo,c)
        if len(geo.neighbors(c))>=3 and p1 and p2:junctions.append((c,p1,p2))
    junctions=junctions[:6];np.random.seed(PREREG['seed']);random.seed(PREREG['seed']);bank=[];sources=[dict(path=str(data_path),sha256=sha(data_path))]
    for group in ('P1','P2'):
        for ji,(cell,g1,g2) in enumerate(junctions):
            goals=g1 if group=='P1' else g2
            for k,offset in enumerate(PREREG[group+'_offsets']):
                seed=PREREG['seed']+(0 if group=='P1' else 1000)+ji*10+k;e.action_space.seed(seed+2000000);s,info=e.reset(seed=seed,options={'task_id':1,'render_goal':False});u=e.unwrapped
                xy=geo.xy(cell)+offset;v=np.zeros(2);source='privileged_geometry_constructed';idx=None
                if k==3:
                    dist=np.linalg.norm(qpos-geo.xy(cell),axis=1);mask=(dist<=.45)&(np.linalg.norm(qvel,axis=1)<=.1)&(~terminal)&np.all(obs==qpos,axis=1)
                    valid=[int(x) for x in np.flatnonzero(mask) if int(x) not in used]
                    if valid:
                        idx=min(valid,key=lambda x:(float(dist[x]),x));used.add(idx);xy=qpos[idx].astype(float);v=qvel[idx].astype(float);source='offline_record_reconstructed_full_physics'
                u.data.qpos[:]=xy;u.data.qvel[:]=v;mujoco.mj_forward(u.model,u.data);scene=snapshot(e,info['goal'],seed)
                bank.append(dict(state_id=f'{group}_J{ji:02d}_S{k}',group=group,cluster=f'J_{cell[0]}_{cell[1]}',source=source,cell=cell,goals=[geo.xy(g) for g in goals],goal_cells=goals,firsts=[geo.firsts(cell,g) for g in goals],routes=[geo.route(cell,g) for g in goals],scene=scene,offline_index=idx,data_action=None if idx is None else data['actions'][idx].copy(),source_path=str(data_path) if idx is not None else str(inspect.getfile(type(u))),source_scene_id=None))
    scene_file=MECH/'runs/C_stitch_SCENES.pkl';ident=json.loads((MECH/'results/SCENE_IDENTITIES.json').read_text());registered=next(x for x in ident['scenes'] if x['path']==str(scene_file));assert sha(scene_file)==registered['sha256'];sources.append(dict(path=str(scene_file),sha256=sha(scene_file)))
    scenes=pickle.load(open(scene_file,'rb'))
    for task in (1,3):
        streams=sorted(set(x['stream'] for x in scenes if x['task']==task))
        for stream in streams:
            selected=sorted([x for x in scenes if x['task']==task and x['stream']==stream],key=lambda x:x['episode'])[:4]
            for scene in selected:
                cell=geo.cell(scene['qpos']);g0=geo.cell(scene['goal']);first0=geo.firsts(cell,g0);alternatives=[]
                for g in geo.cells:
                    if g==cell or g==g0:continue
                    f=geo.firsts(cell,g)
                    if len(f)==1:alternatives.append(((int(f[0] in first0),abs(geo.distances[g][cell]-geo.distances[g0][cell]),g),g))
                g1=min(alternatives)[1];goals=[g0,g1]
                bank.append(dict(state_id=f'P3_T{task}_R{streams.index(stream)}_E{scene["episode"]:02d}',group='P3',cluster=f'T{task}_R{streams.index(stream)}',source='archived_official_initial_state',cell=cell,goals=[scene['goal'].copy(),geo.xy(g1)],goal_cells=goals,firsts=[geo.firsts(cell,g) for g in goals],routes=[geo.route(cell,g) for g in goals],scene=copy.deepcopy(scene),offline_index=None,data_action=None,source_path=str(scene_file),source_scene_id=scene['scene_id']))
    assert len(bank)<=64;dump(ROOT/'runs/FROZEN_BANK.pkl',bank);write(ROOT/'receipts/BANK_FROZEN.json',dict(epoch=time.time(),bank_path=str(ROOT/'runs/FROZEN_BANK.pkl'),bank_sha256=sha(ROOT/'runs/FROZEN_BANK.pkl'),preregistration_sha256=sha(ROOT/'PREREGISTRATION.json'),source_inputs=sources,models_queried=0,count=len(bank),junctions=[x[0] for x in junctions]))
    audit=[];accepted=[]
    for item in bank:
        error=0.;replay=[]
        # Same full physics/RNG/wrappers, same fixed action, twice; not seed matching.
        for _ in range(2):
            s,err=restore(e,item['scene'],item['goals'][0]);error=max(error,err);step(e,np.array([.123,-.234]),'restore_selfcheck');replay.append(snapshot(e,item['goals'][0],item['scene']['reset_seed']))
        assert equal(replay[0],replay[1]),'FULL_STATE_REPEAT_FAILED'
        results=[]
        for gi in (0,1):
            waypoint=geo.xy(item['firsts'][gi][0]);policy=lambda s,g,w=waypoint:np.clip((w-s)/.2,-1,1)
            metric,trace=rollout(e,geo,item,gi,policy,'admission');results.append(metric);dump(ROOT/'runs'/f'admission_{item["state_id"]}_g{gi}.pkl',trace)
        valid=all(x['correct'] for x in results);item['admitted']=valid
        audit.append(dict(state_id=item['state_id'],group=item['group'],cluster=item['cluster'],restore_error=error,full_integration_repeat_exact=True,admitted=valid,goals=results,reason='PASS' if valid else 'REFERENCE_CHANNEL_NOT_ENTERED_WITHIN_25'))
        if valid:accepted.append(item['state_id'])
        guard()
    table=[]
    for item in bank:
        for gi in (0,1):
            goals=item['goals'];table.append(dict(state_id=item['state_id'],group=item['group'],cluster=item['cluster'],source=item['source'],source_path=item['source_path'],source_scene_id=item['source_scene_id'],snapshot_bank_sha256=sha(ROOT/'runs/FROZEN_BANK.pkl'),qpos=json.dumps(item['scene']['qpos'].tolist()),qvel=json.dumps(item['scene']['qvel'].tolist()),integration_sha256=hashlib.sha256(item['scene']['integration'].tobytes()).hexdigest(),reset_seed=item['scene']['reset_seed'],goal_index=gi,goal=json.dumps(goals[gi].tolist()),cell=json.dumps(item['cell']),goal_cell=json.dumps(item['goal_cells'][gi]),BFS_edges=geo.distances[tuple(item['goal_cells'][gi])][tuple(item['cell'])],first_channels=json.dumps(item['firsts'][gi]),reference_route=json.dumps(item['routes'][gi]),euclidean_distance=float(np.linalg.norm(goals[gi]-item['scene']['qpos'])),offline_index=item['offline_index'],offline_action=json.dumps(item['data_action'].tolist()) if item['data_action'] is not None else 'UNMEASURABLE',admitted=item['admitted']))
    rows(ROOT/'SCENE_AND_GOAL_BANK.csv',table);write(ROOT/'ADMISSION_AUDIT.json',dict(status='COMPLETE',policy_outputs_used=False,states=audit,accepted=accepted,group_counts={g:sum(x['admitted'] and x['group']==g for x in bank) for g in ('P1','P2','P3')},P1_junctions=len(set(x['cluster'] for x in bank if x['admitted'] and x['group']=='P1')),state_repeat_tolerance=1e-10,physics_controller=PREREG['controller']))
    write(ROOT/'receipts/PREPARE_EXIT.json',dict(exit_code=0,epoch=time.time(),control_steps=json.loads((ROOT/'receipts/CONTROL_BUDGET.json').read_text())['used'],bank_sha256=sha(ROOT/'runs/FROZEN_BANK.pkl'),admission_sha256=sha(ROOT/'ADMISSION_AUDIT.json'),preregistration_sha256=sha(ROOT/'PREREGISTRATION.json')))
    e.close();print('PREPARE_COMPLETE',len(bank),len(accepted),flush=True)

if __name__=='__main__':main()
