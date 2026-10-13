"""Freeze geometry, complete scenes and independent reference admission before policy query."""
from core import *
import itertools, subprocess, platform

STARTS=[(1,1),(7,4),(1,8),(3,1),(3,6),(3,10),(5,6),(5,8)]
DIST=[2,4,6,8,10,12,14]
def region(c): return ('N' if c[0]<=4 else 'S')+('W' if c[1]<=6 else 'E')
def js(x): return json.dumps(x,default=lambda v:v.tolist() if hasattr(v,'tolist') else int(v))
def main():
    guard(); assert not (ROOT/'runs/GEOMETRY_BANK.pkl').exists()
    entries=model_entries(); e=env_new(); geo=Geometry(e)
    assert len(geo.cells)==46 and max(max(d.values()) for d in geo.distances.values())==19
    supplied=json.loads((ROOT/'GEOMETRY_PREFLIGHT.json').read_text()); counts=[]
    gp_bank=pickle.load(open(GP/'runs/FROZEN_BANK.pkl','rb')); bank=[]; missing=[]
    for si,c in enumerate(STARTS):
        source=sorted([x for x in gp_bank if x['group']=='P3' and tuple(x['cell'])==c],key=lambda x:x['state_id']) if si<2 else []
        if si<2:
            if not source: missing.append(dict(start_cell=c,status='ARCHIVED_STATE_MISSING')); continue
            item=source[0]; scene=copy.deepcopy(item['scene']); source_id=item['state_id']; kind='archived_P3_complete_initial'
            assert geo.cell(scene['qpos'])==c
        else:
            seed=26061301+si; e.action_space.seed(seed+2000000); _,info=e.reset(seed=seed,options={'task_id':1,'render_goal':False})
            e.unwrapped.data.qpos[:]=geo.xy(c); e.unwrapped.data.qvel[:]=0; mujoco.mj_forward(e.unwrapped.model,e.unwrapped.data)
            scene=snapshot(e,info['goal'],seed); source_id=None; kind='privileged_geometry_constructed_complete_state'
        options={d:[] for d in DIST}; channels={}
        for d in DIST:
            all_g=[g for g in geo.cells if geo.distances[g][c]==d]; unique=[g for g in all_g if len(geo.firsts(c,g))==1]
            reg=next(x for x in supplied['rows'] if x['start_ij']==list(c) and x['bfs_edges']==d)
            assert (len(all_g),len(unique))==(reg['all_goal_cells'],reg['unique_shortest_first_exit_cells'])
            for g in unique:
                f=geo.firsts(c,g)[0]; channels.setdefault(f,set()).add(d); options[d].append(g)
            counts.append(dict(start_cell=js(c),distance=d,all_candidates=len(all_g),unique_candidates=len(unique),tie_candidates=len(all_g)-len(unique),candidate_regions=js(sorted(set(region(g) for g in unique)))))
        selected_channels=sorted(channels,key=lambda f:(not(bool(channels[f]&{2,4}) and bool(channels[f]&{8,10,12,14})),-len(channels[f]),f))[:2]
        selected=[]
        for d in DIST:
            goals=[g for g in options[d] if geo.firsts(c,g)[0] in selected_channels]
            combinations=list(itertools.combinations(goals,min(2,len(goals)))) if goals else []
            # Region coverage first as card; channel coverage deterministic secondary tie break, then goal lexicographic.
            chosen=min(combinations,key=lambda gs:(-len(set(region(g) for g in gs)),-len(set(geo.firsts(c,g)[0] for g in gs)),gs)) if combinations else ()
            selected.extend((g,'main','UNIQUE_MAIN') for g in chosen)
            row=counts[-len(DIST)+DIST.index(d)]; row.update(selected_channels=js(selected_channels),selected_goals=js(chosen),selected_count=len(chosen),cross_region_status='CROSS_REGION_AVAILABLE' if len(set(region(g) for g in chosen))>=2 else 'NO_CROSS_REGION_SUPPORT')
        auxiliary=[]
        if si<2:
            g=geo.cell(scene['goal']); auxiliary.append((g,'aux','OFFICIAL_ANCHOR'))
        ties=[]
        for g in geo.cells:
            if len(geo.firsts(c,g))<2 or g in [x[0] for x in auxiliary]: continue
            matches=[x[0] for x in selected if geo.distances[x[0]][c]==geo.distances[g][c]]
            if matches: ties.append((not any(region(x)==region(g) for x in matches),geo.distances[g][c],g))
        for _,_,g in sorted(ties)[:2-len(auxiliary)]: auxiliary.append((g,'aux','MATCHED_TIE'))
        for g,b,role in selected+auxiliary:
            cid=f'S{si:02d}_{b}_d{geo.distances[g][c]:02d}_g{g[0]}_{g[1]}'
            bank.append(dict(condition_id=cid,start_cell=c,goal_cell=g,goal_xy=geo.xy(g),scene=copy.deepcopy(scene),bank=b,role=role,distance=geo.distances[g][c],region=region(g),firsts=geo.firsts(c,g),source=kind,source_state_id=source_id,admitted=False))
    assert sum(x['bank']=='main' for x in bank)<=112 and sum(x['bank']=='aux' for x in bank)<=16 and len(bank)<=128
    dump(ROOT/'runs/GEOMETRY_BANK.pkl',bank)
    write(ROOT/'receipts/GEOMETRY_FREEZE.json',dict(epoch=time.time(),models_queried=0,bank_sha256=sha(ROOT/'runs/GEOMETRY_BANK.pkl'),policy_independent=True,missing=missing,selection='channel common support then coverage then lex; targets max regional coverage then channel coverage then lex; auxiliary same-distance tie regional match then distance then lex'))
    audit=[]
    for item in bank:
        repeats=[]
        for _ in range(2):
            s,err=restore(e,item['scene'],item['goal_xy']); step(e,np.array([.123,-.234]),'restore_selfcheck'); repeats.append(snapshot(e,item['goal_xy'],item['scene']['reset_seed']))
        assert equal(repeats[0],repeats[1])
        s,err=restore(e,item['scene'],item['goal_xy']); target=geo.xy(item['firsts'][0]); states=[s.copy()]; correct=False
        for t in range(25):
            s,_,term,trunc,info=step(e,np.clip((target-s)/.2,-1,1),'reference_admission'); states.append(s.copy())
            if geo.cell(s)!=item['start_cell']: correct=geo.cell(s) in item['firsts']; break
            if term or trunc: break
        item['admitted']=bool(correct)
        audit.append(dict(condition_id=item['condition_id'],admitted=correct,restore_error=err,full_state_repeat_exact=True,reference_steps=len(states)-1,reason='PASS' if correct else 'REFERENCE_CHANNEL_NOT_ENTERED_WITHIN_25'))
        dump(ROOT/'runs'/('admission_'+item['condition_id']+'.pkl'),dict(states=np.asarray(states),initial=item['scene'],goal_xy=item['goal_xy'],reference_firsts=item['firsts'],audit=audit[-1]))
    dump(ROOT/'runs/SCENE_GOAL_BANK.pkl',bank); rows(ROOT/'GEOMETRY_SUPPORT.csv',counts)
    rows(ROOT/'SCENE_GOAL_BANK.csv',[dict(condition_id=x['condition_id'],bank=x['bank'],role=x['role'],start_cell=js(x['start_cell']),goal_cell=js(x['goal_cell']),goal_xy=js(x['goal_xy']),distance=x['distance'],region=x['region'],firsts=js(x['firsts']),qpos=js(x['scene']['qpos']),qvel=js(x['scene']['qvel']),integration_sha256=hashlib.sha256(x['scene']['integration'].tobytes()).hexdigest(),source=x['source'],source_state_id=x['source_state_id'],admitted=x['admitted']) for x in bank])
    write(ROOT/'ADMISSION_AUDIT.json',dict(items=audit,admitted=sum(x['admitted'] for x in bank),total=len(bank),models_queried=0))
    official=OLD/'vendor/saw'; assert subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],text=True).strip()=='3535a16c901e5de1233ac5a586401b3ddfa5e30d'
    assert not subprocess.check_output(['git','-C',str(official),'diff','HEAD','--name-only'],text=True).strip()
    inputs=[dict(path=str(OLD/'vendor/data/pointmaze-large-stitch-v0.npz'),sha256=sha(OLD/'vendor/data/pointmaze-large-stitch-v0.npz')),dict(path=str(GP/'runs/FROZEN_BANK.pkl'),sha256=sha(GP/'runs/FROZEN_BANK.pkl'))]
    for entry in entries:
        cf=(OLD if entry['variant']=='A' else PREV)/'runs/formal_stitch/config.json' if entry['seed']==0 else MECH/f'runs/C_stitch_{entry["variant"]}_s{entry["seed"]}/config.json'
        entry.update(config_path=str(cf),config_sha256=sha(cf)); inputs.append(dict(path=str(cf),sha256=sha(cf))); inputs.append(dict(path=entry['source_index'],sha256=entry['source_index_sha256']))
    write(ROOT/'FROZEN_MANIFEST.json',dict(models=entries,inputs=inputs,bank_sha256=sha(ROOT/'runs/SCENE_GOAL_BANK.pkl'),base_commit='19d1b4b411a00b92866b4512150339472a086f0b',official_commit='3535a16c901e5de1233ac5a586401b3ddfa5e30d',seed=26061301,bootstrap_seed=26061303,bootstrap_draws=5000,devices=[str(d) for d in jax.devices()],affinity=sorted(os.sched_getaffinity(0)),versions=dict(python=platform.python_version(),jax=jax.__version__,numpy=np.__version__,mujoco=mujoco.__version__),state_restore_tolerance=1e-10,thresholds=dict(small_command_mean_norm=.25,near_stationary_max_displacement=.1,displacement_floor=.01,reversal_cosine=-.5,repeat_radius=.1,repeat_min_separation=3,backtracking_path_min=1,backtracking_reversals_min=3,backtracking_repeat_min=2),budget=json.loads((ROOT/'receipts/RESOURCE_START.json').read_text())))
    e.close(); print('PREPARE_COMPLETE',len(bank),sum(x['admitted'] for x in bank),flush=True)
if __name__=='__main__': main()
