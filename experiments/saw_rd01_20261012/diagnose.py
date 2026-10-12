"""Frozen-model D2/D3/D4 probes with train-style indices and independent confirmation."""
import argparse
import csv
import mujoco
import jax.numpy as jnp
from common import *
from data_analysis import geodesic
from qualify import TraceSampler


def restore(kind):
    env,data,val,config=load(f'pointmaze-large-{kind}-v0')
    agent=init_agent(data,config)
    path=ROOT/'runs'/f'formal_{kind}'/'params_1000000.pkl'
    with open(path,'rb') as f: state=pickle.load(f)
    assert state['step']==1000000
    agent=flax.serialization.from_state_dict(agent,state['agent'])
    return env,data,val,config,jax.device_put(agent,jax.devices('cpu')[0]),dict(path=str(path),sha256=sha(path))


@jax.jit
def probe(agent,s,g,w,n,a):
    v1,v2=agent.network.select('value')(s,g)
    w1,w2=agent.network.select('value')(w,g)
    n1,n2=agent.network.select('value')(n,g)
    actor=agent.network.select('actor')(s,g).mode()
    teacher=agent.network.select('low_actor')(s,w).mode()
    v=(v1+v2)/2; wv=(w1+w2)/2; nv=(n1+n2)/2
    log_rho=3*(wv-v); log_awr=3*(nv-v)
    return dict(v1=v1,v2=v2,w1=w1,w2=w2,v=v,wv=wv,waypoint_adv=wv-v,
                log_rho=log_rho,rho_before=jnp.exp(log_rho),rho=jnp.minimum(jnp.exp(log_rho),100.),
                one_step_adv=nv-v,awr_before=jnp.exp(log_awr),awr=jnp.minimum(jnp.exp(log_awr),100.),
                actor=actor,teacher=teacher,mean_squared_difference=jnp.sum((actor-teacher)**2,axis=-1),
                actor_data_mse=jnp.sum((actor-a)**2,axis=-1),teacher_data_mse=jnp.sum((teacher-a)**2,axis=-1))


def rows_to_csv(path,rows):
    keys=list(rows[0]) if rows else []
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n'); w.writeheader(); w.writerows(rows)


def run(kind,phase,do_rollouts=True):
    start=time.monotonic(); cpu_start=time.process_time()
    env,data,val,config,agent,identity=restore(kind); u=env.unwrapped
    is_confirm=phase=='confirmation'; dataset=val if is_confirm else data
    rawpath=ROOT/'vendor/data'/f'pointmaze-large-{kind}-v0{"-val" if is_confirm else ""}.npz'
    raw=np.load(rawpath)
    obs=raw['observations']; qvel=raw['qvel']; config_seed=(111001 if is_confirm else 101001)+(kind=='stitch')
    if is_confirm:
        tracer=TraceSampler(dataset.dataset,config)
        with separate_numpy(config_seed): tracer.sample(32)
        i=tracer.indices; _,low,g,w=tracer.observed
        ids=np.arange(32)+100000
    else:
        # Deterministic preregistered diversity by trajectory horizon/turns, not outcomes.
        idx=np.loadtxt(ROOT/'results'/f'{kind}_sampler_indices.csv',delimiter=',',skiprows=1,dtype=int)
        geom=np.loadtxt(ROOT/'results'/f'{kind}_sampler_geometry.csv',delimiter=',',skiprows=1)
        eligible=np.flatnonzero((idx[:,4]-idx[:,1]>=25)&(np.linalg.norm(obs[idx[:,5]]-obs[idx[:,1]],axis=1)>1.0))
        turn=eligible[geom[eligible,-1]>0]; straight=eligible[geom[eligible,-1]==0]
        chosen=np.asarray([v for pair in zip(turn[:16],straight[:16]) for v in pair],dtype=int)
        if len(chosen)<32: chosen=np.r_[chosen,[x for x in eligible if x not in set(chosen)][:32-len(chosen)]]
        ids,i,_,low,g,w=idx[chosen].T
    n=np.minimum(i+1,len(obs)-1)
    original_g=g.copy()
    goal_vectors=obs[g].copy()
    external=np.arange(len(i))%2==1
    with separate_numpy(config_seed+1000000):
        task_goals=np.asarray([u.add_noise(u.ij_to_xy(u.task_infos[t%5]['goal_ij'])) for t in range(len(i))],dtype=obs.dtype)
    goal_vectors[external]=task_goals[external]
    g=g.copy();g[external]=-1
    end=dataset.terminal_locs[np.searchsorted(dataset.terminal_locs,i)]
    w=w.copy();w[external]=np.minimum(i[external]+25,end[external])
    info=sync(probe(agent,obs[i],goal_vectors,obs[w],obs[n],raw['actions'][i]))
    info={k:np.asarray(v) for k,v in info.items()}
    tuple_rows=[]
    for t in range(len(i)):
        row=dict(dataset=kind,phase=phase,sample_id=int(ids[t]),state_idx=int(i[t]),goal_idx=int(g[t]),waypoint_idx=int(w[t]),
                 original_sampled_goal_idx=int(original_g[t]),goal_source='registered_task_goal' if external[t] else 'original_future_goal',
                 goal_x=float(goal_vectors[t,0]),goal_y=float(goal_vectors[t,1]),goal_distance=float(np.linalg.norm(goal_vectors[t]-obs[i[t]])),
                 high_horizon=-1 if external[t] else int(g[t]-i[t]),waypoint_horizon=int(w[t]-i[t]))
        row.update({k:float(v[t]) for k,v in info.items() if v.ndim==1})
        for k in ('actor','teacher'):
            for d in range(2): row[f'{k}_{d}']=float(info[k][t,d])
        row['weighted_matching_error']=row['rho']*row['mean_squared_difference']
        tuple_rows.append(row)
    rows_to_csv(ROOT/'results'/f'{kind}_{phase}_D4_tuples.csv',tuple_rows)
    # Fourteen actual data future candidates, state itself, and original goal.
    offsets=np.array([0,1,2,3,4,6,8,10,12,14,16,18,20,24,25])
    end=dataset.terminal_locs[np.searchsorted(dataset.terminal_locs,i)]
    candidates=np.minimum(i[:,None]+offsets[None,:],end[:,None])
    candidates=np.c_[candidates,g]
    candidate_vectors=np.concatenate([obs[candidates[:,:15]],goal_vectors[:,None,:]],axis=1)
    flat_s=np.repeat(obs[i],16,axis=0); flat_g=np.repeat(goal_vectors,16,axis=0)
    cinfo=sync(probe(agent,flat_s,flat_g,candidate_vectors.reshape(-1,2),np.repeat(obs[n],16,axis=0),np.repeat(raw['actions'][i],16,axis=0)))
    cinfo={k:np.asarray(v).reshape((len(i),16)+np.asarray(v).shape[1:]) for k,v in cinfo.items()}
    candidate_rows=[]; conflict=[]; disagreement=[]; progress_rank=[]
    for t in range(len(i)):
        geo=geodesic(u,candidate_vectors[t],goal_vectors[t])
        sgeo=float(geodesic(u,obs[[i[t]]],goal_vectors[t])[0])
        rho=cinfo['rho'][t].copy()
        supported=(candidates[t]>i[t])&(candidates[t]<=i[t]+25)
        unique_seen=set()
        for c,wi in enumerate(candidates[t]):
            if int(wi) in unique_seen: supported[c]=False
            unique_seen.add(int(wi))
        rho=rho*supported
        p=rho/(rho.sum()+1e-30)
        teachers=cinfo['teacher'][t]; actor=cinfo['actor'][t,0]; target=(p[:,None]*teachers).sum(0)
        spread=float((p[:,None]*(teachers-target)**2).sum())
        conflict.append(dict(sample_id=int(ids[t]),weighted_teacher_variance=spread,
                             actor_to_weighted_teacher_mse=float(((actor-target)**2).sum()),
                             supported_unique_candidates=int(supported.sum()),
                             weight_effective_candidates=float(1/(np.sum(p**2)+1e-30)),max_weight_fraction=float(p.max())))
        values=cinfo['wv'][t]; head1=cinfo['w1'][t]; head2=cinfo['w2'][t]
        order1=np.argsort(np.argsort(head1)); order2=np.argsort(np.argsort(head2))
        disagreement.append(float(np.mean(order1!=order2)))
        if np.std(values)>0 and np.std(geo)>0: progress_rank.append(float(np.corrcoef(values,-geo)[0,1]))
        for c,wi in enumerate(candidates[t]):
            future_steps=int(wi-i[t]); supported_short=(0<=future_steps<=25)
            row=dict(dataset=kind,phase=phase,sample_id=int(ids[t]),candidate_id=c,state_idx=int(i[t]),goal_idx=int(g[t]),waypoint_idx=int(wi),
                     goal_source='registered_task_goal' if external[t] else 'original_future_goal',
                     goal_x=float(goal_vectors[t,0]),goal_y=float(goal_vectors[t,1]),
                     candidate_source='external_task_goal_reference' if wi<0 else 'recorded_state',
                     actual_data_future_steps=future_steps,actual_original_trajectory_id=int(np.searchsorted(np.flatnonzero(raw['terminals']),i[t])),
                     same_original_trajectory=bool(wi>=0 and np.searchsorted(np.flatnonzero(raw['terminals']),i[t])==np.searchsorted(np.flatnonzero(raw['terminals']),wi)),
                     empirical_within_25_steps=supported_short,cell_progress=float(sgeo-geo[c]),
                     waypoint_distance=float(np.linalg.norm(candidate_vectors[t,c]-obs[i[t]])),
                     unique_candidate_count=len(np.unique(candidates[t])),ranking=int(np.sum(values>values[c])))
            row.update({k:float(v[t,c]) for k,v in cinfo.items() if v.ndim==2})
            candidate_rows.append(row)
    rows_to_csv(ROOT/'results'/f'{kind}_{phase}_D2_candidates.csv',candidate_rows)
    rows_to_csv(ROOT/'results'/f'{kind}_{phase}_D4_conflicts.csv',conflict)
    rollouts=[]; raw_rollouts=[]
    if do_rollouts:
        # 16 matched pairs per dataset/phase; total 128 segments across both phases.
        selected=[t for t in range(len(i)) if np.linalg.norm(obs[w[t]]-obs[i[t]])>u._goal_tol][:16]
        action_fn=jax.jit(lambda a,s,target,low: jax.lax.cond(low,
                          lambda _: a.network.select('low_actor')(s,target).mode(),
                          lambda _: a.network.select('actor')(s,target).mode(),operand=None))
        for t in selected:
            for actor_type in ('low_actor','actor'):
                reset_seed=121001+(kind=='stitch')+1000*is_confirm+int(ids[t])
                with separate_numpy(reset_seed):
                    env.action_space.seed(reset_seed+2000000)
                    env.reset(seed=reset_seed,options={'task_id':1})
                    u.set_state(obs[i[t]].astype(float),qvel[i[t]].astype(float)); u.set_goal(goal_xy=obs[w[t]].astype(float))
                    assert np.allclose(u.get_ob(),obs[i[t]],atol=1e-6)
                    initial_qpos=u.data.qpos.copy(); initial_qvel=u.data.qvel.copy()
                    states=[u.get_ob().copy()]; velocities=[initial_qvel]; actions=[]; contacts=[]
                    success=False; budget=min(25,int(w[t]-i[t])); budget=max(1,budget)
                    for k in range(budget):
                        action=np.clip(np.asarray(action_fn(agent,states[-1],obs[w[t]],actor_type=='low_actor')),-1,1)
                        s,r,term,trunc,inf=env.step(action)
                        states.append(s.copy()); velocities.append(u.data.qvel.copy()); actions.append(action); contacts.append(int(u.data.ncon))
                        if inf['success']: success=True; break
                    row=dict(dataset=kind,phase=phase,sample_id=int(ids[t]),state_idx=int(i[t]),waypoint_idx=int(w[t]),actor_type=actor_type,
                             reset_seed=reset_seed,budget=budget,steps=len(actions),success=success,
                             initial_distance=float(np.linalg.norm(obs[w[t]]-obs[i[t]])),final_distance=float(np.linalg.norm(states[-1]-obs[w[t]])),
                             full_qpos_match_error=float(np.max(np.abs(initial_qpos-raw['qpos'][i[t]]))),
                             full_qvel_match_error=float(np.max(np.abs(initial_qvel-qvel[i[t]]))))
                    assert row['full_qpos_match_error']<1e-5 and row['full_qvel_match_error']<1e-5
                    rollouts.append(row)
                    raw_rollouts.append(dict(**row,states=np.asarray(states),qvel=np.asarray(velocities),actions=np.asarray(actions),contacts=np.asarray(contacts),goal=obs[w[t]]))
        rows_to_csv(ROOT/'results'/f'{kind}_{phase}_D3_rollouts.csv',rollouts)
        with open(ROOT/'runs'/f'{kind}_{phase}_local_rollouts.pkl','wb') as f: pickle.dump(raw_rollouts,f,protocol=4)
    # D4 full 4096 training-rule sample evidence, not solely chosen illustrative states.
    full_summary=None
    if not is_confirm:
        batch=np.load(ROOT/'runs'/f'{kind}_sampler_probe.npz'); chunks=[]
        for start_idx in range(0,4096,256):
            sl=slice(start_idx,start_idx+256)
            x=sync(probe(agent,batch['observations'][sl],batch['high_actor_goals'][sl],batch['high_actor_targets'][sl],batch['next_observations'][sl],batch['actions'][sl]))
            chunks.append({k:np.asarray(v) for k,v in x.items()})
        full={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        np.savez_compressed(ROOT/'runs'/f'{kind}_D4_4096_raw.npz',**full)
        full_summary={}
        for k,v in full.items():
            if v.ndim!=1: continue
            finite=v[np.isfinite(v)]
            full_summary[k]=dict(finite_count=len(finite),nonfinite_count=int((~np.isfinite(v)).sum()),
                                 mean=float(finite.mean()) if len(finite) else None,
                                 q50=float(np.quantile(finite,.5)) if len(finite) else None,
                                 q90=float(np.quantile(finite,.9)) if len(finite) else None,
                                 q99=float(np.quantile(finite,.99)) if len(finite) else None)
    summary=dict(kind=kind,phase=phase,checkpoint=identity,state_goal_groups=len(i),candidates_per_group=16,
                 diagnostic_goal_groups={label:dict(n=int(mask.sum()),goal_distance_mean=float(np.linalg.norm(goal_vectors[mask]-obs[i[mask]],axis=1).mean()),
                                                      value_head_abs_difference_mean=float(np.abs(info['v1'][mask]-info['v2'][mask]).mean()),
                                                      matching_error_mean=float(info['mean_squared_difference'][mask].mean()),
                                                      weighted_matching_error_mean=float((info['rho'][mask]*info['mean_squared_difference'][mask]).mean()),
                                                      rho_saturation=float(np.mean(info['log_rho'][mask]>=np.log(100))))
                                         for label,mask in [('original_future_goal',~external),('registered_task_goal',external)]},
                 tuple_rho_saturation=float(np.mean(info['log_rho']>=np.log(100))),
                 candidate_rho_saturation=float(np.mean(cinfo['log_rho']>=np.log(100))),
                 candidate_rho_before_float32_overflow_fraction=float(np.mean(~np.isfinite(cinfo['rho_before']))),
                 value_head_rank_disagreement=float(np.mean(disagreement)),
                 value_vs_cell_distance_correlation_mean=float(np.mean(progress_rank)) if progress_rank else None,
                 value_geometry_inference='correlation is descriptive, not a correctness label; empirical future candidates verified from full recorded states',
                 D4_matching_error_mean=float(np.mean(info['mean_squared_difference'])),D4_weighted_matching_error_mean=float(np.mean(info['rho']*info['mean_squared_difference'])),
                 D4_teacher_variance_mean=float(np.mean([x['weighted_teacher_variance'] for x in conflict])),
                 D4_teacher_validity='unique empirical short future support; actual low-actor execution checked separately for selected original waypoints',
                 actor_mean_action_clip_fraction=float(np.mean(np.any(np.abs(info['actor'])>1,axis=1))),
                 teacher_mean_action_clip_fraction=float(np.mean(np.any(np.abs(info['teacher'])>1,axis=1))),
                 D4_full_training_style_4096=full_summary,
                 D3_segments=len(rollouts),D3_success={k:float(np.mean([r['success'] for r in rollouts if r['actor_type']==k])) for k in ['low_actor','actor']} if rollouts else None,
                 seconds=time.monotonic()-start,cpu_process_seconds=time.process_time()-cpu_start,
                 confirmation_note='fresh preregistered state-goal RNG on official validation data; validation distribution was used for training logs, not claimed wholly unconsumed' if is_confirm else 'training-style discovery samples')
    write_json(ROOT/'results'/f'{kind}_{phase}_DIAGNOSIS.json',summary)
    env.close()
    print('DIAGNOSIS_DONE',kind,phase,summary['D3_success'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['navigate','stitch'],required=True)
    p.add_argument('--phase',choices=['discovery','confirmation'],required=True); p.add_argument('--no-rollouts',action='store_true')
    a=p.parse_args(); run(a.kind,a.phase,not a.no_rollouts)
