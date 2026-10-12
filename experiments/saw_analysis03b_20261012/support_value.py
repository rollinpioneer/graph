"""Localized support, real-data signal and frozen model queries; never updates."""
import argparse
from evaluate import *
from scipy.spatial import cKDTree
import jax.numpy as jnp

@jax.jit
def query(agent,s,g,w,n,action):
    v1,v2=agent.network.select('value')(s,g);w1,w2=agent.network.select('value')(w,g);n1,n2=agent.network.select('value')(n,g)
    actor=agent.network.select('actor')(s,g).mode();teacher=agent.network.select('low_actor')(s,w).mode()
    adv=(w1+w2-v1-v2)/2;one=(n1+n2-v1-v2)/2;lr=agent.config['kl_alpha']*adv;la=agent.config['awr_alpha']*one
    return dict(v1=v1,v2=v2,w1=w1,w2=w2,waypoint_adv=adv,log_rho=lr,rho=jnp.minimum(jnp.exp(lr),100),
        rho_before=jnp.exp(lr),one_step_adv=one,awr=jnp.minimum(jnp.exp(la),100),log_awr=la,
        actor=actor,teacher=teacher,actor_teacher_mse=jnp.sum((actor-teacher)**2,axis=-1),
        actor_data_mse=jnp.sum((actor-action)**2,axis=-1),teacher_data_mse=jnp.sum((teacher-action)**2,axis=-1))

losses=jax.jit(lambda agent,batch:agent.total_loss(batch)[1])

@jax.jit
def gradients(agent,batch):
    ga=jax.grad(lambda p:agent.actor_loss(batch,p)[0])(agent.network.params)
    gw=jax.grad(lambda p:agent.waypoint_loss(batch,p)[0])(agent.network.params)
    al=jax.tree_util.tree_leaves(ga);wl=jax.tree_util.tree_leaves(gw)
    dot=sum(jnp.sum(a*w) for a,w in zip(al,wl));an=jnp.sqrt(sum(jnp.sum(a*a) for a in al));wn=jnp.sqrt(sum(jnp.sum(w*w) for w in wl))
    return dict(awr_gradient_norm=an,waypoint_gradient_norm=wn,cosine=dot/(an*wn+1e-30))

def prepare():
    with open(ROOT/'runs/E2_SNAPSHOTS.pkl','rb') as f:snaps=pickle.load(f)
    entries=[];rows=[];replays=[]
    for kind in ('navigate','stitch'):
        env,data,val,config=load(f'pointmaze-large-{kind}-v0','A')
        raw=np.load(OLD/'vendor/data'/f'pointmaze-large-{kind}-v0.npz');obs=raw['observations'];vel=raw['qvel'];acts=raw['actions']
        valid=np.flatnonzero(1-raw['terminals']);tree=cKDTree(obs[valid]);terminal=np.flatnonzero(raw['terminals'])
        for snap in [x for x in snaps if x['kind']==kind]:
            check_time();scene=snap['scene'];s=scene['qpos'];g=scene['goal'];v=scene['qvel']
            distance,near=tree.query(s,k=100);ids=valid[np.atleast_1d(near)];i=int(ids[0]);gi=int(valid[tree.query(g,k=1)[1]])
            compact_end=int(data.terminal_locs[np.searchsorted(data.terminal_locs,i)]);rawtraj=int(np.searchsorted(terminal,i))
            pe=np.max(np.abs(obs[ids]-s),axis=1);ve=np.max(np.abs(vel[ids]-v),axis=1);full=(pe<=.15)&(ve<=.5)
            other=(np.searchsorted(terminal,ids)!=rawtraj)&full
            hit=np.linalg.norm(obs-g,axis=1)<=1.0
            offsets=np.array([0,1,2,3,4,6,8,10,12,14,16,18,20,24,25]);cidx=np.minimum(i+offsets,compact_end)
            cv=np.vstack([obs[cidx],g]);ni=min(i+1,len(obs)-1)
            future=np.arange(min(i+1,compact_end),compact_end+1)
            if config['actor_geom_sample']:
                weights=(1-config['discount'])*config['discount']**(future-i-1);weights[-1]=config['discount']**max(0,compact_end-i-1)
            else:
                weights=np.ones(len(future));weights[[0,-1]]=.5 if len(future)>1 else 1.;weights/=weights.sum()
            p_a=float(weights@hit[future]);p_r=float(hit[valid].mean());p_b=.5*p_a+.5*p_r
            same_segment=bool(i<=gi<=compact_end);wi=min(i+25,gi) if same_segment else min(i+25,compact_end)
            batch=dict(observations=obs[[i]],next_observations=obs[[ni]],actions=acts[[i]],value_goals=obs[[gi]],
                high_actor_goals=obs[[gi]],high_actor_targets=obs[[wi]],low_actor_goals=obs[[min(i+25,compact_end)]],
                rewards=np.array([float(i==gi)-1]),masks=np.array([float(i!=gi)]))
            entry=dict(snapshot_id=snap['id'],kind=kind,task=snap['task'],source_success=snap['source_success'],scene=scene,
                state_idx=i,goal_idx=gi,waypoint_idx=wi,candidate_indices=np.r_[cidx,-1],candidate_vectors=cv,
                raw_observation=obs[i].copy(),next_observation=obs[ni].copy(),data_action=acts[i].copy(),batch=batch)
            entries.append(entry)
            row=dict(snapshot_id=snap['id'],kind=kind,task=snap['task'],source_success=snap['source_success'],state_idx=i,goal_idx=gi,
                nearest_position_error=float(pe[0]),nearest_velocity_error=float(ve[0]),nearest_goal_position_error=float(np.linalg.norm(obs[gi]-g)),
                nearby_qpos_qvel_supported_count=int(full.sum()),nearby_other_trajectory_supported_count=int(other.sum()),
                future_goal_region_probability_A=p_a,goal_region_probability_random=p_r,goal_region_probability_B=p_b,
                actual_tuple_A_sampler_supported=same_segment,actual_tuple_B_sampler_supported=True,
                full_integration_available_in_raw_data=False,region_radius=1.0,trajectory_id=rawtraj)
            rows.append(row)
            if snap['episode']==0 and snap['task'] in (1,2,5):
                options=[('nearest',i),('other_trajectory',int(ids[np.flatnonzero(other)[0]]) if other.any() else None)]
                for label,idx in options:
                    if idx is None:replays.append(dict(snapshot_id=snap['id'],kind=kind,chain=label,status='NO_QPOS_QVEL_SUPPORTED_JOIN'));continue
                    end=int(terminal[np.searchsorted(terminal,idx)]);n=min(25,end-idx)
                    s0,err=restore_gate(env,scene);positions=[s0.copy()];velocities=[env.unwrapped.data.qvel.copy()];perr=[];verr=[];actions=[]
                    for j in range(n):
                        consume('E2');check_time();a=acts[idx+j];ss,r,t,tr,info=env.step(a);actions.append(a.copy());positions.append(ss.copy());velocities.append(env.unwrapped.data.qvel.copy())
                        perr.append(float(np.max(np.abs(ss-obs[idx+j+1]))));verr.append(float(np.max(np.abs(env.unwrapped.data.qvel-vel[idx+j+1]))))
                        if t or tr:break
                    record=dict(snapshot_id=snap['id'],kind=kind,chain=label,status='MEASURED',state_idx=idx,steps=len(actions),
                        max_position_error=max(perr,default=0.),max_velocity_error=max(verr,default=0.),
                        approximate_join_gate=bool(perr and max(perr)<=.15 and max(verr)<=.5),
                        strict_full_data_state_reproduction=False,original_far_success=bool(info['success']) if actions else False)
                    replays.append(record);dump(ROOT/'runs'/f'E3_REPLAY_{snap["id"]}_{label}.pkl',dict(**record,initial=scene,states=np.asarray(positions),qvel=np.asarray(velocities),actions=np.asarray(actions),position_errors=perr,velocity_errors=verr))
        raw.close();env.close()
    dump(ROOT/'runs/E3_QUERY_ENTRIES.pkl',entries);csv_rows(ROOT/'results/E3_SUPPORT.csv',rows)
    write(ROOT/'results/E3_DATA_REPLAYS.json',replays);write(ROOT/'receipts/E3_prepare_exit.json',dict(status='COMPLETED',entries=len(entries),replay_chains=len(replays)))
    print('E3_PREPARED',len(entries),flush=True)

def model_queries(variant):
    with open(ROOT/'runs/E3_QUERY_ENTRIES.pkl','rb') as f:entries=pickle.load(f)
    branchfile=ROOT/'results'/('E2_B_BRANCHES.csv' if variant=='B' else 'E2_BRANCHES.csv');trigger=set()
    if branchfile.exists():
        rr=list(csv.DictReader(branchfile.open()));groups={r['snapshot_id']:{x['condition']:x for x in rr if x['snapshot_id']==r['snapshot_id']} for r in rr}
        trigger={k for k,v in groups.items() if v['C2']['local_reached']=='True' and v['C1']['local_reached']=='False'}
    candidates=[];signals=[];grad_rows=[];identities=[];missing=[];start=time.time()
    gradient_batches={}
    if trigger and variant=='B':
        with open(ROOT/'runs/E2_CANDIDATES.pkl','rb') as f:verified={x['snapshot_id']:x for x in pickle.load(f)}
        joint=list(csv.DictReader((ROOT/'results/E3_JOINT_SAMPLING_SUPPORT.csv').open()))
        for kind in ('navigate','stitch'):
            raw=np.load(OLD/'vendor/data'/f'pointmaze-large-{kind}-v0.npz');obs=raw['observations'];act=raw['actions']
            for e in [x for x in entries if x['kind']==kind and x['snapshot_id'] in trigger]:
                near=verified[e['snapshot_id']]['near'];i=near['state_idx'];w=near['waypoint_idx'];g=e['goal_idx']
                j=next(x for x in joint if x['snapshot_id']==e['snapshot_id'] and x['group']=='E2_verified_near')
                assert j['random_waypoint_matches']=='True'
                gradient_batches[e['snapshot_id']]=(dict(observations=obs[[i]],next_observations=obs[[i+1]],actions=act[[i]],value_goals=obs[[g]],
                    high_actor_goals=obs[[g]],high_actor_targets=obs[[w]],low_actor_goals=obs[[w]],rewards=np.array([float(i==g)-1]),masks=np.array([float(i!=g)])),
                    dict(state_idx=i,goal_idx=g,waypoint_idx=w,qpos_match_error=near['qpos_error'],qvel_match_error=near['qvel_error'],
                        tuple_scope='actual E2 verified target and native B random-goal waypoint; raw state approximates physical snapshot'))
            raw.close()
    for kind in ('navigate','stitch'):
        for step in (100000,300000,500000,1000000):
            try:agent,identity=load_checkpoint(kind,variant,step)
            except FileNotFoundError:
                missing.append(dict(kind=kind,step=step));continue
            identities.append(identity)
            for entry in [x for x in entries if x['kind']==kind]:
                check_time();n=16;s=np.repeat(entry['scene']['qpos'][None],n,axis=0).astype(np.float32);g=np.repeat(entry['scene']['goal'][None],n,axis=0).astype(np.float32)
                info={k:np.asarray(v) for k,v in sync(query(agent,s,g,entry['candidate_vectors'].astype(np.float32),np.repeat(entry['next_observation'][None],n,axis=0),np.repeat(entry['data_action'][None],n,axis=0))).items()}
                rank1=np.argsort(np.argsort(-info['w1'],kind='stable'),kind='stable');rank2=np.argsort(np.argsort(-info['w2'],kind='stable'),kind='stable')
                for c in range(16):
                    row=dict(snapshot_id=entry['snapshot_id'],kind=kind,variant=variant,checkpoint_step=step,task=entry['task'],candidate=c,
                        candidate_idx=int(entry['candidate_indices'][c]),state_idx=entry['state_idx'],goal_idx=entry['goal_idx'],
                        candidate_x=float(entry['candidate_vectors'][c,0]),candidate_y=float(entry['candidate_vectors'][c,1]),
                        candidate_source='recorded_short_future' if c<15 else 'far_goal_reference',
                        head1_rank=int(rank1[c]),head2_rank=int(rank2[c]),rho_saturated=bool(info['log_rho'][c]>=np.log(100)),
                        rho_before_finite=bool(np.isfinite(info['rho_before'][c])),rho_before=float(info['rho_before'][c]) if np.isfinite(info['rho_before'][c]) else None)
                    row.update({k:float(v[c]) for k,v in info.items() if v.ndim==1 and k!='rho_before'})
                    for label in ('actor','teacher'):
                        row.update({f'{label}_{d}':float(info[label][c,d]) for d in range(2)})
                    candidates.append(row)
                actual={k:float(v) for k,v in sync(losses(agent,entry['batch'])).items()};signals.append(dict(snapshot_id=entry['snapshot_id'],kind=kind,variant=variant,checkpoint_step=step,state_idx=entry['state_idx'],goal_idx=entry['goal_idx'],waypoint_idx=entry['waypoint_idx'],**actual))
                if entry['snapshot_id'] in trigger:
                    batch,meta=gradient_batches.get(entry['snapshot_id'],(entry['batch'],dict(state_idx=entry['state_idx'],goal_idx=entry['goal_idx'],waypoint_idx=entry['waypoint_idx'],tuple_scope='registered real data tuple')))
                    gr={k:float(v) for k,v in sync(gradients(agent,batch)).items()};grad_rows.append(dict(snapshot_id=entry['snapshot_id'],kind=kind,variant=variant,checkpoint_step=step,**meta,**gr))
            print('E3_QUERY',kind,variant,step,flush=True)
    csv_rows(ROOT/'results'/f'E3_{variant}_CANDIDATES.csv',candidates);csv_rows(ROOT/'results'/f'E3_{variant}_SIGNALS.csv',signals)
    if grad_rows:csv_rows(ROOT/'results'/f'E3_{variant}_GRADIENTS.csv',grad_rows)
    write(ROOT/'receipts'/f'E3_{variant}_exit.json',dict(status='PARTIAL_MISSING_CHECKPOINTS' if missing else 'COMPLETED',missing_steps=missing,candidates=len(candidates),real_tuple_signals=len(signals),gradient_trigger_snapshots=sorted(trigger),gradient_rows=len(grad_rows),seconds=time.time()-start,checkpoints=identities,training_updates=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','A','B']);a=p.parse_args();prepare() if a.mode=='prepare' else model_queries(a.mode)
