"""Native F/R tuple pairs, original loss gradients and bounded physical probes."""
import argparse
from frozen import *
from scipy.spatial import cKDTree
import jax.numpy as jnp

class Trace(HGCDataset):
    def get_observations(self,i):self.indices.append(np.asarray(i).copy());return super().get_observations(i)
    def paired(self,i,seed):
        self.indices=[]
        with separate_numpy(seed):b=super().sample(len(i),i,evaluation=True);rng=np.random.get_state()
        return b,self.indices,rng

@jax.jit
def query(agent,b):
    s=b['observations'];g=b['high_actor_goals'];w=b['high_actor_targets'];n=b['next_observations'];a=b['actions']
    v1,v2=agent.network.select('value')(s,g);n1,n2=agent.network.select('value')(n,g);w1,w2=agent.network.select('value')(w,g);t1,t2=agent.network.select('target_value')(s,g)
    actor=agent.network.select('actor')(s,g);teacher=agent.network.select('low_actor')(s,w);am=actor.mode();tm=teacher.mode()
    awrlog=agent.config['awr_alpha']*(n1+n2-v1-v2)/2;wplog=agent.config['kl_alpha']*(w1+w2-v1-v2)/2
    wa=jnp.minimum(jnp.exp(awrlog),100);ww=jnp.minimum(jnp.exp(wplog),100);kld=jnp.sum((am-tm)**2,axis=-1)
    return dict(v1=v1,v2=v2,target_v1=t1,target_v2=t2,log_awr=awrlog,log_wp=wplog,AWR_weight=wa,WP_weight=ww,
        actor_data_mse=jnp.sum((am-a)**2,axis=-1),teacher_data_mse=jnp.sum((tm-a)**2,axis=-1),actor_teacher_mse=kld,
        AWR_contribution=-wa*actor.log_prob(a),WP_contribution=ww*kld,raw_actor=am,raw_teacher=tm)

def dot(a,b):return sum(jnp.sum(x*y) for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)))
@jax.jit
def gradient_pair(agent,f,r):
    grad=lambda b,fn:jax.grad(lambda p:fn(b,p)[0])(agent.network.params)
    fa=grad(f,agent.actor_loss);fw=grad(f,agent.waypoint_loss);ra=grad(r,agent.actor_loss);rw=grad(r,agent.waypoint_loss)
    ft=jax.tree_util.tree_map(lambda x,y:x+y,fa,fw);rt=jax.tree_util.tree_map(lambda x,y:x+y,ra,rw)
    norm=lambda a:jnp.sqrt(dot(a,a));cos=lambda a,b:dot(a,b)/(norm(a)*norm(b)+1e-30)
    return dict(F_AWR_norm=norm(fa),F_WP_norm=norm(fw),R_AWR_norm=norm(ra),R_WP_norm=norm(rw),F_AWR_WP_cosine=cos(fa,fw),R_AWR_WP_cosine=cos(ra,rw),F_R_total_cosine=cos(ft,rt),F_total_norm=norm(ft),R_total_norm=norm(rt))

def physical_probe(env,scene,agent,policy_goal,target,kind,raw_actions=None):
    s,err=restore_gate(env,scene);states=[s.copy()];actions=[];flags=[]
    count=25 if kind in ('teacher','privileged','data_chain') else 50
    if raw_actions is not None:count=min(count,len(raw_actions))
    initial_target=float(np.linalg.norm(s-target));reached=initial_target<=1.0
    for k in range(count):
        if kind=='privileged':a=np.clip((target-s)/.2,-1,1)
        elif kind=='data_chain':a=raw_actions[k]
        elif kind=='teacher':a=np.clip(np.asarray(agent.network.select('low_actor')(s,target).mode()),-1,1)
        else:a=np.asarray(mode(agent,s,policy_goal))
        control_count();s,r,t,u,info=env.step(a);states.append(s.copy());actions.append(a);flags.append((float(r),bool(t),bool(u),float(info['success'])));reached|=np.linalg.norm(s-target)<=1.0
        if t or u or (kind in ('teacher','privileged') and reached):break
    xy=np.asarray(states);row=dict(policy=kind,steps=len(actions),target_initial_distance=initial_target,target_reached=bool(reached),original_far_success=bool(flags[-1][3]) if flags else False,
        initial_restore_error=err,final_target_distance=float(np.linalg.norm(s-target)),net_displacement=float(np.linalg.norm(s-xy[0])),path_length=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),measurement_status='MEASURED' if flags else 'ZERO_STEPS')
    return row,dict(**row,scene=scene,policy_goal=policy_goal,target=target,states=xy,actions=np.asarray(actions),flags=flags)

def main(kind):
    start=time.time();roles=[(v,s) for v in ('A','B') for s in (300000,500000,1000000)] if kind=='navigate' else [('A',1000000),('B',1000000)]
    agents={role:load_agent(kind,*role)[0] for role in roles}
    with (ROOT/'runs'/f'R_SNAPSHOTS_{kind}.pkl').open('rb') as f:snaps=pickle.load(f)
    env,data,val,cfg=load(f'pointmaze-large-{kind}-v0','A');rawdata=np.load(OLD/'vendor/data'/f'pointmaze-large-{kind}-v0.npz');obs=rawdata['observations'];vel=rawdata['qvel'];acts=rawdata['actions'];valid=data.dataset.valid_idxs
    assert np.array_equal(data.dataset['observations'],obs[:data.dataset.size]),'RAW_COMPACT_INDEX_MAPPING'
    tree=cKDTree(obs[valid]);fconf=dict(cfg,actor_p_trajgoal=1.,actor_p_randomgoal=0.);rconf=dict(cfg,actor_p_trajgoal=0.,actor_p_randomgoal=1.);F=Trace(data.dataset,fconf);R=Trace(data.dataset,rconf)
    support=[];summaries=[];grads=[];physical=[];rawphys=[];paired_ids=[]
    for k,snap in enumerate(snaps):
        sc=snap['scene'];_,neighbors=tree.query(sc['qpos'],k=100);idx=valid[np.atleast_1d(neighbors)];pe=np.max(np.abs(obs[idx]-sc['qpos']),axis=-1);ve=np.max(np.abs(vel[idx]-sc['qvel']),axis=-1);eligible=np.flatnonzero((pe<=.15)&(ve<=.5))
        row=dict(case_id=snap['case_id'],kind=kind,anchor=snap['anchor'],primary_anchor=snap['primary_anchor'],replay_status=snap['status'],neighbors_eligible=len(eligible),status='PASS' if len(eligible) and snap['status']=='PASS' else 'NO_APPROXIMATE_DATA_SUPPORT' if not len(eligible) else 'REPLAY_GATE_FAILED',state_idx=None,qpos_error=None,qvel_error=None)
        if row['status']!='PASS':support.append(row);continue
        choice=int(eligible[0]);i=int(idx[choice]);end=int(data.terminal_locs[np.searchsorted(data.terminal_locs,i)]);row.update(state_idx=i,qpos_error=float(pe[choice]),qvel_error=float(ve[choice]));support.append(row)
        batchidx=np.repeat(i,1024);seed=2030131+(kind=='stitch')*10000+k
        f,fi,fr=F.paired(batchidx,seed);r,ri,rr=R.paired(batchidx,seed)
        assert equal(fr,rr)
        for name in ('observations','actions','next_observations','value_goals','low_actor_goals','rewards','masks'):assert np.array_equal(f[name],r[name]),name
        assert np.array_equal(fi[3],np.minimum(i+25,fi[2]));assert np.array_equal(ri[3],np.repeat(min(i+25,end),1024))
        prefix=f"{snap['case_id']}_t{snap['anchor']}";path=ROOT/'runs'/f'{prefix}_paired_tuples.npz'
        np.savez_compressed(path,**{f'F_{name}':x for name,x in f.items()},**{f'R_{name}':x for name,x in r.items()},state_idx=batchidx,F_goal_idx=fi[2],F_waypoint_idx=fi[3],R_goal_idx=ri[2],R_waypoint_idx=ri[3])
        paired_ids.append(dict(case_id=snap['case_id'],anchor=snap['anchor'],path=str(path),sha256=sha(path),count_per_type=1024,state_idx=i,stream=seed,shared_base_and_rng=True))
        for role,agent in agents.items():
            for label,batch in (('F',f),('R',r)):
                result={name:np.asarray(x) for name,x in sync(query(agent,batch)).items()}
                assert all(np.isfinite(x).all() for x in result.values()),'NONFINITE_OFFLINE_QUERY'
                np.savez_compressed(ROOT/'runs'/f'{prefix}_{role[0]}_{role[1]}_{label}_signals.npz',**result)
                s=dict(case_id=snap['case_id'],kind=kind,anchor=snap['anchor'],primary_anchor=snap['primary_anchor'],variant=role[0],checkpoint_step=role[1],goal_type=label,n=1024,state_idx=i)
                for name,x in result.items():
                    if x.ndim==1:
                        for tag,z in zip(('mean','p10','p50','p90','p99'),(x.mean(),*np.quantile(x,[.1,.5,.9,.99]))):s[name+'_'+tag]=float(z)
                s.update(AWR_saturated=float(np.mean(result['AWR_weight']>=100)),WP_saturated=float(np.mean(result['WP_weight']>=100)),AWR_below_point1=float(np.mean(result['AWR_weight']<.1)),WP_below_point1=float(np.mean(result['WP_weight']<.1)))
                original_terminal=np.flatnonzero(rawdata['terminals']);goal_indices=fi[2] if label=='F' else ri[2]
                s['cross_original_trajectory_fraction']=float(np.mean(np.searchsorted(original_terminal,goal_indices)!=np.searchsorted(original_terminal,i)))
                summaries.append(s)
            if snap['primary_anchor']:
                fb={name:x[:16] for name,x in f.items()};rb={name:x[:16] for name,x in r.items()};g={name:float(x) for name,x in sync(gradient_pair(agent,fb,rb)).items()};assert all(np.isfinite(x) for x in g.values())
                grads.append(dict(case_id=snap['case_id'],kind=kind,anchor=snap['anchor'],variant=role[0],checkpoint_step=role[1],tuple_samples_per_type=16,state_idx=i,**g))
        if snap['primary_anchor']:
            for label,batch,indices in (('F',f,fi),('R',r,ri)):
                goal=batch['high_actor_goals'][0];target=batch['high_actor_targets'][0];wp=int(indices[3][0])
                probedata=acts[i:wp]
                for policy,v in [('privileged',None),('data_chain',None),('teacher','A'),('teacher','B'),('actor','A'),('actor','B')]:
                    agent=agents.get((v,1000000));p,tr=physical_probe(env,sc,agent,goal,target,policy,probedata if policy=='data_chain' else None)
                    p.update(case_id=snap['case_id'],kind=kind,anchor=snap['anchor'],goal_type=label,variant=v,target_idx=wp,qpos_match_error=float(pe[choice]),qvel_match_error=float(ve[choice]),scope='full source state; approximate raw-data join; near success is not far task success')
                    physical.append(p);rawphys.append(dict(**p,trace=tr))
        write(ROOT/'receipts'/f'R_supervision_{kind}_progress.json',dict(anchors=k+1,total_anchors=len(snaps),supported=len(paired_ids),seconds=time.time()-start))
        print('R_SUPERVISION',kind,k+1,'DONE',flush=True)
    env.close();rawdata.close();csv_rows(ROOT/'results'/f'R_SUPPORT_{kind}.csv',support)
    if summaries:csv_rows(ROOT/'results'/f'R_SUPERVISION_AUDIT_{kind}.csv',summaries)
    if grads:csv_rows(ROOT/'results'/f'R_GRADIENTS_{kind}.csv',grads)
    if physical:csv_rows(ROOT/'results'/f'R_TUPLE_PHYSICAL_{kind}.csv',physical)
    write(ROOT/'results'/f'R_TUPLE_IDENTITIES_{kind}.json',paired_ids);dump(ROOT/'runs'/f'R_TUPLE_PHYSICAL_TRACES_{kind}.pkl',rawphys)
    write(ROOT/'receipts'/f'R_supervision_{kind}_exit.json',dict(status='COMPLETED',selected_anchors=len(snaps),supported_anchors=len(paired_ids),summary_rows=len(summaries),gradient_rows=len(grads),physical_probes=len(physical),seconds=time.time()-start,training_updates=0))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['navigate','stitch'],required=True);main(p.parse_args().kind)
