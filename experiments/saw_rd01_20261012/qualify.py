import argparse
import copy
import importlib.metadata
import json
import os
import signal
import time
from pathlib import Path

from common import *


class TraceSampler(HGCDataset):
    def get_observations(self, idxs):
        self.observed.append(np.asarray(idxs).copy())
        return super().get_observations(idxs)

    def sample(self, batch_size, idxs=None, evaluation=False):
        self.observed=[]
        if idxs is None:
            idxs=self.dataset.get_random_idxs(batch_size)
        self.indices=np.asarray(idxs).copy()
        return super().sample(batch_size, idxs, evaluation)


def cpu_audit():
    result={}
    for kind in ('navigate','stitch'):
        name=f'pointmaze-large-{kind}-v0'
        env, data, val, config=load(name)
        u=env.unwrapped
        audit=dict(name=name, env_spec=env.spec.id, max_episode_steps=env.spec.max_episode_steps,
                   observation_shape=env.observation_space.shape, action_shape=env.action_space.shape,
                   action_low=env.action_space.low, action_high=env.action_space.high, action_semantics='qpos += 0.2*clipped_action; qvel zero then 5 MuJoCo steps',
                   goal_tolerance=u._goal_tol, normalization='none', maze_map=u.maze_map,
                   tasks=u.task_infos, config=config, goal_noise=u._noise, qpos_size=u.model.nq, qvel_size=u.model.nv,
                   training_reward='0 at identical sampled state index, otherwise -1', masks='0 at identical goal index, otherwise 1; not dataset terminals',
                   env_reward='1 success, 0 otherwise; terminate_at_goal True', files=[])
        for split in ('','-val'):
            path=ROOT/'vendor/data'/f'{name}{split}.npz'
            with np.load(path) as raw:
                terminals=np.flatnonzero(raw['terminals'])
                valid=np.flatnonzero(1-raw['terminals'])
                traj_id=np.searchsorted(terminals, np.arange(len(raw['observations'])))
                starts=np.r_[0,terminals[:-1]+1]
                step_id=np.arange(len(raw['observations']))-starts[traj_id]
                fields={k:dict(shape=raw[k].shape,dtype=str(raw[k].dtype),finite=bool(np.isfinite(raw[k]).all())) for k in raw.files}
                nextobs=raw['observations'][np.minimum(np.arange(len(traj_id))+1,len(traj_id)-1)]
                indexpath=ROOT/'runs'/f'{kind}{split}_raw_index.npz'
                np.savez_compressed(indexpath, trajectory_id=traj_id,step_id=step_id,valid=1-raw['terminals'],
                                    terminal=raw['terminals'],observation=raw['observations'],action=raw['actions'],
                                    next_observation=nextobs,next_observation_valid=1-raw['terminals'])
                audit['files'].append(dict(path=str(path),sha256=sha(path),fields=fields,trajectories=len(terminals),
                                           trajectory_length_min=int(np.diff(np.r_[-1,terminals]).min()),
                                           trajectory_length_max=int(np.diff(np.r_[-1,terminals]).max()),
                                           raw_index=dict(path=str(indexpath),sha256=sha(indexpath)),
                                           terminal_rows_have_no_valid_next_transition=True))
        tracer=TraceSampler(data.dataset,config)
        with separate_numpy(41001 if kind=='navigate' else 41002):
            batch=tracer.sample(4096)
            i=tracer.indices; vg,low,g,w=tracer.observed
            end=tracer.terminal_locs[np.searchsorted(tracer.terminal_locs,i)]
            checks=dict(valid_transition=bool((data.dataset['valids'][i]>0).all()),
                        low_exact_25_or_terminal=bool(np.array_equal(low,np.minimum(i+25,end))),
                        future_goal_same_official_segment=bool(((g>=i)&(g<=end)).all()),
                        waypoint_exact=bool(np.array_equal(w,np.minimum(i+25,g))),
                        reward_exact=bool(np.array_equal(batch['rewards'],(i==vg).astype(float)-1)),
                        masks_exact=bool(np.array_equal(batch['masks'],(i!=vg).astype(float))))
            assert all(checks.values()), checks
            path=ROOT/'results'/f'{kind}_sampler_indices.csv'
            np.savetxt(path,np.c_[np.arange(4096),i,vg,low,g,w],fmt='%d',delimiter=',',header='sample_id,state_idx,value_goal_idx,low_goal_idx,high_goal_idx,waypoint_idx',comments='')
            np.savez_compressed(ROOT/'runs'/f'{kind}_sampler_probe.npz',**batch,state_idx=i,value_idx=vg,low_idx=low,goal_idx=g,waypoint_idx=w)
            audit['sampler']=dict(checks=checks,samples=4096,current_value_goal_fraction=float(np.mean(i==vg)),
                                  value_goal_other_trajectory_fraction=float(np.mean(np.searchsorted(end if False else tracer.terminal_locs,vg)!=np.searchsorted(tracer.terminal_locs,i))),
                                  high_goal_future_fraction=float(np.mean((g>=i)&(g<=end))),
                                  compact_terminal_remapping='official loader marks both final raw row and predecessor terminal; valid rows exclude final raw row; sampler preserved exactly')
        # Logging must not affect the sampler's next batch or NumPy state.
        np.random.seed(0); state=np.random.get_state(); b1=data.sample(1024); next1=data.sample(1024)
        np.random.set_state(state); b2=data.sample(1024)
        json.dumps({k:float(np.asarray(v).mean()) for k,v in b2.items()})
        with separate_numpy(777): val.sample(1024)
        next2=data.sample(1024)
        audit['sampler_logging_rng_equal']=all(np.array_equal(next1[k],next2[k]) and np.array_equal(b1[k],b2[k]) for k in b1)
        assert audit['sampler_logging_rng_equal']
        # True environment threshold checks; use a free cell center and full state.
        env.action_space.seed(999); np.random.seed(999); env.reset(seed=999,options={'task_id':1})
        center=np.asarray(u.ij_to_xy(tuple(np.argwhere(u.maze_map==0)[0])))
        checks=[]
        for label,offset,expect in [('success',[0,0],1),('non_success',[1.5,0],0),('short_positive',[1.05,0],1)]:
            env.reset(seed=999,options={'task_id':1}); u.set_state(center+np.array(offset),np.zeros(u.model.nv)); u.set_goal(goal_xy=center)
            action=np.array([-1.,0.]) if label=='short_positive' else np.zeros(2)
            ob,r,term,trunc,info=env.step(action)
            checks.append(dict(label=label,success=info['success'],expected=expect,terminated=term,truncated=trunc,distance=float(np.linalg.norm(ob-center))))
            assert info['success']==expect
        audit['evaluator_checks']=checks
        result[kind]=audit
        env.close()
    versions={x:importlib.metadata.version(x) for x in ('ogbench','jax','jaxlib','flax','distrax','optax','numpy','mujoco','dm_control','gymnasium')}
    write_json(ROOT/'results/DATA_AUDIT.json',result)
    write_json(ROOT/'results/SAMPLER_AUDIT.json',{k:dict(sampler=v['sampler'],logging_rng_equal=v['sampler_logging_rng_equal']) for k,v in result.items()})
    write_json(ROOT/'results/dependency_versions.json',versions)
    print('CPU_AUDIT_PASS',versions,flush=True)


def pilot(kind, updates):
    name=f'pointmaze-large-{kind}-v0'
    signal.signal(signal.SIGUSR1,lambda signum,frame: (_ for _ in ()).throw(TimeoutError('resource guard')))
    env,data,val,config=load(name)
    env.action_space.seed(0); env.reset(seed=0, options={'task_id':1})
    agent=init_agent(data,config)
    sync(agent)
    t=time.monotonic(); first=data.sample(1024); sample_seconds=time.monotonic()-t
    t=time.monotonic(); agent,info=agent.update(first); sync((agent,info)); compile_seconds=time.monotonic()-t
    start_agent=agent
    # Identical fixed batches on a copied initial optimizer/parameters/RNG, log on/off.
    fixed=[data.sample(1024) for _ in range(5)]
    for batch in fixed: agent,info=agent.update(batch)
    sync(agent); logged=agent
    agent=start_agent
    for batch in fixed:
        agent,info=agent.update(batch)
        sync(info); json.dumps({k:float(v) for k,v in info.items()})
    sync(agent)
    differences=[float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) for x,y in zip(jax.tree_util.tree_leaves(agent),jax.tree_util.tree_leaves(logged))]
    assert max(differences)==0
    elapsed=[]; sampling=[]; metrics=[]
    for block in range(updates//100):
        t=time.monotonic(); sample_cost=0
        for _ in range(100):
            ts=time.monotonic(); batch=data.sample(1024); sample_cost+=time.monotonic()-ts
            agent,info=agent.update(batch)
        sync((agent,info)); duration=time.monotonic()-t
        elapsed.append(duration); sampling.append(sample_cost)
        metrics.append({k:float(v) for k,v in info.items()})
        print(kind,'PILOT',100*(block+1),round(duration/100,6),'s/update',flush=True)
    t=time.monotonic()
    with separate_numpy(50001):
        batch=val.sample(1024); loss,vi=agent.total_loss(batch); sync((loss,vi))
    val_seconds=time.monotonic()-t
    t=time.monotonic(); cp=checkpoint(agent,env,updates+6,ROOT/'runs'/f'pilot_{kind}'); checkpoint_seconds=time.monotonic()-t
    evaluation=evaluate(agent,name,61001 if kind=='navigate' else 61002,episodes=2,save_dir=ROOT/'runs'/f'pilot_{kind}')
    steady=float(np.median(np.asarray(elapsed[-5:]))/100)
    # Pilot untrained full horizon episodes; conservative measured extrapolation.
    eval_full=evaluation['seconds']*25
    formal_estimate=steady*1000000+compile_seconds+11*eval_full+4*checkpoint_seconds+200*val_seconds
    result=dict(kind=kind,updates=updates+11,batch_size=1024,config=config,devices=[str(d) for d in jax.devices()],
                compile_first_update_seconds=compile_seconds,steady_seconds_per_update=steady,
                synchronized_block_seconds=elapsed,sampling_seconds=sampling,
                validation_seconds=val_seconds,checkpoint_seconds=checkpoint_seconds,checkpoint=cp,
                small_eval_seconds=evaluation['seconds'],small_eval_episodes=10,small_eval_steps=sum(r['steps'] for r in evaluation['records']),
                full_250_episode_eval_estimate_seconds=eval_full,formal_estimate_seconds=formal_estimate,formal_estimate_gpu_hours=formal_estimate/3600,
                logging_update_max_abs_difference=max(differences),fresh_formal_init_required=True,final_metrics=metrics[-1])
    write_json(ROOT/'results'/f'{kind}_qualification.json',result)
    print('QUALIFICATION',kind,json.dumps({k:v for k,v in result.items() if k in ('steady_seconds_per_update','formal_estimate_gpu_hours','compile_first_update_seconds')}),flush=True)
    env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--mode',choices=['audit','pilot'],required=True)
    parser.add_argument('--kind',default='navigate'); parser.add_argument('--updates',type=int,default=1000)
    args=parser.parse_args()
    if args.mode=='audit': cpu_audit()
    else: pilot(args.kind,args.updates)
