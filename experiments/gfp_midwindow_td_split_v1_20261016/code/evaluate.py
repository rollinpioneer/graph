import os,sys,json,time,pickle,hashlib,multiprocessing as mp,random,resource
from pathlib import Path

def worker(conn):
    import numpy as np,ogbench
    from gymnasium.envs.mujoco import MujocoEnv
    original=MujocoEnv.do_simulation
    counts=[0,0]
    def counted(self,ctrl,n_frames):
        counts[0]+=1;counts[1]+=int(n_frames)
        return original(self,ctrl,n_frames)
    MujocoEnv.do_simulation=counted
    env=ogbench.make_env_and_datasets('humanoidmaze-medium-navigate-singletask-task1-v0',env_only=True)
    conn.send(('ready',counts.copy()))
    while True:
        cmd,arg=conn.recv()
        if cmd=='close':break
        if cmd=='reset':
            np.random.seed(arg);random.seed(arg)
            env.action_space.seed(arg)
            ob,info=env.reset(seed=arg)
            conn.send((ob,counts.copy()))
        else:
            ob,r,t,tr,info=env.step(arg)
            conn.send((ob,float(r),bool(t),bool(tr),float(info.get('success',0)),counts.copy()))
    env.close();conn.close()

def main(seed,arm,it,repair=False):
    process_started=time.time()
    assert arm in ('L1a','L1b') and it in (25000,50000) and not repair
    # Only this parent imports JAX; spawned simulators do not initialize CUDA.
    import numpy as np,jax,flax
    from common import ROOT,make_agent,write,sha
    run=ROOT/'runs'/f's{seed}_{arm}';out=run/(f'eval_{it}_repair1' if repair else f'eval_{it}');out.mkdir(exist_ok=False)
    checkpoint_path=run/f'checkpoint_{it}.pkl';before=sha(checkpoint_path)
    ag,cfg,_=make_agent(seed,arm)
    with open(checkpoint_path,'rb') as f:payload=pickle.load(f)
    ag=flax.serialization.from_state_dict(ag,payload['agent'])
    ctx=mp.get_context('spawn');pairs=[ctx.Pipe() for _ in range(10)]
    procs=[ctx.Process(target=worker,args=(c,)) for _,c in pairs]
    for proc in procs:proc.start()
    pipes=[p for p,_ in pairs]
    counts=[p.recv()[1] for p in pipes];episodes=[];t0=time.time();external=0
    try:
        for rnd in range(10):
            seeds=[4242000+rnd*10+k for k in range(10)]
            for p,s in zip(pipes,seeds):p.send(('reset',s))
            replies=[p.recv() for p in pipes];obs=np.stack([x[0] for x in replies]);counts=[x[1] for x in replies]
            hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in obs]
            active=np.ones(10,bool);returns=np.zeros(10);paths=[[x[:2].tolist()] for x in obs]
            # Common action noise schedule, aligned by global local episode time, independent of termination.
            key=jax.random.fold_in(jax.random.PRNGKey(4242),rnd)
            for t in range(1,2001):
                key,sk=jax.random.split(key)
                actions=np.clip(np.asarray(ag.sample_actions(obs,seed=sk)),-1,1)
                indices=np.where(active)[0]
                for k in indices:pipes[k].send(('step',actions[k]))
                for k in indices:
                    ob,r,term,trunc,succ,cnt=pipes[k].recv();obs[k]=ob;counts[k]=cnt;returns[k]+=r;external+=1
                    if t%20==0 or term or trunc:paths[k].append(ob[:2].tolist())
                    if term or trunc:
                        ij=np.floor((ob[:2]+2)/4).astype(int)+1
                        inside=tuple(ij) in ((1,1),(2,1),(1,2),(2,2))
                        episodes.append(dict(episode=rnd*10+int(k),reset_seed=seeds[k],initial_obs_sha256=hashes[k],
                          policy_seed=4242,policy_round=rnd,length=t,return_=float(returns[k]),success=succ,
                          final_xy=ob[:2].tolist(),final_in_start=inside,path=paths[k]))
                        active[k]=False
                assert sum(c[0] for c in counts)<=208333,'physical control call budget'
                if t%100==0:write(out/'progress.json',dict(completed=len(episodes),external_steps=external,control_calls=sum(c[0] for c in counts),seconds=time.time()-t0))
                if not active.any():break
            assert not active.any(),'episode exceeded 2000'
        write(out/'episodes_unverified.json',dict(episodes=episodes,control_calls=sum(c[0] for c in counts),external_steps=external,checkpoint_sha256=before))
        assert len(episodes)==100 and sha(checkpoint_path)==before
        assert sum(c[0] for c in counts)>=external,'unaccounted simulation calls'
        write(out/'results.json',dict(seed=seed,arm=arm,updates=it,checkpoint_sha256=before,optimizer_updates=0,
          success_percent=100*float(np.mean([e['success'] for e in episodes])),
          start_stay_percent=100*float(np.mean([e['final_in_start'] for e in episodes])),
          return_mean=float(np.mean([e['return_'] for e in episodes])),episodes=episodes,
          external_steps=external,control_calls=sum(c[0] for c in counts),physics_substeps=sum(c[1] for c in counts),
          seconds=time.time()-t0,checkpoint_unchanged=True))
    finally:
        for p in pipes:p.send(('close',None))
        for proc in procs:proc.join(30)
        me=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
        write(out/'cost.json',dict(process_wall_seconds=time.time()-process_started,
          cpu_seconds_self=me.ru_utime+me.ru_stime,cpu_seconds_workers=children.ru_utime+children.ru_stime,
          max_rss_self_kib=me.ru_maxrss,max_rss_child_kib=children.ru_maxrss,
          workers_stopped=all(not p.is_alive() for p in procs)))

if __name__=='__main__':main(int(sys.argv[1]),sys.argv[2],int(sys.argv[3]),repair='--repair' in sys.argv)
