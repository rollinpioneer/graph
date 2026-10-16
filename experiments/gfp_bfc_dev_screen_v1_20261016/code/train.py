import os,sys,time,json,pickle,resource,random
import numpy as np
import jax,jax.numpy as jnp,flax
from common import ROOT,FORMAL,write,sha,make_agent

def digest(tree):
    import hashlib
    h=hashlib.sha256()
    for x in jax.tree_util.tree_leaves(tree):
        a=np.asarray(x);h.update(str((a.shape,str(a.dtype))).encode());h.update(a.tobytes())
    return h.hexdigest()

def main(seed,arm):
    run=ROOT/'runs'/f's{seed}_{arm}';run.mkdir(parents=True,exist_ok=False)
    t0=time.time();ag,cfg,source=make_agent(seed,arm);start=int(ag.network.step)-1
    np.random.seed(20261016);random.seed(20261016)
    source_sha=sha(source)
    protocol_sha=sha(ROOT/'protocol.json')
    write(run/'identity.json',dict(seed=seed,arm=arm,source=str(source),source_sha256=source_sha,
          protocol_sha256=protocol_sha,config=cfg,pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES']))
    data={p.stem:jnp.asarray(np.load(p,mmap_mode='r')) for p in (ROOT/'data').glob('*.npy')}
    size=int(json.loads((ROOT/'receipts/assets.json').read_text())['dataset_size'])
    probe=np.load(FORMAL/'analysis_p0p1/out/probe/states.npz')
    obs=jnp.asarray(probe['obs']);z=jnp.asarray(probe['z'])
    reg=np.zeros(len(obs),bool)
    for i,j in [(1,1),(2,1),(1,2),(2,2)]:reg|=(probe['cell_i']==i)&(probe['cell_j']==j)
    @jax.jit
    def assess(a, fixed):
        act=jnp.clip(a.network.select('actor')(obs,z),-1,1)
        q=a.network.select('critic')(obs,actions=act).mean(0)
        qt=a.critic_eval(obs,act)
        grad=jax.grad(lambda aa:a.network.select('critic')(obs,actions=aa).mean(0).sum())(fixed)
        return act,q,qt,grad
    a0=jnp.clip(ag.network.select('actor')(obs,z),-1,1);g0=None;rows=[];info={}
    def snapshot(it):
        nonlocal g0
        a,q,qt,g=map(np.asarray,assess(ag,a0))
        assert all(np.isfinite(v).all() for v in (a,q,qt,g))
        if g0 is None:g0=g.copy()
        cosine=(g*g0).sum(1)/(np.linalg.norm(g,axis=1)*np.linalg.norm(g0,axis=1)+1e-12)
        row=dict(updates=it,step=start+it,seconds=time.time()-t0,V_online_start=float(q[reg].mean()),
          V_online_all=float(q.mean()),V_target_start=float(qt[reg].mean()),
          dqda_norm_start=float(np.linalg.norm(g,axis=1)[reg].mean()),
          dqda_cos_start=float(cosine[reg].mean()),action_shift_start=float(np.linalg.norm(a-np.asarray(a0),axis=1)[reg].mean()),
          rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        row.update({k:float(v) for k,v in info.items()});rows.append(row)
        write(run/'probes.json',rows);write(run/'progress.json',dict(row,phase='training'))
    def save(it):
        state=flax.serialization.to_state_dict(ag)
        payload=dict(agent=state,updates=it,seed=seed,arm=arm,source_sha256=source_sha,protocol_sha256=protocol_sha,
                     numpy_rng=np.random.get_state(),python_rng=random.getstate())
        path=run/f'checkpoint_{it}.pkl'
        with open(str(path)+'.tmp','wb') as f:pickle.dump(payload,f)
        os.replace(str(path)+'.tmp',path)
        with open(path,'rb') as f:p=pickle.load(f)
        restored=flax.serialization.from_state_dict(ag,p['agent'])
        assert digest(restored)==digest(ag) and int(restored.network.step)==start+it+1 and int(restored.network.opt_state[0].count)==start+it
        ag_rng=np.asarray(ag.rng).tolist()
        write(run/f'checkpoint_{it}.json',dict(sha256=sha(path),full_restore='PASS',step=start+it,updates=it,
              reference_included=arm!='R0',state_sha256=digest(ag),agent_rng=ag_rng))
        return restored
    snapshot(0)
    for it in range(4,100001,4):
        before=ag
        ag,info=ag.multi_sample_and_update(4,data,size,256)
        # Check every fused result, including all intermediate BFC losses.
        vals=np.asarray([float(x) for x in info.values()]);assert np.isfinite(vals).all()
        assert bool(info.get('fused_all_finite',True))
        if it==4:
            assert int(ag.network.step)==start+5
            assert digest(ag.network.params)!=digest(before.network.params)
            ag=save(it)
            write(run/'admission.json',dict(passed=True,actual_updates=4,fused_steps=4,full_restore=True,
                  reference_changed=(digest(ag.reference)!=digest(before.reference)) if arm!='R0' else None))
        if it%5000==0:snapshot(it)
        if it in (50000,100000):ag=save(it)
    assert sha(source)==source_sha
    write(run/'complete.json',dict(updates=100000,final_step=int(ag.network.step)-1,seconds=time.time()-t0,
      cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_unchanged=True))

if __name__=='__main__':main(int(sys.argv[1]),sys.argv[2])
