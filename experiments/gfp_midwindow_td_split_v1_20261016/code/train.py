import os, sys, time, pickle, random, resource, json, pathlib
import numpy as np
import jax, jax.numpy as jnp, flax
from common import ROOT, SOURCE, FORMAL, make_agent, load_base, digest, sha, write

def main(arm):
    assert arm in ('L1a','L1b'), 'Only two new arms authorized'
    run = ROOT/'runs'/f's3_{arm}'; run.mkdir(parents=True, exist_ok=False)
    t0 = time.time(); ag, cfg, source = make_agent(3, arm)
    source_hash = sha(source)
    expected = json.loads((SOURCE/'runs/s3_R3/checkpoint_50000.json').read_text())
    assert source_hash == expected['sha256'] == 'ffba11e943f7e192134b86093f4b2dc49efbe239cf2d32e3e884406358de1cca'
    l0, _, _ = make_agent(3, 'L0')
    assert digest(l0) == expected['state_sha256']
    assert digest(ag.network) == digest(l0.network) and digest(ag.rng) == digest(l0.rng)
    assert digest(ag.reference) == digest(l0.reference)
    with open(source, 'rb') as f: payload = pickle.load(f)
    np.random.set_state(payload['numpy_rng']); random.setstate(payload['python_rng'])
    write(run/'identity.json', dict(arm=arm, source=str(source), source_sha256=source_hash,
      protocol_sha256=sha(ROOT/'protocol.json'), config=cfg, intervention=json.loads((ROOT/'protocol.json').read_text())['arms'][arm], initial_state_sha256=digest(ag),
      initial_agent_rng=np.asarray(ag.rng).tolist(), pid=os.getpid(), gpu=os.environ['CUDA_VISIBLE_DEVICES']))
    data = {p.stem:jnp.asarray(np.load(p,mmap_mode='r')) for p in (SOURCE/'data').glob('*.npy')}
    size = json.loads((SOURCE/'receipts/assets.json').read_text())['dataset_size']
    batch = {k:v[:256] for k,v in data.items()}
    key = jax.random.PRNGKey(20261016)
    # Pure loss/gradient admission: no optimizer calls here.
    y0 = l0.td_parts(batch,key)[0]; y1 = ag.td_parts(batch,key)[0]
    assert np.array_equal(np.asarray(y0),np.asarray(y1))
    loss0, grad0 = jax.value_and_grad(lambda p:l0.total_loss(batch,p,key)[0])(l0.network.params)
    loss1, grad1 = jax.value_and_grad(lambda p:ag.total_loss(batch,p,key)[0])(ag.network.params)
    diffs = [float(jnp.max(jnp.abs(a-b))) for a,b in zip(jax.tree_util.tree_leaves(grad0),jax.tree_util.tree_leaves(grad1))]
    assert np.isclose(loss0,loss1,rtol=3e-5,atol=3e-5)
    assert all(np.allclose(a,b,rtol=3e-5,atol=3e-5) for a,b in zip(jax.tree_util.tree_leaves(grad0),jax.tree_util.tree_leaves(grad1)))
    l1,_,_=make_agent(3,'L1')
    assert np.array_equal(np.asarray(l1.td_parts(batch,key)[0]),np.asarray(y0))
    loss_joint,grad_joint=jax.value_and_grad(lambda p:l1.total_loss(batch,p,key)[0])(l1.network.params)
    assert np.isclose(loss_joint,loss0,rtol=3e-5,atol=3e-5)
    assert all(np.allclose(a,b,rtol=3e-5,atol=3e-5) for a,b in zip(jax.tree_util.tree_leaves(grad_joint),jax.tree_util.tree_leaves(grad0)))
    write(run/'zero_update_admission.json',dict(passed=True,td_exact=True,gradient_max_abs=max(diffs),original_L0_L1_checked=True,optimizer_updates=0))
    del l1,grad_joint
    del grad0,grad1,l0
    probe = np.load(FORMAL/'analysis_p0p1/out/probe/states.npz')
    obs,z = jnp.asarray(probe['obs']),jnp.asarray(probe['z'])
    reg = np.zeros(len(obs),bool)
    for i,j in ((1,1),(2,1),(1,2),(2,2)): reg |= (probe['cell_i']==i)&(probe['cell_j']==j)
    historical,_,_ = load_base(3)
    old_actions = jnp.clip(historical.network.select('actor')(obs,z),-1,1)
    mid_actions = jnp.clip(ag.network.select('actor')(obs,z),-1,1)
    @jax.jit
    def assess(a):
        actions = jnp.clip(a.network.select('actor')(obs,z),-1,1)
        q = a.network.select('critic')(obs,actions=actions).mean(0)
        qt = a.critic_eval(obs,actions)
        qfixed = a.network.select('critic')(obs,actions=old_actions).mean(0)
        fn = lambda act:a.network.select('critic')(obs,actions=act).mean(0).sum()
        return actions,q,qt,qfixed,jax.grad(fn)(old_actions),jax.grad(fn)(mid_actions)
    old = list(map(np.asarray,assess(historical)))
    mid = list(map(np.asarray,assess(ag)))
    hist_gap = float(old[1][reg].mean()-old[1].mean())
    hist_norm = float(np.linalg.norm(old[4],axis=1)[reg].mean())
    fixed_ref,fixed_flow = digest(ag.reference),digest(ag.network.params['modules_vabc'])
    write(run/'anchors.json',dict(historical_step=300000,mid_step=350000,historical_gap=hist_gap,
      historical_gradient_norm=hist_norm,probe_sha256=sha(FORMAL/'analysis_p0p1/out/probe/states.npz'),
      historical_action_sha256=digest(old_actions),mid_action_sha256=digest(mid_actions)))
    del historical
    rows=[]; info={}
    def snapshot(it):
        a,q,qt,qf,g,gm = list(map(np.asarray,assess(ag)))
        assert all(np.isfinite(v).all() for v in (a,q,qt,qf,g,gm))
        norms=np.linalg.norm(g,axis=1)
        cosine=(g*old[4]).sum(1)/(norms*np.linalg.norm(old[4],axis=1)+1e-12)
        cm=(gm*mid[5]).sum(1)/(np.linalg.norm(gm,axis=1)*np.linalg.norm(mid[5],axis=1)+1e-12)
        row=dict(updates=it,step=350000+it,seconds=time.time()-t0,V_online_start=float(q[reg].mean()),
          V_online_all=float(q.mean()),V_online_outside=float(q[~reg].mean()),V_target_start=float(qt[reg].mean()),
          V_target_outside=float(qt[~reg].mean()),V_fixed_start=float(qf[reg].mean()),V_fixed_outside=float(qf[~reg].mean()),
          dqda_norm_start=float(norms[reg].mean()),dqda_norm_outside=float(norms[~reg].mean()),
          dqda_cos_start=float(cosine[reg].mean()),mid_gradient_cos_start=float(cm[reg].mean()),
          mid_gradient_norm_start=float(np.linalg.norm(gm,axis=1)[reg].mean()),
          action_shift_start=float(np.linalg.norm(a-np.asarray(old_actions),axis=1)[reg].mean()),
          historical_gap_increase=float(q[reg].mean()-q.mean()-hist_gap),historical_gradient_ratio=float(norms[reg].mean()/hist_norm))
        row['joint_anomaly']=row['historical_gap_increase']>=15 and row['historical_gradient_ratio']>=2
        row.update({k:float(v) for k,v in info.items()})
        assert arm!='L1a' or digest(ag.reference)==fixed_ref
        assert arm!='L1b' or digest(ag.frozen_vabc)==fixed_flow
        rows.append(row);write(run/'probes.json',rows);write(run/'progress.json',dict(row,phase='training'))
    def save(it):
        state=flax.serialization.to_state_dict(ag)
        p=dict(agent=state,updates=it,seed=3,arm=arm,numpy_rng=np.random.get_state(),python_rng=random.getstate(),
               source_sha256=source_hash,protocol_sha256=sha(ROOT/'protocol.json'))
        path=run/f'checkpoint_{it}.pkl'
        with open(str(path)+'.tmp','wb') as f:pickle.dump(p,f)
        os.replace(str(path)+'.tmp',path)
        with open(path,'rb') as f:back=pickle.load(f)
        restored=flax.serialization.from_state_dict(ag,back['agent'])
        assert digest(restored)==digest(ag)
        oldmeta=json.loads((pathlib.Path('/home/xushijie3/work/gfp_midwindow_td_freeze_v1_20261016')/f'runs/s3_L0/checkpoint_{it}.json').read_text())
        assert np.asarray(ag.rng).tolist()==oldmeta['agent_rng'], 'RNG consumption differs from historical control'
        assert int(restored.network.step)==350001+it and int(restored.network.opt_state[0].count)==350000+it
        assert pickle.dumps(back['numpy_rng'])==pickle.dumps(np.random.get_state()) and back['python_rng']==random.getstate()
        write(run/f'checkpoint_{it}.json',dict(sha256=sha(path),state_sha256=digest(ag),full_restore='PASS',step=350000+it,
          updates=it,agent_rng=np.asarray(ag.rng).tolist(),reference_sha256=digest(ag.reference),
          frozen_vabc_sha256=digest(ag.frozen_vabc) if arm=='L1b' else None))
        return restored
    snapshot(0);ag=save(0)
    for it in range(4,50001,4):
        if it==4:before=ag
        ag,info=ag.multi_sample_and_update(4,data,size,256)
        assert bool(info['fused_all_finite']) and all(np.isfinite(float(x)) for x in info.values())
        if it==4:
            assert all(digest(ag.network.params[k])!=digest(before.network.params[k]) for k in ('modules_actor','modules_vabc','modules_critic','modules_target_critic'))
            assert (digest(ag.reference)==fixed_ref) if arm=='L1a' else (digest(ag.reference)!=fixed_ref)
            assert arm!='L1b' or digest(ag.frozen_vabc)==fixed_flow
            ag=save(4)
            write(run/'admission.json',dict(passed=True,actual_updates=4,full_restore=True,online_modules_changed=True,td_reference_frozen=arm=='L1a',td_vabc_frozen=arm=='L1b',unfrozen_reference_changed=arm=='L1b'))
            del before
        if it%5000==0:snapshot(it);ag=save(it)
    assert sha(source)==source_hash
    write(run/'complete.json',dict(updates=50000,final_step=400000,seconds=time.time()-t0,
      cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_unchanged=True))

if __name__=='__main__':main(sys.argv[1])
