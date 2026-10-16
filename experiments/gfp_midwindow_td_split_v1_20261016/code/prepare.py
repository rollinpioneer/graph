"""Prepare a new isolated experiment from the archived paired study; no training."""
import pathlib, shutil, json, hashlib, difflib
ROOT=pathlib.Path(__file__).resolve().parents[1]
OLD=pathlib.Path('/home/xushijie3/work/gfp_midwindow_td_freeze_v1_20261016')
for name in ['common.py','bfc.py','frozen.py','train.py','evaluate.py','manager.py','archive.py']:
    shutil.copy2(OLD/'code'/name,ROOT/'code'/name)
def replace(name,old,new):
    p=ROOT/'code'/name;s=p.read_text();assert old in s,(name,old);p.write_text(s.replace(old,new))
replace('common.py',"arm in ('L0', 'L1')","arm in ('L0', 'L1', 'L1a', 'L1b')")
replace('common.py',"    if arm == 'L1':\n", "    if arm == 'L1a':\n        ag = ag.replace(tau_pi=0.)\n    elif arm in ('L1', 'L1b'):\n")
replace('common.py',"tau_pi=0., epsilon=0., frozen_vabc", "tau_pi=0. if arm == 'L1' else 0.00005, epsilon=0., frozen_vabc")
replace('train.py',"def main(arm):\n", "def main(arm):\n    assert arm in ('L1a','L1b'), 'Only two new arms authorized'\n")
replace('train.py',"assert source_hash == expected['sha256']", "assert source_hash == expected['sha256'] == 'ffba11e943f7e192134b86093f4b2dc49efbe239cf2d32e3e884406358de1cca'")
replace('train.py',"    write(run/'zero_update_admission.json',dict(passed=True,td_exact=True,gradient_max_abs=max(diffs),optimizer_updates=0))", """    l1,_,_=make_agent(3,'L1')
    assert np.array_equal(np.asarray(l1.td_parts(batch,key)[0]),np.asarray(y0))
    loss_joint,grad_joint=jax.value_and_grad(lambda p:l1.total_loss(batch,p,key)[0])(l1.network.params)
    assert np.isclose(loss_joint,loss0,rtol=3e-5,atol=3e-5)
    assert all(np.allclose(a,b,rtol=3e-5,atol=3e-5) for a,b in zip(jax.tree_util.tree_leaves(grad_joint),jax.tree_util.tree_leaves(grad0)))
    write(run/'zero_update_admission.json',dict(passed=True,td_exact=True,gradient_max_abs=max(diffs),original_L0_L1_checked=True,optimizer_updates=0))
    del l1,grad_joint""")
replace('train.py',"protocol_sha256=sha(ROOT/'protocol.json'), config=cfg, initial_state_sha256=digest(ag)","protocol_sha256=sha(ROOT/'protocol.json'), config=cfg, intervention=json.loads((ROOT/'protocol.json').read_text())['arms'][arm], initial_state_sha256=digest(ag)")
replace('train.py',"        assert arm!='L1' or (digest(ag.reference)==fixed_ref and digest(ag.frozen_vabc)==fixed_flow)","        assert arm!='L1a' or digest(ag.reference)==fixed_ref\n        assert arm!='L1b' or digest(ag.frozen_vabc)==fixed_flow")
replace('train.py',"digest(ag.frozen_vabc) if arm=='L1' else None","digest(ag.frozen_vabc) if arm=='L1b' else None")
replace('train.py',"            assert (digest(ag.reference)==fixed_ref) if arm=='L1' else (digest(ag.reference)!=fixed_ref)","            assert (digest(ag.reference)==fixed_ref) if arm=='L1a' else (digest(ag.reference)!=fixed_ref)\n            assert arm!='L1b' or digest(ag.frozen_vabc)==fixed_flow")
replace('train.py',"td_sources_frozen=arm=='L1'","td_reference_frozen=arm=='L1a',td_vabc_frozen=arm=='L1b',unfrozen_reference_changed=arm=='L1b'")
replace('train.py',"        assert digest(restored)==digest(ag)","""        assert digest(restored)==digest(ag)
        oldmeta=json.loads((pathlib.Path('/home/xushijie3/work/gfp_midwindow_td_freeze_v1_20261016')/f'runs/s3_L0/checkpoint_{it}.json').read_text())
        assert np.asarray(ag.rng).tolist()==oldmeta['agent_rng'], 'RNG consumption differs from historical control'""")
replace('train.py',"import os, sys, time, pickle, random, resource, json","import os, sys, time, pickle, random, resource, json, pathlib")
replace('evaluate.py',"import os,sys,json,time,pickle,hashlib,multiprocessing as mp,random","import os,sys,json,time,pickle,hashlib,multiprocessing as mp,random,resource")
replace('evaluate.py',"def main(seed,arm,it,repair=False):\n", "def main(seed,arm,it,repair=False):\n    process_started=time.time()\n    assert arm in ('L1a','L1b') and it in (25000,50000) and not repair\n")
replace('evaluate.py',"        for proc in procs:proc.join(30)","""        for proc in procs:proc.join(30)
        me=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
        write(out/'cost.json',dict(process_wall_seconds=time.time()-process_started,
          cpu_seconds_self=me.ru_utime+me.ru_stime,cpu_seconds_workers=children.ru_utime+children.ru_stime,
          max_rss_self_kib=me.ru_maxrss,max_rss_child_kib=children.ru_maxrss,
          workers_stopped=all(not p.is_alive() for p in procs)))""")
replace('manager.py',"[('L0','4'),('L1','7')]","[('L1a','4'),('L1b','7')]")
replace('manager.py',"write(ROOT/f'runs/s3_{arm}/eval_{it}/exit.json',dict(exit_code=code,pid=e['proc'].pid,time=time.time()))","write(ROOT/f'runs/s3_{arm}/eval_{it}/exit.json',dict(exit_code=code,pid=e['proc'].pid,time=time.time(),dispatch_time=e['started']))")
replace('manager.py',"            e['proc']=subprocess.Popen", "            e['started']=time.time()\n            e['proc']=subprocess.Popen")
replace('manager.py',"        time.sleep(10)","""        active={p.pid for p,_,_ in jobs.values()}|{e['proc'].pid for e in evals.values() if e['proc'] is not None}
        raw=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
        own=[]
        for line in raw.stdout.splitlines():
            parts=line.split(',')
            if len(parts)==2 and parts[0].strip().isdigit() and int(parts[0]) in active:own.append(dict(pid=int(parts[0]),used_mib=int(parts[1])))
        with open(ROOT/'logs/own_gpu_memory.jsonl','a') as f:f.write(json.dumps(dict(time=time.time(),processes=own))+'\\n')
        time.sleep(10)""")
replace('archive.py',"codex/gfp-midwindow-td-freeze-v1-20261016","codex/gfp-midwindow-td-split-v1-20261016")
replace('archive.py',"ROOT/'EXPERIMENT_CARD.md'","ROOT/'EXP_CARD_L1_SPLIT_20261016.md'")
replace('archive.py',"list(d.glob('eval_*/exit.json'))","list(d.glob('eval_*/exit.json'))+list(d.glob('eval_*/cost.json'))")
for d in ['receipts','logs','runs','reports']:(ROOT/d).mkdir(exist_ok=True)
protocol=dict(authorized=True,authorization='User latest explicit Execute instruction; draft card alone is not authorization',
 source='/home/xushijie3/work/gfp_bfc_dev_screen_v1_20261016/runs/s3_R3/checkpoint_50000.pkl',source_sha256='ffba11e943f7e192134b86093f4b2dc49efbe239cf2d32e3e884406358de1cca',
 arms={'L1a':dict(tau_pi=0,epsilon=0,freeze_td_reference=True,freeze_td_vabc=False),'L1b':dict(tau_pi=.00005,epsilon=0,freeze_td_reference=False,freeze_td_vabc=True)},
 updates_per_arm=50000,total_updates_cap=100000,eval_updates=[25000,50000],eval_episodes=100,total_eval_episodes=400,
 max_external_steps=800000,max_control_calls=833332,max_physics_substeps=4166660,probe_every=5000,online_updates=0,
 anomaly_gap_increment=15,anomaly_gradient_ratio=2,anomaly_consecutive=2,behavior_success_max_percent=5,behavior_final_start_min_percent=90,
 strong_benefit_success_min_percent=25,strong_benefit_final_start_max_percent=30,strong_benefit_requires_no_joint_anomaly=True,no_auto_retry=True)
(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
diff=''
for name in ['common.py','bfc.py','frozen.py','train.py','evaluate.py','manager.py']:
 diff+=''.join(difflib.unified_diff((OLD/'code'/name).read_text().splitlines(True),(ROOT/'code'/name).read_text().splitlines(True),fromfile='historical/'+name,tofile='split/'+name))
(ROOT/'reports/code_diff.patch').write_text(diff)
ident=dict(previous_commit='d25e9ceff3e7ad4ef15d32b355e8026173e19585',checkpoint_sha256=hashlib.sha256(pathlib.Path(protocol['source']).read_bytes()).hexdigest(),
 reused_unchanged={n:hashlib.sha256((ROOT/'code'/n).read_bytes()).hexdigest() for n in ['bfc.py','frozen.py']})
assert ident['checkpoint_sha256']==protocol['source_sha256']
for name in ident['reused_unchanged']:assert (ROOT/'code'/name).read_bytes()==(OLD/'code'/name).read_bytes()
(ROOT/'receipts/prelaunch.json').write_text(json.dumps(ident,indent=2))
print('PREPARED_NO_UPDATES')
