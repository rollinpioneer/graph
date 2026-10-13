"""Verify immutable input assets, complete learners, own budgets, and sealed scene identity."""
from preflight import *
def main():
    checks=[];source=json.loads((ROOT/'receipts/TRAINING_SOURCE.json').read_text())
    for f in source['files']:assert sha(ROOT/f['path'])==f['sha256']
    checks.append('C startup source bytes unchanged')
    original=json.loads((ROOT/'results/PREFLIGHT.json').read_text())
    for old in (OLD,EV,PREV):
        recorded=next(x for x in original['original_manifests'] if x['root']==str(old));assert sha(old/'LIGHT_MANIFEST.json')==recorded['manifest_sha256']
        manifest=json.loads((old/'LIGHT_MANIFEST.json').read_text())
        for f in manifest['files']:assert sha(old/f['relative'])==f['sha256']
    official=OLD/'vendor/saw';assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=official,text=True).strip()=='3535a16c901e5de1233ac5a586401b3ddfa5e30d';assert not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=official,text=True).strip()
    for item in json.loads((ROOT/'results/PREFLIGHT.json').read_text())['readonly_checkpoints']:assert sha(item['path'])==item['sha256']
    for item in json.loads((ROOT/'results/DATA_IDENTITIES.json').read_text()):assert sha(item['path'])==item['sha256']
    checks.append('all original light manifests and readonly checkpoint files unchanged')
    cps=[];initial_by_role={}
    for kind,v,seed in [(k,v,s) for k,vs in [('stitch',['A','B']),('navigate',['A'])] for v in vs for s in (1,2)]:
        label=f'C_{kind}_{v}_s{seed}';statusp=ROOT/'results'/f'{label}_training_status.json';resourcep=ROOT/'receipts'/f'{label}_resource.json'
        if not statusp.exists():cps.append(dict(label=label,status='NOT_EXECUTED',actual_updates=None,valid_complete_1M=False));continue
        status=json.loads(statusp.read_text());resource=json.loads(resourcep.read_text());items=json.loads((ROOT/'results'/f'{label}_checkpoints.json').read_text());milestones=[]
        for c in items:
            assert sha(c['path'])==c['sha256']
            with open(c['path'],'rb') as f:s=pickle.load(f)
            assert s['step']==c['step'];assert int(s['agent']['network']['step'])==c['step']+1;assert int(s['agent']['network']['opt_state']['0']['count'])==c['step']
            if c['step']==0:
                expected=jax.random.split(jax.random.PRNGKey(seed),2)[0];assert np.array_equal(s['agent']['rng'],np.asarray(expected))
                initial_by_role[kind,v,seed]=dict(params=digest(s['agent']['network']['params']),numpy_rng=digest(s['numpy_rng']))
            for x in jax.tree_util.tree_leaves(s['agent']):
                a=np.asarray(x)
                if np.issubdtype(a.dtype,np.number):assert np.isfinite(a).all()
            assert all(k in s for k in ('numpy_rng','python_rng','env_rng','action_rng'));milestones.append(c['step'])
        valid=status['status']=='COMPLETE_1M' and status['step']==1000000 and resource['exit_code']==0 and all(s in milestones for s in (0,100000,300000,500000,1000000))
        cps.append(dict(label=label,status=status['status'],actual_updates=status['step'],registered_stages=milestones,resource_exit=resource['exit_code'],valid_complete_1M=valid))
    for seed in (1,2):
        p=ROOT/'results'/f'GPU_UPDATE0_s{seed}.json';assert p.exists();x=json.loads(p.read_text());assert x['agent_optimizer_rng_equal'] is True,x
        assert all(sha(path)==h for path,h in zip(x['paths'],x['sha256']))
    for kind,v in [('stitch','A'),('stitch','B'),('navigate','A')]:
        if (kind,v,1) in initial_by_role and (kind,v,2) in initial_by_role:
            assert initial_by_role[kind,v,1]['params']!=initial_by_role[kind,v,2]['params'];assert initial_by_role[kind,v,1]['numpy_rng']!=initial_by_role[kind,v,2]['numpy_rng']
    checks.append('actual GPU seed1/2 A/B update0 learner identity; all available CP counters, optimizer counts and states finite')
    for c in json.loads((ROOT/'results/SCENE_IDENTITIES.json').read_text())['scenes']:assert sha(c['path'])==c['sha256']
    resources=[json.loads(p.read_text()) for p in (ROOT/'receipts').glob('*_resource.json')];limits={'C-S':39600,'C-N':19800,'R':14400,'C-EVAL':14400,'T':9000,'T-EVAL':14400};used={g:sum(r['residency_seconds'] for r in resources if r['group']==g) for g in limits}
    live=[json.loads(p.read_text()) for p in (ROOT/'receipts').glob('*_active.json')]
    assert all(x['pid']==os.getpid() and x['group']=='R' for x in live),'OTHER_OWN_JOBS_STILL_ACTIVE'
    for x in live:used[x['group']]+=time.time()-x['start']
    assert all(used[g]<=limits[g] for g in limits);assert max(r['peak_aggregate_mib'] for r in resources)<=12288
    for phase in ['C','R','T']:
        p=ROOT/'receipts'/f'{phase}_EPISODE_BUDGET.json'
        if p.exists():x=json.loads(p.read_text());assert x['used']<=x['limit']
    rc=json.loads((ROOT/'receipts/R_CONTROL_BUDGET.json').read_text());assert rc['used']<=rc['limit']
    checks.append('separate own residency, aggregate VRAM, scene hashes, episode/control step caps, no live guarded jobs')
    for p in list((ROOT/'results').glob('Ceval_*_episodes.csv'))+list((ROOT/'results').glob('Reval_*_episodes.csv'))+list((ROOT/'results').glob('Teval_*_episodes.csv')):
        rows=list(csv.DictReader(open(p)));assert len({r['scene_id'] for r in rows})==len(rows);assert max(float(r['initial_restore_error']) for r in rows)<=1e-10;assert all(int(r['steps'])<=1000 for r in rows)
    tcps=[]
    if (ROOT/'receipts/T_AUTHORIZATION.json').exists() and json.loads((ROOT/'receipts/T_AUTHORIZATION.json').read_text())['approved']:
        tsrc=json.loads((ROOT/'receipts/T_TRAINING_SOURCE.json').read_text())
        for name,h in tsrc['files'].items():assert sha(ROOT/name)==h
        identity=json.loads((ROOT/'receipts/T_ACTUAL_UPDATE0.json').read_text());assert identity['status']=='PASS'
        scene_identity=json.loads((ROOT/'results/T_SCENE_IDENTITIES.json').read_text());assert sha(scene_identity['path'])==scene_identity['sha256'];assert scene_identity['count']==500
        with open(scene_identity['path'],'rb') as f:tscenes=pickle.load(f)
        tscene_hash={x['scene_id']:digest(x) for x in tscenes};assert len(tscene_hash)==500
        stream_path=ROOT/'results/T_SHARED_STREAM.json';stream=json.loads(stream_path.read_text());stream_sha=sha(stream_path)
        old_stream=next(x for x in json.loads((ROOT/'LIGHT_MANIFEST.json').read_text())['files'] if x['relative']=='results/T_SHARED_STREAM.json');assert stream_sha==old_stream['sha256']
        assert sha(stream['source']['path'])==stream['source']['sha256']
        for chunk in stream['chunks']:assert sha(chunk['path'])==chunk['sha256']
        with open(stream['source']['path'],'rb') as f:tref=pickle.load(f)
        actor_keys=set(json.loads((ROOT/'results/T_CPU_ADMISSION.json').read_text())['trainable_keys']);actual_initial_hashes=[]
        for arm in range(4):
            p=ROOT/'results'/f'T{arm}_checkpoints.json'
            if p.exists():
                steps=[]
                for c in json.loads(p.read_text()):
                    assert sha(c['path'])==c['sha256'];assert c['frozen_parameters_and_reference_optimizer_equal'];assert c['actor_optimizer_count']==500000+c['step'];assert c['reference_native_step']==500001
                    with open(c['path'],'rb') as f:s=pickle.load(f)
                    assert s['student_step']==s['stream_cursor']==c['step'];assert s['stream_manifest_sha256']==stream_sha
                    assert int(s['agent']['network']['step'])==s['reference_native_step']==500001
                    assert int(s['actor_opt_state']['0']['count'])==500000+c['step']
                    assert digest(s['agent']['network']['opt_state'])==digest(tref['agent']['network']['opt_state'])
                    assert set(s['agent']['network']['params'])==set(tref['agent']['network']['params'])
                    for key in s['agent']['network']['params']:
                        if key not in actor_keys:assert digest(s['agent']['network']['params'][key])==digest(tref['agent']['network']['params'][key])
                    for x in jax.tree_util.tree_leaves((s['agent'],s['actor_opt_state'])):
                        a=np.asarray(x)
                        if np.issubdtype(a.dtype,np.number):assert np.isfinite(a).all()
                    if c['step']==0:actual_initial_hashes.append(hashlib.sha256(pickle.dumps((s['agent'],s['actor_opt_state'],s['actor_rng'],s['numpy_rng'],s['python_rng'],s['stream_cursor'],s['stream_manifest_sha256']),4)).hexdigest())
                    steps.append(c['step'])
                status=json.loads((ROOT/'results'/f'T{arm}_training_status.json').read_text());resource=json.loads((ROOT/'receipts'/f'T{arm}_resource.json').read_text())
                tcps.append(dict(arm=arm,status=status['status'],student_updates=status['student_updates'],registered_stages=steps,resource_exit=resource['exit_code'],valid_complete_100k=status['status']=='COMPLETE_100K' and status['student_updates']==100000 and resource['exit_code']==0 and all(x in steps for x in (0,25000,50000,100000))))
        assert actual_initial_hashes==identity['serialized_agent_actor_optimizer_rng_cursor_sha256'];assert len(set(actual_initial_hashes))==1
        for p in (ROOT/'results').glob('Teval_*_episodes.csv'):
            rows=list(csv.DictReader(open(p)));assert all(x['scene_sha256']==tscene_hash[x['scene_id']] for x in rows)
        checks.append('T actual same initial actor/full optimizer/RNG; frozen modules/reference states unchanged; inherited actor count and sealed stream identity')
    write(ROOT/'results/FINAL_AUDIT.json',dict(status='PASS',checks=checks,C_learners=cps,C_all_six_complete=all(c['valid_complete_1M'] for c in cps),C_initial_seed_identity=[dict(kind=k,variant=v,seed=s,**x) for (k,v,s),x in initial_by_role.items()],T_learners=tcps,T_all_four_complete=len(tcps)==4 and all(x['valid_complete_100k'] for x in tcps),used_residency_seconds=used,hard_limits_seconds=limits,R_control_steps=rc,source=source,SSH_cleanup='pending final publication; actual client exit recorded in local receipt after closing'))
    print('FINAL_AUDIT_PASS',all(c['valid_complete_1M'] for c in cps))
if __name__=='__main__':main()
