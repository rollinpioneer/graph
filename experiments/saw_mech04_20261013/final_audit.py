"""Verify immutable input assets, complete learners, own budgets, and sealed scene identity."""
from preflight import *
def main():
    checks=[];source=json.loads((ROOT/'receipts/TRAINING_SOURCE.json').read_text())
    for f in source['files']:assert sha(ROOT/f['path'])==f['sha256']
    checks.append('C startup source bytes unchanged')
    for old in (OLD,EV,PREV):
        manifest=json.loads((old/'LIGHT_MANIFEST.json').read_text())
        for f in manifest['files']:assert sha(old/f['relative'])==f['sha256']
    official=OLD/'vendor/saw';assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=official,text=True).strip()=='3535a16c901e5de1233ac5a586401b3ddfa5e30d';assert not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=official,text=True).strip()
    for item in json.loads((ROOT/'results/PREFLIGHT.json').read_text())['readonly_checkpoints']:assert sha(item['path'])==item['sha256']
    for item in json.loads((ROOT/'results/DATA_IDENTITIES.json').read_text()):assert sha(item['path'])==item['sha256']
    checks.append('all original light manifests and readonly checkpoint files unchanged')
    cps=[]
    for kind,v,seed in [(k,v,s) for k,vs in [('stitch',['A','B']),('navigate',['A'])] for v in vs for s in (1,2)]:
        label=f'C_{kind}_{v}_s{seed}';statusp=ROOT/'results'/f'{label}_training_status.json';resourcep=ROOT/'receipts'/f'{label}_resource.json'
        if not statusp.exists():cps.append(dict(label=label,status='NOT_EXECUTED',actual_updates=None,valid_complete_1M=False));continue
        status=json.loads(statusp.read_text());resource=json.loads(resourcep.read_text());items=json.loads((ROOT/'results'/f'{label}_checkpoints.json').read_text());milestones=[]
        for c in items:
            assert sha(c['path'])==c['sha256']
            with open(c['path'],'rb') as f:s=pickle.load(f)
            assert s['step']==c['step'];assert int(s['agent']['network']['step'])==c['step']+1;assert int(s['agent']['network']['opt_state']['0']['count'])==c['step']
            for x in jax.tree_util.tree_leaves(s['agent']):
                a=np.asarray(x)
                if np.issubdtype(a.dtype,np.number):assert np.isfinite(a).all()
            assert all(k in s for k in ('numpy_rng','python_rng','env_rng','action_rng'));milestones.append(c['step'])
        valid=status['status']=='COMPLETE_1M' and status['step']==1000000 and resource['exit_code']==0 and all(s in milestones for s in (0,100000,300000,500000,1000000))
        cps.append(dict(label=label,status=status['status'],actual_updates=status['step'],registered_stages=milestones,resource_exit=resource['exit_code'],valid_complete_1M=valid))
    for seed in (1,2):
        p=ROOT/'results'/f'GPU_UPDATE0_s{seed}.json';assert p.exists();x=json.loads(p.read_text());assert x['agent_optimizer_rng_equal'] is True,x
        assert all(sha(path)==h for path,h in zip(x['paths'],x['sha256']))
    checks.append('actual GPU seed1/2 A/B update0 learner identity; all available CP counters, optimizer counts and states finite')
    for c in json.loads((ROOT/'results/SCENE_IDENTITIES.json').read_text())['scenes']:assert sha(c['path'])==c['sha256']
    resources=[json.loads(p.read_text()) for p in (ROOT/'receipts').glob('*_resource.json')];limits={'C-S':39600,'C-N':19800,'R':14400,'C-EVAL':14400,'T':9000,'T-EVAL':14400};used={g:sum(r['residency_seconds'] for r in resources if r['group']==g) for g in limits}
    assert all(used[g]<=limits[g] for g in limits);assert max(r['peak_aggregate_mib'] for r in resources)<=12288
    assert not list((ROOT/'receipts').glob('*_active.json')),'OWN_JOBS_STILL_ACTIVE'
    for phase in ['C','R','T']:
        p=ROOT/'receipts'/f'{phase}_EPISODE_BUDGET.json'
        if p.exists():x=json.loads(p.read_text());assert x['used']<=x['limit']
    rc=json.loads((ROOT/'receipts/R_CONTROL_BUDGET.json').read_text());assert rc['used']<=rc['limit']
    checks.append('separate own residency, aggregate VRAM, scene hashes, episode/control step caps, no live guarded jobs')
    for p in list((ROOT/'results').glob('Ceval_*_episodes.csv'))+list((ROOT/'results').glob('Reval_*_episodes.csv'))+list((ROOT/'results').glob('Teval_*_episodes.csv')):
        rows=list(csv.DictReader(open(p)));assert len({r['scene_id'] for r in rows})==len(rows);assert max(float(r['initial_restore_error']) for r in rows)<=1e-10;assert all(int(r['steps'])<=1000 for r in rows)
    if (ROOT/'receipts/T_AUTHORIZATION.json').exists() and json.loads((ROOT/'receipts/T_AUTHORIZATION.json').read_text())['approved']:
        tsrc=json.loads((ROOT/'receipts/T_TRAINING_SOURCE.json').read_text())
        for name,h in tsrc['files'].items():assert sha(ROOT/name)==h
        identity=json.loads((ROOT/'receipts/T_ACTUAL_UPDATE0.json').read_text());assert identity['status']=='PASS'
        for arm in range(4):
            p=ROOT/'results'/f'T{arm}_checkpoints.json'
            if p.exists():
                for c in json.loads(p.read_text()):
                    assert sha(c['path'])==c['sha256'];assert c['frozen_parameters_and_reference_optimizer_equal'];assert c['actor_optimizer_count']==500000+c['step'];assert c['reference_native_step']==500001
        checks.append('T actual same initial actor/full optimizer/RNG; frozen modules/reference states unchanged; inherited actor count and sealed stream identity')
    write(ROOT/'results/FINAL_AUDIT.json',dict(status='PASS',checks=checks,C_learners=cps,C_all_six_complete=all(c['valid_complete_1M'] for c in cps),used_residency_seconds=used,hard_limits_seconds=limits,R_control_steps=rc,source=source,SSH_cleanup='pending final publication; actual client exit recorded in local receipt after closing'))
    print('FINAL_AUDIT_PASS',all(c['valid_complete_1M'] for c in cps))
if __name__=='__main__':main()
