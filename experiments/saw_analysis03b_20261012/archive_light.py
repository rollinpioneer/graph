"""Verify actual execution and stage an explicit own-experiment lightweight list."""
import hashlib,json,pickle,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;W=ROOT.parents[1]
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def leaves(x):
    if isinstance(x,dict):
        for v in x.values():yield from leaves(v)
    elif isinstance(x,(tuple,list)):
        for v in x:yield from leaves(v)
    elif isinstance(x,np.ndarray):yield x
def main():
    assert not list((ROOT/'receipts').glob('*_active.json')),'GPU_JOB_STILL_ACTIVE'
    budget=json.loads((ROOT/'receipts/E1_budget.json').read_text());assert budget['used']<=1600
    assert json.loads((ROOT/'receipts/E2_budget.json').read_text())['used']<=10000
    resources=[json.loads(p.read_text()) for p in (ROOT/'receipts').glob('*_resource.json')]
    assert len(resources)==4 and sum(x['gpu_residency_upper_seconds'] for x in resources)<=19800
    assert all(x['peak_aggregate_mib']<=12288 for x in resources)
    assert all(x['gpu_residency_upper_seconds']<=9000 for x in resources if x['label'].startswith('formal_'))
    assert all(x['gpu_residency_upper_seconds']<=600 for x in resources if x['label'].startswith('pilot_'))
    assert sum(json.loads(p.read_text())['actual_updates'] for p in (ROOT/'results').glob('*_qualification.json'))==2000
    scientific=[];checkpoints=[]
    for kind in ('navigate','stitch'):
        original=json.loads((Path('/home/xushijie3/work/graph_saw_rd01_20261012/experiments/saw_rd01_20261012')/'runs'/f'formal_{kind}'/'config.json').read_text())['config']
        actual=json.loads((ROOT/'runs'/f'formal_{kind}'/'config.json').read_text())['config']
        diff={k:[original[k],actual[k]] for k in original if original[k]!=actual[k]}
        assert diff=={'actor_p_trajgoal':[1.0,0.5],'actor_p_randomgoal':[0.0,0.5]}
        status=json.loads((ROOT/'results'/f'{kind}_training_status.json').read_text());assert status['step']<=1000000
        for cp in status['checkpoints']:
            assert sha(cp['path'])==cp['sha256']
            with open(cp['path'],'rb') as f:s=pickle.load(f)
            assert s['step']==cp['step'] and set(('agent','numpy_rng','python_rng','env_rng','action_rng','qpos','qvel')).issubset(s)
            assert 'rng' in s['agent'] and set(('params','opt_state','step')).issubset(s['agent']['network'])
            arr=list(leaves(s['agent']));finite=all(np.isfinite(v).all() for v in arr)
            if status['status']=='SINGLE_SEED_B_COMPARISON':assert finite and status['step']==1000000
            optimizer_step=int(s['agent']['network']['step']);matched=optimizer_step==cp['step']+1
            if status['status']=='SINGLE_SEED_B_COMPARISON':assert matched
            checkpoints.append(dict(**cp,kind=kind,finite_arrays=finite,array_count=len(arr),serialized_optimizer_step=optimizer_step,
                actual_updates_from_native_counter=optimizer_step-1,counter_matches_label=matched))
        scientific.append(dict(kind=kind,config_diff=diff,status=status['status'],updates=status['step']))
    source='388ded0a23258ed43e12093825b17fe80b7510fc';sourcefiles=[]
    for name in ('common.py','train.py','resource_guard.py'):
        blob=subprocess.check_output(['git','show',f'{source}:experiments/saw_analysis03b_20261012/{name}'],cwd=W)
        expected=hashlib.sha256(blob).hexdigest();assert sha(ROOT/name)==expected
        sourcefiles.append(dict(path=name,sha256=expected,training_start_commit=source))
    for p in list((ROOT/'results').glob('*.json'))+list((ROOT/'receipts').glob('*.json')):
        json.loads(p.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(f'Invalid JSON numeric:{p}:{x}')))
    index=[]
    for directory in ('runs','logs'):
        for p in sorted((ROOT/directory).rglob('*')):
            if p.is_file():index.append(dict(path=str(p),relative=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
    write(ROOT/'EVIDENCE_INDEX.json',dict(server_root=str(ROOT),files=index,checkpoints=checkpoints,input_references=json.loads((ROOT/'results/READ_ONLY_ASSETS.json').read_text()),training_source=sourcefiles,
        note='Large files remain server-only; weights are not published; raw pre-correction replay gate measurements remain immutable and are interpreted by E3_REPLAY_GATE_AUDIT.json'))
    write(ROOT/'results/ARCHIVE_VALIDATION.json',dict(status='PASS',scientific=scientific,training_sources=sourcefiles,E1_episodes=budget['used'],E2_control_steps=json.loads((ROOT/'receipts/E2_budget.json').read_text())['used'],
        total_GPU_seconds=sum(x['gpu_residency_upper_seconds'] for x in resources),raw_evidence_files=len(index),large_weights_staged=False))
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()
    write(ROOT/'BRANCH_INFO.json',dict(branch='codex/saw-analysis-03b-20261012',url='https://github.com/rollinpioneer/graph/tree/codex/saw-analysis-03b-20261012',
        parent='b0dd744d323b924d02de93f5c487c4afc9565171',pre_archive_commit=head,training_start_commit=source,
        final_commit='verified post-push receipt outside committed manifest'))
    allow=[]
    for p in ROOT.iterdir():
        if p.is_file() and (p.suffix in ('.py','.md') or p.name in ('.gitignore','.gitattributes','requirements.lock.txt','EVIDENCE_INDEX.json','BRANCH_INFO.json')):allow.append(p)
    for directory,suffixes in [('results',('.json','.csv')),('receipts',('.json',)),('figures',('.png','.pdf'))]:
        allow += [p for p in (ROOT/directory).glob('*') if p.is_file() and p.suffix in suffixes]
    allow=sorted(set(allow));assert all('runs/' not in str(p.relative_to(ROOT)) and 'logs/' not in str(p.relative_to(ROOT)) for p in allow)
    write(ROOT/'LIGHT_MANIFEST.json',dict(files=[dict(relative=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in allow],excluded='runs,logs,weights,datasets,SSH secrets,temporary locks,caches',created_unix=time.time()))
    allow.append(ROOT/'LIGHT_MANIFEST.json')
    subprocess.run(['git','add','--']+[str(p.relative_to(W)) for p in allow],cwd=W,check=True)
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=W,text=True).splitlines()
    assert all(p.startswith('experiments/saw_analysis03b_20261012/') for p in staged)
    subprocess.run(['git','diff','--cached','--check'],cwd=W,check=True)
    print('ARCHIVE_VALIDATION_PASS',len(allow),flush=True)
if __name__=='__main__':main()
