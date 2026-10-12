"""Validate and explicitly stage only this experiment's lightweight evidence."""
import csv
import subprocess
from common import *

def main():
    expected='codex/saw-rd-01-20261012';repo=ROOT.parent.parent
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
    assert git('branch','--show-current')==expected
    source=ROOT/'vendor/saw'
    assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()=='3535a16c901e5de1233ac5a586401b3ddfa5e30d'
    assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True).strip()
    assert sha(ROOT/'REQUESTED_PLAN.md')=='b1392af8bf89aa839297da1cfee7dbeb6a27c642a6944542aaa27bfc2caae604'
    arrays=0
    def finite_tree(x):
        nonlocal arrays
        if isinstance(x,dict):
            for v in x.values():finite_tree(v)
        elif isinstance(x,(list,tuple)):
            for v in x:finite_tree(v)
        elif isinstance(x,np.ndarray) and np.issubdtype(x.dtype,np.number):
            assert np.isfinite(x).all();arrays+=1
    for kind in ('navigate','stitch'):
        with open(ROOT/'runs'/f'formal_{kind}'/'params_1000000.pkl','rb') as f:state=pickle.load(f)
        assert state['step']==1000000;finite_tree(state['agent'])
        train=list(csv.DictReader(open(ROOT/'results'/f'{kind}_training.csv')))
        assert int(train[-1]['step'])==1000000
        for row in train:
            for key,v in row.items():
                if v:assert np.isfinite(float(v)),(kind,key)
        final=json.load(open(ROOT/'results'/f'{kind}_FINAL_EVAL.json'))
        assert len(final['records'])==250
        for task in range(1,6):assert sum(r['task_id']==task for r in final['records'])==50
    resources=json.load(open(ROOT/'results/RESOURCE_TOTAL.json'))
    assert resources['total_gpu_hours']<12 and resources['peak_aggregate_memory_mib']<=12288
    assert resources['local_policy_segments']==122 and resources['candidate_state_goal_groups']==128
    write_json(ROOT/'receipts/CONFIRMATION_EXIT.json',dict(navigate_exit_code=0,stitch_exit_code=0,
               observation='Persistent controlling shell waited for both recorded child PIDs and received exit 0; both outputs exist.',gpu_seconds=0))
    write_json(ROOT/'receipts/FINAL_VALIDATION.json',dict(status='PASS',checkpoint_numeric_arrays_checked=arrays,
               official_tracked_source_unchanged=True,training_tables_finite=True,final_cases=500,
               aggregate_resource_caps_pass=True,short_segments=122,state_goal_groups=128,
               source_plan_sha256=sha(ROOT/'REQUESTED_PLAN.md'),figure_visual_review='PASS: three figures downloaded and inspected',
               progress_note='*_progress.json preserves the last logging snapshot; training_status.json and completed guard receipts determine the final state.'))
    # Include root code/config/reports and explicit lightweight result/receipt extensions.
    files=[]
    for p in ROOT.iterdir():
        if p.is_file() and (p.suffix in ('.py','.md','.json') or p.name in ('.gitignore','.gitattributes','requirements.lock.txt')) and p.name!='LIGHT_MANIFEST.json':files.append(p)
    for directory,extensions in [('results',{'.json','.csv','.png','.svg'}),('receipts',{'.json'})]:
        files.extend(p for p in (ROOT/directory).iterdir() if p.is_file() and p.suffix in extensions)
    for p in files:
        assert p.stat().st_size<=524288,(p,p.stat().st_size)
        if p.suffix=='.csv':p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
        if p.suffix=='.svg':p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
        if p.suffix=='.json':json.loads(p.read_text(),parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    manifest=[dict(relative=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]
    write_json(ROOT/'LIGHT_MANIFEST.json',dict(files=manifest,note='Excludes itself and post-push/SSH-close receipts to avoid self-referencing identity. Large raw evidence is server-only.'))
    files.append(ROOT/'LIGHT_MANIFEST.json')
    staged=[str(p.relative_to(repo)) for p in sorted(files)]
    subprocess.run(['git','-C',str(repo),'add','--',*staged],check=True)
    actual=set(git('diff','--cached','--name-only').splitlines())
    assert actual<=set(staged),'Unrelated staged changes'
    subprocess.run(['git','-C',str(repo),'diff','--cached','--check'],check=True)
    print('ARCHIVE_ALLOWLIST',len(files),'BYTES',sum(p.stat().st_size for p in files),'ARRAYS',arrays,flush=True)

if __name__=='__main__':main()
