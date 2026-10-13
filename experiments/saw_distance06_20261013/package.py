"""Allowlisted light publication and immutable raw-path index; never archive weights/data."""
import pathlib,json,hashlib,time,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parent; W=ROOT.parent.parent
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
    assert json.loads((ROOT/'AUDIT06.json').read_text())['status']=='PASS'
    assert json.loads((ROOT/'results/SIGNAL_FORWARD_AUDIT.json').read_text())['status']=='PASS'
    required=['EXPERIMENT_CARD.md','FROZEN_MANIFEST.json','GEOMETRY_SUPPORT.csv','SCENE_GOAL_BANK.csv','BEHAVIOR_ROLLOUTS.csv','DISTANCE_CONTRASTS.csv','REGION_TIE_CONTROLS.csv','SIGNAL_TUPLES.csv','SIGNAL_BY_DISTANCE.csv','CONDITIONAL_ACTION_AUDIT.csv','TRAJECTORY_MORPHOLOGY.csv','REPRODUCTION_REPORT.md','HYPOTHESIS_VERDICT.md','NEXT_DECISION.md','RESOURCE_AND_EXIT_AUDIT.json']
    assert all((ROOT/x).exists() for x in required)
    raw=[]
    for directory in ('runs','logs'):
        for p in sorted((ROOT/directory).rglob('*')):
            if p.is_file():raw.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    write(ROOT/'RAW_ARTIFACT_INDEX.json',dict(epoch=time.time(),files=raw,immutable_prior_model_files_not_copied=True))
    roots=set(required+['CARD_SUPPLIED.md','GEOMETRY_PREFLIGHT.json','geometry_precheck.py','core.py','prepare.py','execute.py','signal_prepare.py','signal06.py','signal_analysis.py','behavior_analysis.py','audit06.py','finalize.py','package.py','run_phase.py','export_signal_figures_v2.py','export_behavior_figures_v2.py','ADMISSION_AUDIT.json','AUDIT06.json','RAW_ARTIFACT_INDEX.json','SAW06.md','README.txt','MANIFEST.json','README.md','.gitignore'])
    selected=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if len(rel.parts)==1 and rel.name in roots:selected.append(p)
        elif rel.parts[0] in ('results','receipts') and p.suffix in ('.csv','.json','.txt'):selected.append(p)
        elif rel.parts[0]=='figures' and p.suffix=='.png':selected.append(p)
    banned={'.npz','.npy','.pkl','.tar','.zip','.pt','.pth','.mp4','.pem'}
    assert all(p.suffix not in banned for p in selected)
    manifest=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in selected]
    write(ROOT/'LIGHT_MANIFEST.json',dict(files=manifest,excludes_self_and_allowlist=True))
    paths=[str(p.relative_to(W)) for p in selected]+[str((ROOT/'LIGHT_MANIFEST.json').relative_to(W)),str((ROOT/'STAGING_ALLOWLIST.json').relative_to(W))]
    write(ROOT/'STAGING_ALLOWLIST.json',dict(paths=paths))
    subprocess.run(['git','-C',str(W),'add','--',*paths],check=True)
    actual=subprocess.check_output(['git','-C',str(W),'diff','--cached','--name-only'],text=True).splitlines()
    assert all(p in paths for p in actual),('UNRELATED_STAGE',set(actual)-set(paths))
    subprocess.run(['git','-C',str(W),'-c','core.whitespace=-blank-at-eol','diff','--cached','--check'],check=True)
    print('PACKAGE_READY',len(selected)+2,'RAW',len(raw),'STAGED',len(actual),'LIGHT_BYTES',sum(p.stat().st_size for p in selected),flush=True)
if __name__=='__main__':main()
