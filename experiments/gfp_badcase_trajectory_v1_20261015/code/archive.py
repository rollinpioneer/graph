import pathlib,json,hashlib,shutil,subprocess
R=pathlib.Path(__file__).resolve().parents[1];W=R/'archive_worktree';prefix='experiments/gfp_badcase_trajectory_v1_20261015'
branch=subprocess.check_output(['git','-C',str(W),'branch','--show-current'],text=True).strip()
assert branch=='codex/gfp-badcase-trajectory-v1-20261015',branch
files=['EXPERIMENT_CARD.md','ACTUAL_COMMANDS.txt']
for folder,exts in [('code',{'.py'}),('configs',{'.json'}),('reports',{'.json','.md','.png','.jpg','.patch'}),('receipts',{'.json'})]:
 for p in sorted((R/folder).iterdir()):
  if p.is_file() and p.suffix in exts:files.append(str(p.relative_to(R)))
records=[]
for f in files:
 p=R/f;assert p.stat().st_size<2000000,(f,p.stat().st_size)
 records.append(dict(path=f,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
manifest=R/'reports/archive_manifest.json';manifest.write_text(json.dumps(records,indent=2))
if 'reports/archive_manifest.json' not in files:files.append('reports/archive_manifest.json')
for f in files:
 dst=W/prefix/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/f,dst)
subprocess.run(['git','-C',str(W),'add','--',*[prefix+'/'+f for f in files]],check=True)
print('ARCHIVE_BRANCH',branch,'FILES',len(files),'BYTES',sum((R/f).stat().st_size for f in files))
subprocess.run(['git','-C',str(W),'diff','--cached','--stat'],check=True)
(R/'receipts/archive_staged_paths.json').write_text(json.dumps(files,indent=2))

