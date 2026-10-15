from pathlib import Path
import json,hashlib,shutil,subprocess,zipfile
R=Path(__file__).resolve().parents[1];W=R/'archive_worktree';branch='codex/gfp-guidance-distance-z01-20261015';rel=Path('experiments/gfp_guidance_distance_z01_20261015')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert subprocess.check_output(['git','branch','--show-current'],cwd=W,text=True).strip()==branch
arrays=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((R/'arrays').glob('*.npz'))]
(R/'receipts/array_index.json').write_text(json.dumps(arrays,indent=2))
files=[R/'EXPERIMENT_CARD.md']
for d,patterns in [('code',['*.py']),('configs',['*.json']),('reports',['*.json','*.csv','*.md','*.png','*.pdf']),('receipts',['*.json','probe_resources.jsonl']),('sources',['**/*'])]:
 for pattern in patterns:
  files.extend(p for p in (R/d).glob(pattern) if p.is_file())
files=sorted(set(files)-{R/'receipts/archive_manifest.json',R/'receipts/git_push.json',R/'receipts/local_bundle.json'})
assert all(p.stat().st_size<2_000_000 for p in files)
manifest=[dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
(R/'receipts/archive_manifest.json').write_text(json.dumps(manifest,indent=2));files.append(R/'receipts/archive_manifest.json')
for p in files:
 dst=W/rel/p.relative_to(R);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
paths=[str(rel/p.relative_to(R)) for p in files]
subprocess.run(['git','add','--']+paths,cwd=W,check=True)
print('ARCHIVE',len(files),sum(p.stat().st_size for p in files),'bytes',flush=True)
subprocess.run(['git','diff','--cached','--stat'],cwd=W,check=True)
with zipfile.ZipFile(R/'handoff.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in files:z.write(p,str(p.relative_to(R)))
b=R/'handoff.zip';print('BUNDLE',b.stat().st_size,sha(b),flush=True)
