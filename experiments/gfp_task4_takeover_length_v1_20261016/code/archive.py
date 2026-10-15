from pathlib import Path
import json,hashlib,shutil,subprocess
R=Path(__file__).resolve().parents[1];A=R/'archive_worktree';prefix=Path('experiments/gfp_task4_takeover_length_v1_20261016');branch='codex/gfp-task4-takeover-length-v1-20261016'
assert subprocess.check_output(['git','-C',str(A),'branch','--show-current'],text=True).strip()==branch
files=[R/'EXPERIMENT_CARD.md']
for pat in ['code/*.py','configs/*.json','reports/*.md','reports/*.json','reports/*.csv','reports/*.png','receipts/*.json']:files.extend(R.glob(pat))
manifest=[]
for p in sorted(set(files)):
 if p.name=='archive_manifest.json':continue
 assert p.stat().st_size<3000000
 rel=p.relative_to(R);dest=A/prefix/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 manifest.append(dict(path=str(rel),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
p=R/'receipts/archive_manifest.json';p.write_text(json.dumps(manifest,indent=2));shutil.copy2(p,A/prefix/'receipts/archive_manifest.json')
paths=[str(prefix/x['path']) for x in manifest]+[str(prefix/'receipts/archive_manifest.json')]
subprocess.run(['git','-C',str(A),'add','--sparse','-f','--',*paths],check=True)
subprocess.run(['git','-C',str(A),'diff','--cached','--check'],check=True)
print(json.dumps(dict(branch=branch,files=len(paths),bytes=sum(x['bytes'] for x in manifest))))
