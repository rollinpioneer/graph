from pathlib import Path
import json,hashlib,shutil,subprocess
R=Path(__file__).resolve().parents[1];A=R/'archive_worktree';PREFIX=Path('experiments/gfp_task4_same_state_v1_20261016');BRANCH='codex/gfp-task4-same-state-v1-20261016'
assert subprocess.check_output(['git','-C',str(A),'branch','--show-current'],text=True).strip()==BRANCH
files=[R/'EXPERIMENT_CARD.md']
for pat in ['code/*.py','configs/*.json','reports/*.md','reports/*.json','reports/*.csv','reports/*.png','reports/continuation/*.md','reports/continuation/*.json','reports/continuation/*.csv','reports/continuation/*.png','receipts/*.json','receipts/*.jsonl','attempts/*/*.json','attempts/*/*.py']:files.extend(R.glob(pat))
manifest=[]
for p in sorted(set(files)):
 if p.name=='archive_manifest.json':continue
 assert p.is_file() and p.stat().st_size<3000000
 rel=p.relative_to(R);dest=A/PREFIX/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 manifest.append(dict(path=str(rel),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
m=R/'receipts/archive_manifest.json';m.write_text(json.dumps(manifest,indent=2));shutil.copy2(m,A/PREFIX/'receipts/archive_manifest.json')
paths=[str(PREFIX/x['path']) for x in manifest]+[str(PREFIX/'receipts/archive_manifest.json')]
subprocess.run(['git','-C',str(A),'add','--sparse','-f','--',*paths],check=True)
subprocess.run(['git','-C',str(A),'diff','--cached','--check'],check=True)
print(json.dumps(dict(branch=BRANCH,files=len(paths),bytes=sum(x['bytes'] for x in manifest))))
