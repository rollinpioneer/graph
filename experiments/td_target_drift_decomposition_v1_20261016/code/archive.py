from pathlib import Path
import json,subprocess,shutil,hashlib
root=Path(__file__).resolve().parents[1];wt=root/'archive_worktree';branch='codex/td-target-drift-decomposition-v1-20261016'
assert subprocess.check_output(['git','branch','--show-current'],cwd=wt,text=True).strip()==branch
selected=[root/'EXPERIMENT_CARD.md']+list((root/'code').glob('*.py'))+list((root/'reports').glob('*.json'))+list((root/'reports').glob('*.md'))+list((root/'receipts').glob('*.json'))
manifest=[];names=[]
for p in selected:
 assert p.stat().st_size<10000000
 rel=p.relative_to(root);out=wt/'experiments/td_target_drift_decomposition_v1_20261016'/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
 names.append(str(out.relative_to(wt)));manifest.append(dict(path=str(rel),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(root/'receipts/archive_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
subprocess.run(['git','add','--']+names,cwd=wt,check=True)
print('ARCHIVED',len(names),sum(x['bytes'] for x in manifest))
