"""Stage only the explicitly generated lightweight manifest. Commit/push require separate review."""
from pathlib import Path
import json,subprocess,shutil,hashlib
R=Path(__file__).resolve().parents[1];A=R/'archive_worktree';prefix=Path('experiments/gfp_task4_baseline_pilot_v1_20261015');D=A/prefix
branch=subprocess.check_output(['git','-C',str(A),'branch','--show-current'],text=True).strip();assert branch=='codex/gfp-task4-baseline-pilot-v1-20261015'
files=[R/'EXPERIMENT_CARD.md']
for pattern in ['code/*.py','configs/*.json','reports/*.md','reports/*.json','reports/*.csv','reports/*.png','reports/*.pdf','reports/start_failure/*.json','reports/start_failure/*.md','reports/start_failure/*.csv','reports/start_failure/*.png','reports/start_failure/*.pdf','receipts/*.json','workspaces/gfp/main.py','workspaces/gfp/training_wiring.py','workspaces/gfp/batch_hooks.py','workspaces/gfp/evaluation_process.py']:
 files.extend(R.glob(pattern))
for name in ['GFP_seed0','GFP_seed1','ReBRAC_seed0','ReBRAC_seed1']:
 d=R/'runs'/name
 for pat in ['status.json','runtime.json','dispatch_admission.json','exit.json','training_progress.json','checkpoint_index.json','matched_exit.json','matched/*.json','native/*.json']:
  files.extend(d.glob(pat))
 if (d/'runtime.json').exists():
  out=Path(json.loads((d/'runtime.json').read_text())['output_dir']);rd=R/'reports'/name;rd.mkdir(exist_ok=True)
  for n in ['train.csv','success.csv','eval.csv','checkpoint_index.jsonl','config.json','actual_agent_config.json']:
   p=out/n
   if p.exists():q=rd/n;shutil.copy2(p,q);files.append(q)
manifest=[]
for p in sorted(set(files)):
 if p.name=='archive_manifest.json' or not p.is_file():continue
 assert p.stat().st_size<3000000,(p,p.stat().st_size)
 rel=p.relative_to(R);dest=D/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);manifest.append(dict(path=str(rel),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(R/'receipts/archive_manifest.json').write_text(json.dumps(manifest,indent=2));shutil.copy2(R/'receipts/archive_manifest.json',D/'receipts/archive_manifest.json')
paths=[str(prefix/x['path']) for x in manifest]+[str(prefix/'receipts/archive_manifest.json')]
subprocess.run(['git','-C',str(A),'add','-f','--']+paths,check=True);subprocess.run(['git','-C',str(A),'diff','--cached','--check'],check=True)
print(json.dumps(dict(branch=branch,files=len(paths),bytes=sum(x['bytes'] for x in manifest))))
