"""Explicit lightweight staging list, exact code identities and raw log indices."""
from common import *
import subprocess
repo=R.parents[1]
official=OLD/'EC-Diffuser'
codes=[]
for name in ['isaac_panda_push_env.py','isaac_env_wrappers.py','env_config/generalization_num_cubes/IsaacPandaPushConfig.yaml']:
 p=official/name;codes.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
for p in sorted(R.glob('*.py')):codes.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
dump('code_identities.json',{'utc':utc(),'base_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),'files':codes})
logs=[];failures=[]
for run in sorted((R/'runs').iterdir()):
 if not run.is_dir():continue
 for p in sorted(run.glob('*.log')):
  logs.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
 result=json.loads((run/'guard_result.json').read_text())
 log=(run/'process.log').read_text(errors='replace')
 selected=[x for x in log.splitlines() if any(k in x for k in ('cannot be used','RuntimeError','Ninja','Traceback'))]
 failures.append({'run':run.name,'exit_code':result['exit_code'],'stop_reason':result['stop_reason'],'log_path':str(run/'process.log'),'diagnostic_lines':selected,
 'state':'technical failure' if result['exit_code'] else 'measurement completed (gate failure can coexist with exit0)'})
dump('runtime_log_inventory.json',logs);dump('failure_and_stop_receipts.json',failures)
# Artifacts are generated for this round only. Neither npz nor unselected log can enter.
files=[]
for p in sorted(R.glob('*')):
 if p.is_file() and p.suffix in ('.py','.md'):files.append(p)
for p in sorted((R/'results').glob('*')):
 if p.is_file() and p.suffix in ('.json','.csv','.png','.md'):files.append(p)
for p in sorted((R/'receipts').glob('*')):
 if p.is_file() and p.suffix in ('.json','.txt'):files.append(p)
for run in sorted((R/'runs').iterdir()):
 if not run.is_dir():continue
 for name in ['guard_result.json','launch.json','gpu_memory.jsonl']:
  p=run/name
  if p.exists():files.append(p)
 for name in ['samples.json','receipt.json']:
  p=run/'data'/name
  if p.exists():files.append(p)
files.append(repo/'experiments/pc_cause04_card_20261012/EXPERIMENT_CARD.md')
manifest=R/'light_files.txt';files.append(manifest)
# Store ASCII paths with one file per line for git add --pathspec-from-file.
relative=[str(p.relative_to(repo)) for p in files]
assert len(relative)==len(set(relative)) and all(p.startswith('experiments/pc_cause04_') for p in relative)
manifest.write_text('\n'.join(relative)+'\n')
for p in files:
 assert p.exists() and '.npz' not in p.name and p.stat().st_size<3*1024*1024
 assert not any(x in p.parts for x in ['.ssh','resources','__pycache__'])
receipt={'utc':utc(),'files':len(files),'bytes':sum(p.stat().st_size for p in files),'staging_manifest_sha256':sha(manifest),
'plot_visual_check':'P2_case, P1_success, P3_pre_cross_trace inspected locally; images actual-step-labelled; trace uses pre-cross speeds',
'original_card_sha256':sha(R/'REQUESTED_CARD.md'),'no_auto_merge_or_force_push':True}
dump('archive_allowlist_receipt.json',receipt)
# Include receipt last, then update list deterministically; no manifest self-hash claim after list mutation.
relative.append(str((R/'results/archive_allowlist_receipt.json').relative_to(repo)))
manifest.write_text('\n'.join(relative)+'\n')
receipt['files']=len(relative);receipt['staging_manifest_sha256']=sha(manifest)
(R/'results/archive_allowlist_receipt.json').write_text(json.dumps(receipt,indent=2))
print('LIGHT_ALLOWLIST',len(relative),'bytes',receipt['bytes'],flush=True)
