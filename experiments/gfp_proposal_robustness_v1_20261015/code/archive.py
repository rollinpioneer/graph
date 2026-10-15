import json,hashlib,subprocess,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];W=R/'archive_worktree';prefix=Path('experiments')/R.name;branch='codex/gfp-proposal-robustness-v1-20261015'
assert subprocess.check_output(['git','-C',str(W),'branch','--show-current'],text=True).strip()==branch
assert not subprocess.check_output(['git','-C',str(W),'status','--porcelain'],text=True).strip()
files=['EXPERIMENT_CARD.md','code/freeze.py','code/probe.py','code/run_once.py','code/analyze.py','code/plot.py','code/report.py','code/archive.py','configs/frozen_protocol.json','configs/states.json','configs/first_actions.json','reports/REPORT.md','reports/summary.json','reports/branch_results.json','reports/old_suffix_derived.json','reports/new_branch_metrics.csv','reports/state_successes.csv','reports/new_suffix_successes.png','reports/branch_index.json','receipts/preflight.json','receipts/protocol_push.json','receipts/code_push.json','receipts/assets_before.json','receipts/device.json','receipts/environment.json','receipts/suffix_key_hashes.json','receipts/dispatch.json','receipts/resource_samples.json','receipts/progress.json','receipts/probe_result.json','receipts/exit.json','receipts/verification.json']
assert all((R/f).is_file() for f in files)
manifest=[dict(path=f,bytes=(R/f).stat().st_size,sha256=hashlib.sha256((R/f).read_bytes()).hexdigest()) for f in files]
assert all(x['bytes']<3_000_000 for x in manifest)
(R/'reports/archive_manifest.json').write_text(json.dumps(manifest,indent=2));files.append('reports/archive_manifest.json')
for f in files:
 dest=W/prefix/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/f,dest)
subprocess.run(['git','-C',str(W),'add','--']+[str(prefix/f) for f in files],check=True)
staged=subprocess.check_output(['git','-C',str(W),'diff','--cached','--name-only'],text=True).splitlines();assert set(staged)<=set(str(prefix/f) for f in files)
subprocess.run(['git','-c','core.whitespace=cr-at-eol','-C',str(W),'diff','--cached','--check'],check=True)
print(subprocess.check_output(['git','-C',str(W),'diff','--cached','--stat'],text=True));print('FILES',len(files),'TOTAL_BYTES',sum(x['bytes'] for x in manifest))
