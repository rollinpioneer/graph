"""Explicit lightweight manifest only; caller separately commits and pushes."""
import pathlib,json,hashlib,shutil,subprocess
R=pathlib.Path(__file__).resolve().parents[1];W=R/'archive_worktree';prefix='experiments/'+R.name
branch='codex/gfp-mechanism-and-condition-screen-v1-20261015'
files=['EXPERIMENT_CARD.md','configs/states.json','code/select_states.py','code/probe.py','code/summarize_probe.py','code/check_public_asset.py','code/inspect_pipeline.py','code/write_probe_report.py','code/plot_probe.py','code/archive.py','reports/FINAL_REPORT.md','reports/MECHANISM_REPORT.md','reports/CONDITION_A_HISTORY.md','reports/CONDITION_B_SAFETY.md','reports/CONDITION_C_CORRECTION.md','reports/branch_results.json','reports/state_summary.csv','reports/probe_summary.json','reports/proposal_distributions.json','reports/pipeline_data_inspection.json','reports/pipeline_data_check.json','reports/cost_receipt.json','reports/local_rescues.png','receipts/protocol_freeze.json','receipts/initial_archive.json','receipts/assets_before.json','receipts/device.json','receipts/progress.json','receipts/probe_result.json','receipts/process_exit.json','receipts/public_data_access.json','receipts/dsrl_availability.json','receipts/source_trajectories_unchanged.json','sources/code_evidence_manifest.json','sources/paper_evidence.json']
def git(*a):return subprocess.check_output(['git','-C',str(W),*a],text=True).strip()
assert git('branch','--show-current')==branch
assert not git('diff','--cached','--name-only')
entries=[]
for name in files:
 p=R/name;assert p.is_file() and p.stat().st_size<2_000_000
 entries.append(dict(path=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
assert sum(x['bytes'] for x in entries)<10_000_000
manifest=dict(branch=branch,files=entries,large_assets_excluded=['branches/*.npz','sources/Pipeline-val.npz','full logs','old checkpoints'],base='83d147e1f976ad8f87848989e3b7dda336c2d724')
(R/'receipts/archive_manifest.json').write_text(json.dumps(manifest,indent=2));files.append('receipts/archive_manifest.json')
for name in files:
 dst=W/prefix/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/name,dst)
git('add','--',*[prefix+'/'+p for p in files])
changed=git('diff','--cached','--name-only').splitlines();assert all(x in [prefix+'/'+p for p in files] for x in changed)
print('FILES',len(files),'BYTES',sum(x['bytes'] for x in entries));print(git('diff','--cached','--stat'))
