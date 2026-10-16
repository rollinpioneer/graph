"""Copy only the explicit lightweight deliverables; do not commit/push automatically."""
from pathlib import Path
import json,hashlib,shutil,subprocess,zipfile,time
R=Path(__file__).resolve().parents[1];W=R/'archive_worktree';B='codex/gfp-task4-global-diagnosis-v1-20261016'
assert subprocess.check_output(['git','branch','--show-current'],cwd=W,text=True).strip()==B
assert json.loads((R/'receipts/probe_exit.json').read_text())['exit_code']==0
p=json.loads((R/'receipts/probe.json').read_text());assert p['optimizer_updates']==p['environment_steps']==p['forbidden_attempts']==0
assert all(x['parameter_optimizer_rng_unchanged'] for x in p['records']) and len(p['records'])==12
data=json.loads((R/'reports/data_identity.json').read_text());prior=json.loads(Path('/home/xushijie3/work/gfp_task4_baseline_pilot_v1_20261015/receipts/data_identity.json').read_text());assert all(x['sha256']==prior[x['split']]['sha256'] for x in data)
files=['EXPERIMENT_CARD.md']+['code/'+n for n in ['data.py','probe.py','run_probe.py','analyze.py','topology.py','finalize.py','report.py','archive.py']]+['reports/'+n for n in ['JOINT_DIAGNOSIS.md','decision_table.json','decision_table.csv','data_support.json','data_identity.json','proposal_fit.json','controlled_contrasts.json','original_batch_guidance.json','analysis_receipt.json','value_topology.json','value_cell_maps.json','first_edge_robustness.json','reused_asset_manifest.json','array_index.json','reused_performance.json','joint_evidence.png','joint_evidence.pdf']]+['receipts/'+n for n in ['freeze.json','data.json','data_export_fix.json','probe.json','probe_launch.json','probe_exit.json','executed_code.json']]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
completion=dict(status='COMPLETED_DIAGNOSE_NO_NEW_EXPERIMENTS',epoch=time.time(),branch=B,optimizer_updates=0,environment_steps=0,model_states=23940,state_noise_trials=p['state_noise_trials'],source_weights_unchanged=True,dataset_identity_matches_prior=True,main_fact_validation='task4 route count 9 train /1 validation agrees with earlier independent count; first-edge Q sign survives stable-posture same-episode check',scope_limit='No causal action-value ground truth; no new training, modules or interactions',analysis_repairs=['Initial data JSON export rejected numpy integer; repeated CPU analysis, first run CPU cost unmeasured','Analysis syntax error before execution fixed; no inference rerun'],code_sha256={n:sha(R/n) for n in files if n.startswith('code/')},logs=[dict(path=str(x),bytes=x.stat().st_size,sha256=sha(x)) for x in sorted((R/'logs').glob('*.log'))])
(R/'receipts/completion.json').write_text(json.dumps(completion,indent=2));files+=['receipts/completion.json']
items=[]
for n in files:
 p=R/n;assert p.is_file() and p.stat().st_size<3_000_000
 dst=W/'experiments'/R.name/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst);items.append(dict(path=n,bytes=p.stat().st_size,sha256=sha(p)))
manifest=dict(branch=B,prefix='experiments/'+R.name,files=items,total_bytes=sum(x['bytes'] for x in items))
(R/'receipts/archive_manifest.json').write_text(json.dumps(manifest,indent=2));shutil.copy2(R/'receipts/archive_manifest.json',W/'experiments'/R.name/'receipts/archive_manifest.json')
files+=['receipts/archive_manifest.json']
with zipfile.ZipFile(R/'handoff.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
 for n in files:z.write(R/n,n)
print('ARCHIVE',len(files),manifest['total_bytes'],'ZIP', (R/'handoff.zip').stat().st_size,sha(R/'handoff.zip'))
