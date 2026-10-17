"""Explicit lightweight archive only; no weights, numeric arrays or full logs."""
from pathlib import Path
import json,hashlib,shutil,subprocess
R=Path(__file__).resolve().parent;W=R/'archive_worktree';B='codex/crr-historical-separation-z0-20261017';prefix='experiments/crr_historical_separation_z0_20261017';T=W/prefix
assert subprocess.check_output(['git','-C',str(W),'branch','--show-current'],text=True).strip()==B
names=['EXPERIMENT_CARD.md','protocol.json','prepare.py','infer.py','analyze.py','archive.py','RESULTS.md','HANDOFF.md','source_manifest.json','checkpoint_manifest.json','configs.json','historical_probes.json','historical_evaluations.json','sample_manifest.json','metrics.csv','sensitivity.csv','decision.json','contrast.png','q_differences.png','gradient_ratio.png','server_artifact_index.json','receipts/preparation.json','receipts/validation.json','receipts/integrity.json','receipts/inference.json','receipts/execution.json','receipts/analysis.json','receipts/s0_fixed_action_identity.json','receipts/s3_fixed_action_identity.json']
names += ['sources/'+n for n in ['s0_config.json','s3_config.json','gfp.py','networks.py','flax_utils.py','exp_b.py','exp_c.py','common.py','B_EVAL_POINTS.txt','C_EVAL_POINTS.txt','D_window_rule.json','data_identity.json']]
selected=[n for n in names if (R/n).is_file()];manifest=[]
for n in selected:
 p=R/n;assert p.stat().st_size<2_000_000,(n,p.stat().st_size)
 q=T/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 manifest.append(dict(path=n,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(R/'archive_manifest.json').write_text(json.dumps(manifest,indent=2));shutil.copy2(R/'archive_manifest.json',T/'archive_manifest.json')
paths=[prefix+'/'+n for n in selected]+[prefix+'/archive_manifest.json']
subprocess.run(['git','-C',str(W),'add','--']+paths,check=True)
staged=subprocess.check_output(['git','-C',str(W),'diff','--cached','--name-only'],text=True).splitlines();assert set(staged)<=set(paths)
print('EXPLICIT_STAGED_FILES',len(staged),'BYTES',sum(m['bytes'] for m in manifest));print('\n'.join(staged))
subprocess.run(['git','-C',str(W),'diff','--cached','--check','--']+[p for p in paths if '/sources/' not in p],check=True)
