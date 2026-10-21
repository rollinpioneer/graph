import json,hashlib
from pathlib import Path
R=Path.home()/'work/grasp/eg_tf32_residual_p2_v1_20261021'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=[R/x for x in ['EXPERIMENT_CARD.md','frozen_config.json','preflight_lock.json','method_lock.json','FINAL_REPORT.md']]+list((R/'code').glob('*.py'))
files += [R/'results'/x for x in ['readout_init.json','invariance_checks.json','training_summary.json','training_controller.json','offline.json','convergence.json','comparisons.json','transitions.json','sequential_decision.json','final_validation.json','family_results.csv','per_scene_results.csv','per_scene_differences.csv']]
losses=[]
for p in sorted((R/'results').glob('train_*.json')):
 x=json.loads(p.read_text());vv=x['losses'];losses.append(dict(family=x['family'],seed=x['seed'],losses_every100=vv[99::100],first100_mean=sum(vv[:100])/100,last100_mean=sum(vv[-100:])/100,source_sha256=sha(p)))
(R/'results/training_losses.json').write_text(json.dumps(losses,indent=2));files.append(R/'results/training_losses.json')
inventory=[dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
retained=[R/'head_inputs.npz',R/'results/evaluation_artifact_manifest.json',R/'results/P0_union_manifest.json',R/'results/offline_per_frame.csv']+list((R/'heads').glob('*/*/step1000.npz'))
inventory_retained=[dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p)) for p in retained]
entries=[dict(path='experiments/'+R.name+'/'+str(p.relative_to(R)),mode='100644',type='blob',content=p.read_text()) for p in files]
manifest=dict(archived=inventory,retained_on_server=inventory_retained,server_root=str(R),note='Full frame caches, checkpoints and detailed artifact manifest stay on server. Their hashes are recorded here; only code and lightweight results are uploaded.')
entries.append(dict(path='experiments/'+R.name+'/ARCHIVE_MANIFEST.json',mode='100644',type='blob',content=json.dumps(manifest,indent=2)))
(R/'ARCHIVE_MANIFEST.json').write_text(json.dumps(manifest,indent=2));(R/'archive_payload.json').write_text(json.dumps(entries));print(json.dumps(dict(files=len(entries),payload_bytes=(R/'archive_payload.json').stat().st_size)))
