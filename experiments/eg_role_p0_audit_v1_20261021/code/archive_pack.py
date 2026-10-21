import json,hashlib,gzip,base64,argparse
from pathlib import Path
B=Path.home()/'work/grasp'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('stage',choices=['P0','P1','P2']);a=p.parse_args();name={'P0':'eg_role_p0_audit_v1_20261021','P1':'eg_gw_p1_v1_20261021','P2':'eg_l1_readout_p2_v1_20261021'}[a.stage];R=B/name
files=[R/'EXPERIMENT_CARD.md',R/'frozen_config.json']+list((R/'code').glob('*.py'))+list((R/'code').glob('*.diff'))
if a.stage=='P0':
 files+=list(R.glob('P0_*.json'))+list(R.glob('P0_*.csv'))+[R/'P0_REPORT.md']
elif a.stage=='P1':
 files+=[R/'FINAL_REPORT.md',R/'method_lock.json']
 for n in ['reproduction.json','offline.json','comparisons.json','transitions.json','development_decision.json','final_validation.json','family_results.csv','per_scene_results.csv','per_scene_differences.csv','training_summary.json','evaluation_artifact_manifest.json','P0_union_manifest.json']:
  files.append(R/'results'/n)
 files+=list((R/'results').glob('train_*.json'))+list((R/'results').glob('controller_*.json'))
else:
 files+=[R/'results/sequential_decision.json'];report=R/'FINAL_REPORT.md';report.write_text('# P2 blocked by preregistered F1\n\nUser approved conditional P2 execution. P0 found affine readout reconstruction max error 0.003219462596 > 0.0001 with non-near ranking inversions. T1 and T2 pass but T0 fails. P2 is BLOCKED, with zero training and zero C3 evaluation. This is not evidence that E_act is scientifically ineffective. No tolerance relaxation, alternative initialization, C4/C5 access or architecture expansion was performed. Configuration was frozen with P1 before P1 C3 results.\n');files.append(report)
entries=[];inventory=[]
for f in files:
 raw=f.read_bytes();rel=f.relative_to(R).as_posix();inventory.append(dict(path=rel,sha256=sha(f),bytes=len(raw)))
 if len(raw)>150000:
  compressed=gzip.compress(raw,mtime=0);entries.append(dict(path=f'experiments/{name}/{rel}.gz',encoding='base64',content=base64.b64encode(compressed).decode()))
 else:entries.append(dict(path=f'experiments/{name}/{rel}',encoding='utf-8',content=raw.decode('utf-8')))
receipt=dict(server_root=str(R),files=inventory,note='Large lightweight tables/manifests are gzip-compressed, exact decompressed bytes match listed SHA256. Model checkpoints, frame-level caches and data stay on server.')
entries.append(dict(path=f'experiments/{name}/ARCHIVE_MANIFEST.json',encoding='utf-8',content=json.dumps(receipt,indent=2)))
payload=R/'archive_payload.json';payload.write_text(json.dumps(entries));print(json.dumps(dict(stage=a.stage,files=len(entries),payload_bytes=payload.stat().st_size,path=str(payload))))
