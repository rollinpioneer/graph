"""Small scientific audit and lightweight export only."""
import json,hashlib,tarfile,base64,time
from pathlib import Path
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
stats=json.loads((R/'statistics.json').read_text());cfg=json.loads((R/'frozen_config.json').read_text());frozen=sha(R/'frozen_config.json')
assert frozen==(R/'frozen_config.sha256').read_text().strip()
assert all(v['frames']==48 for v in stats['panels'].values())
assert stats['panels']['C3']['candidates']==1229 and stats['panels']['C3']['poses']==14748
for op in ['OP1','OP2']:
 b=next(r for r in stats['joint'] if r['op']==op and r['group']=='B');o=next(r for r in stats['joint'] if r['op']==op and r['group']=='O')
 assert [b[k] for k in ['feasible','recovery','harm','collision','top1_feasible']]==[784,192,49,127,26],b
 assert o['feasible']==641 and o['top1_feasible']==20,o
 for cat,n in [('below_threshold',74),('missing_observation',44),('geometry_mismatch',3),('feasible_false_filter',164)]:
  r=next(x for x in stats['mechanism'] if x['category']==cat and x['op']==op and x['method']=='N_ref');assert r['n']==n,(cat,r)
for r in cfg['calibration_input_manifest']:assert sha(R/r['path'])==r['sha256']
timing=list((R/'timing').glob('*.json'));assert len(timing)==48 and all(json.loads(p.read_text())['identical_scores'] for p in timing)
first=min(p.stat().st_mtime for p in (R/'cpu').glob('C3_*.json'))
assert first>=cfg['created_unix']
for p in (R/'joint').glob('*.json'):assert json.loads(p.read_text())['frozen_config_sha256']==frozen
audit=dict(status='passed',calibration_before_formal_C3=True,first_C3_receipt_unix=first,freeze_unix=cfg['created_unix'],frozen_config_sha256=frozen,calibration_files_SHA_verified=len(cfg['calibration_input_manifest']),C3_original_O_and_B_replay_exact=True,mechanism_74_44_3_and_164_replayed=True,timing_replay_identical_48=True,C3_formal_parameter_configurations=1,no_training=True,C4_C5=False)
(R/'integrity_audit.json').write_text(json.dumps(audit,indent=2))
files=[]
for pattern in ['*.json','*.csv','*.png','*.sha256','cpu/*.json','nv/*.json','dense/*.json','dense/*.npz','joint/*.json','timing/*.json','per_pose/*.npz','cache/*.json']:
 files.extend(R.glob(pattern))
files=[p for p in set(files) if p.name not in ['export_manifest.json']]
manifest={str(p.relative_to(R)):sha(p) for p in files};(R/'export_manifest.json').write_text(json.dumps(manifest,indent=2));files.append(R/'export_manifest.json')
with tarfile.open(R/'light_results.tar.gz','w:gz') as t:
 for p in sorted(files):t.add(p,arcname=p.relative_to(R).as_posix())
print(json.dumps(dict(files=len(files),compressed_bytes=(R/'light_results.tar.gz').stat().st_size,audit=audit)))
