"""Data integrity and explicitly requested counts only; no selection rules."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
SCENES=[100,106,115,124,132,139,150,157,160,169,176,186]
FRAMES=[32,96,160,224]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert not (ROOT/'STOP_ON_MISMATCH').exists()
 replay=json.loads((ROOT/'c3_replay.json').read_text());assert replay['passed']
 counts=dict(frames=0,candidates=0,variants=0,original_angle_clean=0,original_angle_feasible=0,original_angle_weak=0,mu_unknown=0,I1_missing=0,original_angle_I1_missing=0)
 manifest=[]
 expected={f'{s:04d}_{f:04d}.npz' for s in SCENES for f in FRAMES}
 assert {p.name for p in (ROOT/'evaluation').glob('*.npz')}==expected
 for s in SCENES:
  for f in FRAMES:
   p=ROOT/'evaluation'/f'{s:04d}_{f:04d}.npz';r=json.loads(p.with_suffix('.json').read_text())
   assert sha(p)==r['output_sha256']
   assert all(r[k] for k in ['collision_match','empty_match','owners_match','candidate_id_match','original_angle_match','pred_preserved'])
   a=np.load(p);n=len(a['candidate_id']);ii=np.arange(n);aa=a['original_angle']
   assert a['pred'].shape==(n,12,17)
   assert all(a[k].shape==(n,12) for k in ['mu','collision','empty','I1','unknown_reason','contact_reason'])
   assert r['candidates']==n and r['variants']==n*12
   assert r['unknown']==int(np.isnan(a['mu']).sum()) and r['contact_missing']==int(np.isnan(a['I1']).sum())
   assert np.all(a['mu'][a['collision']]==-1)
   clean=~a['collision'][ii,aa]&~a['empty'][ii,aa];mu=a['mu'][ii,aa]
   counts['frames']+=1;counts['candidates']+=n;counts['variants']+=n*12
   counts['original_angle_clean']+=int(clean.sum())
   counts['original_angle_feasible']+=int((clean&(mu>0)&(mu<=.4)).sum())
   counts['original_angle_weak']+=int((clean&(mu>.4)).sum())
   counts['mu_unknown']+=r['unknown'];counts['I1_missing']+=r['contact_missing']
   counts['original_angle_I1_missing']+=int(np.isnan(a['I1'][ii,aa]).sum())
   manifest.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),receipt_sha256=sha(p.with_suffix('.json'))))
 assert (counts['frames'],counts['candidates'],counts['variants'])==(48,2247,26964),counts
 summary=dict(c3_replay=dict(passed=replay['passed'],scene=replay['scene'],frame=replay['frame'],candidates=replay['candidates'],variants=replay['variants'],**replay['checks']),**counts,verification=dict(all_48_candidate_ids_match=True,all_48_original_angles_match=True,all_48_owner_mappings_match=True,all_26964_collision_labels_match=True,all_26964_empty_labels_match=True),definitions=dict(original_angle_clean='not collision and not empty',original_angle_feasible='clean and 0 < mu <= 0.4',original_angle_weak='clean and mu > 0.4',unknown='mu is NaN; no force-closure solution (-1) is not unknown'))
 (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
 (ROOT/'output_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
