import json,hashlib,csv
from pathlib import Path
B=Path.home()/'work/grasp';R=B/'eg_f1_precision_diagnosis_v1_20261021';P2=B/'eg_l1_readout_p2_v1_20261021'
d=json.loads((R/'diagnosis.json').read_text());rows=json.loads((R/'full_frames.json').read_text())
summary=[]
for s in sorted({x['scene'] for x in rows}):
 rr=[x for x in rows if x['scene']==s]
 summary.append(dict(scene=s,frames=len(rr),replay_max_error=max(x['replay']['max_error'] for x in rr),affine_max_error=max(x['collapsed_affine']['max_error'] for x in rr),affine_nonnear_inversions=sum(x['collapsed_affine']['nonnear_inversions'] for x in rr)))
with (R/'per_scene.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
(R/'worst_frames.json').write_text(json.dumps(sorted(rows,key=lambda x:x['collapsed_affine']['max_error'],reverse=True)[:10],indent=2))
status=dict(status='BLOCKED_PURE_AFFINE_INITIALIZATION',cache_replay='PASS_EXACT_ALL_3328_FRAMES',F1_original_collapsed_affine='FAIL_UNCHANGED',I1_pure_affine='FAIL',I1_replayed_frozen_score='PASS_BUT_NOT_A_65_PARAMETER_AFFINE_MODEL',training_steps=0,C4_read=False,thresholds_unchanged=True,diagnosis='cuDNN TF32 historical six-output convolution vs collapsed float64 affine readout; weight and cache identities match',evidence_directory=str(R))
(P2/'results/F1_precision_diagnosis.json').write_text(json.dumps(status,indent=2))
sources=[B/'eg_sub25_v1_20261015/runtime/EconomicGrasp/models/modules_economicgrasp.py',B/'eg_sub25_v1_20261015/runtime/EconomicGrasp/models/economicgrasp.py',B/'eg_lqr_build_confirm_v3_20261019/code/infer.py',B/'eg_lqr_build_confirm_v3_20261019/code/scoring.py',B/'eg_role_p0_audit_v1_20261021/code/p0_facts.py']
manifest=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources+[R/'full_frames.json',R/'diagnosis.json']]
(R/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({p.name:p.read_text() for p in [R/'per_scene.csv',R/'worst_frames.json',R/'evidence_manifest.json',P2/'results/F1_precision_diagnosis.json']}))
