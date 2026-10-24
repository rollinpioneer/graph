import json,inspect,hashlib
import run_controls as C
rows=[]
for p in sorted((C.ROOT/'results').glob('*.json')):
 r=json.loads(p.read_text());s=r['scene'];f=r['frame']
 checks={
  'original_cache':C.E.sha(C.OLD/f'cache/C3_{s:04d}_{f:04d}.npz')==r['original_cache_sha256'],
  'original_labels':C.E.sha(C.OLD/f'evaluation/C3_{s:04d}_{f:04d}.npz')==r['original_evaluation_sha256'],
  'prior_angle_results':C.E.sha(C.PREV/f'evaluation/{s:04d}_{f:04d}.npz')==r['prior_variants_sha256']}
 assert all(checks.values());rows.append(dict(scene=s,frame=f,checks=checks))
assert len(rows)==48
data=dict(rows=rows,source_hashes={fn.__name__:hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() for fn in [C.CD.ModelFreeCollisionDetector.detect,C.E.postprocess,C.sample,C.E.EU.collision_detection]},official_CD_file_sha256=C.E.sha(C.EG/'utils/collision_detector.py'),training=False,parameter_updates=0,new_force_closure_evaluations=0,historical_inputs_changed=False)
(C.ROOT/'integrity_audit.json').write_text(json.dumps(data,indent=2));print(json.dumps(data))
