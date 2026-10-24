from pathlib import Path
import json,hashlib,inspect
import evaluate_angles as A
root=Path(__file__).resolve().parent
rows=[]
for p in sorted((root/'evaluation').glob('*.json')):
 r=json.loads(p.read_text());s=r['scene'];f=r['frame']
 rows.append(dict(scene=s,frame=f,cache_unchanged=A.E.sha(A.OLD/f'cache/C3_{s:04d}_{f:04d}.npz')==r['cache_sha256'],evaluation_unchanged=A.E.sha(A.OLD/f'evaluation/C3_{s:04d}_{f:04d}.npz')==r['evaluation_sha256']))
assert len(rows)==48 and all(r['cache_unchanged'] and r['evaluation_unchanged'] for r in rows)
sources={fn.__name__:hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() for fn in [A.E.Direct,A.E.EU.collision_detection,A.E.EU.get_grasp_score,A.E.GE.voxel_sample_points]}
data=dict(rows=rows,source_hashes=sources,history_changed=False,parameter_updates=0,network_inference=False,C4_C5_read=False)
(root/'integrity_audit.json').write_text(json.dumps(data,indent=2));print(json.dumps(data))
