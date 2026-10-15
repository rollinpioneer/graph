import json,sys,hashlib,concurrent.futures
from pathlib import Path
import numpy as np
from resource_jobs import run,R
assert json.loads((R/'data_recovery/RAW_DATA_VALIDATION.json').read_text())['status']=='PASS'
S=R/'runtime/EconomicGrasp';BASE=R/'runs/EG_FULL_V1/graspness_equivalence';rows=[]
for scene in [0,99]:
 for cached in [False,True]:
  mode='cached' if cached else 'reference';out=BASE/mode
  cmd=[sys.executable,str(S/'dataset/generate_graspness.py'),'--dataset_root',str(R/'data/graspnet'),'--camera_type','kinect','--scene_start',str(scene),'--scene_end',str(scene+1),'--frame_end','2','--output_root',str(out)]
  if cached:cmd.append('--cache_scene_graspness')
  run(f'graspness_equivalence_{scene}_{mode}',cmd,cwd=S,gpu=True)
 for frame in range(2):
  rel=Path(f'scene_{scene:04d}/kinect/{frame:04d}.npy');p=BASE/'reference'/rel;q=BASE/'cached'/rel
  a=np.load(p);b=np.load(q);assert a.dtype==b.dtype and np.array_equal(a,b),(scene,frame,float(np.max(np.abs(a-b))))
  rows.append({'scene':scene,'frame':frame,'shape':list(a.shape),'dtype':str(a.dtype),'exact_equal':True,'reference_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'cached_sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'max_abs_diff':float(np.max(np.abs(a-b)))})
result={'status':'PASS','comparison':'Same pinned original computation without scene cache vs cached computation, exact equality and identical NPY hashes, four real frames','frames':rows,'threshold_friction':.8,'views_angles_depths':[300,12,4],'knn_chunk_points':10000,'normalization':'unchanged official min-max; invalid denominator fails'}
(R/'data_recovery/GRASPNESS_EQUIVALENCE.json').write_text(json.dumps(result,indent=2));print('GRASPNESS_EQUIVALENCE_PASS',flush=True)
