"""CPU-only feasibility probe. Official files remain unchanged.
Use a double KD-tree only to shortlist. Reproduce CUDA's float32 sequential
FMA squared distance and first-index tie rule for ambiguous candidates.
64*u relative plus 64*float32.tiny absolute slack conservatively encloses
float32 subtraction/squaring/3-coordinate accumulation errors, including FTZ.
"""
import os,sys,runpy,json,ctypes,ctypes.util,time,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
R=Path(__file__).resolve().parents[1]
S=R/'runtime/EconomicGrasp';sys.path.insert(0,str(S))
from libs.knn import knn_modules
libm=ctypes.CDLL(ctypes.util.find_library('m'));fmaf=libm.fmaf;fmaf.argtypes=[ctypes.c_float]*3;fmaf.restype=ctypes.c_float
libm.fegetround.restype=ctypes.c_int;assert libm.fegetround()==0, 'Require round-to-nearest'
last_ref=None;tree=None;points=None;stats={'queries':0,'ambiguous_queries':0,'max_candidates':0}
def cpu_knn(ref,query,k=1):
 global last_ref,tree,points
 assert k==1 and tuple(ref.shape[:2])==(1,3) and tuple(query.shape[:2])==(1,3)
 if ref is not last_ref:
  points=ref[0].T.contiguous().float().numpy();assert np.isfinite(points).all() and np.max(np.abs(points)) < np.sqrt(np.finfo(np.float32).max)/8;tree=cKDTree(points.astype(np.float64));last_ref=ref
 q=query[0].T.contiguous().float().numpy();assert np.isfinite(q).all() and np.max(np.abs(q)) < np.sqrt(np.finfo(np.float32).max)/8
 if len(points)==1:return torch.ones((1,1,len(q)),dtype=torch.long)
 distances,indices=tree.query(q,k=2,workers=4);chosen=indices[:,0].copy();u=np.finfo(np.float32).eps/2
 radius_squared=distances[:,0]**2*(1+64*u)+64*np.finfo(np.float32).tiny
 ambiguous=np.flatnonzero(distances[:,1]**2<=radius_squared);stats['queries']+=len(q);stats['ambiguous_queries']+=len(ambiguous)
 for row in ambiguous:
  candidates=tree.query_ball_point(q[row],np.sqrt(radius_squared[row])*(1+8*np.finfo(np.float64).eps));stats['max_candidates']=max(stats['max_candidates'],len(candidates));best=None
  for index in sorted(candidates):
   distance=0.0
   for dim in range(3):
    delta=float(np.float32(points[index,dim]-q[row,dim]));distance=fmaf(delta,delta,distance)
   if best is None or distance<best[0]:best=(distance,index)
  assert best is not None;chosen[row]=best[1]
 return torch.from_numpy(chosen.astype(np.int64).reshape(1,1,-1)+1)
knn_modules.knn=cpu_knn
torch.Tensor.cuda=lambda self,*args,**kwargs:self
if __name__=='__main__':
 started=time.time();rows=[]
 for scene in [0,99]:
  output=R/'runs/EG_FULL_V1/cpu_knn_probe'
  sys.argv=[str(S/'dataset/generate_graspness.py'),'--dataset_root',str(R/'data/graspnet'),'--camera_type','kinect','--scene_start',str(scene),'--scene_end',str(scene+1),'--frame_end','2','--output_root',str(output),'--cache_scene_graspness']
  runpy.run_path(str(S/'dataset/generate_graspness.py'),run_name='__main__')
  for frame in range(2):
   rel=Path(f'scene_{scene:04d}/kinect/{frame:04d}.npy');a=np.load(R/'runs/EG_FULL_V1/graspness_equivalence/reference'/rel);p=output/rel;b=np.load(p)
   rows.append({'scene':scene,'frame':frame,'exact_equal':bool(np.array_equal(a,b)),'max_abs_diff':float(np.max(np.abs(a-b))),'cpu_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 result={'status':'PASS' if all(r['exact_equal'] for r in rows) else 'FAIL','probe_only':True,'formal_label_generation_switched':False,'frames':rows,'stats':stats,'wall_seconds':time.time()-started}
 (R/'data_recovery/CPU_KNN_PROBE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);raise SystemExit(0 if result['status']=='PASS' else 2)
