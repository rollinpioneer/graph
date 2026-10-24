"""Evaluation-only dense point geometry audit. Never feeds a detector or calibration."""
import sys,json,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
import pipeline as P
R=P.R
def run(task):
 _,s,f=task;out=R/f'dense/{s:04d}_{f:04d}.npz'
 if out.exists():return
 start=time.time();a=np.load(R/f'cpu/C3_{s:04d}_{f:04d}.npz');rows=a['pred'].reshape(-1,17);owner=np.repeat(a['owner'][a['candidate_id']],12)
 direct=P.E.Direct(s);direct.frame(f);models,_,_=P.E.ge.get_scene_models(s,0)
 points=np.concatenate([P.E.GE.transform_points(m,pose) for m,pose in zip(models,direct.poses)])
 # Table is sampled more densely over the identical official table volume.
 _,_,cam,align=P.E.ge.get_model_poses(s,f)
 table=P.E.GE.create_table_points(1,1,.05,dx=-.5,dy=-.5,dz=-.05,grid_size=.0025)
 table=P.E.GE.transform_points(table,np.linalg.inv(align@cam));allpoints=np.concatenate([points,table]);del points,table
 tree=cKDTree(allpoints);dense=[];robust=[];min_margin=[]
 for row,o in zip(rows,owner):
  pts=allpoints[tree.query_ball_point(row[13:16],.12)];lo=direct.mt[o].min(0)-.05;hi=direct.mt[o].max(0)+.05
  pts=pts[((pts>lo)&(pts<hi)).all(1)];t=(pts-row[13:16])@row[4:13].reshape(3,3)
  dense.append(bool(P.solid_masks(t,row[1],row[3]).any()));robust.append(bool(P.solid_masks(t,row[1],row[3],.00125).any()))
 np.savez_compressed(out,dense_collision=np.array(dense).reshape(-1,12),robust_collision=np.array(robust).reshape(-1,12),candidate_id=a['candidate_id'])
 out.with_suffix('.json').write_text(json.dumps(dict(scene=s,frame=f,poses=len(rows),full_object_points='original model point clouds',table_step=.0025,erosion_margin=.00125,wall_seconds=time.time()-start)))
 print(s,f,len(rows),round(time.time()-start,2),flush=True)
if __name__=='__main__':
 assert (R/'frozen_config.json').exists();(R/'dense').mkdir(exist_ok=True)
 ts=P.tasks('C3')[int(sys.argv[1])::3]
 with ProcessPoolExecutor(int(sys.argv[2])) as pool:list(pool.map(run,ts))
