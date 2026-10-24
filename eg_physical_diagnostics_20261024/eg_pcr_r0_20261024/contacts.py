import os,sys,json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.io import loadmat
from PIL import Image
ROOT=Path.home()/'work/grasp/eg_pcr_r0_20261024'
OLD=ROOT.parent/'eg_pose_quality_diagnosis_v1_20261022'
sys.path.insert(0,str(OLD))
import evaluate as E
from graspnetAPI.utils.utils import CameraInfo,create_point_cloud_from_depth_image

def run(scene,frame):
 start=time.time();a=np.load(OLD/f'cache/C3_{scene:04d}_{frame:04d}.npz');e=np.load(OLD/f'evaluation/C3_{scene:04d}_{frame:04d}.npz')
 pred=a['pred'];assert hashlib.sha256(np.ascontiguousarray(pred[:,1:]).tobytes()).hexdigest()==str(a['geometry_sha256'])==str(e['geometry_sha256'])
 ids=e['final_ids'];ids=ids[(e['P0_mu'][ids]>0)&~e['P0_collision'][ids]&~e['P0_empty'][ids]]
 direct=E.Direct(scene);direct.frame(frame,a['input_points'])
 base=E.D/f'scenes/scene_{scene:04d}/kinect';meta=loadmat(base/f'meta/{frame:04d}.mat');k=meta['intrinsic_matrix'];dep=np.array(Image.open(base/f'depth/{frame:04d}.png'))
 cam=CameraInfo(dep.shape[1],dep.shape[0],k[0,0],k[1,1],k[0,2],k[1,2],float(meta['factor_depth'].squeeze()))
 raw=create_point_cloud_from_depth_image(dep,cam,organized=True)[dep>0];trees=[cKDTree(raw),cKDTree(a['input_points'])]
 rows=[]
 for ix in ids:
  row=dict(scene=scene,frame=frame,candidate_id=int(ix),mu=float(e['P0_mu'][ix]),object_id=int(e['object_id'][ix]),contact_status='unknown')
  o=int(e['P0_owner'][ix]);gl=[np.zeros((0,17)) for _ in direct.models];gl[o]=pred[ix:ix+1].astype(float);gl[o][:,1]=gl[o][:,1].clip(0,.1)
  cs,es,gs=E.EU.collision_detection(gl,direct.mt,direct.dex,direct.poses,direct.allpts,outlier=.05,empty_thresh=10,return_dexgrasps=True)
  assert not cs[o][0] and not es[o][0]
  try:
   ok,ct=gs[o][0].close_fingers(direct.dex[o],check_approach=False)
   if ok and len(ct)==2:
    points=np.stack([v.point for v in ct]);normals=np.stack([v.normal for v in ct]);world=E.GE.transform_points(points,direct.poses[o]);axis=points[1]-points[0];axis/=np.linalg.norm(axis)
    cos=np.array([np.dot(-axis,normals[0]),np.dot(axis,normals[1])]);row.update(contact_status='ok',first_contact_risk=float(np.mean(.5*(1-cos))),contact_points_object=points.tolist(),contact_normals_object=normals.tolist())
    for label,tree in zip(['raw','sampled'],trees):
     distances=tree.query(world)[0];row[label+'_left_m']=float(distances[0]);row[label+'_right_m']=float(distances[1])
   else:row['contact_status']='close_fingers_failed'
  except Exception as ex:row['contact_status']=type(ex).__name__+':'+str(ex)
  rows.append(row)
 dest=ROOT/'contacts';dest.mkdir(exist_ok=True);(dest/f'{scene:04d}_{frame:04d}.json').write_text(json.dumps(dict(rows=rows,seconds=time.time()-start,cache_sha256=E.sha(OLD/f'cache/C3_{scene:04d}_{frame:04d}.npz'),evaluation_sha256=E.sha(OLD/f'evaluation/C3_{scene:04d}_{frame:04d}.npz'))))
 print(scene,frame,len(rows),sum(r['contact_status']=='ok' for r in rows),flush=True)

if __name__=='__main__':
 plan=json.loads((OLD/'study_plan.json').read_text());tasks=[(s,f) for s in plan['C3_scenes'] for f in plan['C3_frames']]
 if sys.argv[1]=='pilot':run(*tasks[0])
 else:
  from concurrent.futures import ProcessPoolExecutor
  shard=int(sys.argv[1]);tasks=tasks[shard::3]
  with ProcessPoolExecutor(4) as pool:list(pool.map(run,*zip(*tasks)))
