"""CPU observation-only contact geometry; no friction or selection analysis."""
import sys,json,time,hashlib,traceback,inspect
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.io import loadmat
from PIL import Image
ROOT=Path(__file__).resolve().parent;B=ROOT.parent.parent
V3=B/'eg_firmness_export_20261024/v3'
sys.path.insert(0,str(B/'eg_pose_quality_diagnosis_v1_20261022'));import evaluate as E
STOP=ROOT/'STOP_ON_FAILURE'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False))
def gt_contacts(rows,object_ids,old_I1,s,f):
 models,dex,ids=E.ge.get_scene_models(s,0);frame_ids,poses,cam,align=E.ge.get_model_poses(s,f)
 assert list(ids)==list(frame_ids) and len(set(ids))==len(ids)
 lookup={int(v):j for j,v in enumerate(ids)};pts_cam=np.full((len(rows),2,3),np.nan);risk=np.full(len(rows),np.nan);errors={}
 for k in np.flatnonzero(np.isfinite(old_I1)):
  row=rows[k];o=lookup[int(object_ids[k])];pose=poses[o];grasp_point=row[13:16];R=row[4:13].reshape(3,3);width=row[1];depth=row[3]
  # Verbatim conversion used in official collision_detection's Dex-Net branch.
  center = np.array([depth, 0, 0]).reshape([3, 1])
  center = np.dot(R, center).reshape([3])
  center = (center + grasp_point).reshape([1,3])
  center = E.EU.transform_points(center, np.linalg.inv(pose)).reshape([3])
  R = np.dot(pose[:3,:3].T, R)
  binormal, approach_angle = E.EU.matrix_to_dexnet_params(R)
  g = E.EU.ParallelJawPtGrasp3D(E.EU.ParallelJawPtGrasp3D.configuration_from_params(center, binormal, width, approach_angle), depth)
  try:
   ok,ct=g.close_fingers(dex[o],check_approach=False)
   if not ok or len(ct)!=2:errors[str(k)]='close_fingers_failed';continue
   pts=np.stack([v.point for v in ct]);ns=np.stack([v.normal for v in ct]);axis=pts[1]-pts[0];length=np.linalg.norm(axis)
   if length<1e-12 or not np.isfinite(ns).all():errors[str(k)]='invalid_contact';continue
   axis/=length;cos=np.array([np.dot(-axis,ns[0]),np.dot(axis,ns[1])]);risk[k]=np.mean(.5*(1-cos))
   pts_cam[k]=E.GE.transform_points(pts,pose)
  except Exception as ex:errors[str(k)]='contact:'+type(ex).__name__
 R=rows[:,4:13].reshape(-1,3,3);T=rows[:,13:16];d=rows[:,3];w=rows[:,1]
 both=np.isfinite(pts_cam).all(axis=(1,2));local=np.einsum('npi,nij->npj',pts_cam-T[:,None,:],R)
 line=np.sqrt((local[:,:,0]-d[:,None])**2+local[:,:,2]**2)
 line_ok=(line<=.001).all(1);order_ok=local[:,0,1]<local[:,1,1];width_ok=(abs(local[:,:,1])<=w[:,None]/2+.001).all(1)
 joint=line_ok&order_ok&width_ok;den=int(both.sum());joint_fraction=float(joint[both].mean()) if den else None
 finite=np.isfinite(risk)&np.isfinite(old_I1);err=float(np.max(abs(risk[finite]-old_I1[finite]))) if finite.any() else None
 checks=dict(gt_two_contact_variants=den,joint_convention_fraction=joint_fraction,line_fraction=float(line_ok[both].mean()) if den else None,order_fraction=float(order_ok[both].mean()) if den else None,width_fraction=float(width_ok[both].mean()) if den else None,convention_passed=bool(den and joint_fraction>=.99),I1_missing_mask_equal=bool(np.array_equal(np.isnan(risk),np.isnan(old_I1))),I1_max_absolute_error=err,I1_passed=bool(np.array_equal(np.isnan(risk),np.isnan(old_I1)) and err is not None and err<1e-7),recompute_scope='stored finite I1 only; originally missing entries retained',contact_errors=errors)
 return pts_cam,checks,ids
def run_frame(task,save=True):
 split,s,f=task;key=f'{split}_{s:04d}_{f:04d}';start=time.time();cpu=time.process_time()
 assert not STOP.exists()
 inp=V3/f'eval/{key}.npz'
 with np.load(inp) as z:
  ids=z['candidate_id'].copy();angles=z['original_angle'].copy();pred=z['pred'].copy();oid=z['object_id'].copy();old_I1=z['I1'].copy()
 n=len(ids);assert pred.shape==(n,12,17) and oid.shape==(n,12)
 rows=pred.reshape(-1,17);gt,checks,scene_ids=gt_contacts(rows,oid.ravel(),old_I1.ravel(),s,f)
 base=dict(split=split,scene=s,frame=f,candidates=n,variants=len(rows),input_sha256=sha(inp),checks=checks)
 # Fail before observation calculations if the required convention/replay gate fails.
 if not checks['convention_passed'] or not checks['I1_passed']:
  base.update(wall_seconds=time.time()-start,cpu_seconds=time.process_time()-cpu)
  write_json(ROOT/f'failure_{key}.json',base)
  np.savez_compressed(ROOT/f'failure_{key}_gt.npz',candidate_id=ids,original_angle=angles,gt_pts_cam=gt.reshape(n,12,2,3))
  raise AssertionError((key,'GT convention or I1 replay gate failed',checks))
 p=E.D/f'scenes/scene_{s:04d}/kinect';meta=loadmat(p/f'meta/{f:04d}.mat');K=meta['intrinsic_matrix'];raw=np.asarray(Image.open(p/f'depth/{f:04d}.png'))
 assert set(map(int,np.asarray(meta['cls_indexes']).ravel()))=={int(i)+1 for i in scene_ids},'segmentation mapping mismatch'
 depth=raw.astype(np.float32)/float(meta['factor_depth'].item());yy,xx=np.indices(raw.shape)
 cloud=np.stack([(xx-K[0,2])*depth/K[0,0],(yy-K[1,2])*depth/K[1,1],depth],-1)
 pixel=np.flatnonzero(depth.ravel()>0);raw_points=cloud.reshape(-1,3)[pixel]
 pc=o3d.geometry.PointCloud();pc.points=o3d.utility.Vector3dVector(raw_points)
 filtered,keep=pc.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
 keep=np.asarray(keep,dtype=int);P=raw_points[keep];pixel=pixel[keep];assert np.array_equal(np.asarray(filtered.points),P)
 tree=cKDTree(P);computed=np.zeros(len(P),bool);normals=np.full((len(P),3),np.nan)
 def normal(i):
  if not computed[i]:
   neighbors=tree.query_ball_point(P[i],.01,return_sorted=True);computed[i]=True
   if len(neighbors)>=10:
    v=P[neighbors];v=v-v.mean(0);_,vectors=np.linalg.eigh(v.T@v/len(v));normal=vectors[:,0]
    if np.dot(normal,-P[i])<0:normal=-normal
    normals[i]=normal
  return normals[i]
 def select(indices,local,R):
  points=np.full((2,3),np.nan);ns=np.full((2,3),np.nan)
  if not len(indices):return 0,np.nan,np.nan,points,ns
  a=int(np.argmin(local[:,1]));b=int(np.argmax(local[:,1]));ia=indices[a];ib=indices[b]
  if local[b,1]-local[a,1]>=.005:
   points[0]=P[ia];points[1]=P[ib];ns[0]=normals[ia];ns[1]=normals[ib]
   axis=points[1]-points[0];axis/=np.linalg.norm(axis);cos=np.array([np.dot(-axis,ns[0]),np.dot(axis,ns[1])]);risk=np.mean(.5*(1-cos))
   return 2,risk,risk,points,ns
  # With <5 mm span, apply the stated side test to the minimum-y point.
  side=0 if np.dot(normals[ia],R[:,1])<0 else 1;i=ia if side==0 else ib
  points[side]=P[i];ns[side]=normals[i];axis=-R[:,1] if side==0 else R[:,1]
  return 1,np.nan,.5*(1-np.dot(normals[i],axis)),points,ns
 N=len(rows);out=dict(candidate_id=ids,original_angle=angles,gt_pts_cam=gt.reshape(n,12,2,3))
 for rmm in [2,3,5]:out[f'I1obs_r{rmm}']=np.full(N,np.nan);out[f'found_r{rmm}']=np.zeros(N,np.int8)
 out.update(I1half_r3=np.full(N,np.nan),I1obs_seg_r3=np.full(N,np.nan),found_seg_r3=np.zeros(N,np.int8),obs_pts_r3=np.full((N,2,3),np.nan),obs_normals_r3=np.full((N,2,3),np.nan))
 # Segmentation is used exclusively in the separately reported seg_r3 branch.
 labels=np.asarray(Image.open(p/f'label/{f:04d}.png')).ravel()[pixel]
 for j,row in enumerate(rows):
  R=row[4:13].reshape(3,3);T=row[13:16];w=row[1];d=row[3];center=T+d*R[:,0]
  for rmm in [2,3,5]:
   rad=rmm/1000;ix=np.asarray(tree.query_ball_point(center,w/2+rad+.001,return_sorted=True),dtype=int)
   local=(P[ix]-T)@R;mask=(abs(local[:,0]-d)<rad)&(abs(local[:,2])<rad)&(abs(local[:,1])<=w/2);ix=ix[mask];local=local[mask]
   valid=np.array([np.isfinite(normal(int(i))).all() for i in ix],bool);ix=ix[valid];local=local[valid]
   found,risk,half,pts,ns=select(ix,local,R);out[f'found_r{rmm}'][j]=found;out[f'I1obs_r{rmm}'][j]=risk
   if rmm==3:
    out['I1half_r3'][j]=half;out['obs_pts_r3'][j]=pts;out['obs_normals_r3'][j]=ns
    mask=labels[ix]==int(oid.ravel()[j])+1;sf,sr,_,_,_=select(ix[mask],local[mask],R);out['found_seg_r3'][j]=sf;out['I1obs_seg_r3'][j]=sr
 for k in list(out):
  if k in ['candidate_id','original_angle','gt_pts_cam']:continue
  out[k]=out[k].reshape((n,12)+out[k].shape[1:])
 checks.update(candidate_id_equal=bool(np.array_equal(out['candidate_id'],ids)),original_angle_equal=bool(np.array_equal(out['original_angle'],angles)))
 rec=dict(**base,raw_points=len(raw_points),filtered_points=len(P),outlier_removed_fraction=1-len(P)/len(raw_points),segmentation_mapping='pixel label = zero-based GraspNet object_id + 1; 0 is background; scene IDs checked against cls_indexes',wall_seconds=time.time()-start,cpu_seconds=time.process_time()-cpu,depth_sha256=sha(p/f'depth/{f:04d}.png'),meta_sha256=sha(p/f'meta/{f:04d}.mat'),segmentation_sha256=sha(p/f'label/{f:04d}.png'),source_sha256=sha(Path(__file__)))
 if save:
  path=ROOT/f'{key}.npz';assert not path.exists();np.savez_compressed(path,**out);rec['output_sha256']=sha(path);write_json(path.with_suffix('.json'),rec)
 print('FRAME '+key,flush=True);return out,rec
def safe_run(task):
 try:return run_frame(task)[1]
 except BaseException:STOP.write_text(traceback.format_exc());raise
def main():
 if sys.argv[1]=='pilot':
  first,r1=run_frame(('C3',100,0));second,r2=run_frame(('C3',100,0),save=False)
  exact={k:bool(first[k].dtype==second[k].dtype and first[k].shape==second[k].shape and np.ascontiguousarray(first[k]).tobytes()==np.ascontiguousarray(second[k]).tobytes()) for k in first}
  repeat=dict(passed=all(exact.values()),frame='C3_0100_0000',fields_bitwise_equal=exact,repeat_wall_seconds=r2['wall_seconds'],repeat_cpu_seconds=r2['cpu_seconds'])
  write_json(ROOT/'repeatability.json',repeat);assert repeat['passed'],repeat
  _,r3=run_frame(('cal',67,0));write_json(ROOT/'pilot_passed.json',dict(passed=True,frames=['C3_0100_0000','cal_0067_0000']))
 else:
  assert json.loads((ROOT/'pilot_passed.json').read_text())['passed'];plan=json.loads((ROOT/'protocol.json').read_text())
  tasks=[(split,s,f) for split in ['cal','C3','P2'] for s in plan['cal_scenes' if split=='cal' else 'test_scenes'] for f in plan[split+'_frames']]
  tasks=[t for t in tasks if not (ROOT/f'{t[0]}_{t[1]:04d}_{t[2]:04d}.json').exists()];workers=int(sys.argv[1]);todo=iter(tasks)
  with ProcessPoolExecutor(workers) as pool:
   pending={pool.submit(safe_run,t) for t in [next(todo,None) for _ in range(workers)] if t is not None}
   while pending:
    done,pending=wait(pending,return_when=FIRST_COMPLETED)
    for f in done:f.result()
    assert not STOP.exists()
    for _ in done:
     task=next(todo,None)
     if task is not None:pending.add(pool.submit(safe_run,task))
if __name__=='__main__':
 try:main()
 except BaseException:STOP.write_text(traceback.format_exc());raise
