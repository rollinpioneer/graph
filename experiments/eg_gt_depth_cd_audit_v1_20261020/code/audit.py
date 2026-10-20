"""Read-only C3 geometry audit. No inference, quality-label generation or fitting."""
import os,sys,json,time,hashlib
from pathlib import Path
import numpy as np
R=Path(os.environ['R']); BASE=R.parent
DV=BASE/'eg_visibility_dv_v1_20261020'
sys.path.insert(0,str(DV/'code'))
from visibility import Observation, native, export, DATA, C3, SEEDS, sha
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import maximum_filter,minimum_filter,uniform_filter,binary_erosion
from scipy.spatial import cKDTree
CODE=BASE/'eg_sub25_v1_20261015/runtime/EconomicGrasp'
sys.path.insert(0,str(CODE))
from utils.collision_detector import ModelFreeCollisionDetector
from utils.data_utils import CameraInfo,create_point_cloud_from_depth_image,get_workspace_mask
from graspnetAPI import GraspGroup
import open3d as o3d
import pandas as pd

def save(path,obj):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def clean(v):
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,(np.floating,float)):return float(v) if np.isfinite(v) else None
 if isinstance(v,np.integer):return int(v)
 if isinstance(v,np.bool_):return bool(v)
 return v
def prepare():
 d=pd.read_csv(DV/'results/DV1_candidates.csv')
 top=d[d['rank']==1]; fail=top[top.collision & ~top['empty']]
 assert len(fail)==1177
 jobs=[]; pairs=[]
 for s in C3:
  sdf=d[d.scene==s]; ff=fail[fail.scene==s]
  successframes=sorted(top[(top.scene==s)&top.valid].frame.unique())
  controls=[int(successframes[i]) for i in np.unique(np.linspace(0,len(successframes)-1,min(16,len(successframes))).astype(int))]
  frames=sorted(set(ff.frame)|set(controls))
  for f in frames:
   group=sdf[sdf.frame==f]
   events=[]
   for row in ff[ff.frame==f].itertuples():
    choices=group[(group['head']==row.head)&group.valid].sort_values('rank').candidate.astype(int).tolist()
    events.append(dict(head=int(row.head),failure=int(row.candidate),success_choices=choices))
   control=group[(group['rank']==1)&group.valid].sort_values('head').candidate.astype(int).tolist() if f in controls else []
   jobs.append(dict(scene=s,frame=int(f),events=events,controls=sorted(set(control))))
 save(R/'selection.json',dict(rule='all 1177 nonempty collision events, same-head same-frame known Top10 valid controls; 16 evenly spaced successful frames/scene',jobs=jobs,events=1177))
 return jobs

def boxes(p,cd=False):
 w=p[1] if cd else np.clip(p[1],0,.1);h=p[2]/2 if cd else .01;d=p[3]
 x0=d-.06 if cd else -.02
 base0=d-.07 if cd else -.03;base1=d-.06 if cd else -.02
 result=[np.array([[x0,d],[-w/2-.01,-w/2],[-h,h]]),
         np.array([[x0,d],[w/2,w/2+.01],[-h,h]]),
         np.array([[base0,base1],[-w/2-.01,w/2+.01],[-h,h]])]
 if cd:result.append(np.array([[d-.12,d-.07],[-w/2-.01,w/2+.01],[-h,h]]))
 return result
def masks(xyz,p,cd=False):
 out=[]
 for j,b in enumerate(boxes(p,cd)):
  m=(xyz>b[:,0]).all(1)&(xyz<b[:,1]).all(1)
  if cd and j>=2:m=(xyz>b[:,0]).all(1)&(xyz[:,0]<=b[0,1])&(xyz[:,1:]<b[1:,1]).all(1)
  out.append(m)
 return np.stack(out,1)
def med(x):return float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan

def shard(task):
 scene,shard_id,shards,pilot=task; t=time.time(); cpu=time.process_time();os.nice(10)
 jobs=[v for v in json.load(open(R/'selection.json'))['jobs'] if v['scene']==scene][shard_id::shards]
 if pilot:jobs=jobs[:1]
 n=native(scene); dense={}
 for oid in n.ge.get_model_poses(scene,0)[0]:
  path=DATA/f'models/{oid:03d}/nontextured.ply'
  dense[oid]=np.asarray(o3d.io.read_point_cloud(str(path)).points)
  assert len(dense[oid])>0
 source_identity={v['frame']:v for v in json.load(open(DV/f'results/identity_C3_{scene}.json'))['frames']}
 rows=[];pairs=[];objects=[];identities=[];checks=[];completed=0
 for job in jobs:
  if time.time()-t>360:break
  f=job['frame'];n.frame(f);obs=Observation(scene,f);folder=DATA/f'scenes/scene_{scene:04d}/kinect'
  depth=np.asarray(Image.open(folder/f'depth/{f:04d}.png'));seg=np.asarray(Image.open(folder/f'label/{f:04d}.png'))
  meta=loadmat(folder/f'meta/{f:04d}.mat');K=meta['intrinsic_matrix']
  camera=CameraInfo(1280.,720.,K[0,0],K[1,1],K[0,2],K[1,2],meta['factor_depth'])
  cloud=create_point_cloud_from_depth_image(depth,camera,organized=True)
  cam=np.load(folder/'camera_poses.npy')[f];align=np.load(folder/'cam0_wrt_table.npy')
  workspace=get_workspace_mask(cloud,seg,trans=align@cam,organized=True,outlier=.02)
  detector=ModelFreeCollisionDetector(cloud[(depth>0)&workspace],voxel_size=.005)
  depthtree=cKDTree(obs.points[obs.depth>0]);cdtree=cKDTree(detector.scene_points)
  nearhole=uniform_filter((depth==0).astype(float),5)>0
  jump=maximum_filter(obs.depth,3)-minimum_filter(np.where(depth>0,obs.depth,1e6),3)>.01
  edge=maximum_filter(seg,3)!=minimum_filter(seg,3)
  dp=DV/f'cache/c3/{scene:04d}/{f:04d}.npz';cached=np.load(dp);ids=cached['ids']
  pred=np.load(export(scene,f))['pred'];owners=n.owners(pred)
  chosen=set(job['controls']); role={i:'background_success' for i in chosen}
  for ev in job['events']:
   i=ev['failure'];chosen.add(i);role[i]='failure'
   same=[j for j in ev['success_choices'] if owners[j]==owners[i]]
   j=(same or ev['success_choices'] or [None])[0]
   if j is not None:chosen.add(j);role.setdefault(j,'paired_success')
   pairs.append(dict(scene=scene,frame=f,head=ev['head'],failure=i,success=j,tier='same_object' if same else 'other_object' if j is not None else 'unpaired'))
  chosen=sorted(chosen);cdflag,ious=detector.detect(GraspGroup(pred[chosen].copy()),approach_dist=.05,collision_thresh=.01,return_ious=True)
  hp=BASE/f'eg_stat_shape_h_v1_20261020/evaluation/development/{scene:04d}/{f:04d}.npz'
  hc=np.load(hp);assert np.array_equal(cdflag,hc['cd_mask'][chosen]),(scene,f,np.array(chosen)[cdflag!=hc['cd_mask'][chosen]].tolist())
  assert sha(export(scene,f))==source_identity[f]['export_sha256']
  reference=[v for v in source_identity[f]['references'] if v['experiment']=='eg_stat_shape_h_v1_20261020'][0]
  assert sha(hp)==reference['npz_sha256']
  checks.append(dict(frame=f,cd_verified=len(chosen)))
  # Pose cross-check: metadata uses 1-based object IDs; XML/API uses 0-based.
  mid=meta['cls_indexes'].ravel().astype(int)-1;poses=meta['poses']
  poseerr=[]
  for oi,oid in enumerate(n.ids):
   mi=np.flatnonzero(mid==oid);assert len(mi)==1
   poseerr.append(float(np.max(np.abs(n.poses[oi][:3]-poses[:,:,mi[0]]))))
  identities.append(dict(frame=f,export_sha256=sha(export(scene,f)),dv_cache_sha256=sha(dp),quality_cache_sha256=sha(hp),depth_sha256=sha(folder/f'depth/{f:04d}.png'),meta_sha256=sha(folder/f'meta/{f:04d}.mat'),xml_meta_pose_max_abs=max(poseerr),principal_truncation_px=(K[:2,2]-np.floor(K[:2,2])).tolist()))
  for ii,i in enumerate(chosen):
   p=pred[i].astype(float);o=int(owners[i]);obj=n.mt[o];crop=((n.allpts>obj.min(0)-.05)&(n.allpts<obj.max(0)+.05)).all(1)
   globalidx=np.flatnonzero(crop);pts=n.allpts[crop];local=(pts-p[13:16])@p[4:13].reshape(3,3)
   gm=masks(local,p);hit=gm.any(1);xyz=local[hit];gpts=pts[hit];indices=globalidx[hit];ct=len(gpts)
   w=np.clip(p[1],0,.1);inner=(np.abs(local[:,2])<.01)&(local[:,0]>-.02)&(local[:,0]<p[3])&(np.abs(local[:,1])<=w/2)
   pos=np.flatnonzero(ids==i);assert len(pos)==1;pos=pos[0]
   assert bool(ct>0 or inner.sum()<10)==bool(cached['collision'][pos])
   assert bool(inner.sum()<10)==bool(cached['empty'][pos])
   dst,res,u,v=obs.query(gpts);cm=masks(xyz,p,True);covered=cm.any(1)
   uv=np.rint(gpts[:,:2]/np.maximum(gpts[:,2:3],1e-12)*np.array([K[0,0],K[1,1]])+K[:2,2]).astype(int)
   inview=(gpts[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<depth.shape[1])&(uv[:,1]>=0)&(uv[:,1]<depth.shape[0])
   gseg=np.full(ct,-1);objmask=indices<len(n.objpts);gseg[objmask]=n.seg[indices[objmask]]
   target=(gseg==o);other=(gseg>=0)&~target
   boundary=np.full(ct,np.inf)
   for k,b in enumerate(boxes(p)):
    m=gm[hit,k];boundary[m]=np.minimum(boundary[m],np.minimum(xyz[m]-b[:,0],b[:,1]-xyz[m]).min(1))
   visible=np.isfinite(res);free=dst==0
   expected=np.full(ct,0);expected[objmask]=np.array(n.ids)[gseg[objmask]]+1
   r=dict(scene=scene,frame=f,candidate=i,role=role[i],owner_id=int(n.ids[o]),mu=float(cached['mu'][pos]),collision=bool(cached['collision'][pos]),empty=bool(cached['empty'][pos]),gt_points=ct,gt_inner=int(inner.sum()),raw_width=p[1],depth=p[3],cd_iou=float(ious[0][ii]),cd_collision=bool(cdflag[ii]),cd_equivalent_volume=float((2*p[2]*.06*.01+p[2]*(p[1]+.02)*(.01+.05))/.005**3))
   # Direct replay count and original CD strict threshold.
   nearby=detector.scene_points[cdtree.query_ball_point(p[13:16],.3)]
   cmdepth=masks((nearby-p[13:16])@p[4:13].reshape(3,3),p,True)
   r['cd_depth_points']=int(cmdepth.any(1).sum())
   assert abs(r['cd_depth_points']/(r['cd_equivalent_volume']+1e-6)-r['cd_iou'])<1e-7
   for name,m in [('left',gm[hit,0]),('right',gm[hit,1]),('base',gm[hit,2]),('target',target),('other',other),('table',~objmask),('cd_covered',covered),('cd_body',cm[:,:3].any(1)),('cd_sweep',cm[:,3]),('boundary2mm',boundary<=.002),('free',free),('surface',dst==1),('occluded',dst==2),('unknown',dst==3),('inview',inview),('hole_near',nearhole[v,u]&inview),('depth_jump',jump[v,u]&inview),('seg_edge',edge[v,u]&inview),('same_segment',(seg[v,u]==expected)&inview),('workspace',workspace[v,u]&inview)]:
    r[name+'_count']=int(m.sum());r[name+'_fraction']=float(m.mean()) if ct else np.nan
   r['residual_mm_median']=med(res)*1000;r['boundary_mm_median']=med(boundary)*1000
   r['nearest_depth_mm_median']=med(depthtree.query(gpts)[0])*1000 if ct else np.nan
   r['free_smooth_same_segment_count']=int((free&~nearhole[v,u]&~jump[v,u]&~edge[v,u]&(seg[v,u]==expected)).sum())
   # Near depth in a small image neighborhood: descriptive projection-offset sensitivity.
   if ct:
    offsets=[(a,b) for a in range(-2,3) for b in range(-2,3)]
    ds=np.stack([obs.depth[(v+b).clip(0,depth.shape[0]-1),(u+a).clip(0,depth.shape[1]-1)] for a,b in offsets])
    rr=np.where(ds>0,np.abs(ds-gpts[:,2]),np.inf)
    r['within5x5_surface_count']=int(((rr.min(0)<=.004)&inview).sum())
   else:r['within5x5_surface_count']=0
   rows.append(clean(r))
  # Registration context uses dense model surfaces and eroded observed object interiors.
  for o in sorted(set(int(owners[i]) for i in chosen)):
   oid=int(n.ids[o]);mask=binary_erosion(seg==oid+1,iterations=2)&(depth>0)&~jump&~nearhole
   vv,uu=np.where(mask)
   if len(vv)<32:continue
   take=np.linspace(0,len(vv)-1,min(2048,len(vv))).astype(int);vv=vv[take];uu=uu[take]
   observed=obs.points[vv,uu];model=dense[oid]@n.poses[o][:3,:3].T+n.poses[o][:3,3]
   tree=cKDTree(model);dist,idx=tree.query(observed);nearest=model[idx];delta=nearest-observed
   pixel=np.stack([nearest[:,0]/nearest[:,2]*K[0,0]+K[0,2]-uu,nearest[:,1]/nearest[:,2]*K[1,1]+K[1,2]-vv],1)
   objects.append(clean(dict(scene=scene,frame=f,object_id=oid,n=len(take),surface_distance_mm_median=med(dist)*1000,surface_distance_mm_p90=float(np.quantile(dist,.9))*1000,dx_mm=med(delta[:,0])*1000,dy_mm=med(delta[:,1])*1000,dz_mm=med(delta[:,2])*1000,du_px=med(pixel[:,0]),dv_px=med(pixel[:,1]),within4mm=float((dist<=.004).mean()),within10mm=float((dist<=.01).mean()))))
  completed+=1
 out=R/('pilot' if pilot else 'results')/f'shard_{scene}_{shard_id}'
 save(str(out)+'.json',clean(dict(rows=rows,pairs=pairs,objects=objects,identities=identities,checks=checks,wall_seconds=time.time()-t,cpu_seconds=time.process_time()-cpu,frames=completed,planned_frames=len(jobs),status='PASS' if completed==len(jobs) else 'BUDGET_STOP')))
 print('DONE',scene,shard_id,len(jobs),round(time.time()-t,2),flush=True)
 return str(out)+'.json'

if __name__=='__main__':
 import argparse
 from concurrent.futures import ProcessPoolExecutor,as_completed
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--pilot',action='store_true');ap.add_argument('--workers',type=int,default=24);a=ap.parse_args()
 if a.prepare:prepare()
 else:
  tasks=[(s,k,2,a.pilot) for s in (C3[:2] if a.pilot else C3) for k in (range(1) if a.pilot else range(2))]
  start=time.time()
  with ProcessPoolExecutor(max_workers=min(a.workers,len(tasks))) as pool:
   for f in as_completed([pool.submit(shard,t) for t in tasks]):f.result()
  save(R/('pilot_receipt.json' if a.pilot else 'run_receipt.json'),dict(status='PASS',max_workers=min(a.workers,len(tasks)),tasks=len(tasks),wall_seconds=time.time()-start,GPU_hours=0,new_quality_labels=0,training=0))
