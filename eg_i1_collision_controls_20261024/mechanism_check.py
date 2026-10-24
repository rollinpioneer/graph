"""Read-only geometric attribution; never changes policies, labels or thresholds."""
import sys,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.io import loadmat
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import run_controls as C
ROOT=C.ROOT

def gt_masks(t,w,d):
 h=(t[:,2]>-.01)&(t[:,2]<.01);x=(t[:,0]>-.02)&(t[:,0]<d)
 l=h&x&(t[:,1]>-(w/2+.01))&(t[:,1]<-w/2)
 r=h&x&(t[:,1]<(w/2+.01))&(t[:,1]>w/2)
 b=h&(t[:,1]>-(w/2+.01))&(t[:,1]<(w/2+.01))&(t[:,0]>-.03)&(t[:,0]<-.02)
 return l,r,b

def eg_masks(t,w,d):
 h=(t[:,2]>-.01)&(t[:,2]<.01);x=(t[:,0]>d-.06)&(t[:,0]<d)
 l=h&x&(t[:,1]>-(w/2+.01))&(t[:,1]<-w/2)
 r=h&x&(t[:,1]<(w/2+.01))&(t[:,1]>w/2)
 y=(t[:,1]>-(w/2+.01))&(t[:,1]<(w/2+.01))
 b=h&y&(t[:,0]<=d-.06)&(t[:,0]>d-.07)
 sweep=h&y&(t[:,0]<=d-.07)&(t[:,0]>d-.12)
 return l,r,b,sweep

def run(task):
 s,f=task;start=time.time();z=np.load(ROOT/f'results/{s:04d}_{f:04d}.npz');a=np.load(C.PREV/f'evaluation/{s:04d}_{f:04d}.npz');original=np.load(C.OLD/f'evaluation/C3_{s:04d}_{f:04d}.npz')
 dcol=list(z['groups']).index('D');selected=[]
 for j,cid in enumerate(z['candidate_id']):
  k=int(z['selected_angle'][j,dcol])
  if a['collision'][j,k] and not a['empty'][j,k]:selected.append((j,k,'D_residual_collision'))
  for kk in range(12):
   if z['EG_cd'][j,kk] and not a['collision'][j,kk] and not a['empty'][j,kk] and 0<a['mu'][j,kk]<=.4:selected.append((j,kk,'feasible_false_filter'))
 if not selected:rows=[]
 else:
  direct=C.E.Direct(s);direct.frame(f)
  labels=np.concatenate([np.full(len(m),i,int) for i,m in enumerate(direct.mt)]+[np.full(len(direct.table_trans),-1,int)])
  base=C.E.D/f'scenes/scene_{s:04d}/kinect';meta=loadmat(base/f'meta/{f:04d}.mat');K=meta['intrinsic_matrix'];depth=np.array(Image.open(base/f'depth/{f:04d}.png'));seg=np.array(Image.open(base/f'label/{f:04d}.png'))
  cam=C.DATA.CameraInfo(depth.shape[1],depth.shape[0],K[0,0],K[1,1],K[0,2],K[1,2],meta['factor_depth']);cloud=C.DATA.create_point_cloud_from_depth_image(depth,cam,organized=True)
  trans=np.load(base/'cam0_wrt_table.npy')@np.load(base/'camera_poses.npy')[f];mask=(depth>0)&C.DATA.get_workspace_mask(cloud,seg,trans=trans,organized=True,outlier=.02)
  detector=C.CD.ModelFreeCollisionDetector(cloud[mask],voxel_size=.005);obs=detector.scene_points;tree=cKDTree(obs);rows=[]
  for j,k,scope in selected:
   p=a['pred'][j,k];R=p[4:13].reshape(3,3);center=p[13:16];w=p[1];d=p[3];owner=int(original['owner'][int(z['candidate_id'][j])]);model=direct.mt[owner]
   low=model.min(0)-.05;high=model.max(0)+.05;crop=((direct.allpts>low)&(direct.allpts<high)).all(1);gtpts=direct.allpts[crop];glabels=labels[crop];t=(gtpts-center)@R
   gm=gt_masks(t,w,d);gs=np.logical_or.reduce(gm)
   assert bool(gs.any())==bool(a['collision'][j,k]),(s,f,j,k,'GT collision geometry')
   ot=(obs-center)@R;om=gt_masks(ot,w,d);obs_in_crop=((obs>low)&(obs<high)).all(1);os=np.logical_or.reduce(om)&obs_in_crop
   em=eg_masks(ot,w,d);es=np.logical_or.reduce(em)
   # Confirm our diagnostic mask exactly reproduces the official EG occupancy.
   volume=(.02*.06*.01*2+.02*(w+.02)*.01+.02*(w+.02)*.05)/(.005**3)
   giou=es.sum()/(volume+1e-6);assert abs(giou-z['EG_ious'][j,k,0])<1e-8
   fullt=(direct.allpts-center)@R;full_eg=eg_masks(fullt,w,d);full_static=np.logical_or.reduce(full_eg[:3]);full_sweep=full_eg[3]
   nn=tree.query(gtpts[gs])[0] if gs.any() else np.array([])
   row=dict(scene=s,frame=f,candidate_id=int(z['candidate_id'][j]),angle_class=k,object_id=int(a['object_id'][j]),scope=scope,
     gt_collision_points=int(gs.sum()),gt_left=int(gm[0].sum()),gt_right=int(gm[1].sum()),gt_bottom=int(gm[2].sum()),
     gt_assigned_object=int((gs&(glabels==owner)).sum()),gt_other_objects=int((gs&(glabels>=0)&(glabels!=owner)).sum()),gt_table=int((gs&(glabels==-1)).sum()),
     observed_points_in_GT_collision_solid=int(os.sum()),observed_GT_points_also_counted_by_EG=int((os&es).sum()),observed_EG_points=int(es.sum()),EG_global_iou=float(giou),
     full_GT_points_in_EG_static=int(full_static.sum()),full_GT_points_in_EG_sweep=int(full_sweep.sum()),full_GT_points_in_EG_union=int((full_static|full_sweep).sum()),
     gt_collision_nearest_observation_min_m=float(nn.min()) if len(nn) else None,gt_collision_nearest_observation_median_m=float(np.median(nn)) if len(nn) else None,
     RC=float(z['RC'][j,k]),RS=float(z['RS'][j,k]))
   rows.append(row)
 dest=ROOT/'mechanism';dest.mkdir(exist_ok=True);(dest/f'{s:04d}_{f:04d}.json').write_text(json.dumps(dict(scene=s,frame=f,rows=rows,wall_seconds=time.time()-start)))
 print(s,f,len(rows),flush=True)

if __name__=='__main__':
 tasks=[(s,f) for s in C.SCENES for f in C.FRAMES][int(sys.argv[1])::3]
 with ProcessPoolExecutor(4) as pool:list(pool.map(run,tasks))
