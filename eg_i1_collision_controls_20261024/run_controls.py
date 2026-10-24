import sys,json,time,hashlib,inspect,importlib.util
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from PIL import Image
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'eg_pose_quality_diagnosis_v1_20261022'
PREV=ROOT.parent/'eg_i1_angle_physics_20261024'
R0=ROOT.parent/'eg_pcr_r0_20261024'
sys.path.insert(0,str(OLD));import evaluate as E
sys.path.insert(0,str(R0));from pcr import sample
EG=ROOT.parent/'eg_full_repro_v1_20261015/runtime/EconomicGrasp'
def module(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
CD=module('official_cd',EG/'utils/collision_detector.py')
DATA=module('official_data',EG/'utils/data_utils.py')
torch.set_num_threads(1)
SCENES=[100,106,115,124,132,139,150,157,160,169,176,186]
FRAMES=[0,64,128,192]

def status(mu,c,e):
 if e:return 'empty'
 if c:return 'collision'
 if not np.isfinite(mu):return 'unknown'
 if 0<mu<=.4:return 'feasible'
 if .4<mu<=1.2:return 'weak'
 return 'no_FC'

def choose(risk,valid,k):
 valid=valid & np.isfinite(risk)
 if not valid.any():return int(k),True
 options=np.flatnonzero(valid);v=risk[options];tie=options[np.isclose(v,v.min(),rtol=0,atol=1e-7)]
 dist=np.minimum((tie-k)%12,(k-tie)%12)
 return int(k if k in tie else tie[np.lexsort((tie,dist))][0]),False

def detect(detector,rows):
 mask=[];parts=[]
 for begin in range(0,len(rows),32):
  c,v=detector.detect(E.GraspGroup(rows[begin:begin+32]),approach_dist=.05,collision_thresh=.01,return_ious=True)
  mask.append(c);parts.append(np.stack(v,axis=1))
 return np.concatenate(mask),np.concatenate(parts)

def run(task):
 s,f=task;dest=ROOT/'results';dest.mkdir(exist_ok=True);out=dest/f'{s:04d}_{f:04d}.npz'
 if out.exists():return dict(scene=s,frame=f,cached=True)
 start=time.time();cpu=time.process_time()
 vp=PREV/f'evaluation/{s:04d}_{f:04d}.npz';cp=OLD/f'cache/C3_{s:04d}_{f:04d}.npz';ep=OLD/f'evaluation/C3_{s:04d}_{f:04d}.npz'
 v=np.load(vp);a=np.load(cp);e=np.load(ep);ids=v['candidate_id'];n=len(ids);poses=v['pred'].reshape(-1,17)
 base=E.D/f'scenes/scene_{s:04d}/kinect';meta=loadmat(base/f'meta/{f:04d}.mat');k=meta['intrinsic_matrix'];depth=np.array(Image.open(base/f'depth/{f:04d}.png'));seg=np.array(Image.open(base/f'label/{f:04d}.png'))
 cam=DATA.CameraInfo(depth.shape[1],depth.shape[0],k[0,0],k[1,1],k[0,2],k[1,2],meta['factor_depth'])
 cloud=DATA.create_point_cloud_from_depth_image(depth,cam,organized=True)
 trans=np.load(base/'cam0_wrt_table.npy')@np.load(base/'camera_poses.npy')[f]
 workspace=(depth>0)&DATA.get_workspace_mask(cloud,seg,trans=trans,organized=True,outlier=.02)
 detector=CD.ModelFreeCollisionDetector(cloud[workspace],voxel_size=.005)
 replay,replay_parts=detect(detector,a['pred']);assert np.array_equal(replay,a['cd_mask']),(s,f,'original CD mismatch')
 cd,ious=detect(detector,poses);cd=cd.reshape(n,12);ious=ious.reshape(n,12,5)
 nominal=v['original_angle'];assert np.array_equal(cd[np.arange(n),nominal],a['cd_mask'][ids])
 RA=np.full(n*12,np.nan);RC=RA.copy();RS=RA.copy();outside=np.zeros(n*12,bool);lefts=RA.copy();rights=RA.copy()
 obj=np.repeat(v['object_id'],12);objects=meta['cls_indexes'].reshape(-1)
 sdf_manifest=[]
 for o in np.unique(obj):
  ix=np.flatnonzero(obj==o);path=R0/f'sdf/{o:03d}/grid_sampled_sdf.npz';grid=np.load(path)
  pose=torch.tensor(meta['poses'][:,:,np.flatnonzero(objects==o+1)[0]],dtype=torch.float32)
  p=torch.tensor(poses[ix],dtype=torch.float32);rot=p[:,4:13].reshape(-1,3,3);width=p[:,1];d=p[:,3]
  left=p[:,13:16]+rot[:,:,0]*(d-.002)[:,None]-rot[:,:,1]*width[:,None]/2
  right=p[:,13:16]+rot[:,:,0]*(d-.002)[:,None]+rot[:,:,1]*width[:,None]/2
  lp=((left-pose[:,3])@pose[:,:3]).requires_grad_();rp=((right-pose[:,3])@pose[:,:3]).requires_grad_()
  ls,ln,lo=sample(lp,grid);rs,rn,ro=sample(rp,grid);ln=ln@pose[:,:3].T;rn=rn@pose[:,:3].T
  ra=.5*(1-F.cosine_similarity(right-left,rn,dim=1))+.5*(1-F.cosine_similarity(left-right,ln,dim=1))
  rc=F.relu(.005-ls)+F.relu(.005-rs);ss=F.relu(ls-.02)+F.relu(rs-.02)
  RA[ix]=ra.detach().numpy();RC[ix]=rc.detach().numpy();RS[ix]=ss.detach().numpy();outside[ix]=(lo|ro).numpy();lefts[ix]=ls.detach().numpy();rights[ix]=rs.detach().numpy()
  sdf_manifest.append(dict(object_id=int(o),sha256=E.sha(path)))
 RA=RA.reshape(n,12);RC=RC.reshape(n,12);RS=RS.reshape(n,12);outside=outside.reshape(n,12)
 oldpcr={int(r['candidate_id']):r for r in json.loads((R0/f'pcr/{s:04d}_{f:04d}.json').read_text())['rows']}
 replay_error={term:float(max(abs(arr[j,nominal[j]]-oldpcr[int(cid)][term]) for j,cid in enumerate(ids))) for term,arr in [('RA',RA),('RC',RC),('RS',RS)]}
 assert max(replay_error.values())<1e-5,(s,f,'PCR replay',replay_error)
 choices=np.zeros((n,7),int);fallback=np.zeros((n,7),bool);groups=list('OABCDEF');I=v['I1'];combined=I+RC+RS;full=RA+RC+RS
 for j,k in enumerate(nominal):
  definitions=[(np.zeros(12),np.eye(12,dtype=bool)[k]),(I[j],np.ones(12,bool)),(I[j],~cd[j]),(combined[j],np.ones(12,bool)),(combined[j],~cd[j]),(I[j],~v['collision'][j]&~v['empty'][j]),(full[j],np.ones(12,bool))]
  for g,(risk,valid) in enumerate(definitions):choices[j,g],fallback[j,g]=choose(risk,valid,int(k))
 # Full original raw candidate pool, including currently unlabelled IDs.
 post=[];fixed=[]
 for gi,group in enumerate(groups):
  pred=a['pred'].astype(np.float64).copy();pred[ids]=v['pred'][np.arange(n),choices[:,gi]]
  assert np.array_equal(pred[:,[0,1,2,3,13,14,15,16]],a['pred'][:,[0,1,2,3,13,14,15,16]])
  m=a['cd_mask'].copy();m[ids]=cd[np.arange(n),choices[:,gi]]
  owner=e['owner'];ni,oi,final=E.postprocess(pred,m,owner)
  if group=='O':assert np.array_equal(final,e['final_ids']),(s,f,'baseline postprocess')
  labels={int(cid):j for j,cid in enumerate(ids)}
  def label(cid):
   if cid in labels:
    j=labels[cid];k=choices[j,gi];mu=float(v['mu'][j,k]);c=bool(v['collision'][j,k]);empty=bool(v['empty'][j,k]);src='existing angle evaluation'
   else:
    mu=float(e['P0_mu'][cid]);c=bool(e['P0_collision'][cid]);empty=bool(e['P0_empty'][cid]);src='existing original evaluation'
   return dict(candidate_id=cid,mu=mu,collision=c,empty=empty,status=status(mu,c,empty),source=src)
  fixed.append(dict(group=group,**label(int(e['final_ids'][0]))))
  post.append(dict(group=group,raw_count=len(pred),post_cd=int((~m).sum()),nms_ids=ni.tolist(),object10_ids=oi.tolist(),final_ids=final.tolist(),final_results=[label(int(cid)) for cid in final],top1=label(int(final[0])) if len(final) else dict(status='no_candidate')))
 np.savez_compressed(out,candidate_id=ids,original_angle=nominal,object_id=v['object_id'],EG_cd=cd,EG_ious=ious,RA=RA,RC=RC,RS=RS,outside_grid=outside,sdf_left=lefts.reshape(n,12),sdf_right=rights.reshape(n,12),groups=np.array(groups),selected_angle=choices,fallback=fallback)
 receipt=dict(scene=s,frame=f,n=n,variants=n*12,CD_replay_exact=True,postprocess_O_exact=True,PCR_replay_error=replay_error,prior_variants_sha256=E.sha(vp),original_cache_sha256=E.sha(cp),original_evaluation_sha256=E.sha(ep),sdf=sdf_manifest,result_sha256=E.sha(out),raw_valid_points=int((depth>0).sum()),workspace_points=int(workspace.sum()),voxel_points=len(detector.scene_points),wall_seconds=time.time()-start,cpu_seconds=time.process_time()-cpu,postprocess=post,fixed_top1=fixed)
 out.with_suffix('.json').write_text(json.dumps(receipt));print(s,f,n,round(time.time()-start,2),flush=True);return {k:receipt[k] for k in ['scene','frame','n','wall_seconds']}

if __name__=='__main__':
 if sys.argv[1]=='pilot':run((100,0))
 else:
  tasks=[(s,f) for s in SCENES for f in FRAMES][int(sys.argv[1])::3]
  with ProcessPoolExecutor(int(sys.argv[2])) as pool:list(pool.map(run,tasks))
