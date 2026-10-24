"""N0 CPU data preparation and observation-only detectors. No optimization/training."""
import sys,json,time,hashlib,importlib.util
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
from scipy.io import loadmat
from PIL import Image
R=Path(__file__).resolve().parent;B=R.parent
OLD=B/'eg_pose_quality_diagnosis_v1_20261022';ANGLE=B/'eg_i1_angle_physics_20261024';CONTROL=B/'eg_i1_collision_controls_20261024'
sys.path.insert(0,str(OLD));import evaluate as E
from proj_sdf_ref import query,classify,gripper_samples,box_samples
def module(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
EG=B/'eg_full_repro_v1_20261015/runtime/EconomicGrasp'
CD=module('n0_cd',EG/'utils/collision_detector.py')
DATA=module('n0_data',EG/'utils/data_utils.py')
PLAN=json.loads((R/'protocol.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tasks(split):
 return [(split,s,f) for s in PLAN['training_scenes' if split=='train' else 'C3_scenes'] for f in PLAN['training_frames' if split=='train' else 'P2_frames' if split=='P2' else 'C3_frames']]
def get_inputs(s,f):
 p=E.D/f'scenes/scene_{s:04d}/kinect';m=loadmat(p/f'meta/{f:04d}.mat');K=m['intrinsic_matrix'];raw=np.asarray(Image.open(p/f'depth/{f:04d}.png'))
 depth=raw.astype(np.float32)/float(m['factor_depth'].item());yy,xx=np.indices(raw.shape)
 cloud=np.stack([(xx-K[0,2])*depth/K[0,0],(yy-K[1,2])*depth/K[1,1],depth],-1)
 return p,m,K,depth,cloud
def solid_masks(t,w,d,margin=0.):
 z=(t[:,2]>-.01+margin)&(t[:,2]<.01-margin);x=(t[:,0]>-.02+margin)&(t[:,0]<d-margin)
 l=z&x&(t[:,1]>-w/2-.01+margin)&(t[:,1]<-w/2-margin)
 r=z&x&(t[:,1]>w/2+margin)&(t[:,1]<w/2+.01-margin)
 b=z&(t[:,0]>-.03+margin)&(t[:,0]<-.02-margin)&(t[:,1]>-w/2-.01+margin)&(t[:,1]<w/2+.01-margin)
 return l|r|b
def official_labels(direct,rows,owner):
 c=np.zeros(len(rows),bool);e=c.copy()
 for o in np.unique(owner):
  ids=np.flatnonzero(owner==o)
  for ix in np.array_split(ids,max(1,int(np.ceil(len(ids)/24)))):
   gl=[np.empty((0,17)) for _ in direct.mt];gl[o]=rows[ix]
   cc,ee=E.EU.collision_detection(gl,direct.mt,direct.dex,direct.poses,direct.allpts,outlier=.05,empty_thresh=10)
   c[ix]=cc[o];e[ix]=ee[o]
 return c,e
def eg_detect(detector,rows):
 masks=[];ious=[]
 for k in range(0,len(rows),24):
  c,p=detector.detect(E.GraspGroup(rows[k:k+24]),approach_dist=.05,collision_thresh=.01,return_ious=True)
  masks.append(c);ious.append(np.stack(p,1))
 return np.concatenate(masks),np.concatenate(ious)
def surfaces(row):
 w,d=row[1],row[3];parts=[]
 for lo,hi in [([-.02,-w/2-.01,-.01],[d,-w/2,.01]),([-.02,w/2,-.01],[d,w/2+.01,.01]),([-.03,-w/2-.01,-.01],[-.02,w/2+.01,.01])]:
  for axis in range(3):
   other=[i for i in range(3) if i!=axis];a=[np.linspace(lo[i],hi[i],int(np.ceil((hi[i]-lo[i])/.0025))+1) for i in other]
   uv=np.stack(np.meshgrid(*a,indexing='ij'),-1).reshape(-1,2)
   for value in [lo[axis],hi[axis]]:
    q=np.zeros((len(uv),3));q[:,other]=uv;q[:,axis]=value;parts.append(q)
 return np.concatenate(parts)@row[4:13].reshape(3,3).T+row[13:16]
def scores(rows,depth,K,full_detector):
 start=time.perf_counter();cd,ratios=eg_detect(full_detector,rows);obs=full_detector.scene_points;tree=cKDTree(obs)
 w=rows[:,1];vol=np.stack([np.full(len(rows),.02*.06*.01),np.full(len(rows),.02*.06*.01),.02*(w+.02)*.01,.02*(w+.02)*.05],1)/.005**3
 count=np.rint(ratios[:,1:]*(vol+1e-6));result=dict(full_EG=cd,R1=ratios[:,0],R2=count.max(1),R3=np.zeros(len(rows)),R4=np.zeros(len(rows)),ref_occ=np.zeros(len(rows)),ref_unk=np.zeros(len(rows)))
 for i,row in enumerate(rows):
  local=obs[tree.query_ball_point(row[13:16],.16)]
  t=(local-row[13:16])@row[4:13].reshape(3,3);result['R3'][i]=solid_masks(t,row[1],row[3]).sum()
  result['R4'][i]=-tree.query(surfaces(row))[0].min()
  pts=gripper_samples(row);phi,weight=query(pts,depth,K);st=classify(phi,weight)
  result['ref_occ'][i]=np.mean(st==1);result['ref_unk'][i]=np.mean(st==2)
 result['detector_seconds']=np.array(time.perf_counter()-start)
 return result
def run(task):
 split,s,f=task;dest=R/'cpu';dest.mkdir(exist_ok=True);out=dest/f'{split}_{s:04d}_{f:04d}.npz'
 if out.exists():return
 start=time.time();p,m,K,depth,cloud=get_inputs(s,f)
 cp=(OLD if split=='C3' else R)/f'cache/{split}_{s:04d}_{f:04d}.npz'
 a=np.load(cp);raw=a['pred'].astype(float);direct=E.Direct(s);direct.frame(f)
 if split=='C3':
  ev=np.load(OLD/f'evaluation/C3_{s:04d}_{f:04d}.npz');v=np.load(ANGLE/f'evaluation/{s:04d}_{f:04d}.npz');control=np.load(CONTROL/f'results/{s:04d}_{f:04d}.npz')
  owner=ev['owner'];ids=v['candidate_id'];nom=v['original_angle'];poses=v['pred'];coll=v['collision'];empty=v['empty'];eg=control['EG_cd'];mu=v['mu']
  assert np.array_equal(control['candidate_id'],ids)
  # Directly replay a fixed small slice of historical official collision labels.
  ri=np.arange(min(24,len(poses.reshape(-1,17))));cc,ee=official_labels(direct,poses.reshape(-1,17)[ri],np.repeat(owner[ids],12)[ri])
  assert np.array_equal(cc,coll.ravel()[ri]) and np.array_equal(ee,empty.ravel()[ri])
 else:
  owner=direct.owners(raw);_,_,ids=E.postprocess(raw,a['cd_mask'],owner);nom=a['angle_index'][ids];poses=np.repeat(raw[ids,None,:],12,axis=1)
  for j,k in enumerate(nom):
   for ang in range(12):
    if ang!=k:poses[j,ang,4:13]=(raw[ids[j],4:13].reshape(3,3)@E.rx(np.array([(ang-k)*np.pi/12]))[0]).ravel()
  coll,empty=official_labels(direct,poses.reshape(-1,17),np.repeat(owner[ids],12));coll=coll.reshape(-1,12);empty=empty.reshape(-1,12);mu=np.full(coll.shape,np.nan)
  # Original EG workspace mask is a baseline only; not an input to reference detectors.
  seg=np.asarray(Image.open(p/f'label/{f:04d}.png'));trans=np.load(p/'cam0_wrt_table.npy')@np.load(p/'camera_poses.npy')[f]
  workspace=(depth>0)&DATA.get_workspace_mask(cloud,seg,trans=trans,organized=True,outlier=.02)
  baseline=CD.ModelFreeCollisionDetector(cloud[workspace],voxel_size=.005)
  replay,_=eg_detect(baseline,raw);assert np.array_equal(replay,a['cd_mask']),(split,s,f,'EG replay')
  eg,_=eg_detect(baseline,poses.reshape(-1,17));eg=eg.reshape(-1,12)
 flat=poses.reshape(-1,17);both=np.concatenate([flat,raw]) if split=='C3' else flat
 detector=CD.ModelFreeCollisionDetector(cloud[depth>0],voxel_size=.005);data=scores(both,depth,K,detector)
 data.update(candidate_id=ids,original_angle=nom,pred=poses,raw=raw,owner=owner,object_id=np.array(direct.ids)[owner[ids]],collision=coll,empty=empty,mu=mu,EG_cd=eg,raw_EG=a['cd_mask'],variant_count=np.array(len(flat)))
 np.savez_compressed(out,**data)
 rec=dict(split=split,scene=s,frame=f,candidates=len(ids),variants=len(flat),scores=len(both),wall_seconds=time.time()-start,detector_seconds=float(data['detector_seconds']),cache_sha256=sha(cp),protocol_sha256=sha(R/'protocol.json'),result_sha256=sha(out),GT_inputs_used_for_detector=False)
 out.with_suffix('.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec),flush=True)
if __name__=='__main__':
 split=sys.argv[1];index=int(sys.argv[2]);workers=int(sys.argv[3]);ts=tasks(split)
 if index==-1:ts=ts[:1]
 else:ts=ts[index::3]
 if split!='train':assert (R/'frozen_config.json').exists(),'Freeze calibration before evaluation'
 with ProcessPoolExecutor(workers) as pool:list(pool.map(run,ts))
