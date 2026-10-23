"""Bounded direct-pose evaluation of pre-registered fixed-ID interventions."""
import os,sys,json,time,hashlib,concurrent.futures
from pathlib import Path
import numpy as np
B=Path.home()/'work/grasp';R=B/'eg_module_controlled_diagnosis_20261023';OLD=B/'eg_pose_quality_diagnosis_v1_20261022';SRC=B/'eg_full_repro_v1_20261015/runtime/EconomicGrasp';D=B/'eg_sub25_v1_20261015/data/graspnet'
args=sys.argv[1:];sys.path[:0]=[str(OLD),str(SRC)];sys.argv=['eval','--dataset_root',str(D),'--camera','kinect','--batch_size','1']
from evaluate import Direct,postprocess,rx,sha
from graspnetAPI import GraspGroup
from utils.collision_detector import ModelFreeCollisionDetector
from PIL import Image
from scipy.io import loadmat
from utils.data_utils import CameraInfo,create_point_cloud_from_depth_image,get_workspace_mask
P=json.loads((R/'protocol.json').read_text())
def rawcloud(s,f):
 p=D/f'scenes/scene_{s:04d}/kinect';depth=np.array(Image.open(p/f'depth/{f:04d}.png'));seg=np.array(Image.open(p/f'label/{f:04d}.png'));m=loadmat(p/f'meta/{f:04d}.mat');intr=m['intrinsic_matrix'];camera=CameraInfo(1280.,720.,intr[0][0],intr[1][1],intr[0][2],intr[1][2],m['factor_depth'])
 cloud=create_point_cloud_from_depth_image(depth,camera,organized=True);cam=np.load(p/'camera_poses.npy');align=np.load(p/'cam0_wrt_table.npy');mask=(depth>0)&get_workspace_mask(cloud,seg,trans=align@cam[f],organized=True,outlier=.02);return cloud[mask]
def job(t):
 split,s,f=t;tag=f'{split}_{s:04d}_{f:04d}';dest=R/f'evaluation/{tag}.npz'
 if dest.exists():return {'frame':tag,'cached':True}
 start=time.time();cp=time.process_time();a=dict(np.load(OLD/f'cache/{tag}.npz'));e=dict(np.load(OLD/f'evaluation/{tag}.npz'));z=np.load(R/f'cache/{tag}.npz');direct=Direct(s);direct.frame(f);ids=z['mechanism_ids'];out={'mechanism_ids':ids};calls=0
 for variant in ['H0','H1_half','H1_off','H2_half','H2_off','H3','H4','N']+(['E1'] if split=='train' else []):
  if variant.startswith('H1') or variant=='H0':result={k:e['P0_'+k][ids] for k in ['mu','collision','empty','owner','object_id','unknown_reason']}
  else:
   pred=z[variant+'_pred'];assert np.array_equal(pred[:,13:16],a['pred'][:,13:16]);assert np.max(abs(pred[:,4:13].reshape(-1,3,3)[:,:,0]-a['pred'][:,4:13].reshape(-1,3,3)[:,:,0]))<2e-6
   result=direct.score(pred[ids]);calls+=len(ids)
  for k,v in result.items():out[variant+'_'+k]=v
 if split=='train':
  gi=z['grid_ids'];base=a['pred'][gi];gr=np.repeat(base,48,axis=0).astype(float);aa=np.tile(np.repeat(np.arange(12),4),len(gi));dd=np.tile(np.arange(4),12*len(gi));rb=np.repeat(a['view_rotation'][gi].astype(float),48,axis=0);gr[:,4:13]=(rb@rx(aa*np.pi/12)).reshape(-1,9);gr[:,3]=(dd+1)*.01
  res=direct.score(gr);calls+=len(gr);out['grid_ids']=gi;out['grid_angle']=aa;out['grid_depth']=dd
  for k,v in res.items():out['grid_'+k]=v
  pos=np.arange(len(gi))*48+a['angle_index'][gi]*4+a['depth_index'][gi]
  for k in ['mu','collision','empty']:
   assert np.array_equal(res[k][pos],e['P0_'+k][gi]),('grid_baseline',tag,k,res[k][pos],e['P0_'+k][gi])
  out['grid_baseline_positions']=pos
 else:
  detector=ModelFreeCollisionDetector(rawcloud(s,f),voxel_size=.005)
  for variant in ['H1_off','H2_off']:
   pred=z[variant+'_pred'];cd=detector.detect(GraspGroup(pred.copy()),approach_dist=.05,collision_thresh=.01) if variant=='H2_off' else a['cd_mask']
   owner=direct.owners(pred);ni,oi,fi=postprocess(pred,cd,owner);out['aux_'+variant+'_ids']=fi;out['aux_'+variant+'_nms_ids']=ni;out['aux_'+variant+'_obj10_ids']=oi;out['aux_'+variant+'_cd']=cd
   result=direct.score(pred[fi]);calls+=len(fi)
   for k,v in result.items():out['aux_'+variant+'_'+k]=v
 cap=128*8+16*48 if split=='train' else 50*7
 assert calls<=cap,(calls,cap)
 dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest,**out)
 meta=dict(frame=tag,calls=calls,unknown=sum(int(np.isnan(v).sum()) for k,v in out.items() if k.endswith('_mu')),cpu_seconds=time.process_time()-cp,seconds=time.time()-start,cache_sha256=sha(R/f'cache/{tag}.npz'),result_sha256=sha(dest),grid_baseline_verified=split=='train')
 dest.with_suffix('.json').write_text(json.dumps(meta,indent=2));print('DONE',json.dumps(meta),flush=True);return meta
if __name__=='__main__':
 pairs=[('train',s,f) for s in P['train_probe_scenes'] for f in P['train_probe_frames']]+[('C3',s,f) for s in P['c3_scenes'] for f in P['c3_frames']]
 if args[0]=='pilot':pairs=pairs[:2];workers=2
 else:shard,nshard,workers=map(int,args);pairs=[t for i,t in enumerate(pairs) if i%nshard==shard]
 with concurrent.futures.ProcessPoolExecutor(workers) as ex:
  for result in ex.map(job,pairs):pass
 (R/('eval_'+args[0]+'_complete.json')).write_text(json.dumps({'frames':len(pairs),'complete':True}));print('COMPLETE',args,flush=True)
