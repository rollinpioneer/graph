import sys,json,time,importlib.util
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from PIL import Image
ROOT=Path.home()/'work/grasp/eg_pcr_r0_20261024';OLD=ROOT.parent/'eg_pose_quality_diagnosis_v1_20261022'
sys.path.insert(0,str(OLD));import evaluate as E
p=ROOT.parent/'eg_full_repro_v1_20261015/runtime/EconomicGrasp/utils/collision_detector.py'
sp=importlib.util.spec_from_file_location('eg_cd',p);mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
sp2=importlib.util.spec_from_file_location('eg_data',p.with_name('data_utils.py'));data=importlib.util.module_from_spec(sp2);sp2.loader.exec_module(data)

def run(s,f):
 dest=ROOT/'rng_evaluation';dest.mkdir(exist_ok=True)
 pred=np.load(ROOT/f'rng_predictions/scene_{s:04d}/kinect/{f:04d}.npy');a=np.load(OLD/f'cache/C3_{s:04d}_{f:04d}.npz')
 direct=E.Direct(s);direct.frame(f);owner=direct.owners(pred)
 base=E.D/f'scenes/scene_{s:04d}/kinect';meta=loadmat(base/f'meta/{f:04d}.mat');k=meta['intrinsic_matrix'];depth=np.array(Image.open(base/f'depth/{f:04d}.png'));seg=np.array(Image.open(base/f'label/{f:04d}.png'))
 camera=data.CameraInfo(1280,720,k[0,0],k[1,1],k[0,2],k[1,2],meta['factor_depth']);cloud=data.create_point_cloud_from_depth_image(depth,camera,organized=True)
 transform=np.load(base/'cam0_wrt_table.npy')@np.load(base/'camera_poses.npy')[f];mask=(depth>0)&data.get_workspace_mask(cloud,seg,trans=transform,organized=True,outlier=.02)
 detector=mod.ModelFreeCollisionDetector(cloud[mask],voxel_size=.005)
 replay=detector.detect(E.GraspGroup(a['pred']),approach_dist=.05,collision_thresh=.01);assert np.array_equal(replay,a['cd_mask'])
 cd=detector.detect(E.GraspGroup(pred),approach_dist=.05,collision_thresh=.01)
 stages={};keep=[]
 for name,apply in [('native',False),('unified',True)]:
  nms,obj,final=E.postprocess(pred,cd,owner,apply_cd=apply);stages[name]=dict(nms=nms.tolist(),obj10=obj.tolist(),final=final.tolist());keep.extend(final.tolist())
 ids=np.array(sorted(set(keep)),int);r=direct.score(pred[ids]);np.savez_compressed(dest/f'{s:04d}_{f:04d}.npz',pred=pred,candidate_id=ids,cd_mask=cd,**r)
 (dest/f'{s:04d}_{f:04d}.json').write_text(json.dumps(dict(scene=s,frame=f,raw_count=len(pred),stages=stages,native='official test_graspnet.sh has collision_detect disabled; official analytic NMS/obj10/top50 evaluation',unified='same EG raw workspace cloud and visible CD voxel=.005; NMS=.03m,pi/6; obj10/top50',eg_cd_replay_exact=True,unknown=int(np.sum(r['unknown_reason']!='')))))
 print(s,f,len(pred),len(ids),flush=True)
if __name__=='__main__':
 plan=json.loads((OLD/'study_plan.json').read_text());tasks=[(s,f) for s in plan['C3_scenes'] for f in plan['C3_frames']]
 if sys.argv[1]=='pilot':run(*tasks[0])
 else:
  for s,f in tasks[int(sys.argv[1])::3]:run(s,f)
