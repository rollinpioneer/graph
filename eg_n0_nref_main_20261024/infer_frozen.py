"""Official inference plus read-only label-function return tracing. No training mode."""
import os,sys,json,time,hashlib,random,inspect
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
B=Path.home()/'work/grasp';ROOT=B/'eg_n0_nref_main_20261024'
SRC=B/'eg_full_repro_v1_20261015/runtime/EconomicGrasp';D=B/'eg_sub25_v1_20261015/data/graspnet'
MODE=sys.argv[1];sys.path.insert(0,str(SRC));sys.path.insert(0,str(SRC/'libs/pointnet2'))
sys.argv=['diagnostic','--dataset_root',str(D),'--camera','kinect','--batch_size','1']
from dataset.graspnet_dataset import GraspNetDataset,collate_fn
from models.economicgrasp import economicgrasp,pred_decode
from utils.label_generation import process_grasp_labels
from utils.collision_detector import ModelFreeCollisionDetector
from utils.loss_utils import batch_viewpoint_params_to_matrix
from graspnetAPI import GraspGroup

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def cpu(x):return x.detach().cpu().numpy()
def dataset(s,f):
 ds=object.__new__(GraspNetDataset);ds.root=str(D);ds.camera='kinect';ds.num_points=20000;ds.voxel_size=.005;ds.remove_outlier=True;ds.augment=False;ds.load_label=False
 p=D/f'scenes/scene_{s:04d}/kinect'
 ds.colorpath=[str(p/f'rgb/{f:04d}.png')];ds.depthpath=[str(p/f'depth/{f:04d}.png')];ds.labelpath=[str(p/f'label/{f:04d}.png')];ds.metapath=[str(p/f'meta/{f:04d}.mat')];ds.scenename=[f'scene_{s:04d}'];ds.frameid=[f]
 return ds

plan=json.loads((ROOT/'protocol.json').read_text());torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
ck=B/'eg_full_repro_v1_20261015/downloads/economicgrasp_kinect.tar';cksha=sha(ck)
assert cksha=='a47b717ad41870e9b34e85b1487de2a3f0f134ddc1a71a98a1b55bca7fc5804a'
net=economicgrasp(seed_feat_dim=512,is_training=False).cuda();state=torch.load(ck,map_location='cpu',weights_only=True);net.load_state_dict(state['model_state_dict'],strict=True);net.eval()
assert not net.is_training and not net.view.is_training and not net.training
pairs=[('train',s,f) for s in plan['training_scenes'] for f in plan['training_frames']]
if MODE=='pilot':pairs=[('train',s,f) for s,f in plan['initial_check']]
if MODE=='P2':pairs=[('P2',s,f) for s in plan['C3_scenes'] for f in plan['P2_frames']]
(ROOT/'cache').mkdir(exist_ok=True)
for split,s,f in pairs:
 dest=ROOT/f'cache/{split}_{s:04d}_{f:04d}.npz'
 if dest.exists():print('CACHED',s,f,flush=True);continue
 start=time.time();seed=plan['seed']+s*1000+f;np.random.seed(seed);random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 ds=dataset(s,f);sample=ds.get_data(0);data={k:v.cuda() for k,v in collate_fn([sample]).items()}
 with torch.no_grad():end=net(data);pred=cpu(pred_decode(end)[0])
 assert pred.shape==(1024,17) and np.isfinite(pred).all()
 out=dict(pred=pred,candidate_id=np.arange(1024),input_points=sample['point_clouds'],input_seg=sample['seg'],input_colors=sample['cloud_colors'],logits=cpu(end['grasp_score_pred'])[0].reshape(6,1024).T,angle_logits=cpu(end['grasp_angle_pred'])[0],depth_logits=cpu(end['grasp_depth_pred'])[0],raw_width=cpu(end['grasp_width_pred'])[0].reshape(1024),view_index=cpu(end['grasp_top_view_inds'])[0],view_xyz=cpu(end['grasp_top_view_xyz'])[0],view_rotation=cpu(end['grasp_top_view_rot'])[0],checkpoint_sha256=np.array(cksha),scene=np.array(s),frame=np.array(f),seed=np.array(seed))
 out['angle_index']=out['angle_logits'].reshape(-1,1024)[:12].argmax(0);out['depth_index']=out['depth_logits'].reshape(-1,1024)[:4].argmax(0)
 assert np.allclose(pred[:,3],(out['depth_index']+1)*.01,atol=1e-7)
 assert np.allclose(pred[:,1],np.clip(1.2*out['raw_width']/10,0,.1),atol=1e-7)
 assert np.allclose(pred[:,0],out['logits']@np.arange(6)*.2,atol=3e-6)
 rot=pred[:,4:13].reshape(-1,3,3);assert np.max(abs(rot.transpose(0,2,1)@rot-np.eye(3)))<2e-5
 cloud,_=ds.get_data(0,return_raw_cloud=True)
 detector=ModelFreeCollisionDetector(cloud,voxel_size=.005)
 out['cd_mask']=detector.detect(GraspGroup(pred.copy()),approach_dist=.05,collision_thresh=.01)
 out['geometry_sha256']=np.array(hashlib.sha256(np.ascontiguousarray(pred[:,1:]).tobytes()).hexdigest())
 np.savez_compressed(dest,**out)
 paths=ds.colorpath+ds.depthpath+ds.labelpath+ds.metapath+[str(D/f'scenes/scene_{s:04d}/kinect/annotations/{f:04d}.xml')]
 meta=dict(scene=s,frame=f,split=split,sha256=sha(dest),seconds=time.time()-start,gpu=torch.cuda.get_device_name(),torch=torch.__version__,checkpoint_epoch=state['epoch'],tf32_matmul=True,tf32_cudnn=True,training_mode=False,valid_matches=None,inputs=[dict(path=p,sha256=sha(p)) for p in paths])
 dest.with_suffix('.json').write_text(json.dumps(meta,indent=2));print('FRAME '+json.dumps(meta),flush=True)
 del data,end,out;torch.cuda.empty_cache()
print('COMPLETE',MODE,flush=True)
