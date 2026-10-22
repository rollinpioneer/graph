"""Official inference plus read-only label-function return tracing. No training mode."""
import os,sys,json,time,hashlib,random,inspect
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
B=Path.home()/'work/grasp';ROOT=B/'eg_pose_quality_diagnosis_v1_20261022'
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

def label_probe(s,f,end):
 path=D/f'economic_grasp_label_300views/scene_{s:04d}_labels.npz'
 with np.load(path) as z:lab={k:z[k].copy() for k in z.files}
 meta=sio.loadmat(D/f'scenes/scene_{s:04d}/kinect/meta/{f:04d}.mat');obj=meta['cls_indexes'].reshape(-1).astype(int)-1
 probe={'xyz_graspable':end['xyz_graspable'].clone(),'grasp_top_view_inds':end['grasp_top_view_inds'].clone()}
 names={'grasp_points_list':('points',1),'grasp_rotations_list':('rotations',1),'grasp_depth_list':('depth',1),'grasp_scores_list':('scores',10),'grasp_widths_list':('widths',1000),'view_graspness_list':('vgraspness',1),'top_view_index_list':('topview',1)}
 order=[]
 for key,(field,scale) in names.items():
  vals=[]
  for oi in range(len(obj)):
   ix=np.flatnonzero(lab['pointid']==oi)
   if key=='grasp_points_list':order.extend(ix.tolist())
   value=lab[field][ix].astype(np.float32)/scale
   if field in ['rotations','depth','topview']:value=value.astype(np.int32)
   vals.append(torch.from_numpy(value).cuda())
  probe[key]=[vals]
 probe['object_poses_list']=[[torch.from_numpy(meta['poses'][:,:,i].astype(np.float32)).cuda() for i in range(len(obj))]]
 trace={}
 def profile(frame,event,arg):
  if event=='return' and frame.f_code is process_grasp_labels.__code__:
   for k in ['nn_inds','pid','vid','valid_point_mask','valid_view_mask','dist','top_grasp_views_rot']:
    trace[k]=cpu(frame.f_locals[k])
 old_tf32=torch.backends.cuda.matmul.allow_tf32
 torch.backends.cuda.matmul.allow_tf32=False
 sys.setprofile(profile)
 try:
  with torch.no_grad():_,probe=process_grasp_labels(probe)
 finally:
  sys.setprofile(None)
  torch.backends.cuda.matmul.allow_tf32=old_tf32
 out={k:cpu(probe[k])[0] for k in ['batch_grasp_point','batch_grasp_rotations','batch_grasp_depth','batch_grasp_score','batch_grasp_width','batch_valid_mask']}
 out.update({k:v for k,v in trace.items() if k not in ['pid','vid']})
 ix=np.array(order)[trace['nn_inds']];slot=np.full(1024,-1,int);slot[trace['pid']]=trace['vid']
 out['label_point_index']=ix;out['label_slot']=slot;out['matched_object_id']=obj[lab['pointid'][ix]];out['label_object_point']=lab['points'][ix]
 out['label_source_view']=np.where(slot>=0,lab['topview'][ix,np.maximum(slot,0)],-1)
 out['label_sha256']=np.array(sha(path));out['object_ids_meta']=obj
 # Caller retains the original network outputs; label tracing never alters them.
 out['view_rotation_error_deg']=np.rad2deg(np.arccos(np.clip((out['top_grasp_views_rot'][:,:,0]*cpu(end['grasp_top_view_rot'])[0,:,:,0]).sum(1),-1,1)))
 return out

plan=json.loads((ROOT/'study_plan.json').read_text());torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
ck=B/'eg_full_repro_v1_20261015/downloads/economicgrasp_kinect.tar';cksha=sha(ck)
assert cksha=='a47b717ad41870e9b34e85b1487de2a3f0f134ddc1a71a98a1b55bca7fc5804a'
net=economicgrasp(seed_feat_dim=512,is_training=False).cuda();state=torch.load(ck,map_location='cpu',weights_only=True);net.load_state_dict(state['model_state_dict'],strict=True);net.eval()
assert not net.is_training and not net.view.is_training and not net.training
pairs=[('C3',s,f) for s in plan['C3_scenes'] for f in plan['C3_frames']]+[('train',s,f) for s in plan['training_scenes'] for f in plan['training_frames']]
if MODE=='pilot':pairs=[('C3',100,0),('train',plan['training_scenes'][0],0)]
if MODE=='extension':pairs=[('train',s,f) for s in plan['training_scenes'] for f in [64,192]]
if MODE=='reprobe':pairs=[('train',s,f) for s in plan['training_scenes'] for f in plan['training_frames']]
(ROOT/'cache').mkdir(exist_ok=True)
for split,s,f in pairs:
 dest=ROOT/f'cache/{split}_{s:04d}_{f:04d}.npz'
 if MODE=='reprobe':
  with np.load(dest) as z:out={k:z[k].copy() for k in z.files}
  end={'xyz_graspable':torch.from_numpy(out['pred'][:,13:16].copy()).cuda()[None],
       'grasp_top_view_inds':torch.from_numpy(out['view_index']).cuda()[None],
       'grasp_top_view_rot':torch.from_numpy(out['view_rotation']).cuda()[None]}
  old_valid=out['batch_valid_mask'].copy();old_target=out['batch_grasp_score'].copy();old_point=out['label_point_index'].copy()
  out['probe_tf32_valid_mask']=old_valid;out['probe_tf32_label_point_index']=old_point;out['probe_tf32_score_target']=old_target
  geom=str(out['geometry_sha256']);out.update(label_probe(s,f,end))
  assert geom==hashlib.sha256(np.ascontiguousarray(out['pred'][:,1:]).tobytes()).hexdigest()
  np.savez_compressed(dest,**out);receipt=json.loads(dest.with_suffix('.json').read_text());receipt['sha256']=sha(dest)
  receipt['label_probe_matmul_tf32']=False;receipt['probe_valid_mask_changes']=int((old_valid!=out['batch_valid_mask']).sum());receipt['probe_target_changes']=int((old_target!=out['batch_grasp_score']).sum());receipt['probe_matched_point_changes']=int((old_point!=out['label_point_index']).sum());receipt['valid_matches']=int(out['batch_valid_mask'].sum())
  receipt['reference_rotation_orthogonality_error']=float(abs(out['top_grasp_views_rot'].transpose(0,2,1)@out['top_grasp_views_rot']-np.eye(3)).max());dest.with_suffix('.json').write_text(json.dumps(receipt,indent=2))
  print('REPROBE '+json.dumps({k:v for k,v in receipt.items() if k!='inputs'}),flush=True);continue
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
 if split=='train':
  out.update(label_probe(s,f,end))
  with torch.no_grad():assert np.array_equal(pred,cpu(pred_decode(end)[0]))
 cloud,_=ds.get_data(0,return_raw_cloud=True)
 detector=ModelFreeCollisionDetector(cloud,voxel_size=.005)
 out['cd_mask']=detector.detect(GraspGroup(pred.copy()),approach_dist=.05,collision_thresh=.01)
 out['geometry_sha256']=np.array(hashlib.sha256(np.ascontiguousarray(pred[:,1:]).tobytes()).hexdigest())
 np.savez_compressed(dest,**out)
 paths=ds.colorpath+ds.depthpath+ds.labelpath+ds.metapath+[str(D/f'scenes/scene_{s:04d}/kinect/annotations/{f:04d}.xml')]
 meta=dict(scene=s,frame=f,split=split,sha256=sha(dest),seconds=time.time()-start,gpu=torch.cuda.get_device_name(),torch=torch.__version__,checkpoint_epoch=state['epoch'],tf32_matmul=True,tf32_cudnn=True,training_mode=False,valid_matches=int(out['batch_valid_mask'].sum()) if split=='train' else None,inputs=[dict(path=p,sha256=sha(p)) for p in paths])
 dest.with_suffix('.json').write_text(json.dumps(meta,indent=2));print('FRAME '+json.dumps(meta),flush=True)
 del data,end,out;torch.cuda.empty_cache()
print('COMPLETE',MODE,flush=True)
