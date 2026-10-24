"""Frozen EG forward hooks; export only, no learning or angle selection."""
import os,sys,json,time,hashlib,random,traceback
from pathlib import Path
import numpy as np
import torch
B=Path.home()/'work/grasp';ROOT=B/'eg_firmness_export_20261024'
N0=B/'eg_n0_nref_main_20261024';OLD=B/'eg_pose_quality_diagnosis_v1_20261022';ANGLE=B/'eg_i1_angle_physics_20261024'
SRC=B/'eg_full_repro_v1_20261015/runtime/EconomicGrasp';D=B/'eg_sub25_v1_20261015/data/graspnet'
ARGS=sys.argv[1:];sys.path.insert(0,str(SRC));sys.path.insert(0,str(SRC/'libs/pointnet2'))
sys.argv=['diagnostic','--dataset_root',str(D),'--camera','kinect','--batch_size','1']
from dataset.graspnet_dataset import GraspNetDataset,collate_fn
from models.economicgrasp import economicgrasp,pred_decode
from utils.collision_detector import ModelFreeCollisionDetector
from graspnetAPI import GraspGroup
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def cpu(x):return x.detach().cpu().numpy()
def save_json(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False))
def dataset(s,f):
 ds=object.__new__(GraspNetDataset);ds.root=str(D);ds.camera='kinect';ds.num_points=20000;ds.voxel_size=.005;ds.remove_outlier=True;ds.augment=False;ds.load_label=False
 p=D/f'scenes/scene_{s:04d}/kinect'
 ds.colorpath=[str(p/f'rgb/{f:04d}.png')];ds.depthpath=[str(p/f'depth/{f:04d}.png')];ds.labelpath=[str(p/f'label/{f:04d}.png')];ds.metapath=[str(p/f'meta/{f:04d}.mat')];ds.scenename=[f'scene_{s:04d}'];ds.frameid=[f]
 return ds

def main():
 plan=json.loads((ROOT/'protocol.json').read_text());torch.set_num_threads(2)
 torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 ck=B/'eg_full_repro_v1_20261015/downloads/economicgrasp_kinect.tar';cksha=sha(ck)
 assert cksha=='a47b717ad41870e9b34e85b1487de2a3f0f134ddc1a71a98a1b55bca7fc5804a'
 net=economicgrasp(seed_feat_dim=512,is_training=False).cuda();state=torch.load(ck,map_location='cpu',weights_only=True);net.load_state_dict(state['model_state_dict'],strict=True);net.eval()
 assert not net.is_training and not net.view.is_training and not net.training
 saved={}
 def vp_hook(module,args):
  assert tuple(args[0].shape)==(1,256,1024)
  saved['vp']=cpu(args[0])[0].T.copy()
 def ang_hook(module,args,result):
  assert tuple(result.shape)==(1024,4,64)
  saved['ang']=cpu(result)[:,0,:].copy()
 handles=[net.grasp_head.register_forward_pre_hook(vp_hook),net.grasp_head.global_interaction_module.register_forward_hook(ang_hook)]
 for folder in ['cache','raw_features','features','receipts']:(ROOT/folder).mkdir(exist_ok=True)
 pairs=[('cal',s,f) for s in plan['cal_scenes'] for f in plan['cal_frames']]+[(split,s,f) for split in ['C3','P2'] for s in plan['test_scenes'] for f in plan[split+'_frames']]+[('probe',s,f) for s in plan['probe_train_scenes']+plan['probe_validation_scenes'] for f in plan['probe_frames']]
 if ARGS[0]=='pilot':pairs=[('cal',plan['cal_scenes'][0],0),('C3',100,0),('P2',100,32)]
 else:
  assert (ROOT/'pilot_passed.json').is_file()
  pairs=pairs[int(ARGS[0])::int(ARGS[1])]
 records=[]
 for split,s,f in pairs:
  assert not (ROOT/'STOP_ON_MISMATCH').exists()
  key=f'{split}_{s:04d}_{f:04d}';receipt_path=ROOT/f'receipts/infer_{key}.json'
  if receipt_path.exists():continue
  start=time.time();seed=20261024+s*1000+f;np.random.seed(seed);random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  cp=None;reference=None
  if split=='probe':
   ds=dataset(s,f);sample=ds.get_data(0)
  else:
   cp=(OLD if split=='C3' else N0)/f'cache/{"train" if split=="cal" else split}_{s:04d}_{f:04d}.npz';reference=np.load(cp)
   assert 'input_points' in reference.files,(str(cp),reference.files)
   points=reference['input_points'];sample=dict(point_clouds=points,coordinates_for_voxel=points/.005,seg=reference['input_seg'],cloud_colors=reference['input_colors'])
   assert str(reference['checkpoint_sha256'])==cksha
  data={k:v.cuda() for k,v in collate_fn([sample]).items()};saved.clear()
  torch.cuda.synchronize();forward_start=time.perf_counter()
  with torch.no_grad():end=net(data);pred=cpu(pred_decode(end)[0])
  torch.cuda.synchronize();forward_seconds=time.perf_counter()-forward_start
  assert pred.shape==(1024,17) and np.isfinite(pred).all()
  angles=cpu(end['grasp_angle_pred'])[0];depths=cpu(end['grasp_depth_pred'])[0];scores=cpu(end['grasp_score_pred'])[0].reshape(6,1024).T;width=cpu(end['grasp_width_pred'])[0].reshape(1024)
  ai=angles[:12].argmax(0);di=depths[:4].argmax(0)
  assert np.allclose(pred[:,3],(di+1)*.01,atol=1e-7)
  assert np.allclose(pred[:,1],np.clip(1.2*width/10,0,.1),atol=1e-7)
  assert np.allclose(pred[:,0],scores@np.arange(6)*.2,atol=3e-6)
  rot=pred[:,4:13].reshape(-1,3,3);assert np.max(abs(rot.transpose(0,2,1)@rot-np.eye(3)))<2e-5
  saved['logit']=angles[:12].T.copy()
  for k,shape in [('vp',(1024,256)),('ang',(1024,64)),('logit',(1024,12))]:
   assert saved[k].shape==shape and saved[k].dtype==np.float32 and np.isfinite(saved[k]).all()
  rec=dict(split=split,scene=s,frame=f,seed=seed,checkpoint_sha256=cksha,gpu=torch.cuda.get_device_name(),torch=torch.__version__,tf32_matmul=True,tf32_cudnn=True,training_mode=False,forward_seconds=forward_seconds,hook_vp_shape=[1,256,1024],hook_ang_shape=[1024,4,64])
  if split=='probe':
   out=dict(pred=pred,candidate_id=np.arange(1024),input_points=sample['point_clouds'],input_seg=sample['seg'],input_colors=sample['cloud_colors'],logits=scores,angle_logits=angles,depth_logits=depths,raw_width=width,view_index=cpu(end['grasp_top_view_inds'])[0],view_xyz=cpu(end['grasp_top_view_xyz'])[0],view_rotation=cpu(end['grasp_top_view_rot'])[0],checkpoint_sha256=np.array(cksha),scene=np.array(s),frame=np.array(f),seed=np.array(seed),angle_index=ai,depth_index=di)
   cloud,_=ds.get_data(0,return_raw_cloud=True);detector=ModelFreeCollisionDetector(cloud,voxel_size=.005)
   out['cd_mask']=detector.detect(GraspGroup(pred.copy()),approach_dist=.05,collision_thresh=.01)
   out['geometry_sha256']=np.array(hashlib.sha256(np.ascontiguousarray(pred[:,1:]).tobytes()).hexdigest())
   dest=ROOT/f'cache/{key}.npz';assert not dest.exists();np.savez_compressed(dest,**out)
   rawf=ROOT/f'raw_features/{key}.npz';assert not rawf.exists();np.savez_compressed(rawf,**saved)
   rec.update(cache_sha256=sha(dest),raw_features_sha256=sha(rawf))
  else:
   err=float(np.max(np.abs(pred-reference['pred'])));angle_equal=bool(np.array_equal(ai,reference['angle_index']))
   rec.update(input_sha256=sha(cp),pred_max_absolute_error=err,angle_index_equal=angle_equal)
   if err>1e-5 or not angle_equal:
    save_json(ROOT/f'failure_{key}.json',rec);raise AssertionError((key,'prediction replay failed',err,angle_equal))
   ep=ANGLE/f'evaluation/{s:04d}_{f:04d}.npz' if split=='C3' else N0/f'per_pose/{"train" if split=="cal" else split}_{s:04d}_{f:04d}.npz'
   e=np.load(ep);ids=e['candidate_id'];nom=e['original_angle'];assert np.array_equal(ai[ids],nom)
   rec['candidate_reference_sha256']=sha(ep)
   if split=='C3':
    error=float(np.max(np.abs(saved['logit'][ids]-e['angle_logits'][:12].T)));rec['C3_logit_max_absolute_error']=error
    if error>1e-4:
     save_json(ROOT/f'failure_{key}.json',rec);raise AssertionError((key,'C3 logit replay failed',error))
   dest=ROOT/f'features/{key}.npz';assert not dest.exists();np.savez_compressed(dest,candidate_id=ids,original_angle=nom,**{k:v[ids] for k,v in saved.items()})
   rec.update(candidates=len(ids),features_sha256=sha(dest))
  rec['wall_seconds']=time.time()-start;save_json(receipt_path,rec);records.append(rec)
  print('FRAME '+json.dumps(rec),flush=True)
  del data,end;torch.cuda.empty_cache()
 for h in handles:h.remove()
 if ARGS[0]=='pilot':save_json(ROOT/'pilot_passed.json',dict(passed=True,frames=records))
 print('COMPLETE',ARGS,flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  (ROOT/'STOP_ON_MISMATCH').write_text(traceback.format_exc());raise
