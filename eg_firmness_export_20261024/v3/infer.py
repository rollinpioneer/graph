"""V3: all splits use fresh A100 inference and same-forward features."""
import sys,json,time,random,hashlib,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent));import infer_features as I
STOP=ROOT/'STOP_ON_FAILURE'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,data):p.write_text(json.dumps(data,indent=2,allow_nan=False))
def tasks(plan):
 return [(split,s,f) for split in ['cal','C3','P2','probe'] for s in (plan['probe_train_scenes']+plan['probe_validation_scenes'] if split=='probe' else plan['cal_scenes'] if split=='cal' else plan['test_scenes']) for f in plan[split+'_frames']]
def main(args):
 plan=json.loads((ROOT/'protocol.json').read_text());torch.set_num_threads(2)
 torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 ck=I.B/'eg_full_repro_v1_20261015/downloads/economicgrasp_kinect.tar';cksha=I.sha(ck);assert cksha==plan['checkpoint_sha256']
 assert 'A100' in torch.cuda.get_device_name()
 net=I.economicgrasp(seed_feat_dim=512,is_training=False).cuda();state=torch.load(ck,map_location='cpu',weights_only=True);net.load_state_dict(state['model_state_dict'],strict=True);net.eval()
 assert not net.training and not net.is_training and not net.view.is_training
 captured={}
 def vp_hook(module,args):
  assert tuple(args[0].shape)==(1,256,1024);captured['vp']=I.cpu(args[0])[0].T.copy()
 def ang_hook(module,args,result):
  assert tuple(result.shape)==(1024,4,64);captured['ang']=I.cpu(result)[:,0,:].copy()
 handles=[net.grasp_head.register_forward_pre_hook(vp_hook),net.grasp_head.global_interaction_module.register_forward_hook(ang_hook)]
 for name in ['cache','raw_features','receipts']:(ROOT/name).mkdir(exist_ok=True)
 def forward(split,s,f):
  assert not STOP.exists();start=time.time();seed=20261024+s*1000+f
  np.random.seed(seed);random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  ds=I.dataset(s,f);sample=ds.get_data(0);data={k:v.cuda() for k,v in I.collate_fn([sample]).items()};captured.clear()
  torch.cuda.synchronize();t=time.perf_counter()
  with torch.no_grad():end=net(data);pred=I.cpu(I.pred_decode(end)[0])
  torch.cuda.synchronize();forward_seconds=time.perf_counter()-t
  assert pred.shape==(1024,17) and np.isfinite(pred).all()
  out=dict(pred=pred,candidate_id=np.arange(1024),input_points=sample['point_clouds'],input_seg=sample['seg'],input_colors=sample['cloud_colors'],logits=I.cpu(end['grasp_score_pred'])[0].reshape(6,1024).T,angle_logits=I.cpu(end['grasp_angle_pred'])[0],depth_logits=I.cpu(end['grasp_depth_pred'])[0],raw_width=I.cpu(end['grasp_width_pred'])[0].reshape(1024),view_index=I.cpu(end['grasp_top_view_inds'])[0],view_xyz=I.cpu(end['grasp_top_view_xyz'])[0],view_rotation=I.cpu(end['grasp_top_view_rot'])[0],checkpoint_sha256=np.array(cksha),scene=np.array(s),frame=np.array(f),seed=np.array(seed))
  out['angle_index']=out['angle_logits'][:12].argmax(0);out['depth_index']=out['depth_logits'][:4].argmax(0)
  assert np.allclose(pred[:,3],(out['depth_index']+1)*.01,atol=1e-7)
  assert np.allclose(pred[:,1],np.clip(1.2*out['raw_width']/10,0,.1),atol=1e-7)
  assert np.allclose(pred[:,0],out['logits']@np.arange(6)*.2,atol=3e-6)
  rot=pred[:,4:13].reshape(-1,3,3);assert np.max(abs(rot.transpose(0,2,1)@rot-np.eye(3)))<2e-5
  features=dict(vp=captured['vp'],ang=captured['ang'],logit=out['angle_logits'][:12].T.copy())
  for k,d in [('vp',256),('ang',64),('logit',12)]:assert features[k].shape==(1024,d) and features[k].dtype==np.float32 and np.isfinite(features[k]).all()
  cloud,_=ds.get_data(0,return_raw_cloud=True);detector=I.ModelFreeCollisionDetector(cloud,voxel_size=.005)
  out['cd_mask']=detector.detect(I.GraspGroup(pred.copy()),approach_dist=.05,collision_thresh=.01)
  out['geometry_sha256']=np.array(hashlib.sha256(np.ascontiguousarray(pred[:,1:]).tobytes()).hexdigest())
  rec=dict(split=split,scene=s,frame=f,seed=seed,checkpoint_sha256=cksha,gpu=torch.cuda.get_device_name(),torch=torch.__version__,forward_seconds=forward_seconds,wall_seconds=time.time()-start,training=False,tf32=True,view_override=False,source_infer_sha256=sha(Path(I.__file__)),script_sha256=sha(Path(__file__)))
  del data,end;torch.cuda.empty_cache();return out,features,rec
 def save(task,out,features,rec):
  split,s,f=task;key=f'{split}_{s:04d}_{f:04d}';cp=ROOT/f'cache/{key}.npz';fp=ROOT/f'raw_features/{key}.npz'
  assert not cp.exists() and not fp.exists()
  np.savez_compressed(cp,**out);np.savez_compressed(fp,**features)
  rec.update(cache_sha256=sha(cp),raw_features_sha256=sha(fp));write_json(ROOT/f'receipts/infer_{key}.json',rec)
  print('FRAME '+key,flush=True)
 if args[0]=='pilot':
  task=('C3',100,0);a,fa,ra=forward(*task);b,fb,rb=forward(*task)
  equal={k:bool(np.ascontiguousarray(x).tobytes()==np.ascontiguousarray(y).tobytes()) for k,x,y in [('pred',a['pred'],b['pred']),*[(k,fa[k],fb[k]) for k in ['vp','ang','logit']]]}
  report=dict(passed=all(equal.values()),frame='C3_0100_0000',bitwise_equal=equal,gpu=ra['gpu'],forward_seconds=[ra['forward_seconds'],rb['forward_seconds']],wall_seconds=[ra['wall_seconds'],rb['wall_seconds']],first_hashes={k:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for k,x in [('pred',a['pred']),*fa.items()]},second_hashes={k:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for k,x in [('pred',b['pred']),*fb.items()]})
  write_json(ROOT/'repeatability.json',report);assert report['passed'],report;save(task,a,fa,ra)
 else:
  assert json.loads((ROOT/'repeatability.json').read_text())['passed']
  for task in tasks(plan)[int(args[0])::int(args[1])]:
   split,s,f=task
   if (ROOT/f'receipts/infer_{split}_{s:04d}_{f:04d}.json').exists():continue
   a,fa,ra=forward(*task);save(task,a,fa,ra)
 for handle in handles:handle.remove()
 print('COMPLETE',args,flush=True)
if __name__=='__main__':
 try:main(I.ARGS)
 except BaseException:STOP.write_text(traceback.format_exc());raise
