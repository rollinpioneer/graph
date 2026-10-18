"""Conditional S2 only: frozen batch1 inference, compressed per-frame caches."""
from core import *
import argparse,torch
p=argparse.ArgumentParser();p.add_argument('--scene',type=int,required=True);a=p.parse_args();assert a.scene in sum(CFG['confirmation'].values(),[])
assert json.loads((R/'results/development_decision.json').read_text())['pass_stage'] and json.loads((R/'results/transfer_decision.json').read_text())['pass_stage']
sys.path[:0]=[str(CODE),str(CODE/'libs/pointnet2')];sys.argv=[sys.argv[0],'--dataset_root',str(DATA),'--camera','kinect','--model','economicgrasp','--batch_size','1']
from dataset.graspnet_dataset import GraspNetDataset,collate_fn
from models.economicgrasp import economicgrasp,pred_decode
from utils.label_generation import batch_viewpoint_params_to_matrix
os.nice(10);torch.set_num_threads(1);torch.manual_seed(20261018);torch.cuda.manual_seed_all(20261018);start=time.time();net=economicgrasp(seed_feat_dim=512,is_training=False).cuda().eval();net.load_state_dict(torch.load(OLD/'runs/EG_SUB25_V1/train/economicgrasp_epoch10.tar',map_location='cpu',weights_only=False)['model_state_dict']);net.requires_grad_(False);cap={}
def group(m,i,o):cap['g']=o.detach().clone()
def token(m,i):cap['t']=i[0].detach().clone()
hooks=[net.cy_group.register_forward_hook(group),net.grasp_head.conv_score.register_forward_pre_hook(token)]
data=GraspNetDataset(str(DATA),split='train',camera='kinect',num_points=20000,voxel_size=.005,remove_outlier=True,augment=False,load_label=False);folder=R/f'exports/subset/scene_{a.scene:04d}';folder.mkdir(parents=True,exist_ok=True);inputs=R/f'confirmation_inputs/scene_{a.scene:04d}';inputs.mkdir(parents=True,exist_ok=True);times=[]
for f in CFG['confirmation_frames']:
 for attr,sub,ext in [('colorpath','rgb','png'),('depthpath','depth','png'),('labelpath','label','png'),('metapath','meta','mat')]:
  path=DATA/f'scenes/scene_{a.scene:04d}/kinect/{sub}/{f:04d}.{ext}';assert path.is_file(),path;getattr(data,attr)[0]=str(path)
 data.scenename[0]=f'scene_{a.scene:04d}';data.frameid[0]=f
 inp=inputs/f'{f:04d}.npz'
 if inp.exists():
  z=np.load(inp);sample={k:z[k] for k in z.files}
 else:
  seed=int.from_bytes(hashlib.sha256('|'.join(map(str,[20261018,'input',a.scene,f])).encode()).digest()[:4],'little');np.random.seed(seed);sample=data.get_data(0);np.savez_compressed(inp,**sample)
 batch={k:v.cuda() for k,v in collate_fn([sample]).items()};torch.cuda.synchronize();t=time.perf_counter()
 with torch.no_grad():ep=net(dict(batch));pred=pred_decode(ep)[0].cpu().numpy()
 torch.cuda.synchronize();times.append(time.perf_counter()-t);feature=torch.cat([cap['t'],cap['g']],1)[0].T.cpu().numpy();al=ep['grasp_angle_pred'][0,:12].cpu().numpy().T;dl=ep['grasp_depth_pred'][0,:4].cpu().numpy().T;ai=al.argmax(1);feature=np.column_stack([feature,np.sin(2*ai*np.pi/12),np.cos(2*ai*np.pi/12),pred[:,3]/.04,pred[:,1]/.1]).astype(np.float32)
 if f==0:
  for h in hooks:h.remove()
  with torch.no_grad():plain=pred_decode(net(dict(batch)))[0].cpu().numpy()
  assert np.array_equal(pred,plain);hooks=[net.cy_group.register_forward_hook(group),net.grasp_head.conv_score.register_forward_pre_hook(token)]
 ar=np.argsort(-al,axis=1,kind='stable')[:,:3];dr=np.argsort(-dl,axis=1,kind='stable')[:,:2];hy=np.repeat(pred[:,None,:],6,1);approach=-ep['grasp_top_view_xyz'][0]
 for k,(x,y) in enumerate([(x,y) for x in range(3) for y in range(2)]):
  rot=batch_viewpoint_params_to_matrix(approach,torch.as_tensor(ar[:,x],device='cuda')*np.pi/12).reshape(1024,9).cpu().numpy();hy[:,k,4:13]=rot;hy[:,k,3]=(dr[:,y]+1)*np.float32(.01)
 assert np.allclose(hy[:,0],pred,rtol=0,atol=1e-7);hy[:,0]=pred
 np.savez_compressed(folder/f'{f:04d}.npz',pred=pred,feature=feature,angle_logits=al,depth_logits=dl,hypotheses=hy,input_sha256=np.array(sha(inp)))
 if f%64==0:print('EXPORT',a.scene,f,flush=True)
write(R/f'results/export_confirmation_{a.scene}.json',dict(status='PASS',frames=256,batch=1,hook_exact=True,seconds=time.time()-start,forward_decode_seconds=times,gpu_peak_bytes=torch.cuda.max_memory_allocated(),layout='cache follows existing NQS layout; core.export returns group256 then token64 as specified'))
