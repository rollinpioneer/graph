from core import *
import torch
sys.path[:0]=[str(CODE),str(CODE/'libs/pointnet2')]
sys.argv=[sys.argv[0],'--dataset_root',str(DATA),'--camera','kinect','--model','economicgrasp','--batch_size','4','--split_manifest',str(OLD/'data_recovery/frozen_split.json')]
from dataset.graspnet_dataset import collate_fn
from models.economicgrasp import economicgrasp,pred_decode
from utils.label_generation import batch_viewpoint_params_to_matrix
os.nice(10);torch.set_num_threads(1);torch.manual_seed(20261015);torch.cuda.manual_seed_all(20261015)
net=economicgrasp(seed_feat_dim=512,is_training=False).cuda().eval();net.load_state_dict(torch.load(OLD/'runs/EG_SUB25_V1/train/economicgrasp_epoch10.tar',map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
cohort=json.loads((R/'results/cohort.json').read_text());start=time.time();maxdiff=0.;maxsym=0.
for s in SCENES:
 for row in [x for x in cohort if x['scene']==s]:
  f=row['frame'];samples=[]
  for j in range(f,f+4):
   with np.load(E/f'inputs/dev/scene_{s:04d}/kinect/{j:04d}.npz') as z:samples.append({k:z[k] for k in z.files})
  batch={k:v.cuda() for k,v in collate_fn(samples).items()}
  with torch.no_grad():ep=net(batch);pred=pred_decode(ep)[0].cpu().numpy()
  base=np.load(E/f'predictions/subset/dev/scene_{s:04d}/kinect/{f:04d}.npy');assert np.allclose(pred,base,rtol=2e-4,atol=2e-5),(s,f,float(np.max(abs(pred-base))))
  maxdiff=max(maxdiff,float(np.max(abs(pred-base))));i=row['raw_index'];g=np.load(E/f'results/subset/dev/scene_{s:04d}/evaluated/{f:04d}.npz')['grasps'][0]
  al=ep['grasp_angle_pred'][0,:12,i].cpu().numpy();dl=ep['grasp_depth_pred'][0,:4,i].cpu().numpy();ai=int(al.argmax());di=int(dl.argmax());assert abs(g[3]-(di+1)*.01)<1e-7
  # Check all mapped class rotations modulo parallel-jaw pi symmetry.
  approach=-ep['grasp_top_view_xyz'][0,i].repeat(12,1);angles=torch.arange(12,device='cuda')*np.pi/12
  direct=batch_viewpoint_params_to_matrix(approach,angles).cpu().numpy();v=grid(g)[::4,4:13].reshape(12,3,3)
  for k in range(12):
   mat=direct[(ai+k)%12];err=min(np.max(abs(v[k]-mat)),np.max(abs(v[k]-mat@np.diag([1,-1,-1]))));maxsym=max(maxsym,float(err));assert err<2e-5,(s,f,k,err)
  assert np.allclose(direct[ai],g[4:13].reshape(3,3),rtol=0,atol=2e-5)
  np.savez(R/f'replay/{s:04d}_{f:04d}.npz',angle_logits=al,depth_logits=dl,angle_index=ai,depth_index=di,raw_index=i)
 print('REPLAY',s,flush=True)
write(R/'results/replay_receipt.json',dict(status='PASS',frames=640,inference_frames=2560,training_updates=0,max_abs_raw_prediction_difference=maxdiff,max_rotation_symmetry_error=maxsym,seconds=time.time()-start,gpu_peak_bytes=torch.cuda.max_memory_allocated()))
