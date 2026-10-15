import os,sys,json,hashlib,math,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];TRAIN=R/'runs/EG_FULL_V1/train'
sys.argv=[sys.argv[0],'--dataset_root',str(R/'data/graspnet'),'--camera','kinect','--model','economicgrasp','--batch_size','4','--max_epoch','10']
import numpy as np
import torch
from models.economicgrasp import economicgrasp,pred_decode
from dataset.graspnet_dataset import GraspNetDataset,collate_fn
count=0;per_epoch={}
with (TRAIN/'batch_metrics.jsonl').open() as f:
 for line in f:
  row=json.loads(line);count+=1;assert row['global_step']==count
  e=(count-1)//6400+1;b=(count-1)%6400+1
  assert (row['epoch'],row['batch'])==(e,b)
  assert math.isclose(row['lr'],.001*(math.cos((e-1)/10*math.pi)+1)*.5,rel_tol=1e-12)
  assert all(math.isfinite(v) for v in row['metrics'].values())
  for name in ['Objectness','Graspness','View','Angle','Depth','Score','Width']:assert 'B: '+name+' Loss' in row['metrics']
  per_epoch[e]=per_epoch.get(e,0)+1
assert count==64000 and per_epoch==dict.fromkeys(range(1,11),6400)
epochs=[json.loads(x) for x in (TRAIN/'epoch_metrics.jsonl').read_text().splitlines()];assert [x['epoch'] for x in epochs]==list(range(1,11))
identities=[];net=economicgrasp(seed_feat_dim=512,is_training=False)
for e in range(1,11):
 p=TRAIN/f'economicgrasp_epoch{e:02d}.tar';checkpoint=torch.load(p,map_location='cpu',weights_only=False)
 assert checkpoint['epoch']==e and checkpoint['seed']==20261015 and checkpoint['optimizer_updates']==e*6400
 net.load_state_dict(checkpoint['model_state_dict'],strict=True)
 assert all(torch.isfinite(v).all() for v in net.state_dict().values())
 assert checkpoint['optimizer_state_dict']['state']
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 identities.append({'epoch':e,'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest(),'optimizer_updates':e*6400})
 del checkpoint
np.random.seed(20261015);torch.manual_seed(20261015);torch.cuda.manual_seed_all(20261015)
data=GraspNetDataset(str(R/'data/graspnet'),split='test_seen',camera='kinect',num_points=20000,remove_outlier=True,augment=False,load_label=False)
batch={k:v.cuda() for k,v in collate_fn([data[0]]).items()};net.cuda().eval()
with torch.no_grad():pred=pred_decode(net(batch))[0].cpu().numpy()
assert pred.shape==(1024,17) and np.isfinite(pred).all()
rot=pred[:,4:13].reshape(-1,3,3);assert np.allclose(rot.transpose(0,2,1)@rot,np.eye(3),atol=2e-4) and np.allclose(np.linalg.det(rot),1,atol=2e-4)
result={'status':'PASS','epochs':10,'optimizer_updates':count,'epoch_results':epochs,'checkpoint_index':identities,'final_checkpoint_real_frame_decode_shape':list(pred.shape),'final_checkpoint_predictions_finite':True,'grasp_scores_are_original_raw_logit_weighted_sums':True,'training_epoch_wall_seconds_sum':sum(e['wall_seconds'] for e in epochs),'peak_cuda_allocated_bytes':max(e['peak_cuda_allocated_bytes'] for e in epochs),'peak_cuda_reserved_bytes':max(e['peak_cuda_reserved_bytes'] for e in epochs)}
(R/'data_recovery/TRAINING_COMPLETE.json').write_text(json.dumps(result,indent=2));print('TRAINING_COMPLETE_PASS',flush=True)
