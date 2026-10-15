"""Two genuine batch-4 updates, isolated from formal random initialization."""
import os,sys,json,time,random,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];OUT=R/'runs/EG_SUB25_V1/smoke';OUT.mkdir(parents=True,exist_ok=True)
sys.argv=[sys.argv[0],'--dataset_root',str(R/'data/graspnet'),'--camera','kinect','--model','economicgrasp','--batch_size','4','--max_epoch','10','--split_manifest',str(R/'data_recovery/frozen_split.json')]
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from dataset.graspnet_dataset import GraspNetDataset,collate_fn
from models.economicgrasp import economicgrasp,pred_decode
from models.loss_economicgrasp import get_loss
SEED=20261015;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
assert json.loads((R/'data_recovery/SUPERVISION_VALIDATION.json').read_text())['status']=='PASS'
started=time.time();torch.cuda.reset_peak_memory_stats()
dataset=GraspNetDataset(str(R/'data/graspnet'),camera='kinect',split='train',voxel_size=.005,num_points=20000,remove_outlier=True,augment=True)
assert len(dataset)==6400
manifest=json.loads((R/'data_recovery/frozen_split.json').read_text())
indices=[manifest['train_scene_ids'].index(s)*128+manifest['train_frame_ids'].index(f) for s,f in manifest['smoke_pairs']]
assert len(set(indices))==80
def my_worker_init_fn(worker_id):np.random.seed(np.random.get_state()[1][0]+worker_id)
loader=DataLoader(Subset(dataset,indices),batch_size=4,shuffle=True,num_workers=2,worker_init_fn=my_worker_init_fn,collate_fn=collate_fn)
net=economicgrasp(seed_feat_dim=512,is_training=True).cuda();net.train();optimizer=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=0)
def cuda(value):
 if torch.is_tensor(value):return value.cuda()
 if isinstance(value,dict):return {k:cuda(v) for k,v in value.items()}
 if isinstance(value,list):return [cuda(v) for v in value]
 raise TypeError(type(value))
rows=[];initial={k:v.detach().cpu().clone() for k,v in net.named_parameters()}
np.random.seed(SEED)
for step,batch in enumerate(loader,1):
 data=cuda(batch);optimizer.zero_grad();end=net(data);end['epoch']=0;loss,end=get_loss(end)
 metrics={k:float(v.detach().cpu()) for k,v in end.items() if k.startswith(('A:','B:','D:')) and torch.is_tensor(v) and v.numel()==1}
 assert torch.isfinite(loss) and all(np.isfinite(v) for v in metrics.values()),metrics
 expected=['Objectness','Graspness','View','Angle','Depth','Score','Width']
 assert all('B: '+name+' Loss' in metrics for name in expected)
 loss.backward();grads=[p.grad for p in net.parameters() if p.grad is not None]
 assert grads and all(torch.isfinite(g).all() for g in grads)
 grad_norm=float(torch.stack([g.detach().float().norm() for g in grads]).norm())
 assert grad_norm>0
 optimizer.step();assert all(torch.isfinite(p).all() for p in net.parameters())
 changed=sum(not torch.equal(initial[k],v.detach().cpu()) for k,v in net.named_parameters());assert changed>0
 rows.append({'step':step,'metrics':metrics,'gradient_norm':grad_norm,'changed_parameters':changed})
 print(json.dumps(rows[-1]),flush=True)
 del end,loss,data,batch
assert len(rows)==20
checkpoint=OUT/'smoke_checkpoint.tar';assert not checkpoint.exists(),'Preserve previous smoke; inspect before retry'
torch.save({'epoch':0,'model_state_dict':net.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'seed':SEED,'smoke_only':True},checkpoint)
loaded=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert loaded['smoke_only'] and loaded['epoch']==0
restored=economicgrasp(seed_feat_dim=512,is_training=False)
restored.load_state_dict(loaded['model_state_dict'],strict=True)
assert all(torch.equal(v.cpu(),restored.state_dict()[k]) for k,v in net.state_dict().items())
restored_optimizer=torch.optim.Adam(restored.parameters(),lr=.001);restored_optimizer.load_state_dict(loaded['optimizer_state_dict'])
assert len(restored_optimizer.state)==len(optimizer.state)>0
with checkpoint.open('rb') as f:
 h=hashlib.sha256()
 for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 digest=h.hexdigest()
torch.cuda.synchronize()
result={'status':'PASS','seed':SEED,'dataset_images':len(dataset),'indices':indices,'batch_size':4,'updates':20,'metrics':rows,'checkpoint':str(checkpoint),'checkpoint_sha256':digest,'checkpoint_model_and_optimizer_reloaded':True,'wall_seconds':time.time()-started,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'formal_training_initialized_from_smoke':False}
(OUT/'smoke_results.json').write_text(json.dumps(result,indent=2));(R/'data_recovery/SMOKE_STATUS.json').write_text(json.dumps(result,indent=2));print('REAL_SMOKE_PASS',flush=True)
