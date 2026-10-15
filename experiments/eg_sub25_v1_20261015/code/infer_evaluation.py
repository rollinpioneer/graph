import os,sys,json,time,random,hashlib,argparse,shutil
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
R=Path(__file__).resolve().parents[1];D=R/'data_recovery';OUT=R/'runs/EG_SUB25_V1/evaluation'
parser=argparse.ArgumentParser();parser.add_argument('--partition');parser.add_argument('--role');args=parser.parse_args()
config=json.loads((D/'evaluation_run_config.json').read_text());part=args.partition;role=args.role;scenes=config['partitions'][part];assert role in config['inference_models'][part]
sys.argv=[sys.argv[0],'--dataset_root',str(R/'data/graspnet'),'--camera','kinect','--model','economicgrasp','--batch_size','4','--split_manifest',str(D/'frozen_split.json')]
from dataset.graspnet_dataset import GraspNetDataset,collate_fn
from models.economicgrasp import economicgrasp,pred_decode
from graspnetAPI import GraspGroup
SEED=20261015;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);os.nice(10)
cache=OUT/'inputs'/part;target=OUT/'predictions'/role/part;target.mkdir(parents=True,exist_ok=True);done=target/'INFERENCE_COMPLETE.json';assert not done.exists()
class Cached(Dataset):
 def __len__(self):return len(scenes)*256
 def __getitem__(self,index):
  scene,frame=scenes[index//256],index%256
  with np.load(cache/f'scene_{scene:04d}/kinect/{frame:04d}.npz') as z:return {k:z[k] for k in z.files}
if role=='subset':data=GraspNetDataset(str(R/'data/graspnet'),split=part,camera='kinect',num_points=20000,voxel_size=.005,remove_outlier=True,augment=False,load_label=False)
else:
 assert (OUT/'predictions/subset'/part/'INFERENCE_COMPLETE.json').exists();data=Cached()
assert len(data)==len(scenes)*256
net=economicgrasp(seed_feat_dim=512,is_training=False);checkpoint=torch.load(config['models'][role]['path'],map_location='cpu',weights_only=False);net.load_state_dict(checkpoint['model_state_dict'],strict=True);del checkpoint;net.cuda().eval()
def worker_init(i):np.random.seed(np.random.get_state()[1][0]+i)
loader=DataLoader(data,batch_size=4,shuffle=False,num_workers=2,worker_init_fn=worker_init,collate_fn=collate_fn)
rows=[];started=time.time();torch.cuda.reset_peak_memory_stats()
for bi,batch in enumerate(loader):
 arrays={k:v.numpy() for k,v in batch.items()};batch={k:v.cuda() for k,v in batch.items()}
 torch.cuda.synchronize();tic=time.perf_counter()
 with torch.no_grad():predictions=pred_decode(net(batch))
 torch.cuda.synchronize();elapsed=time.perf_counter()-tic
 for j,pred in enumerate(predictions):
  idx=bi*4+j;scene,frame=scenes[idx//256],idx%256;rel=Path(f'scene_{scene:04d}/kinect/{frame:04d}')
  inp=cache/rel.with_suffix('.npz');inp.parent.mkdir(parents=True,exist_ok=True)
  if role=='subset':
   if inp.exists():
    with np.load(inp) as z:assert all(np.array_equal(z[k],arrays[k][j]) for k in arrays)
   else:
    assert shutil.disk_usage(R).free>25*1024**3
    temp=inp.with_suffix('.partial.npz');np.savez(temp,**{k:v[j] for k,v in arrays.items()});temp.replace(inp)
  ihash=hashlib.sha256(inp.read_bytes()).hexdigest();prediction=pred.detach().cpu().numpy();assert prediction.shape==(1024,17) and np.isfinite(prediction).all()
  dest=(target/rel).with_suffix('.npy');dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();GraspGroup(prediction).save_npy(str(dest))
  row={'partition':part,'model':role,'scene':scene,'frame':frame,'input_sha256':ihash,'prediction_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'count':len(prediction),'score_min':float(prediction[:,0].min()),'score_max':float(prediction[:,0].max()),'batch':bi,'batch_size':4,'batch_forward_decode_seconds':elapsed};rows.append(row)
  with (target/'inference_metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 if bi%32==0:print('INFERENCE',role,part,bi*4+4,'/',len(data),flush=True)
assert len(rows)==len(data)
result={'status':'PASS','role':role,'partition':part,'frames':len(rows),'scene_ids':scenes,'checkpoint_sha256':config['models'][role]['sha256'],'wall_seconds':time.time()-started,'forward_decode_seconds':sum(x['batch_forward_decode_seconds']/4 for x in rows),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'input_manifest_sha256':hashlib.sha256(''.join(x['input_sha256'] for x in rows).encode()).hexdigest()}
done.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
