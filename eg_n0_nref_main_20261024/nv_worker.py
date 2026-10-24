"""Official nvblox comparator; single-frame depth only, independent runtime."""
import sys,json,time,os,subprocess
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from PIL import Image
from nvblox_torch.mapper import Mapper,QueryType
from nvblox_torch.mapper_params import MapperParams
from nvblox_torch.sensor import Sensor
from proj_sdf_ref import gripper_samples,classify
R=Path(__file__).resolve().parent;D=R.parent/'eg_sub25_v1_20261015/data/graspnet'
P=json.loads((R/'protocol.json').read_text())
def run(split,s,f):
 src=R/f'cpu/{split}_{s:04d}_{f:04d}.npz';out=R/f'nv/{split}_{s:04d}_{f:04d}.npz'
 if out.exists():return
 a=np.load(src);rows=a['pred'].reshape(-1,17)
 if split=='C3':rows=np.concatenate([rows,a['raw']])
 p=D/f'scenes/scene_{s:04d}/kinect';meta=loadmat(p/f'meta/{f:04d}.mat',variable_names=['intrinsic_matrix','factor_depth'])
 K=meta['intrinsic_matrix'];depth=np.asarray(Image.open(p/f'depth/{f:04d}.png'),dtype=np.float32)/float(meta['factor_depth'].item())
 param=MapperParams();pi=param.get_projective_integrator_params();pi.projective_integrator_truncation_distance_vox=4.;pi.projective_integrator_max_integration_distance_m=7.;param.set_projective_integrator_params(pi)
 v=param.get_view_calculator_params();v.raycast_subsampling_factor=1;param.set_view_calculator_params(v)
 m=Mapper(.0025,mapper_parameters=param);cam=Sensor.from_camera(fu=float(K[0,0]),fv=float(K[1,1]),cu=float(K[0,2]),cv=float(K[1,2]),width=depth.shape[1],height=depth.shape[0])
 torch.cuda.synchronize();ts=time.perf_counter();m.add_depth_frame(torch.tensor(depth,device='cuda'),torch.eye(4),cam);torch.cuda.synchronize();integ=time.perf_counter()-ts
 occ=[];unk=[];sample_seconds=0.;query_seconds=0.
 for start in range(0,len(rows),24):
  ts=time.perf_counter();pts=[gripper_samples(r) for r in rows[start:start+24]];lengths=[len(x) for x in pts];offset=np.r_[0,np.cumsum(lengths)[:-1]];cloud=np.concatenate(pts);sample_seconds+=time.perf_counter()-ts
  ts=time.perf_counter();val=m.query_layer(QueryType.TSDF,torch.tensor(cloud,device='cuda'),mapper_id=0).cpu().numpy();query_seconds+=time.perf_counter()-ts;states=classify(val[:,0],val[:,1])
  occ.extend(np.add.reduceat((states==1).astype(int),offset)/lengths);unk.extend(np.add.reduceat((states==2).astype(int),offset)/lengths)
 try:
  procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
  memory=[int(line.split(',')[1]) for line in procs.splitlines() if int(line.split(',')[0])==os.getpid()]
 except Exception:memory=[]
 np.savez_compressed(out,nv_occ=np.array(occ),nv_unk=np.array(unk))
 rec=dict(split=split,scene=s,frame=f,poses=len(rows),integration_seconds=integ,sampling_seconds=sample_seconds,query_seconds=query_seconds,process_GPU_MiB=memory,gpu=torch.cuda.get_device_name(),torch=torch.__version__)
 out.with_suffix('.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec),flush=True)
 del m;torch.cuda.empty_cache()
if __name__=='__main__':
 torch.set_num_threads(2);split=sys.argv[1];(R/'nv').mkdir(exist_ok=True)
 if split!='train':assert (R/'frozen_config.json').exists()
 pending=[(s,f) for s in P['training_scenes' if split=='train' else 'C3_scenes'] for f in P['training_frames' if split=='train' else 'P2_frames' if split=='P2' else 'C3_frames']]
 began=time.time()
 while pending:
  ready=[t for t in pending if (R/f'cpu/{split}_{t[0]:04d}_{t[1]:04d}.json').exists()]
  for s,f in ready:run(split,s,f);pending.remove((s,f))
  if not ready:
   if time.time()-began>14400:raise TimeoutError(pending)
   time.sleep(5)
