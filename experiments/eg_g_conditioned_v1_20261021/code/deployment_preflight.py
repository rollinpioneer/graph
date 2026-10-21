"""One-frame full-candidate check and no-update GPU cost probe; no optimizer."""
import os,sys,time,json,copy,hashlib,subprocess
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from adapters import Adapter,attributes
torch.set_num_threads(1)
R=Path(os.environ['TASK_ROOT']);BASE=R.parent
L=BASE/'eg_lg1_local_geometry_v1_20261020';P=BASE/'eg_sub25_b_f05_repl_v1_20261019';FAST=BASE/'eg_lg1_c4_confirm_speed_v1_20261020'
os.environ['R']=str(FAST);sys.path.insert(0,str(FAST/'code'))
from geometry_fast import visible_fast,geom_mask,CachedTree
from common import GeoScorer,geom_input
start=time.time();D=dict(np.load(BASE/'eg_b_global_rank_pilot_v1_20261020/prepared.npz'));GG=dict(np.load(L/'prepared_geometry.npz'));N=dict(np.load(R/'cache/attributes.npz'))
scene=int(D['scene'][0]);frame=int(D['frame'][0]);ep=P/f'B/exports/subset/scene_{scene:04d}/{frame:04d}.npz'
with np.load(ep) as z:pred=z['pred'];feat=z['feature320'];act=z['original_action']
cloud,obs=visible_fast(scene,frame,pred);tree=CachedTree(cKDTree(cloud),pred);raw=np.asarray([geom_mask(cloud,tree,p) for p in pred],np.float32)
att=attributes(pred[:,None,:],pred[:,0])[:,0];assert att.shape==(1024,14)
assert np.array_equal(att[D['ids'][0]],N['raw'][0,:,0])
aa=torch.from_numpy((att-N['mean'])/N['std'])
trainX=np.concatenate([np.broadcast_to(((D['feature'][0]-D['mean'])/D['std'])[:,None,:],(128,6,320)),D['action'][0],GG['geometry'][0]],-1).astype(np.float32)
trainA=(N['raw'][0]-N['mean'])/N['std'];y=(((D['mu'][0,...,None]>0)&(D['mu'][0,...,None]<=np.array([.2,.4,.6,.8,1,1.2])+1e-7)).mean(-1)).astype(np.float32)
assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False
load_before=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
tx=torch.from_numpy(trainX).cuda();ta=torch.from_numpy(trainA).cuda();ty=torch.from_numpy(y).cuda();results=[]
for h in json.loads((L/'method_lock.json').read_text())['new_heads']:
 if h['family']!='G':continue
 head=GeoScorer(Path(h['path']),h['sha256']);base=head.net
 x=torch.from_numpy(np.concatenate([(feat-head.ck['mean'])/head.ck['std'],act,geom_input(raw,head.ck['geom_norm'])],-1).astype(np.float32))
 for family in ['cap','concat','FiLM']:
  torch.manual_seed(h['seed']);m=Adapter(copy.deepcopy(base),family).eval()
  with torch.no_grad():cpu_err=float((m(x,aa)-base(x)).abs().max())
  assert cpu_err==0
  m.cuda();gx=x.cuda();ga=aa.cuda()
  with torch.no_grad():gpu_err=float((m(gx,ga)-m.base(gx)).abs().max())
  assert gpu_err==0
  before={k:v.clone() for k,v in m.state_dict().items()};dur=[];torch.cuda.reset_peak_memory_stats()
  for repeat in range(12):
   m.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.perf_counter();loss=((m(tx,ta).squeeze(-1)-ty)**2).mean();loss.backward();torch.cuda.synchronize()
   if repeat>=2:dur.append(time.perf_counter()-t)
  assert torch.isfinite(loss) and all(p.grad is None for p in m.base.parameters())
  assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())
  results.append(dict(seed=h['seed'],family=family,CPU_initial_error=cpu_err,GPU_initial_error=gpu_err,forward_backward_median_seconds=float(np.median(dur)),peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,parameters_unchanged=True))
  del m,gx,ga,before
result=dict(status='PASS',scene=scene,frame=frame,candidates=1024,device=torch.cuda.get_device_name(),load_before=load_before,load_after=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),formal_updates=0,optimizer_steps=0,checkpoint_created=False,results=results,wall_seconds=time.time()-start,estimated_9000_step_forward_backward_seconds=sum(r['forward_backward_median_seconds']*1000 for r in results),estimate_excludes_optimizer_IO_contention=True)
(R/'results/deployment_preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
