"""Approved fixed-1000-step adapters; exact original mixed candidate-group stream."""
import os,sys,json,time,hashlib,argparse,copy
from pathlib import Path
import numpy as np
import torch
from adapters import Adapter
torch.set_num_threads(1)
R=Path(os.environ['TASK_ROOT']);BASE=R.parent;L=BASE/'eg_lg1_local_geometry_v1_20261020';RP=BASE/'eg_b_global_rank_pilot_v1_20261020'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
p=argparse.ArgumentParser();p.add_argument('--family',choices=['cap','concat','FiLM'],required=True);p.add_argument('--seed',type=int,choices=[20261018,20261019,20261020],required=True);a=p.parse_args()
assert json.loads((R/'authorization_P1_P3.json').read_text())['approved']
path=R/f'heads/{a.family}/{a.seed}/step1000.pt';assert not path.exists(),'Already trained; never retrain a completed head'
path.parent.mkdir(parents=True,exist_ok=True);assert path.parent.is_dir()
os.nice(10);start=time.time();torch.backends.cuda.matmul.allow_tf32=False
D=dict(np.load(RP/'prepared.npz'));GG=dict(np.load(L/'prepared_geometry.npz'));N=dict(np.load(R/'cache/attributes.npz'));p0=json.loads((R/'results/P0_RESULT.json').read_text())
assert sha(RP/'prepared.npz')==p0['source_prepared_sha256'] and sha(L/'prepared_geometry.npz')==p0['source_geometry_sha256'] and sha(R/'cache/attributes.npz')==p0['attribute_sha256']
h=next(v for v in json.loads((L/'method_lock.json').read_text())['new_heads'] if v['family']=='G' and v['seed']==a.seed);assert sha(h['path'])==h['sha256'];ck=torch.load(h['path'],map_location='cpu',weights_only=False)
base=torch.nn.Sequential(torch.nn.Linear(390,128),torch.nn.ReLU(),torch.nn.Linear(128,1),torch.nn.Sigmoid());base.load_state_dict(ck['model']);base.eval()
nf=(D['feature']-D['mean'])/D['std'];X=np.concatenate([np.broadcast_to(nf[:,:,None,:],(256,128,6,320)),D['action'],GG['geometry']],-1).astype(np.float32);Y=((D['mu'][...,None]>0)&(D['mu'][...,None]<=np.array([.2,.4,.6,.8,1,1.2])+1e-7)).mean(-1).astype(np.float32);A=(N['raw']-N['mean'])/N['std'];fit=D['role']=='fit'
xf=torch.from_numpy(X[fit].reshape(-1,6,390));af=torch.from_numpy(A[fit].reshape(-1,6,14));yf=torch.from_numpy(Y[fit].reshape(-1,6));assert len(xf)==24576
torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);net=Adapter(copy.deepcopy(base),a.family);frozen={k:v.clone() for k,v in net.base.state_dict().items()};initial=hashlib.sha256(b''.join(v.numpy().tobytes() for v in net.state_dict().values())).hexdigest();net.cuda();torch.cuda.reset_peak_memory_stats();opt=torch.optim.AdamW([v for v in net.parameters() if v.requires_grad],lr=.001,weight_decay=.0001)
rng=np.random.default_rng(a.seed);order=rng.permutation(len(xf));cur=0;stream=hashlib.sha256();losses=[]
for step in range(1,1001):
 if cur+128>len(order):order=rng.permutation(len(xf));cur=0
 ix=order[cur:cur+128];cur+=128;stream.update(ix.tobytes());net.train();pred=net(xf[ix].cuda(),af[ix].cuda()).squeeze(-1);loss=((pred-yf[ix].cuda())**2).mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();assert all(v.grad is None for v in net.base.parameters());opt.step();losses.append(float(loss.item()))
 if step%100==0:print('TRAIN',a.family,a.seed,step,losses[-1],flush=True)
assert stream.hexdigest()==ck['sample_stream_sha256'];assert all(torch.equal(v.cpu(),frozen[k]) for k,v in net.base.state_dict().items())
state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};path.parent.mkdir(parents=True,exist_ok=True)
torch.save(dict(model=state,family=a.family,seed=a.seed,step=1000,mean=D['mean'],std=D['std'],geom_norm=ck['geom_norm'],attribute_mean=N['mean'],attribute_std=N['std'],G_sha256=h['sha256'],initial_sha256=initial,sample_stream_sha256=stream.hexdigest()),path)
loaded=torch.load(path,map_location='cpu',weights_only=False);reload=Adapter(copy.deepcopy(base),a.family);reload.load_state_dict(loaded['model']);reload.eval();net.eval();i=int(np.flatnonzero(D['role']=='model_val')[0]);xx=torch.from_numpy(X[i]);aa=torch.from_numpy(A[i])
with torch.no_grad():ref=net(xx.cuda(),aa.cuda()).cpu();test=reload(xx,aa)
assert torch.allclose(ref,test,atol=1e-6,rtol=1e-6)
receipt=dict(status='PASS',family=a.family,seed=a.seed,step=1000,checkpoint_relative=str(path.relative_to(R)),checkpoint_sha256=sha(path),G_sha256=h['sha256'],frozen_G_unchanged=True,initial_sha256=initial,sample_stream_sha256=stream.hexdigest(),start=start,end=time.time(),seconds=time.time()-start,pid=os.getpid(),GPU_slot=os.environ['CUDA_VISIBLE_DEVICES'],device=torch.cuda.get_device_name(),peak_GPU_bytes=torch.cuda.max_memory_allocated(),trainable_parameters=sum(v.numel() for v in net.parameters() if v.requires_grad),reload_scoring_pass=True,losses=losses,new_training_quality_labels=0,EG_updates=0)
save(R/f'results/train_{a.family}_{a.seed}.json',receipt);print('HEAD_DONE',a.family,a.seed,receipt['seconds'],flush=True)
