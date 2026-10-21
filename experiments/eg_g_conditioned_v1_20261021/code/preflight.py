"""CPU-only identity, pose correspondence, initialization and cost preflight."""
import os,sys,json,time,hashlib,copy,resource
from pathlib import Path
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
from adapters import Adapter,attributes
torch.set_num_threads(1)
R=Path(os.environ['TASK_ROOT']);BASE=R.parent
P=BASE/'eg_sub25_b_f05_repl_v1_20261019';L=BASE/'eg_lg1_local_geometry_v1_20261020';RP=BASE/'eg_b_global_rank_pilot_v1_20261020'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def load_export(i):
 t=time.time();s=int(D['scene'][i]);f=int(D['frame'][i]);p=P/f'B/exports/subset/scene_{s:04d}/{f:04d}.npz';lp=P/f'B/labels/{s:04d}_{f:04d}.npz'
 with np.load(p) as z:
  pred=z['pred'];hyp=z['hypotheses'][D['ids'][i]];act=z['actions'][D['ids'][i]];feature=z['feature320'][D['ids'][i]]
  assert np.array_equal(hyp[:,0],pred[D['ids'][i]]) and np.array_equal(act,D['action'][i]) and np.array_equal(feature,D['feature'][i])
  att=attributes(hyp,pred[D['ids'][i],0]);assert att.shape==(128,6,14) and np.isfinite(att).all()
  assert np.array_equal(att[:,0,:9],pred[D['ids'][i],4:13])
  order=np.lexsort((np.arange(1024),-pred[:,0]));assert np.array_equal(order[:64],D['ids'][i,:64])
 with np.load(lp) as z:assert str(z['export_sha256'])==sha(p) and np.array_equal(z['q'],D['mu'][i])
 return i,att,dict(scene=s,frame=f,seconds=time.time()-t,export_sha256=sha(p),label_sha256=sha(lp),role=str(D['role'][i]))
if __name__=='__main__':
 t=time.time();os.nice(10);D=dict(np.load(RP/'prepared.npz'));GG=dict(np.load(L/'prepared_geometry.npz'))
 prep=json.loads((L/'results/preparation.json').read_text());assert sha(RP/'prepared.npz')==prep['prepared_source_sha256'];assert sha(L/'prepared_geometry.npz')==prep['geometry_cache_sha256']
 workers=int(sys.argv[1]);atts=np.empty((256,128,6,14),np.float32);receipts=[]
 with ProcessPoolExecutor(max_workers=workers) as ex:
  for i,a,r in ex.map(load_export,range(256)):atts[i]=a;receipts.append(r)
 fit=D['role']=='fit';raw=atts[fit].reshape(-1,14);mean=raw.mean(0);std=raw.std(0).clip(1e-6);norm=(atts-mean)/std
 assert np.isfinite(norm).all();np.savez_compressed(R/'cache/attributes.npz',raw=atts,mean=mean,std=std,role=D['role'],scene=D['scene'],frame=D['frame'])
 X=np.concatenate([np.broadcast_to(((D['feature']-D['mean'])/D['std'])[:,:,None,:],(256,128,6,320)),D['action'],GG['geometry']],-1).astype(np.float32)
 lock=json.loads((L/'method_lock.json').read_text());tests=[];counts={};streams=[];cost=[]
 for h in lock['new_heads']:
  if h['family']!='G':continue
  assert sha(h['path'])==h['sha256'];ck=torch.load(h['path'],map_location='cpu',weights_only=False)
  net=torch.nn.Sequential(torch.nn.Linear(390,128),torch.nn.ReLU(),torch.nn.Linear(128,1),torch.nn.Sigmoid());net.load_state_dict(ck['model']);net.eval()
  assert np.array_equal(ck['mean'],D['mean']) and np.array_equal(ck['std'],D['std'])
  rng=np.random.default_rng(h['seed']);order=rng.permutation(24576);cur=0;stream=hashlib.sha256()
  for step in range(1000):
   if cur+128>len(order):order=rng.permutation(len(order));cur=0
   ix=order[cur:cur+128];cur+=128;stream.update(ix.tobytes())
  assert stream.hexdigest()==ck['sample_stream_sha256'];streams.append(dict(seed=h['seed'],sha256=stream.hexdigest()))
  for family in ['cap','concat','FiLM']:
   torch.manual_seed(h['seed']);m=Adapter(copy.deepcopy(net),family);m.eval();counts[family]=sum(p.numel() for p in m.parameters() if p.requires_grad)
   maximum=0.
   with torch.no_grad():
    for i in range(256):
     x=torch.from_numpy(X[i]);a=torch.from_numpy(norm[i]);v=m(x,a);ref=net(x);maximum=max(maximum,float((v-ref).abs().max()))
   assert maximum==0.
   # Backpropagation check only; no optimizer, no weight update/checkpoint.
   before={k:v.clone() for k,v in m.state_dict().items()};x=torch.from_numpy(X[0]);a=torch.from_numpy(norm[0]);y=torch.from_numpy(((D['mu'][0,...,None]>0)&(D['mu'][0,...,None]<=np.array([.2,.4,.6,.8,1.,1.2])+1e-7)).mean(-1).astype(np.float32))
   timer=time.time();loss=((m(x,a).squeeze(-1)-y)**2).mean();loss.backward();sec=time.time()-timer
   assert torch.isfinite(loss) and all(p.grad is None for p in m.base.parameters());assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())
   assert any(p.grad is not None and p.grad.abs().max()>0 for p in m.parameters() if p.requires_grad)
   tests.append(dict(seed=h['seed'],family=family,max_initial_error=maximum,finite_loss=float(loss),frozen_G_gradient_none=True,parameters_unchanged=True));cost.append(sec)
 assert max(counts.values())/min(counts.values())<=1.05
 result=dict(status='PASS',formal_training_updates=0,new_quality_labels=0,EG_forward=0,C4_reads=0,GPU_process_hours=0,frames=256,poses=196608,workers=workers,worker_seconds=sum(r['seconds'] for r in receipts),wall_seconds=time.time()-t,source_prepared_sha256=sha(RP/'prepared.npz'),source_geometry_sha256=sha(L/'prepared_geometry.npz'),attribute_sha256=sha(R/'cache/attributes.npz'),attributes_fit_only=True,parameter_counts=counts,stream_checks=streams,initialization_checks=tests,CPU_forward_backward_seconds=cost,normalization_mean=mean.tolist(),normalization_std=std.tolist(),proposed_GPU_hours_cap=2,proposed_CPU_worker_hours_cap=60,proposed_storage_GiB_cap=4,training_status='AWAITING_SEPARATE_APPROVAL')
 save(R/'results/P0_RESULT.json',result);save(R/'results/input_identities.json',receipts);print(json.dumps(result),flush=True)
