from compact import *
import argparse,resource

def main():
 p=argparse.ArgumentParser();p.add_argument('--family',choices=['SF','H'],required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True);a=p.parse_args()
 t=time.time();os.nice(10);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert json.load(open(R/'results/preparation.json'))['status']=='PASS'
 cfg=json.load(open(R/'frozen_config.json'));d=dict(np.load(RP/'prepared.npz'));fit=d['role']=='fit';norm=dict(np.load(R/'C_normalization.npz'))
 orig=np.concatenate([np.broadcast_to(((d['feature']-d['mean'])/d['std'])[:,:,None],(256,128,6,320)),d['action']],-1).astype(np.float32)
 x=torch.from_numpy(orig[fit].reshape(-1,6,324));y=torch.from_numpy(utility(d['mu'][fit]).reshape(-1,6));fitframes=np.flatnonzero(fit)
 groups=np.column_stack([np.repeat(d['scene'][fit],128),np.repeat(d['frame'][fit],128),d['ids'][fit].reshape(-1)]).astype(np.int64)
 cache={k:np.load(R/f'cache/{k}.npy',mmap_mode='r') for k in ['compact','tokens','mask']}
 net=initialize(a.family,a.seed);initial=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in net.state_dict().values())).hexdigest();net=net.cuda();torch.cuda.reset_peak_memory_stats()
 optim=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(a.seed);order=rng.permutation(len(x));cursor=0;stream=hashlib.sha256();identity=hashlib.sha256();losses=[]
 out=R/f'heads/{a.family}/{a.seed}';out.mkdir(parents=True,exist_ok=True);assert not list(out.glob('step*.pt'))
 def forward(xx,fi,ci):
  xx=xx.cuda()
  args=[xx,torch.from_numpy(cache['tokens'][fi,ci]).cuda(),torch.from_numpy(cache['mask'][fi,ci]).cuda()]
  if a.family=='H':args.append(torch.from_numpy(cache['compact'][fi,ci]).cuda())
  return net(*args)
 def save(step):
  value=dict(model={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},optimizer=optim.state_dict(),mean=d['mean'],std=d['std'],geom_norm=norm,seed=a.seed,family=a.family,step=step,initial_state_sha256=initial,sample_stream_sha256=stream.hexdigest(),identity_stream_sha256=identity.hexdigest(),rng_state=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),order=order,cursor=cursor,losses=losses,config_sha256=sha(R/'frozen_config.json'))
  dest=out/f'step{step}.pt';tmp=dest.with_suffix('.partial.pt');torch.save(value,tmp);tmp.replace(dest)
 for step in range(1,1001):
  if cursor+128>len(order):order=rng.permutation(len(x));cursor=0
  ix=order[cursor:cursor+128];cursor+=128;stream.update(ix.tobytes());identity.update(groups[ix].tobytes());net.train();pred=forward(x[ix],fitframes[ix//128],ix%128);target=y[ix].cuda();assert pred.shape==target.shape==(128,6)
  loss=((pred-target)**2).mean();assert torch.isfinite(loss);optim.zero_grad();loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters());optim.step();losses.append(float(loss.detach()))
  if step%100==0:print('TRAIN',a.family,a.seed,step,losses[-1],round(time.time()-t,2),flush=True)
  if step%250==0:save(step)
 spec=next(v for v in cfg['training']['sampler_streams'] if v['seed']==a.seed);assert stream.hexdigest()==spec['index_stream_sha256'];assert identity.hexdigest()==spec['scene_frame_candidate_stream_sha256']
 # Fixed final checkpoint only; independent reload and held-out diagnostics do not choose a model.
 ck=torch.load(out/'step1000.pt',map_location='cpu',weights_only=False);restored=initialize(a.family,a.seed);restored.load_state_dict(ck['model']);restored=restored.cuda().eval();net.eval();diagnostics={}
 with torch.no_grad():
  for role in ['model_val','gate_cal']:
   indices=np.flatnonzero(d['role']==role);q=[]
   for fi in indices:
    xx=torch.from_numpy(orig[fi]).cuda()
    args=[xx,torch.from_numpy(cache['tokens'][fi].copy()).cuda(),torch.from_numpy(cache['mask'][fi].copy()).cuda()]
    if a.family=='H':args.append(torch.from_numpy(cache['compact'][fi].copy()).cuda())
    pred=net(*args);again=restored(*args)
    assert torch.equal(pred,again);q.append(pred.cpu().numpy())
   q=np.stack(q);target=utility(d['mu'][indices]);diagnostics[role]=dict(MSE=float(np.mean((q-target)**2)),poses=int(q.size),Q_min=float(q.min()),Q_max=float(q.max()))
 result=dict(status='PASS',family=a.family,seed=a.seed,step=1000,checkpoint=str(out/'step1000.pt'),checkpoint_sha256=sha(out/'step1000.pt'),parameters=sum(p.numel() for p in net.parameters()),initial_state_sha256=initial,sample_stream_sha256=stream.hexdigest(),identity_stream_sha256=identity.hexdigest(),seconds=time.time()-t,cpu_seconds=time.process_time(),GPU_slot=os.environ['CUDA_VISIBLE_DEVICES'],peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,diagnostics=diagnostics,losses=losses,reload_exact=True,new_GT=0)
 write(R/f'results/train_{a.family}_{a.seed}.json',result);print('HEAD_DONE',a.family,a.seed,result['seconds'],flush=True)

if __name__=='__main__':main()
