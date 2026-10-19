from repl_common import *
import torch,resource
start=time.time();torch.set_num_threads(1);device='cuda';torch.cuda.reset_peak_memory_stats();os.nice(10);split=json.loads((R/'B_head_split.json').read_text())
def load(scenes):
 out={k:[] for k in ['feature','action','mu','scene']}
 for s in scenes:
  for frame in split['frame_ids']:
   p=R/f'B/labels/{s:04d}_{frame:04d}.npz'
   with np.load(p) as z:
    assert z['q'].shape==(128,6) and np.isfinite(z['q']).all();out['feature'].append(z['feature']);out['action'].append(z['action']);out['mu'].append(z['q']);out['scene'].append(np.full(128,s))
 return {k:np.concatenate(v) for k,v in out.items()}
fit=load(split['fit']);val=load(split['model_val']);gate=load(split['gate_cal']);assert sum(len(x['mu'])*6 for x in [fit,val,gate])==196608
mean=fit['feature'].mean(0);std=fit['feature'].std(0).clip(1e-6)
def inputs(d):return torch.from_numpy(np.concatenate([np.broadcast_to(((d['feature']-mean)/std)[:,None,:],(len(d['feature']),6,320)),d['action']],-1).copy())
xf=inputs(fit);yf=torch.from_numpy(utility(fit['mu']));xv=inputs(val);xg=inputs(gate)
def predict(model,x):
 with torch.no_grad():return np.concatenate([model(x[i:i+512].to(device)).squeeze(-1).cpu().numpy() for i in range(0,len(x),512)])
def metrics(pred,d):
 y=utility(d['mu']);loss=(pred-y)**2
 return dict(scene_equal_mse=float(np.mean([loss[d['scene']==s].mean() for s in np.unique(d['scene'])])),mse_by_scene={str(s):float(loss[d['scene']==s].mean()) for s in np.unique(d['scene'])})
heads=[]
for seed in SEEDS:
 torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed);model=torch.nn.Sequential(torch.nn.Linear(324,128),torch.nn.ReLU(),torch.nn.Linear(128,1),torch.nn.Sigmoid()).to(device);assert sum(p.numel() for p in model.parameters())==41729
 optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);order=rng.permutation(len(xf));cursor=0;stream=hashlib.sha256();history=[];losses=[]
 for step in range(1,1001):
  if cursor+128>len(order):order=rng.permutation(len(xf));cursor=0
  ix=order[cursor:cursor+128];cursor+=128;stream.update(ix.tobytes());model.train();out=model(xf[ix].to(device)).squeeze(-1);loss=((out-yf[ix].to(device))**2).mean();assert torch.isfinite(loss);optim.zero_grad();loss.backward();optim.step();losses.append(float(loss.item()))
  if step%100==0:
   model.eval();m=metrics(predict(model,xv),val);m.update(step=step,train_batch_mse=float(loss.item()));history.append(m);print('HEAD',seed,m,flush=True)
 model.eval();path=R/f'B/heads/{seed}/step1000.pt';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();torch.save(dict(model={k:v.cpu().clone() for k,v in model.state_dict().items()},mean=mean,std=std,seed=seed,step=1000,metrics=history[-1]),path)
 ck=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);assert metrics(predict(model,xv),val)['scene_equal_mse']==history[-1]['scene_equal_mse']
 # Compare independently reloaded CPU scorer with the saved GPU model on a bounded slice.
 scorer=FrozenScorer(path,sha256(path));q=scorer.quality(val['feature'][:16],val['action'][:16,0]);reference=predict(model,xv[:16])[:,0];assert np.allclose(q,reference,atol=1e-6,rtol=1e-6)
 heads.append(dict(seed=seed,path=str(path),sha256=sha256(path),step=1000,fit_groups=len(xf),history=history,training_losses=losses,sample_stream_sha256=stream.hexdigest(),gate_diagnostics=metrics(predict(model,xg),gate),normalization_from='B fit samples only',reload_scoring_pass=True))
write(R/'results/B_heads.json',dict(status='PASS',heads=heads,seconds=time.time()-start,peak_torch_gpu_bytes=torch.cuda.max_memory_allocated(),rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,actions=196608,model_selection='fixed step1000; no gate/model-val selection'))
