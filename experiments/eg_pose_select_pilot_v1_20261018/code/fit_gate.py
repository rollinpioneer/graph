from core import *
import torch,resource
os.nice(10);torch.set_num_threads(1);start=time.time();device='cuda';torch.cuda.reset_peak_memory_stats()
def load(scenes):
 fs=[];acts=[];qs=[];sc=[];ids=[]
 for s in scenes:
  for f in CFG['fit_frames']:
   with np.load(R/f'labels/{s:04d}_{f:04d}.npz') as z:
    assert np.isfinite(z['q']).all();fs.append(z['feature']);acts.append(z['action']);qs.append(z['q']);sc.extend([s]*128);ids.append(z['actions'])
 return dict(feature=np.concatenate(fs),action=np.concatenate(acts),mu=np.concatenate(qs),scene=np.array(sc),ids=np.concatenate(ids))
assert json.loads((R/'results/label_queue.json').read_text())['status']=='PASS'
import subprocess
subprocess.run([sys.executable,str(R/'code/label_summary.py')],check=True)
fit=load(CFG['fit']);val=load(CFG['model_val']);gate=load(CFG['gate_cal']);mean=fit['feature'].mean(0);std=fit['feature'].std(0).clip(1e-6);constant=float(utility(fit['mu']).mean())
def inputs(d):return torch.from_numpy(np.concatenate([np.broadcast_to(((d['feature']-mean)/std)[:,None,:],(len(d['feature']),6,320)),d['action']],-1).copy())
xf=inputs(fit);yf=torch.from_numpy(utility(fit['mu']));xv=inputs(val);xg=inputs(gate)
def predict(model,x):
 with torch.no_grad():return np.concatenate([model(x[i:i+512].to(device)).squeeze(-1).cpu().numpy() for i in range(0,len(x),512)])
def metrics(pred,d):
 y=utility(d['mu']);loss=(pred-y)**2;sel=pred.argmax(1);reg=y.max(1)-y[np.arange(len(y)),sel];good=(d['mu'][:,0]>0)&(d['mu'][:,0]<=.4000001)
 return dict(scene_equal_mse=float(np.mean([loss[d['scene']==s].mean() for s in np.unique(d['scene'])])),mse_by_scene={str(s):float(loss[d['scene']==s].mean()) for s in np.unique(d['scene'])},local_regret=float(reg.mean()),original_success_group_mse=float(loss[good].mean()) if good.any() else None,original_failure_group_mse=float(loss[~good].mean()) if (~good).any() else None,constant_mse=float(((constant-y)**2).mean()))
selected=[];all_gate=[]
for seed in SEEDS:
 torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed);model=make_model().to(device);assert sum(p.numel() for p in model.parameters())==41729;optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);history=[];train_losses=[];order=rng.permutation(len(xf));cursor=0;stream=hashlib.sha256();best=None;folder=R/f'heads/{seed}';folder.mkdir(exist_ok=True)
 for step in range(1,1001):
  if cursor+128>len(order):order=rng.permutation(len(xf));cursor=0
  ix=order[cursor:cursor+128];cursor+=128;stream.update(ix.tobytes());model.train();pred=model(xf[ix].to(device)).squeeze(-1);loss=((pred-yf[ix].to(device))**2).mean();assert torch.isfinite(loss);optim.zero_grad();loss.backward();optim.step();train_losses.append(float(loss.item()))
  if step%100==0:
   model.eval();m=metrics(predict(model,xv),val);m.update(step=step,train_batch_mse=float(loss.item()));history.append(m);torch.save(dict(model={k:v.cpu().clone() for k,v in model.state_dict().items()},mean=mean,std=std,step=step,seed=seed,metrics=m),folder/f'step{step}.pt')
   if best is None or m['scene_equal_mse']<best['scene_equal_mse']:best=m
 path=folder/f'step{best["step"]}.pt';ck=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);model.eval();reload_metrics=metrics(predict(model,xv),val);assert abs(reload_metrics['scene_equal_mse']-best['scene_equal_mse'])<1e-8;gpred=predict(model,xg);np.savez(folder/'gate_predictions.npz',pred=gpred)
 selected.append(dict(seed=seed,selected_step=best['step'],checkpoint_sha256=sha(path),history=history,sample_stream_sha256=stream.hexdigest(),fit_groups=len(xf),model_val_groups=len(xv),metrics=best,training_losses=train_losses,checkpoint_reload_exact=True))
 for tau in [0.,1/12,1/6,1/3]:
  si=select(gpred,gate['ids'],'S4',tau);og=(gate['mu'][:,0]>0)&(gate['mu'][:,0]<=.4000001);mu=gate['mu'][np.arange(len(si)),si];ng=(mu>0)&(mu<=.4000001);scene_rows=[]
  for s in CFG['gate_cal']:
   mask=gate['scene']==s;n=int(mask.sum());ns=int((mask&og).sum());nf=n-ns;rr=int((mask&~og&ng).sum());dd=int((mask&og&~ng).sum());scene_rows.append(dict(scene=s,N=n,original_success=ns,original_failure=nf,R=rr,D=dd,Net=(rr-dd)/n,Damage=dd/ns if ns else None,modified=int((mask&(si!=0)).sum())))
  damages=[x['Damage'] for x in scene_rows if x['Damage'] is not None];all_gate.append(dict(seed=seed,tau=tau,scenes=scene_rows,Net=float(np.mean([x['Net'] for x in scene_rows])),Damage=float(np.mean(damages)) if damages else None,both_classes_present=bool(og.any() and (~og).any())))
write(R/'results/selected_heads.json',selected)
choices=[]
for tau in [0.,1/12,1/6,1/3]:
 group=[x for x in all_gate if x['tau']==tau]
 if all(x['both_classes_present'] and x['Damage'] is not None and x['Damage']<=.02 and x['Net']>0 for x in group):choices.append(tau)
chosen=min(choices) if choices else None
write(R/'results/gate_calibration.json',dict(status='PASS' if chosen is not None else 'NO_USABLE_FINITE_THRESHOLD',tau=chosen,tau_null_means='infinity: preserve original action',selection_uses='gate-cal only; S4; all three seeds; scene equal Damage<=2% and Net>0',rows=all_gate))
write(R/'results/fit_complete.json',dict(status='PASS',heads=3,backbone_updates=0,seconds=time.time()-start,gpu_peak_bytes=torch.cuda.max_memory_allocated(),rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,normalization_sha256=hashlib.sha256(mean.tobytes()+std.tobytes()).hexdigest()))
print('TRAIN_COMPLETE_TAU',chosen,flush=True)
