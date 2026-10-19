from common import *
import torch,argparse,resource
parser=argparse.ArgumentParser();parser.add_argument('--family',choices=['M','R'],required=True);parser.add_argument('--seed',type=int,choices=SEEDS,required=True);a=parser.parse_args()
start=time.time();torch.set_num_threads(1);os.nice(10);device='cuda';torch.cuda.reset_peak_memory_stats()
d=data();fit=np.flatnonzero(d['role']=='fit');assert len(fit)==192
nf=(d['feature']-d['mean'])/d['std'];x=torch.from_numpy(np.concatenate([np.broadcast_to(nf[:,:,None,:],(256,128,6,320)),d['action']],-1).copy());y=torch.from_numpy(utility(d['mu']))
torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);net=model();assert sum(p.numel() for p in net.parameters())==41729
h=hashlib.sha256()
for k,v in net.state_dict().items():h.update(k.encode());h.update(v.numpy().tobytes())
initial_sha=h.hexdigest();net=net.to(device);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
schedule=np.load(R/f'frame_schedule_{a.seed}.npy');assert len(schedule)==1000 and set(schedule)<=set(fit)

def predict(indices):
 net.eval()
 with torch.no_grad():return np.concatenate([net(x[ix:ix+1].to(device)).squeeze(-1).cpu().numpy() for ix in indices])

def diagnostics(pred,indices):
 pred=pred.reshape(len(indices),128,6);target=utility(d['mu'][indices]);mse=((pred-target)**2).mean((1,2));sc=d['scene'][indices]
 result=dict(six_action_MSE=float(np.mean([mse[sc==s].mean() for s in np.unique(sc)])),original_action_MSE=float(((pred[:,:,0]-target[:,:,0])**2).mean()))
 for name in ['all_GW','mu04_vs_mu06','no_closure']:
  values=[]
  for k,i in enumerate(indices):
   g,w=d['g'][i].copy(),d['w'][i].copy()
   if name=='mu04_vs_mu06':g &= np.isclose(d['mu'][i,:,0],.4);w &= np.isclose(d['mu'][i,:,0],.6)
   if name=='no_closure':w &= d['mu'][i,:,0]==-1
   q=pred[k,:,0];diff=q[g,None]-q[None,w]
   if diff.size:values.append((int(d['scene'][i]),float(((diff<0)+.5*(diff==0)).mean())))
  result[name]=dict(inversion=float(np.mean([np.mean([v for s,v in values if s==scene]) for scene in sorted({s for s,v in values})])) if values else None,supported_frames=len(values),supported_scenes=len({s for s,v in values}))
 return result

out=R/f'heads/{a.family}/{a.seed}';out.mkdir(parents=True,exist_ok=True);assert not (out/'step1000.pt').exists()
history=[];stream=hashlib.sha256();paired_frames=pair_count=0
with (out/'train_steps.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=['step','scene','frame','MSE','rank','total','pairs']);writer.writeheader()
 for step,i in enumerate(schedule,1):
  i=int(i);stream.update(np.array([d['scene'][i],d['frame'][i]],np.int64).tobytes());net.train();pred=net(x[i].to(device)).squeeze(-1);mse=((pred-y[i].to(device))**2).mean();g=torch.from_numpy(d['g'][i]).to(device);w=torch.from_numpy(d['w'][i]).to(device);rank=rank_loss(pred[:,0],g,w);loss=mse+rank if a.family=='R' else mse
  assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step();n=int(g.sum()*w.sum());paired_frames+=n>0;pair_count+=n;writer.writerow(dict(step=step,scene=int(d['scene'][i]),frame=int(d['frame'][i]),MSE=float(mse.item()),rank=float(rank.item()),total=float(loss.item()),pairs=n))
  if step%100==0:
   f.flush();idx=np.flatnonzero(d['role']=='model_val');v=diagnostics(predict(idx),idx);v.update(step=step,batch_MSE=float(mse.item()),batch_rank=float(rank.item()),batch_total=float(loss.item()),cumulative_paired_frames=paired_frames,cumulative_pairs=pair_count);history.append(v);print('TRAIN',a.family,a.seed,json.dumps(v),flush=True)
net.eval();path=out/'step1000.pt';torch.save(dict(model={k:v.cpu().clone() for k,v in net.state_dict().items()},optimizer=opt.state_dict(),mean=d['mean'],std=d['std'],seed=a.seed,family=a.family,step=1000,initial_state_sha256=initial_sha,sample_stream_sha256=stream.hexdigest(),schedule_sha256=sha(R/f'frame_schedule_{a.seed}.npy'),margin=1/6,lambda_rank=int(a.family=='R'),source_prepared_sha256=sha(R/'prepared.npz'),frozen_config_sha256=sha(R/'frozen_config.json')),path)
val=np.flatnonzero(d['role']=='model_val');ref=predict(val[:8]).reshape(8,128,6)[:,:,0].reshape(1024);scorer=FrozenScorer(path,sha(path));cpu=scorer.quality(d['feature'][val[:8]].reshape(1024,320),d['action'][val[:8],:,0].reshape(1024,4));assert np.allclose(ref,cpu,atol=1e-6,rtol=1e-6)
write(R/f'results/train_{a.family}_{a.seed}.json',dict(status='PASS',family=a.family,seed=a.seed,step=1000,checkpoint=str(path),checkpoint_sha256=sha(path),initial_state_sha256=initial_sha,sample_stream_sha256=stream.hexdigest(),schedule_sha256=sha(R/f'frame_schedule_{a.seed}.npy'),history=history,seconds=time.time()-start,peak_torch_gpu_bytes=torch.cuda.max_memory_allocated(),GPU_slot=os.environ['CUDA_VISIBLE_DEVICES'],cuda_name=torch.cuda.get_device_name(),torch_version=torch.__version__,reload_score_1024_pass=True))
print('TRAIN_COMPLETE',a.family,a.seed,time.time()-start,flush=True)
