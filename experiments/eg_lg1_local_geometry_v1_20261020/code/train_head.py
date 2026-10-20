from common import *
import torch,argparse,resource
a=argparse.ArgumentParser();a.add_argument('--seed',type=int,choices=SEEDS,required=True);a.add_argument('--family',choices=['G','Gshuffle'],required=True);args=a.parse_args();t=time.time();os.nice(10);torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats()
cfg=json.load(open(R/'frozen_config.json'));prep=json.load(open(R/'results/preparation.json'));assert prep['status']=='PASS';d={k:v for k,v in np.load(RP/'prepared.npz').items()};z=np.load(R/'prepared_geometry.npz');gg=z['geometry'].copy();norm={k:z[k] for k in ['median','mean','std']}
if args.family=='Gshuffle':
 for i,(s,f) in enumerate(zip(d['scene'],d['frame'])):gg[i]=gg[i,permutation(s,f,128)]
nf=(d['feature']-d['mean'])/d['std'];X=np.concatenate([np.broadcast_to(nf[:,:,None,:],(256,128,6,320)),d['action'],gg],-1).copy();y=utility(d['mu']);fit=d['role']=='fit';xf=torch.from_numpy(X[fit].reshape(-1,6,390));yf=torch.from_numpy(y[fit].reshape(-1,6));assert len(xf)==24576
torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);base=network(324);net=network(390)
with torch.no_grad():net[0].weight[:,:324].copy_(base[0].weight);net[0].bias.copy_(base[0].bias);net[2].load_state_dict(base[2].state_dict())
initial=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in net.state_dict().values())).hexdigest();net=net.cuda();optim=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(args.seed);order=rng.permutation(len(xf));cursor=0;stream=hashlib.sha256();losses=[]
for step in range(1,1001):
 if cursor+128>len(order):order=rng.permutation(len(xf));cursor=0
 ix=order[cursor:cursor+128];cursor+=128;stream.update(ix.tobytes());net.train();pred=net(xf[ix].cuda()).squeeze(-1);loss=((pred-yf[ix].cuda())**2).mean();assert torch.isfinite(loss);optim.zero_grad();loss.backward();optim.step();losses.append(float(loss.item()))
 if step%100==0:print('TRAIN',args.family,args.seed,step,float(loss.item()),flush=True)
expected=next(v['old_stream'] for v in prep['old_streams'] if v['seed']==args.seed);assert stream.hexdigest()==expected
path=R/f'heads/{args.family}/{args.seed}/step1000.pt';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
torch.save(dict(model={k:v.cpu().clone() for k,v in net.state_dict().items()},mean=d['mean'],std=d['std'],geom_norm=norm,seed=args.seed,family=args.family,step=1000,initial_state_sha256=initial,sample_stream_sha256=stream.hexdigest()),path)
net.eval();i=int(np.flatnonzero(d['role']=='model_val')[0]);scorer=GeoScorer(path,sha(path));ep=P/f"B/exports/subset/scene_{d['scene'][i]:04d}/{d['frame'][i]:04d}.npz"
with np.load(R/f"geometry/head/{d['scene'][i]:04d}/{d['frame'][i]:04d}.npz") as geo:raw=geo['raw'][:,0]
q=scorer.quality(d['feature'][i],d['action'][i,:,0],raw,int(d['scene'][i]),int(d['frame'][i]))
with torch.no_grad():ref=net(torch.from_numpy(X[i,:,0]).cuda()).squeeze(-1).cpu().numpy()
assert np.allclose(q,ref,atol=1e-6,rtol=1e-6)
write(R/f'results/train_{args.family}_{args.seed}.json',dict(status='PASS',family=args.family,seed=args.seed,step=1000,checkpoint=str(path),checkpoint_sha256=sha(path),initial_state_sha256=initial,sample_stream_sha256=stream.hexdigest(),seconds=time.time()-t,peak_gpu_bytes=torch.cuda.max_memory_allocated(),parameters=sum(p.numel() for p in net.parameters()),GPU_slot=os.environ['CUDA_VISIBLE_DEVICES'],reload_scoring_pass=True,losses=losses))
print('HEAD_DONE',args.family,args.seed,time.time()-t,flush=True)
