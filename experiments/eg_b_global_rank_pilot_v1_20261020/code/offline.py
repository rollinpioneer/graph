from common import *
import torch,argparse
p=argparse.ArgumentParser();p.add_argument('--family',choices=['O','M','R'],required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True);a=p.parse_args();t=time.time();torch.set_num_threads(1);os.nice(10);d=data()
path=P/f'B/heads/{a.seed}/step1000.pt' if a.family=='O' else R/f'heads/{a.family}/{a.seed}/step1000.pt';ck=torch.load(path,map_location='cpu',weights_only=False);net=model();net.load_state_dict(ck['model']);net.eval();assert np.array_equal(ck['mean'],d['mean']) and np.array_equal(ck['std'],d['std'])
rows=[]
for role in ['model_val','gate_cal']:
 indices=np.flatnonzero(d['role']==role);pred=[]
 for i in indices:
  nf=(d['feature'][i]-d['mean'])/d['std'];x=np.concatenate([np.broadcast_to(nf[:,None,:],(128,6,320)),d['action'][i]],-1).copy()
  with torch.no_grad():pred.append(net(torch.from_numpy(x)).squeeze(-1).numpy())
 pred=np.stack(pred);target=utility(d['mu'][indices])
 for zone,ix in [('all',np.arange(128)),('top64',np.arange(64)),('random64',np.arange(64,128))]:
  mse=((pred[:,ix]-target[:,ix])**2).mean((1,2));mse0=((pred[:,ix,0]-target[:,ix,0])**2).mean(1);scenes=d['scene'][indices];qs=pred[:,ix,0];best=ix[qs.argmax(1)];sample_mu=d['mu'][indices,best,0];sample_util=utility(sample_mu)
  base=dict(family=a.family,seed=a.seed,role=role,zone=zone,six_action_MSE=float(np.mean([mse[scenes==s].mean() for s in np.unique(scenes)])),original_action_MSE=float(np.mean([mse0[scenes==s].mean() for s in np.unique(scenes)])),sample_pool_Q_top1_valid=float(((sample_mu>0)&(sample_mu<=.4000001)).mean()),sample_pool_Q_top1_utility=float(sample_util.mean()))
  for stratum in ['all_GW','mu04_vs_mu06','no_closure']:
   vals=[]
   for k,i in enumerate(indices):
    g,w=d['g'][i].copy(),d['w'][i].copy();mask=np.zeros(128,bool);mask[ix]=True;g &= mask;w &= mask
    if stratum=='mu04_vs_mu06':g &= np.isclose(d['mu'][i,:,0],.4);w &= np.isclose(d['mu'][i,:,0],.6)
    if stratum=='no_closure':w &= d['mu'][i,:,0]==-1
    q=pred[k,:,0];diff=q[g,None]-q[None,w]
    if diff.size:vals.append((int(d['scene'][i]),float(((diff<0)+.5*(diff==0)).mean())))
   rows.append(dict(**base,stratum=stratum,inversion=float(np.mean([np.mean([v for s,v in vals if s==scene]) for scene in sorted({s for s,v in vals})])) if vals else None,supported_scenes=len({s for s,v in vals}),supported_frames=len(vals)))
write(R/f'results/offline_{a.family}_{a.seed}.json',dict(family=a.family,seed=a.seed,rows=rows,seconds=time.time()-t,checkpoint_sha256=sha(path),sampled_pool_only=True,checkpoint_selection=False));print('OFFLINE_DONE',a.family,a.seed,time.time()-t,flush=True)
