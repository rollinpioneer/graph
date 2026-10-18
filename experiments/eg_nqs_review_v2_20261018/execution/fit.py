import os,argparse,time,hashlib,json,resource
import numpy as np
from common import *
p=argparse.ArgumentParser();p.add_argument('--role',default='subset');p.add_argument('--seed',type=int,required=True);a=p.parse_args()
import torch
from torch import nn
torch.set_num_threads(1);os.nice(10);start=time.time();device='cuda' if torch.cuda.is_available() else 'cpu'
def load(scenes):
    xs=[];ys=[];counts=[]
    for s in scenes:
        n=0
        for f in frames(s):
            lp=R/f'labels/{a.role}/scene_{s:04d}/{f:04d}.npz';ep=R/f'exports/{a.role}/scene_{s:04d}/{f:04d}.npz'
            with np.load(lp) as z,np.load(ep) as e:
                assert str(z['export_sha256'])==sha(ep)
                xs.append(e['feature'][z['ids']]);ys.append(np.column_stack([z[k] for k in ['E','B6','A']]));n+=len(z['ids'])
        counts.append(dict(scene=s,n=n))
    return np.concatenate(xs),np.concatenate(ys),counts
x,y,counts=load(SPLIT['probe_fit']);cx,cy,ccounts=load(SPLIT['probe_calibration'])
assert len(x)>0 and len(cx)>0 and np.isfinite(x).all() and np.isfinite(cx).all()
mean=x[:,:320].mean(0,dtype=np.float64).astype(np.float32);std=np.maximum(x[:,:320].std(0,dtype=np.float64),1e-6).astype(np.float32)
x[:,:320]=(x[:,:320]-mean)/std;cx[:,:320]=(cx[:,:320]-mean)/std
x=torch.from_numpy(x).to(device);y=torch.from_numpy(y).to(device);cx=torch.from_numpy(cx).to(device);cy=torch.from_numpy(cy).to(device)
def net():return nn.Sequential(nn.Linear(324,128),nn.ReLU(),nn.Linear(128,1),nn.Sigmoid()).to(device)
records=[]
for j,arm in enumerate(['E','B6','A']):
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    model=net();assert sum(v.numel() for v in model.parameters())==41729
    initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(a.seed);order=rng.permutation(len(x));off=0;stream=hashlib.sha256();history=[]
    folder=R/f'heads/{a.role}/{arm}_{a.seed}';folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'COMPLETE.json').exists()
    torch.save(dict(model=model.state_dict(),mean=mean,std=std,seed=a.seed,arm=arm,updates=0),folder/'initial.pt')
    t=time.perf_counter()
    for step in range(1,1001):
        if off>=len(order):order=rng.permutation(len(x));off=0
        ix=order[off:off+1024];off+=len(ix);stream.update(ix.astype('<i8').tobytes())
        ii=torch.from_numpy(ix).to(device);prediction=model(x[ii]).squeeze(-1);loss=((prediction-y[ii,j])**2).mean()
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step%250==0:
            with torch.no_grad():tr=((model(x).squeeze(-1)-y[:,j])**2).mean().item();cl=((model(cx).squeeze(-1)-cy[:,j])**2).mean().item()
            assert np.isfinite([tr,cl]).all();history.append(dict(step=step,train_mse=tr,cal_mse=cl,seconds=time.perf_counter()-t))
            torch.save(dict(model=model.state_dict(),mean=mean,std=std,seed=a.seed,arm=arm,updates=step,initial_hash=initial),folder/f'step{step}.pt')
    # Minimal saved-checkpoint reload fidelity.
    reloaded=net();reloaded.load_state_dict(torch.load(folder/'step1000.pt',map_location=device,weights_only=False)['model'])
    with torch.no_grad():assert torch.equal(model(cx[:128]),reloaded(cx[:128]))
    rec=dict(status='PASS',role=a.role,arm=arm,seed=a.seed,updates=1000,fit_samples=len(x),cal_samples=len(cx),fit_scene_counts=counts,cal_scene_counts=ccounts,initial_hash=initial,batch_stream_hash=stream.hexdigest(),history=history,constant_cal_mse=float(((cy[:,j]-y[:,j].mean())**2).mean().item()),wall_seconds=time.perf_counter()-t,checkpoint_sha256=sha(folder/'step1000.pt'),device=device,checkpoint_reload_exact=True)
    write_json(folder/'COMPLETE.json',rec);records.append(rec);print(json.dumps(rec),flush=True)
assert len({r['initial_hash'] for r in records})==1 and len({r['batch_stream_hash'] for r in records})==1
write_json(R/f'receipts/fit_{a.role}_{a.seed}.json',dict(status='PASS',runs=records,wall_seconds=time.time()-start,cuda_peak=torch.cuda.max_memory_allocated() if device=='cuda' else 0,rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
