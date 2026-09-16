"""Small supervised policy trainer. Not the original SARM training recipe."""
from __future__ import annotations
import copy,json,random,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .env import ACTIONS
from .weights import METHODS
from .io import new_dir,dump,sha

class Policy(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(obs_dim,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,len(ACTIONS)))
    def forward(self,x): return self.net(x)

def train_one(weights_root,out,method,seed,steps=5000,batch_size=256,device='cpu'):
    if method not in METHODS: raise ValueError(method)
    if steps<1 or batch_size<1: raise ValueError('training settings')
    p=Path(weights_root); meta=json.loads((p/'manifest.json').read_text())
    if sha(p/'train.npz')!=meta['dataset_npz_sha256']: raise ValueError('training cache changed')
    with np.load(p/'train.npz',allow_pickle=False) as d:
        x=d['obs']; y=d['actions']; w=d[method]
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w<=0): raise ValueError('bad dataset')
    mean=x.mean(0); std=x.std(0); std=np.maximum(std,1e-6)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    if device.startswith('cuda'):
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; choose --device cpu explicitly')
        torch.cuda.manual_seed_all(seed)
    model=Policy(x.shape[1]).to(device)
    # Store the actual initialization, making across-method identity verifiable.
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    init_hash=hashlib.sha256(b''.join(k.encode()+v.numpy().tobytes() for k,v in sorted(initial.items()))).hexdigest()
    opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    xt=torch.as_tensor((x-mean)/std,device=device); yt=torch.as_tensor(y,device=device)
    wt=torch.as_tensor(w,device=device); sampler=np.random.default_rng(seed+500)
    losses=[]; first_indices=[]
    for step in range(steps):
        ids=sampler.integers(0,len(x),size=batch_size)
        if step<3: first_indices.append(ids.tolist())
        idx=torch.as_tensor(ids,device=device)
        ce=nn.functional.cross_entropy(model(xt[idx]),yt[idx],reduction='none')
        # weights have training-global mean one; no per-batch denominator bias.
        loss=(ce*wt[idx]).mean()
        if not torch.isfinite(loss): raise RuntimeError('nonfinite optimization')
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step%100==0 or step==steps-1: losses.append(dict(step=step+1,loss=float(loss.item())))
    out=new_dir(out)
    ckpt=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
        initial_state=initial,mean=torch.as_tensor(mean),std=torch.as_tensor(std),obs_dim=x.shape[1],
        method=method,seed=seed,steps=steps,actions=list(ACTIONS))
    torch.save(ckpt,out/'policy.pt')
    dump(out/'train.json',dict(method=method,policy_seed=seed,steps=steps,batch_size=batch_size,
        dataset_hash=meta['dataset_npz_sha256'],checkpoint_hash=sha(out/'policy.pt'),initial_state_sha256=init_hash,
        first_batch_indices=first_indices,losses=losses,device=device,checkpoint_rule='FIXED_LAST_STEP',
        policy_evidence='NOT_EVALUATED',confirmation_passed=False))
    return out
