import os,sys,json,hashlib,time,csv
from pathlib import Path
import numpy as np
R=Path(os.environ['R']);D=R;G=R.parent;P=G/'eg_sub25_b_f05_repl_v1_20261019';DG=G/'eg_light_head_d7_diag_v1_20261020';RP=G/'eg_b_global_rank_pilot_v1_20261020';DATA=G/'eg_sub25_v1_20261015/data/graspnet';CODE=G/'eg_sub25_v1_20261015/runtime/EconomicGrasp'
SEEDS=[20261018,20261019,20261020]
sys.path.insert(0,str(G/'eg_lqr_build_confirm_v3_20261019/code'))
from scoring import fused_score,replace_score,FrozenScorer
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.partial.json');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)
def csv_write(p,rows):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def utility(mu):return ((mu[...,None]>0)&(mu[...,None]<=np.array([.2,.4,.6,.8,1.,1.2])+1e-7)).mean(-1).astype(np.float32)
def permutation(scene,frame,n):
 seed=int.from_bytes(hashlib.sha256(f'EG-LG-1|shuffle|20261020|{scene}|{frame}|{n}'.encode()).digest()[:8],'little');return np.random.default_rng(seed).permutation(n)
def geom_input(raw,norm):
 missing=~np.isfinite(raw);x=np.concatenate([np.where(missing,norm['median'],raw),missing.astype(np.float32)],-1)
 return ((x-norm['mean'])/norm['std']).astype(np.float32)
def network(dim):
 import torch
 return torch.nn.Sequential(torch.nn.Linear(dim,128),torch.nn.ReLU(),torch.nn.Linear(128,1),torch.nn.Sigmoid())
class GeoScorer:
 def __init__(self,path,expected):
  import torch
  assert sha(path)==expected;ck=torch.load(path,map_location='cpu',weights_only=False);self.net=network(390);self.net.load_state_dict(ck['model']);self.net.eval();self.ck=ck
 def quality(self,feat,act,raw,scene,frame):
  import torch
  gg=geom_input(raw,self.ck['geom_norm'])
  if self.ck['family']=='Gshuffle':gg=gg[permutation(scene,frame,len(gg))]
  x=np.concatenate([(feat-self.ck['mean'])/self.ck['std'],act,gg],-1).astype(np.float32)
  with torch.no_grad():return self.net(torch.from_numpy(x)).squeeze(-1).numpy()
