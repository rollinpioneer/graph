import os,sys,json,csv,hashlib,time
from pathlib import Path
import numpy as np

R=Path(os.environ['RANK_ROOT']); G=R.parent
P=G/'eg_sub25_b_f05_repl_v1_20261019'
V=G/'eg_lqr_build_confirm_v3_20261019'
CODE=G/'eg_sub25_v1_20261015/runtime/EconomicGrasp'
DATA=G/'eg_sub25_v1_20261015/data/graspnet'
SEEDS=[20261018,20261019,20261020]
FR=np.array([.2,.4,.6,.8,1.,1.2])
sys.path.insert(0,str(V/'code'))
from scoring import fused_score,replace_score,FrozenScorer

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(4*1024**2),b''):h.update(v)
 return h.hexdigest()

def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix('.partial.json');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)

def csv_write(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def utility(mu):return ((mu[...,None]>0)&(mu[...,None]<=FR+1e-7)).mean(-1).astype(np.float32)

def masks(mu,c,e):
 g=(~c)&(~e)&(mu>0)&(mu<=.4000001)
 w=(~c)&(~e)&((mu>.4000001)|(mu==-1))
 return g,w

def model():
 import torch
 return torch.nn.Sequential(torch.nn.Linear(324,128),torch.nn.ReLU(),torch.nn.Linear(128,1),torch.nn.Sigmoid())

def rank_loss(pred,g,w):
 import torch
 if not bool(g.any()) or not bool(w.any()):return pred.sum()*0
 return torch.relu(1/6-(pred[g,None]-pred[None,w])).square().mean()

def data():
 with np.load(R/'prepared.npz') as z:return {k:z[k].copy() for k in z.files}
