from pathlib import Path
import hashlib,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
WORK=Path('/home/xushijie3/work')
DATA=WORK/'flow_critic_comparative_train_v1_20261014/data'
GLOBAL=WORK/'gfp_task4_global_diagnosis_v1_20261016'
def write(name,obj):
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def raw(split):
 p=DATA/('humanoidmaze-medium-navigate-v0'+('-val' if split=='val' else '')+'.npz')
 with np.load(p) as d:o=d['observations'].astype('float32');a=np.clip(d['actions'],-1+1e-5,1-1e-5).astype('float32');t=d['terminals'].astype(bool)
 ends=np.flatnonzero(t);tid=np.searchsorted(ends,np.arange(len(o)))
 return o,a,t,tid,p
def region(o):
 cell=((o[:,:2]+6)/4).astype(int)
 return ((cell[:,0]>=1)&(cell[:,0]<=2)&(cell[:,1]>=1)&(cell[:,1]<=2)).astype('int32')
