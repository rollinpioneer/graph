import os,sys,json,time,hashlib
from pathlib import Path
import numpy as np
G=Path(os.environ['GRASP_ROOT']);R=Path(os.environ['SELECT_ROOT']);N=G/'eg_nqs_pilot_v2_20261018';OLD=G/'eg_sub25_v1_20261015';CODE=OLD/'runtime/EconomicGrasp';DATA=OLD/'data/graspnet';CFG=json.loads((R/'frozen_config.json').read_text());SEEDS=CFG['seeds'];ORDER=[0,2,4,1,3,5];SETS={'A3':[0,1,2],'D2':[0,3],'S4':[0,1,2,3],'J6':list(range(6)),'S4-zero':[0,1,2,3]}
import importlib.util
spec=importlib.util.spec_from_file_location('native_reference',N/'code/common.py');native_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(native_module);Native=native_module.Native;ap=native_module.ap

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.partial.json');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def sample_ids(pred,s,f):
 ids=np.lexsort((np.arange(len(pred)),-pred[:,0]));seed=int.from_bytes(hashlib.sha256(f"EG-POSE-SELECT-PILOT-V1|sample|20261018|{s:04d}|{f:04d}".encode()).digest()[:8],'little');return np.r_[ids[:64],np.random.default_rng(seed).choice(ids[64:],64,replace=False)]
def export(s,f):
 source=R/f'exports/subset/scene_{s:04d}/{f:04d}.npz'
 if not source.exists():source=N/f'exports/subset/scene_{s:04d}/{f:04d}.npz'
 with np.load(source) as z:
  feat=np.concatenate([z['feature'][:,64:320],z['feature'][:,:64]],axis=1);pred=z['pred'].copy();hyp=z['hypotheses'][:,ORDER].copy();al=z['angle_logits'];dl=z['depth_logits']
 ar=np.argsort(-al,axis=1,kind='stable')[:,:3];dr=np.argsort(-dl,axis=1,kind='stable')[:,:2]
 ai=np.stack([ar[:,i] for j in range(2) for i in range(3)],1);di=np.stack([dr[:,j] for j in range(2) for i in range(3)],1)
 assert np.array_equal(hyp[:,0],pred)
 assert np.all(hyp[:,:,0]==pred[:,None,0]) and np.array_equal(hyp[:,:,1],np.repeat(pred[:,None,1],6,1))
 act=np.stack([np.sin(2*ai*np.pi/12),np.cos(2*ai*np.pi/12),(di+1)/4,hyp[:,:,1]/.1],-1).astype(np.float32)
 return pred,hyp,feat,act,ai*4+di

def utility(q):return ((q[...,None]>0)&(q[...,None]<=np.array([.2,.4,.6,.8,1.,1.2])+1e-7)).mean(-1).astype(np.float32)
def make_model():
 import torch
 return torch.nn.Sequential(torch.nn.Linear(324,128),torch.nn.ReLU(),torch.nn.Linear(128,1),torch.nn.Sigmoid())
def select(q,actions,policy,tau):
 ids=np.array(SETS[policy]);nonzero=ids[ids!=0]
 ranked=nonzero[np.argsort(actions[:,nonzero],axis=1,kind='stable')]
 ranked=np.column_stack([np.zeros(len(q),int),ranked])
 values=np.take_along_axis(q,ranked,axis=1);best=ranked[np.arange(len(q)),values.argmax(1)]
 return np.where(q[np.arange(len(q)),best]-q[:,0]>(0 if policy=='S4-zero' else tau),best,0)
