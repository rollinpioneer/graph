from common import *
from geometry_fast import visible_fast,CachedTree
from geometry_fast import _lib
from scipy.spatial import cKDTree
import torch
from torch import nn

COLUMNS=[0,1,2,3,4,5,6,7,8,9,10,11,17,18,21,22,23,24,25,29,32]
QUOTAS=[24,24,16,16,48]

class RegionTree:
 # S only: query a conservative sphere containing the union of all five regions.
 # C must retain the complete 0.16 m neighborhood for its local-count features.
 def __init__(self,tree,poses):
  p=np.asarray(poses,dtype=np.float64);rot=p[:,4:13].reshape(-1,3,3);inverse=np.linalg.inv(rot);w=np.clip(p[:,1],0,.1);h=p[:,2];d=p[:,3]
  lower=np.column_stack([d-.12,-w/2-.01,-h/2]);upper=np.column_stack([d,w/2+.01,h/2]);bits=np.array([[x,y,z] for x in [0,1] for y in [0,1] for z in [0,1]])
  corners=lower[:,None,:]+bits[None,:,:]*(upper-lower)[:,None,:];world=corners@inverse;radius=np.minimum(.16,np.sqrt((world*world).sum(-1)).max(-1)+1e-6)
  centers={};radii={}
  for pose,r in zip(poses,radius):
   key=pose[13:16].tobytes();centers[key]=pose[13:16];radii[key]=max(radii.get(key,0),float(r))
  result=tree.query_ball_point(np.asarray(list(centers.values())),np.array([radii[k] for k in centers]),return_sorted=False,workers=1)
  self.lookup={k:np.asarray(q,np.int64) for k,q in zip(centers,result)}
 def query_ball_point(self,c,r):
  assert r==.16
  return self.lookup[c.tobytes()]

def regions(points,p):
 w=np.clip(p[1],0,.1);h=p[2];d=p[3];x,y,z=points.T if len(points) else (np.array([]),)*3
 height=np.abs(z)<h/2;finger=(x>d-.06)&(x<d);side=np.abs(y)<w/2+.01
 return [height&finger&(y>-w/2-.01)&(y<-w/2),height&finger&(y>w/2)&(y<w/2+.01),height&side&(x<=d-.06)&(x>d-.07),height&side&(x<=d-.07)&(x>d-.12),height&finger&(np.abs(y)<=w/2)]

def compact_values(points,p,masks):
 counts=np.array([v.sum() for v in masks],float);inner=points[masks[4]]
 return compact_statistics(points,p,counts,inner)

def compact_fast(points,p):
 w=np.clip(p[1],0,.1);counts=np.zeros(5,np.int64);indices=np.empty(len(points),np.int64)
 n=_lib.masks(points.ctypes.data,len(points),float(w),float(p[2]),float(p[3]),counts.ctypes.data,indices.ctypes.data)
 return compact_statistics(points,p,counts.astype(float),points[indices[:n]])

def compact_statistics(points,p,counts,inner):
 contacts=[];cx=[]
 for sign in [-1,1]:
  if len(inner):
   edge=inner[:,1].min() if sign<0 else inner[:,1].max();band=inner[np.abs(inner[:,1]-edge)<=.005]
  else:band=np.empty((0,3))
  contacts.append(len(band));cx.append(float(band[:,0].mean()) if len(band) else np.nan)
 span=float(np.ptp(inner[:,1])) if len(inner) else np.nan;center=inner.mean(0) if len(inner) else [np.nan]*3;w=np.clip(p[1],0,.1)
 out=list(np.log1p(counts))+list(counts/max(1,len(points)))+list(np.log1p(contacts))+cx+[span,w-span]+list(center)+[abs(contacts[0]-contacts[1])/max(1,sum(contacts)),np.log1p(len(points))]
 assert len(out)==21
 return np.asarray(out,np.float32)

def point_values(points,p,masks):
 # Preserve every old selected point and slot, then fill only previously empty slots.
 out=np.zeros((128,8),np.float32);valid=np.zeros(128,bool)
 selected=np.full(128,-1,np.int64);region_of=np.full(128,-1,np.int64)
 offset=0
 for region,(mask,quota) in enumerate(zip(masks,QUOTAS)):
  ids=np.flatnonzero(mask);n=min(len(ids),quota)
  if n:
   v=points[ids];order=np.lexsort((ids,v[:,2],v[:,1],v[:,0]))
   chosen=ids[order[np.floor((np.arange(n)+.5)*len(ids)/n).astype(np.int64)]]
   selected[offset:offset+n]=chosen;region_of[offset:offset+n]=region
  offset+=quota
 used=np.zeros(len(points),bool);used[selected[selected>=0]]=True
 # Frozen priority: inner, approach, palm, left, right; no replacement/duplicates.
 for region in [4,3,2,0,1]:
  slots=np.flatnonzero(selected<0);ids=np.flatnonzero(masks[region]&~used)
  n=min(len(slots),len(ids))
  if not n:continue
  v=points[ids];order=np.lexsort((ids,v[:,2],v[:,1],v[:,0]))
  chosen=ids[order[np.floor((np.arange(n)+.5)*len(ids)/n).astype(np.int64)]]
  selected[slots[:n]]=chosen;region_of[slots[:n]]=region;used[chosen]=True
 valid=selected>=0;sample=points[selected[valid]];w=np.clip(p[1],0,.1);h=float(p[2]);d=float(p[3])
 out[valid,:3]=np.column_stack([(sample[:,0]-d)/.12,sample[:,1]/(w/2+.01),sample[:,2]/max(h/2,1e-6)])
 out[np.flatnonzero(valid),3+region_of[valid]]=1
 assert len(np.unique(selected[valid]))==valid.sum()
 assert valid.sum()==min(128,int(np.logical_or.reduce(masks).sum()))
 assert np.isfinite(out).all() and np.all(out[~valid]==0)
 return out,valid

def extract(scene,frame,pred,poses=None,mode='both',timing=None):
 assert mode in ['C','S','both'];poses=pred if poses is None else poses;t=time.perf_counter();cloud,identity=visible_fast(scene,frame,pred);t1=time.perf_counter();tree=(RegionTree if mode=='S' else CachedTree)(cKDTree(cloud),poses);t2=time.perf_counter();cs=[];tokens=[];valid=[]
 for p in poses:
  local=(cloud[tree.query_ball_point(p[13:16],.16)]-p[13:16])@p[4:13].reshape(3,3)
  if mode=='C':cs.append(compact_fast(local,p));continue
  mask=regions(local,p)
  if mode=='both':cs.append(compact_values(local,p,mask))
  if mode in ['S','both']:
   v,m=point_values(local,p,mask);tokens.append(v);valid.append(m)
 result=(np.asarray(cs,np.float32) if cs else None,np.asarray(tokens,np.float32) if tokens else None,np.asarray(valid,bool) if valid else None,identity)
 if timing is not None:timing.update(preprocess=t1-t,neighborhood=t2-t1,geometry=time.perf_counter()-t2,total=time.perf_counter()-t)
 return result

def project_norm(norm):
 ix=COLUMNS+[33+i for i in COLUMNS]
 return dict(median=np.asarray(norm['median'])[COLUMNS].copy(),mean=np.asarray(norm['mean'])[ix].copy(),std=np.asarray(norm['std'])[ix].copy())

class CompactHead(nn.Module):
 def __init__(self):
  super().__init__();self.score=nn.Sequential(nn.Linear(366,128),nn.ReLU(),nn.Linear(128,1),nn.Sigmoid())
 def forward(self,original,geometry):return self.score(torch.cat([original,geometry],-1)).squeeze(-1)

class PointHead(nn.Module):
 def __init__(self):
  super().__init__();self.point=nn.Sequential(nn.Linear(8,32),nn.ReLU(),nn.Linear(32,64),nn.ReLU());self.compress=nn.Sequential(nn.Linear(128,64),nn.ReLU());self.score=nn.Sequential(nn.Linear(388,128),nn.ReLU(),nn.Linear(128,1),nn.Sigmoid())
 def encode(self,points,mask):
  feature=self.point(points);weights=mask[...,None];mean=(feature*weights).sum(-2)/weights.sum(-2).clamp_min(1)
  maximum=feature.masked_fill(~weights,-torch.inf).amax(-2);maximum=torch.where(mask.any(-1,keepdim=True),maximum,torch.zeros_like(maximum))
  return self.compress(torch.cat([maximum,mean],-1))
 def forward(self,original,points,mask):return self.score(torch.cat([original,self.encode(points,mask)],-1)).squeeze(-1)

class HybridHead(PointHead):
 def __init__(self):
  super().__init__()
  # Encoder is instantiated identically to SF before replacing only the scoring input.
  self.score=nn.Sequential(nn.Linear(430,128),nn.ReLU(),nn.Linear(128,1),nn.Sigmoid())
 def forward(self,original,points,mask,geometry):
  return self.score(torch.cat([original,geometry,self.encode(points,mask)],-1)).squeeze(-1)

def initialize(family,seed):
 assert family in ['SF','H']
 torch.manual_seed(seed);base=network(324);model=PointHead() if family=='SF' else HybridHead()
 with torch.no_grad():
  model.score[0].weight[:,:324].copy_(base[0].weight);model.score[0].bias.copy_(base[0].bias);model.score[2].load_state_dict(base[2].state_dict())
 return model
