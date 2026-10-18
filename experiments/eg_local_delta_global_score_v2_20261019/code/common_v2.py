import os,sys,json,time,hashlib,resource
from pathlib import Path
import numpy as np
import torch
P=Path(os.environ['PREV_ROOT']);R=Path(os.environ['V2_ROOT']);sys.path.insert(0,str(P/'code'))
from core import utility,select,make_model as old_model,sha,write,export,Native,ap,N,CODE,DATA
CFG=json.loads((R/'frozen_config.json').read_text());SEEDS=CFG['seeds'];torch.set_num_threads(1)
class Film(torch.nn.Module):
 def __init__(self):
  super().__init__();self.f=torch.nn.Linear(320,124);self.e=torch.nn.Linear(4,248);self.out=torch.nn.Linear(124,1)
 def forward(self,x):
  z=torch.relu(self.f(x[...,:320]));g,b=self.e(x[...,320:]).chunk(2,-1);return torch.sigmoid(self.out(torch.relu((1+g)*z+b)))
def model(family):return Film() if family=='H2' else old_model()
def load(group):
 fs=[];acts=[];ys=[];mus=[];ids=[];sc=[]
 for s in CFG[group]:
  for f in CFG['fit_frames']:
   with np.load(P/f'labels/{s:04d}_{f:04d}.npz') as z:
    fs.append(z['feature']);acts.append(z['action']);mus.append(z['q']);ys.append(utility(z['q']));ids.append(z['actions']);sc.extend([s]*128)
 return dict(feature=np.concatenate(fs),action=np.concatenate(acts),y=np.concatenate(ys),mu=np.concatenate(mus),ids=np.concatenate(ids),scene=np.array(sc))
def inputs(d,mean,std):return torch.from_numpy(np.concatenate([np.broadcast_to(((d['feature']-mean)/std)[:,None,:],(len(d['feature']),6,320)),d['action']],-1).copy())
def predict(m,x):
 m.eval()
 with torch.no_grad():return np.concatenate([m(x[i:i+512]).squeeze(-1).numpy() for i in range(0,len(x),512)])
def metrics(q,d,tau=0):
 ch=select(q,d['ids'],'S4',tau);y=d['y'];rows=[]
 for s in np.unique(d['scene']):
  mask=d['scene']==s;ix=np.flatnonzero(mask);u=y[ix,ch[ix]];u0=y[ix,0];og=(d['mu'][ix,0]>0)&(d['mu'][ix,0]<=.4000001);ng=u>=np.float32(5/6);rr=int((~og&ng).sum());dd=int((og&~ng).sum());ns=int(og.sum());n=len(ix)
  rows.append(dict(scene=int(s),N=n,original_success=ns,R=rr,D=dd,Net=(rr-dd)/n,Damage=dd/ns if ns else None,modified=int((ch[ix]!=0).sum()),utility_gain=float((u-u0).mean()),regret=float((y[ix][:,[0,1,2,3]].max(1)-u).mean()),mse=float(((q[ix]-y[ix])**2).mean())))
 good=(d['mu'][:,0]>0)&(d['mu'][:,0]<=.4000001)
 return dict(scenes=rows,Net=float(np.mean([x['Net'] for x in rows])),Damage=float(np.mean([x['Damage'] for x in rows if x['Damage'] is not None])) if any(x['Damage'] is not None for x in rows) else None,regret=float(np.mean([x['regret'] for x in rows])),mse=float(np.mean([x['mse'] for x in rows])),utility_gain=float(np.mean([x['utility_gain'] for x in rows])),modified=sum(x['modified'] for x in rows),both_classes=bool(good.any() and (~good).any()))
def thresholds(q):
 margins=np.maximum(q[:,[0,1,2,3]].max(1)-q[:,0],0);positive=margins[margins>0]
 return {str(k):None if k==0 or not len(positive) else 0. if k==1 else float(np.quantile(positive,1-k,method='linear')) for k in CFG['kappa']}
def calibrate(records,rule):
 grid=[0.,1/12,1/6,1/3] if rule=='absolute' else CFG['kappa'][:-1];rows=[];chosen=None
 for value in grid:
  seeds=[]
  for seed,rec in records.items():
   tau=value if rule=='absolute' else rec['thresholds'][str(value)];z=metrics(rec['gate_q'],rec['gate_data'],float('inf') if tau is None else tau);seeds.append(dict(seed=seed,tau=tau,**z))
  passed=all(x['both_classes'] and x['Damage'] is not None and x['Damage']<=.02 and x['Net']>0 for x in seeds);rows.append(dict(grid_value=value,pass_all=passed,seeds=seeds))
  if passed and chosen is None:chosen=dict(grid_value=value,thresholds={str(x['seed']):x['tau'] for x in seeds})
 return dict(rule=rule,chosen=chosen,rows=rows,status='PASS' if chosen else 'NO_USABLE_THRESHOLD')
def state_hash(m):
 h=hashlib.sha256()
 for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
