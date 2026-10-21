import os,sys,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
torch.set_num_threads(1)
R=Path(__file__).resolve().parents[2]; BASE=R.parent
sys.path.insert(0,str(R/'code'))
from adapters import Adapter
out=R/'diagnosis_existing/results';out.mkdir(parents=True,exist_ok=True)
D=dict(np.load(BASE/'eg_b_global_rank_pilot_v1_20261020/prepared.npz'))
GG=dict(np.load(BASE/'eg_lg1_local_geometry_v1_20261020/prepared_geometry.npz'))
N=dict(np.load(R/'cache/attributes.npz'))
X=np.concatenate([np.broadcast_to(((D['feature']-D['mean'])/D['std'])[:,:,None,:],(256,128,6,320)),D['action'],GG['geometry']],-1).astype('float32')
A=((N['raw']-N['mean'])/N['std']).astype('float32')
Y=((D['mu'][...,None]>0)&(D['mu'][...,None]<=np.array([.2,.4,.6,.8,1,1.2])+1e-7)).mean(-1)
records=[];mechanisms=[];preds={}
def infer(net,att):
 with torch.no_grad():
  return np.concatenate([net(torch.from_numpy(X[i:i+16]),torch.from_numpy(att[i:i+16])).squeeze(-1).numpy() for i in range(0,256,16)])
def metrics(q):
 rows=[]
 for role in ['fit','model_val','gate_cal']:
  ix=np.flatnonzero(D['role']==role); m=[]
  for i in ix:
   mu=D['mu'][i,:64,0]; col=D['collision'][i,:64,0]
   good=(mu>0)&(mu<=.4000001)&~col
   weak=(mu>=.5999999)&~col; valid=good|weak
   auc=roc_auc_score(good[valid],q[i,:64,0][valid]) if good.any() and weak.any() else None
   inv=np.mean(q[i,:64,0][good][:,None]<=q[i,:64,0][weak][None,:]) if good.any() and weak.any() else None
   m.append(dict(scene=int(D['scene'][i]),frame=int(D['frame'][i]),auc=auc,inversion=inv))
  aucs=[v['auc'] for v in m if v['auc'] is not None];ivs=[v['inversion'] for v in m if v['inversion'] is not None]
  rows.append(dict(role=role,mse6=float(((q[ix]-Y[ix])**2).mean()),mse_original=float(((q[ix,:,0]-Y[ix,:,0])**2).mean()),top64_auc=float(np.mean(aucs)),top64_inversion=float(np.mean(ivs)),frame_metrics=m))
 return rows
lock=json.loads((R/'method_lock.json').read_text())
for seed in [20261018,20261019,20261020]:
 hs=[h for h in lock['new_heads'] if h['seed']==seed]
 gh=next(h for h in hs if h['family']=='G')
 ck=torch.load(gh['path'],map_location='cpu',weights_only=False)
 base=torch.nn.Sequential(torch.nn.Linear(390,128),torch.nn.ReLU(),torch.nn.Linear(128,1),torch.nn.Sigmoid());base.load_state_dict(ck['model'])
 class B(torch.nn.Module):
  def forward(self,x,a):return base(x)
 qg=infer(B(),A);preds[('G',seed)]=qg;records.append(dict(family='G',seed=seed,metrics=metrics(qg)))
 for fam in ['cap','concat','FiLM']:
  h=next(h for h in hs if h['family']==fam);assert hashlib.sha256(Path(h['path']).read_bytes()).hexdigest()==h['sha256']
  ck=torch.load(h['path'],map_location='cpu',weights_only=False);net=Adapter(base,fam);net.load_state_dict(ck['model']);net.eval()
  q=infer(net,A);preds[(fam,seed)]=q;records.append(dict(family=fam,seed=seed,metrics=metrics(q)))
  if fam=='FiLM':
   with torch.no_grad():
    xt=torch.from_numpy(X);at=torch.from_numpy(A);hh=net.base[1](net.base[0](xt));ga,be=net.mod(net.encoder(at)).chunk(2,-1);w=net.base[2].weight[0]
    mult=(w*hh*ga).sum(-1).numpy();add=(w*be).sum(-1).numpy()
    identity=float(np.max(np.abs(q-torch.sigmoid(net.base[2](hh).squeeze(-1)+torch.from_numpy(mult+add)).numpy())))
   for role in ['fit','model_val','gate_cal']:
    ix=D['role']==role
    mechanisms.append(dict(seed=seed,role=role,gamma_logit_rms=float(np.sqrt((mult[ix]**2).mean())),beta_logit_rms=float(np.sqrt((add[ix]**2).mean())),total_logit_rms=float(np.sqrt(((mult[ix]+add[ix])**2).mean())),identity_max_error=identity,q_delta_rms=float(np.sqrt(((q[ix]-qg[ix])**2).mean()))))
 for fam in ['cap','concat','FiLM']:
  logs=json.loads((R/f'results/train_{fam}_{seed}.json').read_text())['losses']
  mechanisms.append(dict(seed=seed,family=fam,loss_first100=float(np.mean(logs[:100])),loss_last100=float(np.mean(logs[-100:])),loss_prev100=float(np.mean(logs[-200:-100]))))
json.dump(dict(records=records,mechanisms=mechanisms,no_training=True,no_new_labels=True),open(out/'offline.json','w'),indent=2)
# Fully reuse cached selections and quality; never invoke official geometry or create labels.
trans={};quality_stats={}
for p in sorted((R/'evaluation/development').glob('*/*.json')):
 rows={v['configuration']:v for v in json.loads(p.read_text())}
 for seed in [20261018,20261019,20261020]:
  g=rows[f'G_{seed}_CD']
  for fam in ['cap','concat','FiLM']:
   v=rows[f'{fam}_{seed}_CD'];key=f'{fam}_{seed}';t=trans.setdefault(key,dict(frames=0,changed=0,success_to_failure=0,failure_to_success=0,changed_both_success=0,changed_both_failure=0,regression_categories={},scenes={}))
   t['frames']+=1;changed=g['top_candidate']!=v['top_candidate'];t['changed']+=changed
   sg=g['top1_valid'];sv=v['top1_valid']
   if sg and not sv:
    t['success_to_failure']+=1;c=v['top1_category'];t['regression_categories'][c]=t['regression_categories'].get(c,0)+1
   if not sg and sv:t['failure_to_success']+=1
   t['changed_both_success']+=bool(changed and sg and sv);t['changed_both_failure']+=bool(changed and not sg and not sv)
   sc=t['scenes'].setdefault(str(v['scene']),dict(regressions=0,recoveries=0))
   sc['regressions']+=int(sg and not sv);sc['recoveries']+=int(not sg and sv)
json.dump(trans,open(out/'transitions.json','w'),indent=2)
print('DONE',len(records),'head metrics',sum(t['frames'] for t in trans.values()),'paired frame records',flush=True)

