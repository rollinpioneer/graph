from pathlib import Path
import numpy as np,json,csv
from scipy.special import expit
from scipy.stats import rankdata
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]; B=200
features={s:dict(np.load(R/'arrays'/f'{s}_samples.npz')) for s in ['train','validation']}
rows=[]; effects=[]; rngrows=[]; eta=[]
def boot_setup(t):
 groups,inv=np.unique(t,return_inverse=True);n=len(groups)
 w=np.random.default_rng(2026101516).multinomial(n,np.full(n,1/n),size=B)
 return inv,n,w

def metric(y,sel,inv,n,w):
 sel=sel & np.isfinite(y);num=np.bincount(inv,weights=np.where(sel,y,0),minlength=n);den=np.bincount(inv,weights=sel,minlength=n)
 if den.sum()==0:return dict(mean=None,ci95=[None,None],n=0,trajectories=0)
 boot=(w@num)/np.maximum(w@den,1)
 return dict(mean=float(num.sum()/den.sum()),ci95=np.quantile(boot,[.025,.975]).tolist(),n=int(den.sum()),trajectories=int((den>0).sum()),_boot=boot)

def public(m):return {k:v for k,v in m.items() if not k.startswith('_')}

def fitted(y,x,t):
 ok=np.isfinite(y)&np.all(np.isfinite(x),axis=1); y=y[ok];x=x[ok].copy();t=t[ok]
 inv,n,w=boot_setup(t);ct=np.bincount(inv,minlength=n)
 x-=np.column_stack([np.bincount(inv,weights=x[:,k],minlength=n)/ct for k in range(x.shape[1])])[inv]
 y=y-np.bincount(inv,weights=y,minlength=n)[inv]/ct[inv]
 # Scale controls for numerical conditioning; first coefficient remains in original units.
 scale=np.std(x,axis=0);scale[scale<1e-10]=1;x/=scale
 xx=np.zeros((n,x.shape[1],x.shape[1]));xy=np.zeros((n,x.shape[1]))
 for i in range(x.shape[1]):
  xy[:,i]=np.bincount(inv,weights=x[:,i]*y,minlength=n)
  for j in range(x.shape[1]):xx[:,i,j]=np.bincount(inv,weights=x[:,i]*x[:,j],minlength=n)
 solve=lambda a,b:np.linalg.pinv(a,rcond=1e-9)@b
 coef=solve(xx.sum(0),xy.sum(0))/scale
 bs=np.array([solve(np.einsum('n,nij->ij',a,xx),a@xy)/scale for a in w])
 return dict(coef=coef.tolist(),ci95=np.quantile(bs,[.025,.975],axis=0).T.tolist(),n=len(y),trajectories=n,design_condition=float(np.linalg.cond(xx.sum(0))))

for seed in [0,1]:
 for cp in [200000,600000,1000000]:
  z=dict(np.load(R/'arrays'/f's{seed}_{cp}.npz'))
  for split_id,split in enumerate(['train','validation']):
   sel=(z['repeat']==0)&(z['split']==split_id);a={k:v[sel] for k,v in z.items()};f=features[split];d=f['bfs'];t=f['trajectory'];g=a['g'];inv,n,w=boot_setup(t)
   neutral=((g>=.4)&(g<=.6)).astype(float);agree=(np.sign(a['member_delta'][:,0])==np.sign(a['member_delta'][:,1])).astype(float)
   valid=d>=0;near=valid&(d<=3);far=d>=8
   nm=metric(neutral,near,inv,n,w);fm=metric(neutral,far,inv,n,w)
   controls=np.column_stack([f['euclidean'],f['euclidean']**2,f['head_height'],f['upright'],f['speed']])
   adjusted=fitted(neutral[valid],np.column_stack([d,controls])[valid],t[valid])
   effect=dict(seed=seed,checkpoint=cp,split=split,near_neutral=public(nm),far_neutral=public(fm),far_minus_near_neutral=float(fm['mean']-nm['mean']),far_minus_near_ci95=np.quantile(fm['_boot']-nm['_boot'],[.025,.975]).tolist(),adjusted_neutral_per_bfs_cell=adjusted,denominator_batch_range=[float(a['denom'].min()),float(a['denom'].max())],proxy={})
   for label,m in [('near',near),('far',far)]:
    progress=f['progress100'];useful=m&(progress>=1);unhelpful=m&(progress<=0)
    gm1=metric(g,useful,inv,n,w);gm0=metric(g,unhelpful,inv,n,w)
    ok=m&np.isfinite(progress)
    adjproxy=fitted(progress[ok],np.column_stack([g,d,controls])[ok],t[ok])
    y=progress[ok]>=1;s=g[ok];r=rankdata(s);auc=float((r[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(~y).sum()))
    effect['proxy'][label]=dict(useful_guidance=public(gm1),unhelpful_guidance=public(gm0),guidance_gap=float(gm1['mean']-gm0['mean']),gap_ci95=np.quantile(gm1['_boot']-gm0['_boot'],[.025,.975]).tolist(),rank_auc=auc,controlled_progress_per_guidance=adjproxy)
   effects.append(effect)
   for label,lo,hi in [('0-3',0,3),('4-7',4,7),('8-11',8,11),('12+',12,99)]:
    m=(d>=lo)&(d<=hi)
    if not m.any():continue
    result=dict(seed=seed,checkpoint=cp,split=split,bfs_bin=label,neutral=public(metric(neutral,m,inv,n,w)),mean_abs_gap=public(metric(abs(a['delta']),m,inv,n,w)),ranking_agreement=public(metric(agree,m,inv,n,w)),saturation=public(metric(((g<.01)|(g>.99)).astype(float),m,inv,n,w)),g_mean=public(metric(g,m,inv,n,w)),gap_quantiles=np.quantile(a['delta'][m],[.1,.25,.5,.75,.9]).tolist(),abs_gap_median=float(np.median(abs(a['delta'][m]))),g_quantiles=np.quantile(g[m],[.1,.25,.5,.75,.9]).tolist(),actor_data_l2=float(a['actor_data_l2'][m].mean()),vabc_data_l2=float(a['vabc_data_l2'][m].mean()))
    rows.append(result)
   for mult in [.1,.3,1,3,10]:
    hypothetical=expit(a['logit']/mult)
    eta.append(dict(seed=seed,checkpoint=cp,split=split,multiplier=mult,near_neutral=float(((hypothetical[near]>=.4)&(hypothetical[near]<=.6)).mean()),far_neutral=float(((hypothetical[far]>=.4)&(hypothetical[far]<=.6)).mean()),near_saturation=float(((hypothetical[near]<.01)|(hypothetical[near]>.99)).mean()),far_saturation=float(((hypothetical[far]<.01)|(hypothetical[far]>.99)).mean())))
   for rep in [1,2]:
    rs=(z['repeat']==rep)&(z['split']==split_id);idx=z['sample_index'][rs];rg=z['g'][rs];rd=d[idx]
    rngrows.append(dict(seed=seed,checkpoint=cp,split=split,repeat=rep,near_neutral=float(((rg[rd<=3]>=.4)&(rg[rd<=3]<=.6)).mean()),far_neutral=float(((rg[rd>=8]>=.4)&(rg[rd>=8]<=.6)).mean()),mean_abs_weight_change=float(abs(rg-g[idx]).mean()),above_half_flip_rate=float(((rg>.5)!=(g[idx]>.5)).mean()),reference_base_near_neutral=float(neutral[idx][rd<=3].mean()),reference_base_far_neutral=float(neutral[idx][rd>=8].mean())))
  print('ANALYZED',seed,cp,flush=True)
for name,obj in [('distance_metrics',rows),('controlled_effects',effects),('rng_sensitivity',rngrows),('eta_counterfactual',eta)]:
 (R/'reports'/f'{name}.json').write_text(json.dumps(obj,indent=2,allow_nan=False))
