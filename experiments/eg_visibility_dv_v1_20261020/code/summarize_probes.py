from common import *
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

def bootstrap(values,scenes):
 x=np.asarray(values,float);s=np.asarray(scenes);ok=np.isfinite(x);x=x[ok];s=s[ok]
 if not len(x):return dict(mean=None,ci95=None,n=0)
 rng=np.random.default_rng(20261020);groups=[np.flatnonzero((s>=lo)&(s<hi)) for lo,hi in [(0,100),(100,130),(130,160),(160,190)]]
 groups=[g for g in groups if len(g)]
 sample=np.concatenate([rng.choice(g,(10000,len(g)),replace=True) for g in groups],1)
 draws=x[sample].mean(1)
 return dict(mean=float(x.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),n=len(x),positive_scenes=int((x>0).sum()))
z=np.load(R/'probes/head_index.npz');rows=[]
for p in sorted((R/'probes').glob('*_head.npy')):
 key,kind,target=p.stem.removesuffix('_head').split('_');t=int(target[1:])-1;pred=np.load(p)
 for scene in np.unique(z['scene']):
  if z['role'][np.flatnonzero(z['scene']==scene)[0]]=='fit':continue
  for frame in np.unique(z['frame'][z['scene']==scene]):
   mask=(z['scene']==scene)&(z['frame']==frame)
   mask&=~z['cd'] if t==0 else ~z['collision'] if t==1 else np.ones(len(mask),bool)
   y=z['y'][mask,t];pr=pred[mask]
   value=float(roc_auc_score(y,pr)) if t<2 and len(np.unique(y))==2 else float(spearmanr(y,pr).statistic) if t==2 and np.std(y)>0 and np.std(pr)>0 else np.nan
   rows.append([int(scene),int(frame),key,kind,t+1,len(y),value])
columns=['scene','frame','features','probe','target','n','metric']
head=pd.DataFrame(rows,columns=columns);head.to_csv(R/'results/heldout_probe_frames.csv',index=False)
df=pd.concat([head]+[pd.read_csv(R/f'results/c3_probe_{s}.csv') for s in C3],ignore_index=True)
roles={int(s):str(z['role'][np.flatnonzero(z['scene']==s)[0]]) for s in np.unique(z['scene'])}
df['pool']=[roles.get(int(s),'C3') for s in df.scene]
scene=df.groupby(['pool','scene','features','probe','target'],as_index=False).agg(metric=('metric','mean'),valid_frames=('metric','count'),candidates=('n','sum'))
scene.to_csv(R/'results/probe_per_scene.csv',index=False)
contrasts=[];absolute=[]
for pool in ['model_val','gate_cal','heldout8','C3']:
 part=scene[scene.pool.isin(['model_val','gate_cal'])] if pool=='heldout8' else scene[scene.pool==pool]
 for (probe,target),g in part.groupby(['probe','target']):
  table=g.pivot(index='scene',columns='features',values='metric')
  for feature in table:
   absolute.append(dict(pool=pool,probe=probe,target=int(target),features=feature,**bootstrap(table[feature],table.index)))
  if 'FG' not in table:continue
  for feature in table:
   if feature=='FG':continue
   delta=table[feature]-table['FG'];contrasts.append(dict(pool=pool,probe=probe,target=int(target),contrast=feature+'-FG',**bootstrap(delta,table.index)))
write(R/'results/probe_absolute.json',absolute);write(R/'results/probe_contrasts.json',contrasts)
decisions={}
for probe in ['linear','mlp']:
 def get(pool,feature,target):
  return next(x for x in contrasts if x['pool']==pool and x['probe']==probe and x['target']==target and x['contrast']==feature+'-FG')
 v=[];k=[]
 for pool in ['heldout8','C3']:
  vv=get(pool,'FGV',1);ss=get(pool,'shuffle',1);kk=get(pool,'FGK',2)
  vp=vv['mean'] is not None and vv['mean']>=.03 and vv['ci95'][0]>0 and ss['mean'] is not None and ss['mean']<=.005
  kp=kk['mean'] is not None and kk['mean']>=.02 and kk['ci95'][0]>0
  v.append(bool(vp));k.append(bool(kp))
 decisions[probe]=dict(V_pass=all(v),K_pass=all(k),V_by_pool=v,K_by_pool=k)
decisions['robust_V_pass']=all(decisions[p]['V_pass'] for p in ['linear','mlp']);decisions['robust_K_pass']=all(decisions[p]['K_pass'] for p in ['linear','mlp'])
write(R/'results/DV2_decision.json',decisions);print(json.dumps(decisions),flush=True)
