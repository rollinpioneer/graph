from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from scipy.stats import spearmanr
O=Path(__file__).parent
d=pd.read_csv(O/'eg_final_candidates.csv',low_memory=False)
c=pd.DataFrame([r for p in sorted((O/'contacts').glob('*.json')) for r in json.loads(p.read_text())['rows']])
x=d[d.P0_category.isin(['feasible','weak_quality'])].merge(c,on=['scene','frame','candidate_id','object_id'],validate='one_to_one')
assert len(x)==1229 and c.shape[0]==1229
x['weak']=x.P0_category.eq('weak_quality');x['EG_risk']=-x.original_score
for mode in ['raw','sampled']:
 x[mode+'_support_m']=x[[mode+'_left_m',mode+'_right_m']].max(axis=1,skipna=False)
 x[mode+'_low_support']=x[mode+'_support_m']>.01
if (O/'pcr').exists():
 p=pd.DataFrame([r for p in sorted((O/'pcr').glob('*.json')) for r in json.loads(p.read_text())['rows']])
 x=x.merge(p,on=['scene','frame','candidate_id','object_id'],validate='one_to_one')
 assert len(x)==1229 and x[['scene','frame']].drop_duplicates().shape[0]==48, 'PCR panel is not complete yet'
rng=np.random.default_rng(20261024)
scenes=sorted(x.scene.unique());indices=np.concatenate([rng.integers(i,i+4,size=(10000,4)) for i in [0,4,8]],axis=1)
def ci(values):
 vals=np.array(values,dtype=float);sample=np.nanmean(vals[indices],axis=1);return dict(mean=float(np.nanmean(vals)),ci95=np.nanquantile(sample,[.025,.975]).tolist())
obs=[]
for s,z in x.groupby('scene'):
 for mode in ['raw','sampled']:
  valid=z[z.contact_status.eq('ok')];a=valid[valid.weak][mode+'_low_support'].mean();b=valid[~valid.weak][mode+'_low_support'].mean()
  obs.append(dict(scene=int(s),mode=mode,weak_low=float(a),feasible_low=float(b),difference=float(a-b),missing=int((~z.contact_status.eq('ok')).sum())))
obs=pd.DataFrame(obs);obs.to_csv(O/'observability_scene.csv',index=False)
stats={'n':len(x),'contact_status':x.contact_status.value_counts().to_dict(),'observability':{mode:dict(difference=ci(obs[obs['mode'].eq(mode)].difference),positive_scenes=int((obs[obs['mode'].eq(mode)].difference>0).sum()),groups={cat:dict(n=len(z),low_support=int(z[mode+'_low_support'].sum()),median_mm=float(z[mode+'_support_m'].median()*1000)) for cat,z in x.groupby('P0_category')}) for mode in ['raw','sampled']}}
metrics=[];pairs=[]
risks=[v for v in ['EG_risk','first_contact_risk','RA','RC','RS','PCR'] if v in x]
for risk in risks:
 for s,z in x.groupby('scene'):
  z=z[z[risk].notna()];metrics.append(dict(risk=risk,scene=int(s),n=len(z),auc=float(roc_auc_score(z.weak,z[risk])),ap=float(average_precision_score(z.weak,z[risk])),rho=float(spearmanr(z[risk],z.P0_mu).statistic) if z[risk].nunique()>1 else None))
 for (s,f,o),z in x.groupby(['scene','frame','object_id']):
  bad=z[z.weak];good=z[~z.weak]
  for _,b in bad.iterrows():
   for _,g in good.iterrows():
    if b.original_score>g.original_score and np.isfinite(b[risk]) and np.isfinite(g[risk]):pairs.append(dict(risk=risk,scene=int(s),frame=int(f),object_id=int(o),bad_id=int(b.candidate_id),good_id=int(g.candidate_id),correct=float(b[risk]>g[risk])+.5*float(b[risk]==g[risk])))
m=pd.DataFrame(metrics);p=pd.DataFrame(pairs);m.to_csv(O/'risk_scene_metrics.csv',index=False);p.to_csv(O/'inversion_pairs.csv',index=False)
stats['risks']={risk:{k:ci(m[m.risk.eq(risk)][k]) for k in ['auc','ap','rho']} for risk in risks}
stats['pair_correct']={risk:dict(n=len(z),scene_macro=ci(z.groupby('scene').correct.mean().reindex(scenes))) for risk,z in p.groupby('risk')}
x.to_csv(O/'candidate_results.csv',index=False);(O/'statistics.json').write_text(json.dumps(stats,indent=2))
print(json.dumps(stats,indent=2))
