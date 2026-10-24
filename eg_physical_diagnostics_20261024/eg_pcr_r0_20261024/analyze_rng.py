from pathlib import Path
import json
import numpy as np,pandas as pd
O=Path(__file__).parent;rows=[];candidates=[]
for p in sorted((O/'rng_evaluation').glob('*.json')):
 meta=json.loads(p.read_text());z=np.load(p.with_suffix('.npz'));ix={int(k):j for j,k in enumerate(z['candidate_id'])}
 cat=np.where(z['unknown_reason']!='','unknown',np.where(z['empty'],'empty',np.where(z['collision'],'collision',np.where((z['mu']>0)&(z['mu']<=.4000001),'feasible',np.where(z['mu']>0,'weak_quality','no_force_closure')))))
 for stage,st in meta['stages'].items():
  ids=st['final'];js=np.array([ix[k] for k in ids]);cc=cat[js];top=js[0];obj=z['object_id'][top];clean=js[~z['collision'][js]&~z['empty'][js]][:10]
  rows.append(dict(scene=meta['scene'],frame=meta['frame'],stage=stage,raw_count=meta['raw_count'],top_id=int(ids[0]),top_object=int(obj),top_mu=float(z['mu'][top]),top_category=cat[top],same_object_alternative=bool(np.any((cat[js]=='feasible')&(z['object_id'][js]==obj))),weak_ff10=float((cat[clean]=='weak_quality').mean()),all_friction_ff10=float(np.isin(cat[clean],['weak_quality','no_force_closure']).mean()),clean10_n=len(clean),unknown=int((cc=='unknown').sum())))
  for rank,(cid,j) in enumerate(zip(ids,js),1):candidates.append(dict(scene=meta['scene'],frame=meta['frame'],stage=stage,candidate_id=cid,rank=rank,object_id=int(z['object_id'][j]),mu=float(z['mu'][j]),category=cat[j],score=float(z['pred'][cid,0])))
d=pd.DataFrame(rows);assert len(d)==96;d.to_csv(O/'rng_frame_results.csv',index=False);pd.DataFrame(candidates).to_csv(O/'rng_final_candidates.csv',index=False)
rng=np.random.default_rng(20261024);idx=np.concatenate([rng.integers(i,i+4,(10000,4)) for i in [0,4,8]],axis=1)
def ci(x):
 v=np.array(x);return dict(mean=float(v.mean()),ci95=np.quantile(v[idx].mean(1),[.025,.975]).tolist())
out={}
for stage,z in d.groupby('stage'):
 weak=z[z.top_category.eq('weak_quality')];out[stage]=dict(top1_categories=z.top_category.value_counts().to_dict(),top1_feasible=ci(z.assign(ok=z.top_category.eq('feasible')).groupby('scene').ok.mean()),weak_scenes=int(weak.scene.nunique()),weak_objects=int(weak.top_object.nunique()),weak_with_same_object_alternative=int(weak.same_object_alternative.sum()),weak_ff10=ci(z.groupby('scene').weak_ff10.mean()),all_friction_ff10=ci(z.groupby('scene').all_friction_ff10.mean()),unknown=int(z.unknown.sum()),raw_candidates=int(z.raw_count.sum()))
(O/'rng_statistics.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
