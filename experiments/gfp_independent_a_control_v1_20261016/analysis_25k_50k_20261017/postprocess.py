import json
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent
a=np.load(p/'frozen_outputs.npz')
q=np.load('/home/xushijie3/work/gfp_reference_support_diagnostic_v1_20261016/arrays/queries.npz')
rows=[]
for s in np.unique(q['split']):
 for r in (0,1):
  m=(q['split']==s)&(q['region']==r)
  x,y=[np.stack([a[f'50000_{z}_{k}']-a[f'25000_{z}_{k}'] for z in (0,1)])[:,m] for k in ('actor','vabc')]
  cos=(x*y).sum(-1)/(np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1)+1e-12)
  rows.append(dict(split=str(s),region='start' if r else 'outside',delta_cosine_mean=float(cos.mean()),delta_cosine_positive_fraction=float((cos>0).mean()),delta_difference_rms=float(np.sqrt(np.mean((x-y)**2))),pooled_delta_cosine=float((x*y).sum()/np.sqrt((x*x).sum()*(y*y).sum()))))
(p/'delta_alignment.json').write_text(json.dumps(rows,indent=2))
