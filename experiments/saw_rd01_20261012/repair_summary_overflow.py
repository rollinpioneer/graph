"""Reaggregate retained raw probes in float64, without querying models again."""
from common import *
for kind in ('navigate','stitch'):
    raw=np.load(ROOT/'runs'/f'{kind}_D4_4096_raw.npz')
    path=ROOT/'results'/f'{kind}_discovery_DIAGNOSIS.json'
    result=json.load(open(path))
    for k,item in result['D4_full_training_style_4096'].items():
        values=raw[k];finite=values[np.isfinite(values)]
        item['mean']=float(np.mean(finite,dtype=np.float64)) if len(finite) else None
    result['numeric_aggregation_note']='Full-probe finite-value means accumulated in float64 from retained raw arrays; avoids float32 sum overflow. Actual float32 untruncated exponential overflow counts remain unchanged. No new model queries, samples, optimization or rollouts.'
    write_json(path,result)
print('RAW_SUMMARY_REAGGREGATED_NO_NEW_PROBES',flush=True)
