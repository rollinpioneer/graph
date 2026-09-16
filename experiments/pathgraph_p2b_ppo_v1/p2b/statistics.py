from __future__ import annotations
import numpy as np

def crossed_bootstrap(delta,confidence=.975,resamples=10000,seed=20260916):
    d=np.asarray(delta,dtype=float)
    if d.ndim!=2 or not np.isfinite(d).all() or min(d.shape)<2:raise ValueError('need complete seed x family matrix')
    rng=np.random.default_rng(seed);a,b=d.shape;out=np.empty(resamples)
    for i in range(resamples):
        si=rng.integers(a,size=a);fi=rng.integers(b,size=b)
        out[i]=d[np.ix_(si,fi)].mean()
    q=(1-confidence)/2
    return {'mean':float(d.mean()),'low':float(np.quantile(out,q)),
            'high':float(np.quantile(out,1-q)),'confidence':confidence,
            'n_policy_seeds':a,'n_eval_families':b,'unit':'crossed_seed_and_family'}

def normalized_auc(steps,rates):
    x=np.asarray(steps,dtype=float);y=np.asarray(rates,dtype=float)
    if len(x)<2 or len(x)!=len(y) or x[0]!=0 or not np.all(np.diff(x)>0):raise ValueError('bad checkpoints')
    return float(np.sum((y[1:]+y[:-1])*.5*np.diff(x))/(x[-1]-x[0]))

def first_threshold(steps,rates,target=.8,consecutive=2):
    for i in range(len(rates)-consecutive+1):
        if all(float(v)>=target for v in rates[i:i+consecutive]):
            return {'reached':True,'first_observed_step':int(steps[i]),'validated_at_step':int(steps[i+consecutive-1]),'right_censored':False}
    return {'reached':False,'first_observed_step':None,'validated_at_step':None,'right_censored':True,'last_observed_step':int(steps[-1])}
