"""Freeze R_d, R_do, R_de coefficients (fit on training frames only) for the P2 confirmation."""
import sys, json, runpy, numpy as np, glob
g = runpy.run_path(sys.argv[3], run_name='testA_main')
TR=g['TR']; features=g['features']; fit_logit=g['fit_logit']; variant_mask=g['variant_mask']
TR['eg_ratio']=np.concatenate([np.load(f)['R1'] for f in sorted(glob.glob(f"{sys.argv[1]}/eg_n0_nref_main_20261024/per_pose/train_*.npz"))])
def design(A, names, stats=None):
    n=len(A['original_angle']); ix=np.arange(n); nom=A['original_angle']
    X,_=features(A,'d'); cols=[]
    for nm in names:
        v=A[nm].astype(float); cols += [v, v - v[ix,nom][:,None]]
    if not cols: return X, None
    C=np.stack(cols,-1)
    if stats is None: f=C.reshape(-1,C.shape[-1]); stats=(f.mean(0), f.std(0)+1e-12)
    return np.concatenate([X,(C-stats[0])/stats[1]],-1), stats
m=variant_mask(TR); y=TR['collision'][m].astype(float); out={}
ex=json.load(open(f'{sys.argv[2]}/exploratory.json'))
for name,cols in {'R_d':[], 'R_do':['ref_occ'], 'R_de':['eg_ratio']}.items():
    X,st=design(TR,cols); w=fit_logit(X[m],y)
    out[name]=dict(features=['d==%d'%j for j in range(1,7)]+sum([[c,'delta_'+c] for c in cols],[]), w=w.tolist(),
                   mean=None if st is None else st[0].tolist(), std=None if st is None else st[1].tolist(),
                   C3_matched_lambda=ex[{'R_d':'d','R_do':'do','R_de':'de'}[name]]['lam'])
json.dump(out,open(f'{sys.argv[2]}/frozen_models_P2.json','w'),indent=1)
print(json.dumps(out,indent=1))
