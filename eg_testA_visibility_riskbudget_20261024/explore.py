"""Exploratory (NOT preregistered) decomposition after Test A primary = VIS-NO."""
import sys, json, runpy, numpy as np
g = runpy.run_path(sys.argv[3], run_name='testA_main')  # re-executes frozen primary; reuses its objects
C3, TR, IX, NOM = g['C3'], g['TR'], g['IX'], g['NOM']
features, fit_logit, auroc, variant_mask = g['features'], g['fit_logit'], g['auroc'], g['variant_mask']
choose, outcomes, counts, scene_rate, interval, LAMS = g['choose'], g['outcomes'], g['counts'], g['scene_rate'], g['interval'], g['LAMS']
def extra(A, names, stats=None):
    n=len(A['original_angle']); ix=np.arange(n); nom=A['original_angle']
    X,_=features(A,'d'); cols=[]
    for nm in names:
        v=A[nm].astype(float); cols += [v, v - v[ix,nom][:,None]]
    if not cols: return X, None
    C=np.stack(cols,-1)
    if stats is None: f=C.reshape(-1,C.shape[-1]); stats=(f.mean(0), f.std(0)+1e-12)
    return np.concatenate([X,(C-stats[0])/stats[1]],-1), stats
for A in (TR, C3):
    A['eg_ratio'] = None
# EG ratio R1 lives in per_pose too
import glob, os
def addR1(A, split):
    vals=[np.load(f)['R1'] for f in sorted(glob.glob(f"{sys.argv[1]}/eg_n0_nref_main_20261024/per_pose/{split}_*.npz"))]
    A['eg_ratio']=np.concatenate(vals)
addR1(TR,'train'); addR1(C3,'C3')
mtr=variant_mask(TR); ytr=TR['collision'][mtr].astype(float); mc3=variant_mask(C3); yc3=C3['collision'][mc3].astype(float)
specs={'d':[], 'de':['eg_ratio'], 'dv':['ref_unk'], 'do':['ref_occ'], 'deo':['eg_ratio','ref_occ'], 'dvo':['ref_unk','ref_occ'], 'devo':['eg_ratio','ref_unk','ref_occ']}
res={}
def match(p):
    rec=[(lam,)+tuple(int(outcomes(choose(p,lam))[k].sum()) for k in ['recovery','collision','harm']) for lam in LAMS]
    gap=np.array([abs(r[1]-141) for r in rec]); i=np.flatnonzero(gap==gap.min())[-1]; return rec[i][0]
base=None
for name,cols in specs.items():
    X,st=extra(TR,cols); w=fit_logit(X[mtr],ytr); Xc,_=extra(C3,cols,st); z=Xc@w; p=1/(1+np.exp(-z))
    lam=match(p); o=outcomes(choose(p,lam)); r=dict(train_auc=auroc(X[mtr]@w,ytr), c3_auc=auroc(z[mc3],yc3), lam=lam, **counts(o))
    if name=='d': base=o
    r['minus_d_coll_rate']=interval(scene_rate(o['collision'])-scene_rate(base['collision']))
    res[name]=r
# constant-risk control: move only if I1 gain > tau
p=np.ones_like(C3['I1']); lam=match(p); o=outcomes(choose(p,lam)); res['gain_threshold_only']=dict(lam=lam,**counts(o), minus_d_coll_rate=interval(scene_rate(o['collision'])-scene_rate(base['collision'])))
json.dump(res,open(f'{sys.argv[2]}/exploratory.json','w'),indent=1)
for k,v in res.items(): print(k, {kk:(round(vv,3) if isinstance(vv,float) else vv) for kk,vv in v.items() if kk in ['train_auc','c3_auc','recovery','collision','harm','feasible','coll_noobs']}, {kk:round(vv*100,2) for kk,vv in v['minus_d_coll_rate'].items()})
# pairwise extras (exploratory)
def sel(name):
    cols=specs[name]; X,st=extra(TR,cols); w=fit_logit(X[mtr],ytr); Xc,_=extra(C3,cols,st); p=1/(1+np.exp(-(Xc@w))); return outcomes(choose(p,res[name]['lam']))
O={k:sel(k) for k in ['do','de','dvo']}; L3o=outcomes(g['L3'])
pairs={'do_minus_de':(O['do'],O['de']),'do_minus_L3':(O['do'],L3o),'dvo_minus_L3':(O['dvo'],L3o)}
out={}
for k,(a,b) in pairs.items():
    out[k]={m:interval(scene_rate(a[m])-scene_rate(b[m])) for m in ['collision','harm','feasible','recovery']}
json.dump(out,open(f'{sys.argv[2]}/exploratory_pairs.json','w'),indent=1)
for k,v in out.items(): print('PAIR',k,{m:[round(x*100,2) for x in vv.values()] for m,vv in v.items()})
