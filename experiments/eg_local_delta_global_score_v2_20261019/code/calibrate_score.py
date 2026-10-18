from common_v2 import *
from scipy.special import expit,logit
from scipy.optimize import minimize
start=time.time();cpu=time.process_time();assert json.loads((R/'results/S0.json').read_text())['H0_eligible'];scores=[];targets=[];weights=[]
for s in CFG['fit']:
 for f in CFG['fit_frames']:
  with np.load(P/f'labels/{s:04d}_{f:04d}.npz') as z:ids=z['ids'];u=utility(z['q'])[:,0];expected=str(z['export_sha256'])
  source=N/f'exports/subset/scene_{s:04d}/{f:04d}.npz';assert sha(source)==expected
  with np.load(source) as z:scores.append(z['pred'][ids,0].astype(np.float64))
  targets.append(u.astype(np.float64));weights.append(np.r_[np.ones(64),np.full(64,15.)])
s=np.concatenate(scores);y=np.concatenate(targets);w=np.concatenate(weights);w/=w.sum();a0=1/max(float(s.std()),1e-6);b0=float(logit(np.clip(np.dot(w,y),1e-8,1-1e-8))-a0*np.dot(w,s))
def objective(v):
 a=np.exp(v[0]);q=expit(a*s+v[1]);diff=q-y;g=2*w*diff*q*(1-q);return np.dot(w,diff**2),np.array([np.dot(g,a*s),g.sum()])
result=minimize(objective,[np.log(a0),b0],jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-12,'gtol':1e-9});assert result.success,result.message;a=float(np.exp(result.x[0]));b=float(result.x[1]);assert np.isfinite(a) and a>0 and np.isfinite(b)
q=expit(a*s+b);assert len(np.unique(q))==len(np.unique(s)),'Calibration created score ties'
gate=json.loads((R/'results/S0.json').read_text())['chosen'];heads=[]
for seed in SEEDS:
 h=json.loads((R/f'results/H0_{seed}.json').read_text())['selected']['regret'];heads.append(dict(seed=seed,family='H0',step=h['step'],sha256=h['checkpoint_sha256'],tau=gate['thresholds'][str(seed)]))
write(R/'results/score_calibration.json',dict(status='PASS',a=a,b=b,objective=float(result.fun),optimizer=str(result.message),iterations=int(result.nit),samples=len(s),weight_top64=1,weight_random64=15,scene_equal=True,score_min=float(s.min()),score_max=float(s.max()),strict_order_preserved_on_fit=True,seconds=time.time()-start,cpu_seconds=time.process_time()-cpu))
write(R/'results/locked_method.json',dict(family='H0',selection='S4 regret',gate='model-val quantile / gate-cal maximum feasible kappa',kappa=gate['grid_value'],heads=heads,score_calibration=dict(a=a,b=b),H1='NOT_RUN_H0_ELIGIBLE',H2='NOT_RUN_H0_ELIGIBLE',new_heads=0,confirmation='NOT_OPENED_REQUIRES_SEPARATE_APPROVAL'))
print('CALIBRATED',a,b,result.fun,flush=True)
