from p2_residual import *
from scipy.stats import rankdata
d=dict(np.load(R/'head_inputs.npz'));cal=json.loads((R/'results/readout_init.json').read_text())['calibration'];fit=d['role']=='fit';rows=[];convergence=[];training=[]
for fam in FAMILIES:
 for seed in SEEDS:
  theta=np.load(R/f'heads/{fam}/{seed}/step1000.npz')['theta'];qq=np.asarray([predict(fam,theta,t,s,logit,cal) for t,s,logit in zip(d['T'],d['S'],d[f'logit_{seed}'])]);target=d['Y'].max(-1) if fam=='Epot' else d['Y'][:,:,0]
  for i,q in enumerate(qq):
   mu=d['mu'][i,:64,0];eligible=(~d['collision'][i,:64,0])&(((mu>0)&(mu<=.4000001))|(mu>=.5999999));labels=mu[eligible]<=.4000001;scores=q[:64][eligible];n1=int(labels.sum());n0=len(labels)-n1
   auc=float((rankdata(scores)[labels].sum()-n1*(n1+1)/2)/(n1*n0)) if n1 and n0 else None
   rows.append(dict(family=fam,seed=seed,scene=int(d['scene'][i]),frame=int(d['frame'][i]),role=str(d['role'][i]),u0_MSE=float(np.mean((q-d['Y'][i,:,0])**2)),m_MSE=float(np.mean((q-d['Y'][i].max(-1))**2)),decision_AUROC=auc))
  if fam!='Gcap65':
   x=np.column_stack([d['T'][fit].reshape(-1,64),np.ones(int(fit.sum())*128)]);calib=cal[fam];yy=(target[fit].reshape(-1)-calib['beta'])/calib['alpha']-d['S'][fit].reshape(-1);coef=np.linalg.lstsq(x,yy,rcond=None)[0];ols=calib['alpha']*(d['S'][fit].reshape(-1)+x@coef)+calib['beta'];best=float(np.mean((ols-target[fit].reshape(-1))**2));actual=float(np.mean((qq[fit]-target[fit])**2))
   convergence.append(dict(family=fam,seed=seed,step1000_fit_MSE=actual,OLS_fit_MSE=best,MSE_gap=actual-best,OLS_deployed=False))
  training.append(json.loads((R/f'results/train_{fam}_{seed}.json').read_text()))
summary=[]
for fam in FAMILIES:
 for role in ['fit','model_val','gate_cal']:
  rr=[x for x in rows if x['family']==fam and x['role']==role];summary.append(dict(family=fam,role=role,**{k:float(np.mean([x[k] for x in rr if x[k] is not None])) for k in ['u0_MSE','m_MSE','decision_AUROC']}))
write(R/'results/offline.json',dict(summary=summary,diagnostic_only=True));csv_write(R/'results/offline_per_frame.csv',rows);write(R/'results/convergence.json',dict(rows=convergence,Gcap65_OLS='not applicable to sigmoid adapter',diagnostic_only=True));write(R/'results/training_summary.json',dict(heads=[{k:v for k,v in x.items() if k!='losses'} for x in training],GPU_process_seconds=sum(x['seconds'] for x in training),effective_updates=9000));print(json.dumps(dict(offline=summary,convergence=convergence)))
