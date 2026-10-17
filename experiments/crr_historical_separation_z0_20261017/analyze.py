"""Three predefined curve families. Intervals describe fixed-data variability only."""
from pathlib import Path
import numpy as np,json,csv,time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;t=time.monotonic()
D=dict(np.load(R/'arrays/samples.npz'));hist=json.loads((R/'historical_probes.json').read_text());ev=json.loads((R/'historical_evaluations.json').read_text());P=json.loads((R/'protocol.json').read_text())
assert json.loads((R/'receipts/execution.json').read_text())['exit_code']==0
ids,inv=np.unique(D['trajectory'],return_inverse=True);rng=np.random.default_rng(2026101731)
counts=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=400);w=counts[:,inv].astype(float)
masks=[D['start'],~D['start']];weights=[w*m for m in masks];den=[x.sum(1) for x in weights];assert all(np.min(x)>0 for x in den)
def means(x):return [float(x[m].mean()) for m in masks]
def bs(x):return [ww@x/dd for ww,dd in zip(weights,den)]
def ci(x):return [float(y) for y in np.quantile(x,[.025,.975])]
rows=[];sens=[]
for seed in [0,3]:
 fork=np.load(R/f'arrays/s{seed}_B0_000000.npy')
 for delta in range(0,100001,10000):
  a=np.load(R/f'arrays/s{seed}_B0_{delta:06d}.npy');b=np.load(R/f'arrays/s{seed}_B1a_{delta:06d}.npy')
  for j in range(2):
   qa,qb=a[j,0],b[j,0];ga,gb=a[j,3],b[j,3];ma,mb=means(qa),means(qb);diff=qa-qb;md=means(diff);ad=means(np.abs(diff));da=means(qa-fork[j,0]);db=means(qb-fork[j,0]);mg0=means(ga);mg1=means(gb)
   bootdiff=bs(diff);bootga=bs(ga)[0];bootgb=bs(gb)[0]
   r=dict(seed=seed,latent=j,delta=delta,step=delta+(600000 if seed==0 else 300000),C_B0=ma[0]-ma[1],C_B1a=mb[0]-mb[1],delta_C=md[0]-md[1],Q_abs_start=ad[0],Q_abs_nonstart=ad[1],Q_signed_start=md[0],Q_signed_nonstart=md[1],drift_B0_start=da[0],drift_B0_nonstart=da[1],drift_B1a_start=db[0],drift_B1a_nonstart=db[1],G_B0_start=mg0[0],G_B1a_start=mg1[0],G_B0_nonstart=mg0[1],G_B1a_nonstart=mg1[1],gradient_ratio=mg0[0]/mg1[0],fixed_actor_Q_signed_start=means(a[j,1]-b[j,1])[0],fixed_vabc_Q_signed_start=means(a[j,2]-b[j,2])[0])
   for key,val in [('delta_C',bootdiff[0]-bootdiff[1]),('gradient_ratio',bootga/bootgb)]:r[key+'_lo'],r[key+'_hi']=ci(val)
   rows.append(r)
   for half in [0,1]:
    start=masks[0]&(D['half']==half);non=masks[1]&(D['half']==half)
    sens.append(dict(seed=seed,latent=j,half=half,delta=delta,delta_C=float(diff[start].mean()-diff[non].mean()),Q_abs_start=float(np.abs(diff[start]).mean()),gradient_ratio=float(ga[start].mean()/gb[start].mean())))
def writecsv(name,data):
 with open(R/name,'w',newline='') as f:wri=csv.DictWriter(f,fieldnames=list(data[0]));wri.writeheader();wri.writerows(data)
writecsv('metrics.csv',rows);writecsv('sensitivity.csv',sens)
alarms={}
for seed in [0,3]:
 hs=hist[f's{seed}_B0']['log'];base=600000 if seed==0 else 300000
 if seed==0: hit=[v['step'] for v in hs if v['V_online_start']-hs[0]['V_online_start']>=20]
 else:hit=[v['step'] for v in hs if (v['V_online_start']-v['V_online_all'])-(hs[0]['V_online_start']-hs[0]['V_online_all'])>=15]
 alarms[str(seed)]=dict(original_10k_delta=P['original_alarm_delta'][str(seed)],first_existing_5k_original_rule_delta=(min(hit)-base if hit else None),s3_additional_joint_alarm_delta=35000 if seed==3 else None)
def annotate(ax,seed,behavior=True):
 coarse=P['original_alarm_delta'][str(seed)]/1000;fine=alarms[str(seed)]['first_existing_5k_original_rule_delta']/1000
 ax.axvline(coarse,color='firebrick',ls='--',lw=1.2,label='Original 10k alarm')
 if fine!=coarse:ax.axvline(fine,color='purple',ls=':',lw=1.4,label='Existing 5k alarm')
 ax.set_xlim(-1,102);ax.set_xticks(range(0,101,10));ax.grid(alpha=.2);ax.set_xlabel('Updates after fork (k)')
 if behavior:
  for v in ev:
   if v['seed']!=seed:continue
   yy=.98 if v['arm'] in ['B0','fork'] else .89
   ax.plot(v['delta']/1000,yy,marker='v',color='black' if v['arm']!='B1a' else 'gray',transform=ax.get_xaxis_transform(),ms=4)
   ax.annotate(f"{v['arm']} {v['success_percent']:g}%",(v['delta']/1000,yy),xycoords=('data','axes fraction'),xytext=(-2,-10),textcoords='offset points',ha='right',va='top',fontsize=7)
def series(seed,j):return [r for r in rows if r['seed']==seed and r['latent']==j]
colors=['#147ba6','#dc7a19']
for name,key,ylabel,baseline in [('contrast','delta_C','Target-Q contrast difference: B0 minus B1a',0),('gradient_ratio','gradient_ratio','Online action-gradient norm ratio: B0 / B1a',1)]:
 fig,axs=plt.subplots(2,1,figsize=(10.6,8),sharex=True)
 for ax,seed in zip(axs,[0,3]):
  for j in range(2):
   ss=series(seed,j);x=np.array([r['delta']/1000 for r in ss]);y=np.array([r[key] for r in ss]);ax.plot(x,y,'o-',color=colors[j],label=f'Fixed latent {j}',ms=3);ax.fill_between(x,[r[key+'_lo'] for r in ss],[r[key+'_hi'] for r in ss],color=colors[j],alpha=.12)
  ax.axhline(baseline,color='gray',lw=.8);annotate(ax,seed);ax.set_title(f'Seed {seed}: fork {600 if seed==0 else 300}k; fixed actions');ax.set_ylabel(ylabel,fontsize=9);ax.legend(loc='lower left',fontsize=8,ncol=2)
 fig.suptitle('Historical separation screen | no new training or environment calls\nBands: paired data-trajectory bootstrap, not training-seed uncertainty',fontsize=10);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(R/f'{name}.png',dpi=170);plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(15,8),sharex=True)
for i,seed in enumerate([0,3]):
 for j in range(2):
  ss=series(seed,j);x=[r['delta']/1000 for r in ss]
  axs[i,0].plot(x,[r['Q_abs_start'] for r in ss],color=colors[j],label=f'z{j} start');axs[i,0].plot(x,[r['Q_abs_nonstart'] for r in ss],'--',color=colors[j],label=f'z{j} nonstart')
  for col,region in [(1,'start'),(2,'nonstart')]:
   axs[i,col].plot(x,[r[f'drift_B0_{region}'] for r in ss],color=colors[j],label=f'z{j} B0');axs[i,col].plot(x,[r[f'drift_B1a_{region}'] for r in ss],'--',color=colors[j],label=f'z{j} B1a')
 for col,title in enumerate(['Mean absolute B0-B1a Q difference','Start: signed Q drift from fork','Nonstart: signed Q drift from fork']):
  ax=axs[i,col];annotate(ax,seed);ax.axhline(0,color='gray',lw=.6);ax.set_title(f's{seed} | {title}',fontsize=9);ax.legend(loc='lower left',fontsize=7,ncol=2)
fig.suptitle('Fixed Actor + VaBC action scores from Target Critic; Q units, no action resampling',fontsize=11);fig.tight_layout(rect=(0,0,1,.96));fig.savefig(R/'q_differences.png',dpi=170);plt.close(fig)
(R/'alarm_timing.json').write_text(json.dumps(alarms,indent=2))
(R/'receipts/analysis.json').write_text(json.dumps(dict(status='COMPLETE',seconds=time.monotonic()-t,trajectory_clusters=len(ids),bootstrap_replicates=400,metric_rows=len(rows),sensitivity_rows=len(sens),training_updates=0,environment_steps=0,posthoc_numeric_risk_threshold_fitted=False),indent=2))
print('ANALYSIS_COMPLETE',len(rows),len(sens),alarms,flush=True)
for seed in [0,3]:
 for delta in range(10000,100001,10000):
  rr=[r for r in rows if r['seed']==seed and r['delta']==delta]
  print(seed,delta,'DeltaC',*[round(r['delta_C'],3) for r in rr],'CI',*[list(np.round([r['delta_C_lo'],r['delta_C_hi']],3)) for r in rr],'G ratio',*[round(r['gradient_ratio'],3) for r in rr],'Qabs(start,non)',[(round(r['Q_abs_start'],2),round(r['Q_abs_nonstart'],2)) for r in rr],flush=True)
