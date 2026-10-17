"""Apply the frozen FACB-Z0 decision; never search or change the bandwidth."""
from pathlib import Path
import json,csv,time,hashlib,resource
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;t=time.monotonic()
def load(n):return json.loads((R/n).read_text())
def save(n,x):(R/n).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ex=load('receipts/inference_exit.json');assert ex['exit_code']==0 and not ex['timeout']
assert sha(R/'calibration.json')==load('receipts/calibration_lock.json')['sha256']
cal=load('calibration.json');c=cal['c'];s=dict(np.load(R/'arrays/samples.npz'));v=np.load(R/'arrays/anchor_values.npy')
rows=[r for r in load('checkpoint_manifest.json') if r['arm']=='G'];qs={r['id']:np.load(R/f'arrays/{r["id"]}_q.npy') for r in rows}
assert all(q.shape==(2,7168) and np.isfinite(q).all() for q in qs.values())
reg={'all_unique':np.ones(len(s['pool']),bool),'global_uniform':s['pool']=='global','start_dedicated':s['pool']=='start','near_goal_dedicated':s['pool']=='near_goal','all_next_start':s['start'],'all_next_outside_start':~s['start']}
quantiles=[0,.01,.05,.25,.5,.75,.95,.99,1]
def stats(d):
 d=np.asarray(d).reshape(-1);assert len(d)>0 and np.isfinite(d).all()
 return dict(n=int(d.size),upper=float((d>c).mean()),lower=float((d<-c).mean()),mean=float(d.mean()),std=float(d.std()),quantile_levels=quantiles,quantiles=np.quantile(d,quantiles,method='linear').tolist())
details=[]
for row in rows:
 d=qs[row['id']]-v[row['seed']][None,:]
 for label,m in reg.items():
  for repeat in ('pooled',0,1):
   a=d[:,m] if repeat=='pooled' else d[repeat,m]
   details.append(dict(id=row['id'],seed=row['seed'],step=row['step'],group=row['group'],region=label,noise=repeat,**stats(a)))
healthy=[r for r in rows if r['group']=='healthy_test'];islands=[r for r in rows if r['group']=='island']
def combined(items,region='all_unique',noise='pooled'):
 ds=[qs[r['id']]-v[r['seed']][None,:] for r in items];m=reg[region]
 return stats(np.concatenate([(d[:,m] if noise=='pooled' else d[noise,m]).reshape(-1) for d in ds]))
healthy_stats={region:{str(n):combined(healthy,region,n) for n in ('pooled',0,1)} for region in reg}
island_stats={r['id']:{str(n):combined([r],'start_dedicated',n) for n in ('pooled',0,1)} for r in islands}
passes=[]
for n in ('pooled',0,1):
 passes.append(dict(noise=n,healthy_upper_pass=healthy_stats['all_unique'][str(n)]['upper']<=.01,islands_upper_pass={r['id']:island_stats[r['id']][str(n)]['upper']>.5 for r in islands}))
go=all(p['healthy_upper_pass'] and all(p['islands_upper_pass'].values()) for p in passes)
decision='GO_FOR_CONTROLLED_SCREEN' if go else 'NO_GO_PROXY_ANCHOR_SEPARABILITY'
lower=healthy_stats['all_unique']['pooled']['lower']
recommendation=('upper_only_candidate' if lower>.05 else 'one_sided_and_two_sided_candidates') if go else 'no_training_with_current_proxy'
seed_summary={str(seed):{region:combined([r for r in healthy if r['seed']==seed],region) for region in reg} for seed in (4,5,6,7)}
# Cross-seed anchors use exactly the original c; no calibration or selection.
sensitivity=[]
for row in [r for r in rows if r['group'] in ('healthy_test','island')]:
 region='start_dedicated' if row['group']=='island' else 'all_unique';m=reg[region]
 for anchor_seed in range(8):
  sensitivity.append(dict(id=row['id'],group=row['group'],anchor_seed=anchor_seed,own_seed=anchor_seed==row['seed'],region=region,**stats(qs[row['id']][:,m]-v[anchor_seed,m][None,:])))
# Overlap is descriptive, not an extra decision rule or a tuned classifier.
healthy_start=np.concatenate([(qs[r['id']]-v[r['seed']][None,:])[:,reg['start_dedicated']].reshape(-1) for r in healthy])
overlap={}
for r in islands:
 x=(qs[r['id']]-v[r['seed']][None,:])[:,reg['start_dedicated']].reshape(-1)
 edges=np.linspace(min(x.min(),healthy_start.min()),max(x.max(),healthy_start.max()),201)
 h=np.histogram(healthy_start,edges)[0]/healthy_start.size;j=np.histogram(x,edges)[0]/x.size
 overlap[r['id']]=dict(histogram_overlap_coefficient=float(np.minimum(h,j).sum()),bins=200,note='Descriptive density overlap; not a true-value error or extra decision gate')
result=dict(decision=decision,calibration=cal,threshold_checks=passes,healthy_overall=healthy_stats,healthy_by_seed=seed_summary,island_start=island_stats,lower_bound_priority=recommendation,healthy_lower_above_5percent=lower>.05,overlap=overlap,details=details,caveats=['I-arm V is a proxy, not independent FACB pretraining','d=qG-V measures model disagreement, not true value error','No policy efficacy or causal training conclusion','Shares transitions across models and noise repeats; counts are descriptive, not independent sample inference','Main calibration/healthy stages 600k/800k/1M; auxiliary 400k/700k stage matching cannot recalibrate c'])
cald=np.concatenate([(qs[x['id']]-v[x['seed']][None,:]).reshape(-1) for x in rows if x['group']=='calibration'])
result['calibration_signed_audit']=dict(negative_fraction=float((cald<0).mean()),upper_trigger=float((cald>c).mean()),lower_trigger=float((cald<-c).mean()),mean=float(cald.mean()),median=float(np.median(cald)),top_absolute_1percent_negative_fraction=float((cald[np.abs(cald)>=c]<0).mean()))
save('separability.json',result);save('anchor_sensitivity.json',sensitivity)
with (R/'trigger_rates.csv').open('w',newline='') as f:
 fields=['id','seed','step','group','region','noise','n','upper','lower','mean','std'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in details)
fig,axs=plt.subplots(2,2,figsize=(12,9))
for ax,region,title in [(axs[0,0],'global_uniform','Uniform global states'),(axs[0,1],'start_dedicated','Start-region states')]:
 arrays=[]
 for label,rs,color in [('Healthy s4-7',healthy,'#2471a3'),('Island s0',islands[:1],'#c0392b'),('Island s3',islands[1:],'#8e44ad')]:
  d=np.concatenate([(qs[r['id']]-v[r['seed']][None,:])[:,reg[region]].reshape(-1) for r in rs]);arrays.append(d)
  ax.hist(d,bins=70,density=True,histtype='step',linewidth=1.5,label=label,color=color)
 ax.axvline(c,color='black',linestyle='--',label='Frozen +c');ax.axvline(-c,color='gray',linestyle=':')
 ax.set(title=title,xlabel='d = native GFP target score - frozen proxy V',ylabel='Density');ax.legend(fontsize=8)
regions=['global_uniform','start_dedicated','near_goal_dedicated'];labels=['Global','Start','Near goal'];x=np.arange(3)
for shift,key,color in [(-.16,'upper','#c0392b'),(.16,'lower','#2471a3')]:
 axs[1,0].bar(x+shift,[100*healthy_stats[r]['pooled'][key] for r in regions],.32,label=key,color=color)
axs[1,0].set(xticks=x,xticklabels=labels,ylabel='Trigger percent',title='Held-out healthy: upper and lower');axs[1,0].legend()
names=['Healthy','Island s0','Island s3']
for shift,n,color in [(-.16,0,'#2471a3'),(.16,1,'#d35400')]:
 vals=[100*healthy_stats['all_unique'][str(n)]['upper']]+[100*island_stats[r['id']][str(n)]['upper'] for r in islands]
 axs[1,1].bar(np.arange(3)+shift,vals,.32,label=f'Noise {n+1}',color=color)
axs[1,1].set(xticks=np.arange(3),xticklabels=names,ylabel='Upper trigger percent',title='Frozen threshold: two noise repeats');axs[1,1].axhline(50,color='gray',linestyle=':');axs[1,1].legend()
fig.suptitle('FACB-Z0: proxy-anchor separability (no training)',fontsize=14);fig.tight_layout();fig.savefig(R/'distributions.png',dpi=140);plt.close(fig)
save('receipts/analysis_costs.json',dict(wall_seconds=time.monotonic()-t,cpu_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,cpu_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,training_updates=0,environment_steps=0))
print(json.dumps(dict(decision=decision,c=c,kappa=cal['kappa'],healthy_upper=healthy_stats['all_unique']['pooled']['upper'],healthy_lower=lower,islands=island_stats,priority=recommendation)),flush=True)
