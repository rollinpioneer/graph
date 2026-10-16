from util import *
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
t0=time.time();q=np.load(ROOT/'arrays/queries.npz');rel=json.loads((ROOT/'reports/reliability.json').read_text());steps=list(range(0,50001,5000));rows=[];groups={}
for split in ['train','val']:
 for reg in [0,1]:
  mask=(q['split']==split)&(q['region']==reg);ep,inv=np.unique(q['trajectory'][mask],return_inverse=True)
  count=np.bincount(inv);rng=np.random.default_rng(82101);weights=rng.multinomial(len(ep),np.full(len(ep),1/len(ep)),size=1000)
  groups[(split,reg)]=(mask,inv,count,weights,len(ep))
def interval(v,group):
 _,inv,count,w,_=group;s=np.bincount(inv,weights=v,minlength=len(count));boot=(w@s)/(w@count)
 return np.quantile(boot,[.025,.975]).tolist()
for arm in ['L0','L1a','L1b']:
 base=np.load(ROOT/'arrays'/f'{arm}_0.npz')
 for it in steps:
  d=np.load(ROOT/'arrays'/f'{arm}_{it}.npz')
  for (split,reg),group in groups.items():
   mask=group[0]
   for mi,seed in enumerate([1701,1702]):
    threshold=rel[mi]['cutoffs'][reg]
    for repeat in [0,1]:
     v=d['nll'][mi,repeat,mask];v0=base['nll'][mi,repeat,mask];real=d['real_nll'][mi,mask]
     delta=v-v0;tail=(v>threshold).astype(float);tail0=(v0>threshold).astype(float)
     rows.append(dict(arm=arm,updates=it,split=split,region='start' if reg else 'outside',density_seed=seed,noise_repeat=repeat,n=len(v),trajectories=group[4],nll=float(v.mean()),nll_minus_real=float((v-real).mean()),tail_fraction=float(tail.mean()),actual_data_tail=float((real>threshold).mean()),delta_nll=float(delta.mean()),delta_nll_ci=interval(delta,group),delta_tail=float((tail-tail0).mean()),delta_tail_ci=interval(tail-tail0,group),action_rms_shift=float(np.sqrt(np.mean((d['actions'][repeat,mask]-base['actions'][repeat,mask])**2)))))
write('reports/curves.json',rows)
with open(ROOT/'reports/curves.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
compact=[]
for arm in ['L0','L1a','L1b']:
 for split in ['train','val']:
  for region in ['start','outside']:
   for it in [0,10000,20000,25000,40000,45000,50000]:
    rr=[x for x in rows if x['arm']==arm and x['split']==split and x['region']==region and x['updates']==it]
    compact.append(dict(arm=arm,split=split,region=region,updates=it,mean_nll=np.mean([x['nll'] for x in rr]),nll_minus_real=np.mean([x['nll_minus_real'] for x in rr]),tail_fraction=np.mean([x['tail_fraction'] for x in rr]),delta_nll=np.mean([x['delta_nll'] for x in rr]),delta_nll_range=[min(x['delta_nll'] for x in rr),max(x['delta_nll'] for x in rr)],delta_tail=np.mean([x['delta_tail'] for x in rr]),action_rms_shift=np.mean([x['action_rms_shift'] for x in rr])))
write('reports/compact.json',compact)
# Cluster intervals here average density fits/noise first; fits are NOT independent experiments.
fig,ax=plt.subplots(2,2,figsize=(12,8),sharex=True)
colors={'L0':'#c43b3b','L1a':'#267d4b','L1b':'#c18a14'}
for ci,reg in enumerate([1,0]):
 group=groups[('val',reg)];mask=group[0]
 for arm in colors:
  base=np.load(ROOT/'arrays'/f'{arm}_0.npz');means=[];cis=[];tails=[]
  for it in steps:
   d=np.load(ROOT/'arrays'/f'{arm}_{it}.npz');delta=(d['nll'][:,:,mask]-base['nll'][:,:,mask]).mean((0,1));means.append(delta.mean());cis.append(interval(delta,group))
   tails.append(np.mean([np.mean(d['nll'][mi,:,mask]>rel[mi]['cutoffs'][reg]) for mi in range(2)]))
  x=np.array(steps)/1000;cis=np.array(cis)
  ax[0,ci].plot(x,means,label=arm,color=colors[arm]);ax[0,ci].fill_between(x,cis[:,0],cis[:,1],alpha=.15,color=colors[arm]);ax[1,ci].plot(x,np.array(tails)*100,color=colors[arm],label=arm)
 for ai in range(2):
  ax[ai,ci].axvline(25,color='gray',linestyle=':',label='behavior already poor by +25k' if ai==0 else None);ax[ai,ci].grid(alpha=.2)
 ax[0,ci].set_title(('Start' if reg else 'Outside')+' | held-out trajectories');ax[1,ci].set_xlabel('Continuation updates (k)')
ax[0,0].set_ylabel('Change in NLL / action dimension\n(higher = less modeled support)');ax[1,0].set_ylabel('Above calibrated 95% data threshold (%)');ax[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT/'reports/support_curves.png',dpi=170);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(11,4))
for mi,s in enumerate([1701,1702]):
 r=rel[mi];names=['ordinary']+[v['group'] for v in r['positive_controls']];rates=[r['actual_test_reject']]+[v['false_reject'] or 0 for v in r['positive_controls']]
 axs[mi].bar(names,np.array(rates)*100);axs[mi].axhline(5,color='black',ls=':');axs[mi].axhline(20,color='red',ls='--');axs[mi].set_title(f'Density fit {s}: observed-action rejection');axs[mi].tick_params(axis='x',rotation=40);axs[mi].set_ylabel('%')
fig.tight_layout();fig.savefig(ROOT/'reports/evaluator_reliability.png',dpi=170);plt.close(fig)
write('receipts/analysis.json',dict(seconds=time.time()-t0,cluster_bootstrap_replicates=1000,gfp_updates=0,environment_steps=0,rows=len(rows)))
print('ANALYZED',len(rows),flush=True)
