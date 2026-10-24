import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
def status(mu,c,e):
 if e:return 'empty'
 if c:return 'collision'
 if not np.isfinite(mu):return 'unknown'
 if 0<mu<=.4:return 'feasible'
 if .4<mu<=1.2:return 'weak'
 return 'no_FC'
rows=[];variants=[];receipts=[]
checks=json.loads((ROOT/'gradient_checks.json').read_text())
probe={(r['scene'],r['frame'],r['candidate_id'],r['temperature'],r['bound']):r for r in checks['probes']}
for file in sorted((ROOT/'evaluation').glob('*.npz')):
 a=np.load(file);s,f=map(int,file.stem.split('_'));receipts.append(json.loads(file.with_suffix('.json').read_text()))
 for j,cid in enumerate(a['candidate_id']):
  k=int(a['original_angle'][j]);rr=a['I1'][j];mu=a['mu'][j];co=a['collision'][j];em=a['empty'][j];defined=np.isfinite(rr);q=a['angle_logits'][:12,j];prob=softmax(q)
  b=status(mu[k],co[k],em[k]);circle=np.minimum((np.arange(12)-k)%12,(k-np.arange(12))%12)
  def choose_i1(mask):
   options=np.flatnonzero(mask&defined);v=rr[options];best=v.min();tie=options[np.isclose(v,best,rtol=0,atol=1e-7)]
   return k if k in tie else int(tie[np.lexsort((tie,circle[tie]))][0])
  def choose_fc(mask):
   valid=mask&~co&~em&np.isfinite(mu)&(mu>0)
   if not valid.any():return k
   best=mu[valid].min();tie=np.flatnonzero(valid&(mu==best))
   return k if k in tie else int(tie[np.lexsort((tie,circle[tie]))][0])
  choices={'frozen_EG':k,'I1_full12_oracle':choose_i1(np.ones(12,bool)),
    'I1_neighbor3_oracle':choose_i1(circle<=1),'I1_neighbor5_oracle':choose_i1(circle<=2),
    'FC_full12_oracle':choose_fc(np.ones(12,bool)),
    'FC_neighbor3_oracle':choose_fc(circle<=1),'FC_neighbor5_oracle':choose_fc(circle<=2),
    'I1_collision_filtered_oracle':choose_i1(~co&~em)}
  for t in [1.,.5,2.]:
   for bound in [.25,.5,1.]:choices[f'GT_logit_T{t}_b{bound}']=probe[(s,f,int(cid),t,bound)]['new_class']
  for policy,new in choices.items():
   newstatus=status(mu[new],co[new],em[new]);validmu=b in ['feasible','weak'] and newstatus in ['feasible','weak']
   rows.append(dict(scene=s,frame=f,candidate_id=int(cid),object_id=int(a['object_id'][j]),policy=policy,old_class=k,new_class=new,
     old_status=b,new_status=newstatus,old_mu=mu[k],new_mu=mu[new],old_I1=rr[k],new_I1=rr[new],angle_changed=new!=k,
     feasible_gain=int(newstatus=='feasible')-int(b=='feasible'),weak_recovered=b=='weak' and newstatus=='feasible',
     feasible_harmed=b=='feasible' and newstatus!='feasible',collision=newstatus=='collision',empty=newstatus=='empty',unknown=newstatus=='unknown',
     mu_lower=validmu and mu[new]<mu[k],mu_higher=validmu and mu[new]>mu[k],paired_valid_mu=validmu,
     I1_down=np.isfinite(rr[new]) and rr[new]<rr[k]-1e-7,missing_contact_mass=float(prob[~defined].sum()),defined_classes=int(defined.sum()),score=float(a['score'][j])))
  for kk in range(12):variants.append(dict(scene=s,frame=f,candidate_id=int(cid),object_id=int(a['object_id'][j]),angle_class=kk,nominal=kk==k,neighbors=int(circle[kk]),mu=mu[kk],status=status(mu[kk],co[kk],em[kk]),I1=rr[kk],contact_status=str(a['contact_reason'][j,kk]),reason=str(a['unknown_reason'][j,kk])))
df=pd.DataFrame(rows);v=pd.DataFrame(variants)
assert len(receipts)==48 and sum(r['n'] for r in receipts)==1229 and all(r['nominal_replay'] for r in receipts)

def ci(values):
 values=values.dropna();rng=np.random.default_rng(20261024);draw=[]
 for lo,hi in [(100,129),(130,159),(160,189)]:
  a=values[(values.index>=lo)&(values.index<=hi)].to_numpy()
  if len(a):draw.append(rng.choice(a,(10000,len(a)),replace=True))
 boot=np.concatenate(draw,axis=1).mean(1)
 return dict(mean=float(values.mean()),lo=float(np.quantile(boot,.025)),hi=float(np.quantile(boot,.975)),n_scenes=len(values))
summ={};scene_rows=[]
for policy,z in df.groupby('policy',sort=False):
 oldgood=z.old_status=='feasible';oldweak=z.old_status=='weak'
 sr=z.groupby('scene').agg(n=('candidate_id','size'),net_feasible=('feasible_gain','mean'),collision=('collision','mean'),empty=('empty','mean'))
 sr['weak_recovery']=z[oldweak].groupby('scene').weak_recovered.mean();sr['feasible_harm']=z[oldgood].groupby('scene').feasible_harmed.mean()
 for s,r in sr.iterrows():scene_rows.append(dict(policy=policy,scene=s,**r.to_dict()))
 stats={key:ci(sr[key]) for key in ['net_feasible','weak_recovery','feasible_harm','collision','empty']}
 summ[policy]=dict(n=len(z),changed=int(z.angle_changed.sum()),statuses=z.new_status.value_counts().to_dict(),weak_recovered=int(z.weak_recovered.sum()),feasible_harmed=int(z.feasible_harmed.sum()),
     I1_down=int(z.I1_down.sum()),I1_down_but_not_mu_lower=int((z.I1_down&~z.mu_lower).sum()),mu_lower=int(z.mu_lower.sum()),mu_higher=int(z.mu_higher.sum()),paired_valid_mu=int(z.paired_valid_mu.sum()),
     valid_pair_mean_mu_delta=float((z.loc[z.paired_valid_mu,'new_mu']-z.loc[z.paired_valid_mu,'old_mu']).mean()),scene_ci=stats)
primary=summ['I1_full12_oracle'];reachable=summ['GT_logit_T1.0_b0.5']
def passes(s):
 c=s['scene_ci'];return c['net_feasible']['lo']>0 and c['weak_recovery']['lo']>0 and c['collision']['mean']<=.05 and c['empty']['mean']<=.05 and c['feasible_harm']['mean']<=.05
gate=passes(primary) and passes(reachable)
summary=dict(n=1229,variants=len(v),scenes=12,frames=48,baseline={'feasible':641,'weak':588},policy=summ,
   contact_status=v.contact_status.value_counts().to_dict(),variant_status=v.status.value_counts().to_dict(),
   undefined_I1=int(v.I1.isna().sum()),max_nominal_I1_error=max(r['I1_replay_error'] for r in receipts),unknown_quality=int((v.status=='unknown').sum()),
   cpu_seconds=sum(r['cpu_seconds'] for r in receipts),gate_passed=gate,training_started=False,
   gradient=dict(n=len(checks['rows']),nonzero=sum(r['gradient_max']>1e-9 for r in checks['rows']),max_fd_error=max(r['finite_difference_error'] for r in checks['rows']),max_analytic_error=max(r['analytic_error'] for r in checks['rows'])))
# Fixed original Top-1: no post-hoc reranking and no reassignment of candidate IDs.
history=pd.read_csv(ROOT.parent/'eg_pcr_r0_20261024/eg_final_candidates.csv');top=history[history.final_rank==1].copy()
assert len(top)==48
top_rows=[]
for policy,z in df.groupby('policy',sort=False):
 lookup={(int(r.scene),int(r.frame),int(r.candidate_id)):r for r in z.itertuples()}
 for r in top.itertuples():
  k=(int(r.scene),int(r.frame),int(r.candidate_id));new=lookup.get(k);st=status(r.P0_mu,r.P0_collision,r.P0_empty)
  top_rows.append(dict(policy=policy,scene=k[0],frame=k[1],candidate_id=k[2],old_status=st,new_status=new.new_status if new else st,changed=bool(new.angle_changed) if new else False))
tops=pd.DataFrame(top_rows);summary['fixed_original_top1']={p:z.new_status.value_counts().to_dict() for p,z in tops.groupby('policy')}
df.to_csv(ROOT/'candidate_policy_results.csv',index=False);v.to_csv(ROOT/'angle_variants.csv',index=False);pd.DataFrame(scene_rows).to_csv(ROOT/'scene_statistics.csv',index=False);tops.to_csv(ROOT/'fixed_top1_results.csv',index=False)
(ROOT/'statistics.json').write_text(json.dumps(summary,indent=2));pd.DataFrame(checks['rows']).to_csv(ROOT/'gradient_check.csv',index=False);pd.DataFrame(checks['probes']).to_csv(ROOT/'logit_probes.csv',index=False)
names=['frozen_EG','I1_full12_oracle','I1_neighbor3_oracle','GT_logit_T1.0_b0.5','FC_full12_oracle'];labels=['Frozen EG','I1 / 12 (GT)','I1 / 3 (GT)','Logit probe (GT)','FC oracle']
fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
bottom=np.zeros(len(names));colors={'feasible':'#39855d','weak':'#e2ac40','no_FC':'#9178ad','collision':'#c65351','empty':'#805d4b','unknown':'#999999'}
for st,col in colors.items():
 counts=np.array([summ[p]['statuses'].get(st,0) for p in names]);axes[0].bar(np.arange(len(names)),counts,bottom=bottom,label=st,color=col);bottom+=counts
axes[0].set(xticks=np.arange(len(names)),xticklabels=labels,ylabel='Candidates',title='Fixed IDs and scores; angle changes only');axes[0].tick_params(axis='x',labelrotation=25);axes[0].legend(fontsize=7)
for ax,metric,title in [(axes[1],'net_feasible','Net feasible gain'),(axes[2],'feasible_harm','Harm to originally feasible')]:
 vals=[summ[p]['scene_ci'][metric] for p in names[1:]];means=[x['mean']*100 for x in vals]
 ax.bar(range(4),means,color=['#3d79a6','#70a5ca','#d49b3b','#39855d']);ax.errorbar(range(4),means,yerr=[[100*(x['mean']-x['lo']) for x in vals],[100*(x['hi']-x['mean']) for x in vals]],fmt='none',color='black',capsize=3)
 ax.axhline(0,color='gray',lw=1);ax.set(xticks=range(4),xticklabels=labels[1:],ylabel='Scene mean percentage points',title=title);ax.tick_params(axis='x',labelrotation=25)
fig.savefig(ROOT/'physical_effect_summary.png',dpi=180);plt.close(fig)
print(json.dumps({k:summary[k] for k in ['n','variants','gate_passed','gradient','variant_status','contact_status']},indent=2))
for p in names:print(p,json.dumps(summ[p]))
