import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'eg_i1_angle_physics_20261024'

def status(mu,c,e):
 if e:return 'empty'
 if c:return 'collision'
 if not np.isfinite(mu):return 'unknown'
 return 'feasible' if 0<mu<=.4 else 'weak' if .4<mu<=1.2 else 'no_FC'

def ci(values):
 values=values.dropna();rng=np.random.default_rng(20261024);draw=[]
 if not len(values):return dict(mean=None,lo=None,hi=None,n_scenes=0)
 for lo,hi in [(100,129),(130,159),(160,189)]:
  x=values[(values.index>=lo)&(values.index<=hi)].to_numpy()
  if len(x):draw.append(rng.choice(x,(10000,len(x)),replace=True))
 boot=np.concatenate(draw,axis=1).mean(1)
 return dict(mean=float(values.mean()),lo=float(np.quantile(boot,.025)),hi=float(np.quantile(boot,.975)),n_scenes=len(values))

variants=[];selected=[];fixed=[];pipeline=[];receipts=[];stage_rows=[]
for file in sorted((ROOT/'results').glob('*.npz')):
 z=np.load(file);receipt=json.loads(file.with_suffix('.json').read_text());receipts.append(receipt);s=receipt['scene'];f=receipt['frame'];a=np.load(PREV/f'evaluation/{s:04d}_{f:04d}.npz')
 assert receipt['prior_variants_sha256']==hashlib.sha256((PREV/f'evaluation/{s:04d}_{f:04d}.npz').read_bytes()).hexdigest()
 for j,cid in enumerate(z['candidate_id']):
  old=int(z['original_angle'][j]);oldstatus=status(a['mu'][j,old],a['collision'][j,old],a['empty'][j,old]);per=[]
  for k in range(12):
   st=status(a['mu'][j,k],a['collision'][j,k],a['empty'][j,k]);parts=z['EG_ious'][j,k];eg=bool(z['EG_cd'][j,k]);gt=bool(a['collision'][j,k]);emp=bool(a['empty'][j,k])
   r=dict(scene=s,frame=f,candidate_id=int(cid),object_id=int(a['object_id'][j]),angle_class=k,nominal=k==old,old_status=oldstatus,old_mu=float(a['mu'][j,old]),
      mu=float(a['mu'][j,k]),status=st,GT_collision=gt,GT_empty=emp,EG_cd=eg,I1=float(a['I1'][j,k]),RA=float(z['RA'][j,k]),RC=float(z['RC'][j,k]),RS=float(z['RS'][j,k]),outside_grid=bool(z['outside_grid'][j,k]),
      sdf_left=float(z['sdf_left'][j,k]),sdf_right=float(z['sdf_right'][j,k]),EG_global_iou=float(parts[0]),EG_left_iou=float(parts[1]),EG_right_iou=float(parts[2]),EG_bottom_iou=float(parts[3]),EG_shifting_iou=float(parts[4]),
      gt_nonempty_collision=gt and not emp,gt_clean=not gt and not emp,gt_feasible=st=='feasible',miss=gt and not emp and not eg,false_filter=not gt and not emp and eg,feasible_filtered=st=='feasible' and eg,
      no_observed_EG_occupancy=parts[0]==0,approach_only_occupancy=bool((parts[1:4]==0).all() and parts[4]>0))
   variants.append(r);per.append(r)
  for g,group in enumerate(z['groups']):
   k=int(z['selected_angle'][j,g]);r=dict(per[k]);st=r['status'];r.update(group=str(group),original_angle=old,changed=k!=old,fallback=bool(z['fallback'][j,g]),feasible_gain=int(st=='feasible')-int(oldstatus=='feasible'),
      weak_recovered=oldstatus=='weak' and st=='feasible',feasible_harmed=oldstatus=='feasible' and st!='feasible',paired_valid_mu=st in ['feasible','weak'],
      mu_lower=st in ['feasible','weak'] and r['mu']<r['old_mu'],mu_higher=st in ['feasible','weak'] and r['mu']>r['old_mu'],
      collision_category=st=='collision',empty_category=st=='empty',no_FC=st=='no_FC',feasible=st=='feasible')
   selected.append(r)
 for r in receipt['fixed_top1']:fixed.append(dict(scene=s,frame=f,**r))
 for p in receipt['postprocess']:
  pipeline.append(dict(scene=s,frame=f,group=p['group'],**p['top1']))
  stage_rows.append(dict(scene=s,frame=f,group=p['group'],raw=p['raw_count'],post_cd=p['post_cd'],post_nms=len(p['nms_ids']),post_object10=len(p['object10_ids']),post_top50=len(p['final_ids']),unknown_final=sum(r['status']=='unknown' for r in p['final_results'])))
assert len(receipts)==48
v=pd.DataFrame(variants);a=pd.DataFrame(selected);fixed=pd.DataFrame(fixed);p=pd.DataFrame(pipeline);stage=pd.DataFrame(stage_rows)
assert len(v)==14748 and len(a)==1229*7
previous=pd.read_csv(PREV/'candidate_policy_results.csv')
for group,policy in [('A','I1_full12_oracle'),('E','I1_collision_filtered_oracle')]:
 aa=a[a.group==group];bb=previous[previous.policy==policy];m=aa.merge(bb,on=['scene','frame','candidate_id'],suffixes=('_now','_previous'),validate='one_to_one')
 assert (m.angle_class==m.new_class).all() and (m.status==m.new_status).all(),group

groups={};scenes=[]
for group,z in a.groupby('group',sort=False):
 sr=z.groupby('scene').agg(n=('candidate_id','size'),net_feasible=('feasible_gain','mean'),feasible=('feasible','mean'),collision=('collision_category','mean'),empty=('empty_category','mean'),no_FC=('no_FC','mean'))
 sr['recovery']=z[z.old_status=='weak'].groupby('scene').weak_recovered.mean();sr['harm']=z[z.old_status=='feasible'].groupby('scene').feasible_harmed.mean()
 for s,r in sr.iterrows():scenes.append(dict(group=group,scene=s,**r.to_dict()))
 valid=z[z.paired_valid_mu]
 groups[group]=dict(n=len(z),statuses=z.status.value_counts().to_dict(),changed=int(z.changed.sum()),fallback=int(z.fallback.sum()),recovery=int(z.weak_recovered.sum()),harm=int(z.feasible_harmed.sum()),
    mu_lower=int(z.mu_lower.sum()),mu_higher=int(z.mu_higher.sum()),paired_valid_mu=len(valid),mean_mu_delta_given_valid=float((valid.mu-valid.old_mu).mean()),
    picked_EG_rejected=int(z.EG_cd.sum()),picked_outside_SDF=int(z.outside_grid.sum()),scene_ci={k:ci(sr[k]) for k in ['net_feasible','feasible','collision','empty','no_FC','recovery','harm']})
sc=pd.DataFrame(scenes);contrasts={}
for g1,g0 in [('B','A'),('C','A'),('D','A'),('D','B'),('D','C'),('F','O'),('B','E'),('D','E')]:
 z1=sc[sc.group==g1].set_index('scene');z0=sc[sc.group==g0].set_index('scene')
 contrasts[g1+'-'+g0]={key:ci(z1[key]-z0[key]) for key in ['net_feasible','collision','recovery','harm']}

def detector_summary(z):
 rows=[]
 for s,x in z.groupby('scene'):
  rows.append(dict(scene=s,GT_collision=int(x.gt_nonempty_collision.sum()),miss=int(x['miss'].sum()),GT_clean=int(x.gt_clean.sum()),false_filter=int(x.false_filter.sum()),GT_feasible=int(x.gt_feasible.sum()),feasible_filtered=int(x.feasible_filtered.sum())))
 ss=pd.DataFrame(rows).set_index('scene')
 result=dict(n=len(z),GT_collision=int(z.gt_nonempty_collision.sum()),miss=int(z['miss'].sum()),GT_clean=int(z.gt_clean.sum()),false_filter=int(z.false_filter.sum()),GT_feasible=int(z.gt_feasible.sum()),feasible_filtered=int(z.feasible_filtered.sum()))
 for name,num,den in [('miss_rate','miss','GT_collision'),('false_filter_rate','false_filter','GT_clean'),('feasible_filter_rate','feasible_filtered','GT_feasible')]:
  result[name]=dict(pooled=result[num]/result[den] if result[den] else None,scene_ci=ci(ss[num]/ss[den].replace(0,np.nan)))
 misses=z[z['miss']];fp=z[z.false_filter]
 result['misses_with_zero_EG_volume_occupancy']=int(misses.no_observed_EG_occupancy.sum())
 result['misses_with_positive_subthreshold_occupancy']=int((~misses.no_observed_EG_occupancy).sum())
 result['false_filters_approach_only_occupancy']=int(fp.approach_only_occupancy.sum())
 return result,rows

detectors={};detector_scenes=[]
for scope,z in [('all_variants',v),('nonnominal_variants',v[~v.nominal]),('selected_A',a[a.group=='A']),('selected_B',a[a.group=='B']),('selected_D',a[a.group=='D'])]:
 summary,rs=detector_summary(z);detectors[scope]=summary;detector_scenes.extend([dict(scope=scope,**r) for r in rs])

def top_summary(z):
 result={}
 for group,x in z.groupby('group'):
  good=x.status=='feasible';unknown=x.status=='unknown';xx=x.assign(good=good,unknown=unknown)
  result[group]=dict(statuses=x.status.value_counts().to_dict(),feasible_rate_lower=float(good.sum()/48),feasible_rate_upper=float((good.sum()+unknown.sum())/48),scene_ci_lower=ci(xx.groupby('scene').good.mean()),unknown=int(unknown.sum()))
 return result
top={'fixed_original_top1':top_summary(fixed),'full_pipeline_top1':top_summary(p)}
assert top['full_pipeline_top1']['O']['statuses']==dict(feasible=20,weak=13,collision=8,no_FC=6,empty=1)
for g in 'OABCDEF':
 xx=p[p.group==g].set_index(['scene','frame']);oo=p[p.group=='O'].set_index(['scene','frame'])
 top['full_pipeline_top1'][g]['changed_id']=int((xx.candidate_id!=oo.candidate_id).sum())
 top['full_pipeline_top1'][g]['paired_vs_O_feasible_lower']=ci(((xx.status=='feasible').astype(int)-(oo.status=='feasible').astype(int)).groupby(level='scene').mean())

passed=[]
for g in 'BCDF':
 x=groups[g]['scene_ci'];delta=contrasts.get(g+'-A')
 if delta is None:
  s1=sc[sc.group==g].set_index('scene');s0=sc[sc.group=='A'].set_index('scene');delta={'collision':ci(s1.collision-s0.collision)}
 if x['net_feasible']['lo']>0 and delta['collision']['hi']<0 and x['harm']['mean']<=.05 and x['collision']['mean']<=.05 and x['recovery']['mean']>0:passed.append(g)
summary=dict(n=1229,variants=14748,scenes=12,frames=48,groups=groups,contrasts=contrasts,detector=detectors,top1=top,diagnostic_gate_pass_groups=passed,
    max_PCR_replay_error={key:max(r['PCR_replay_error'][key] for r in receipts) for key in ['RA','RC','RS']},all_nominal_CD_exact=all(r['CD_replay_exact'] for r in receipts),all_O_postprocess_exact=all(r['postprocess_O_exact'] for r in receipts),
    cpu_seconds=sum(r['cpu_seconds'] for r in receipts),new_force_closure_evaluations=0,training=False,parameter_updates=0,GT_information_in_selection=True,
    fallback_total=int(a.fallback.sum()),postprocess_unknown_final=int(stage.unknown_final.sum()))
v.to_csv(ROOT/'variant_controls.csv',index=False);a.to_csv(ROOT/'selected_candidates.csv',index=False);sc.to_csv(ROOT/'scene_statistics.csv',index=False);pd.DataFrame(detector_scenes).to_csv(ROOT/'detector_scene_counts.csv',index=False)
fixed.to_csv(ROOT/'fixed_top1.csv',index=False);p.to_csv(ROOT/'pipeline_top1.csv',index=False);stage.to_csv(ROOT/'postprocess_stages.csv',index=False)
(ROOT/'statistics.json').write_text(json.dumps(summary,indent=2))
fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained');names=list('OABCDEF');bottom=np.zeros(7)
colors={'feasible':'#3c865f','weak':'#deaf48','collision':'#c95852','no_FC':'#8a76ac','empty':'#755846','unknown':'#a0a0a0'}
for st,c in colors.items():
 counts=np.array([groups[g]['statuses'].get(st,0) for g in names]);axes[0].bar(names,counts,bottom=bottom,label=st,color=c);bottom+=counts
axes[0].set(ylabel='Candidates',title='Fixed IDs / unchanged scores');axes[0].legend(fontsize=7)
for ax,metric,title in [(axes[1],'net_feasible','Net feasible gain'),(axes[2],'harm','Harm to original feasible')]:
 vals=[groups[g]['scene_ci'][metric] for g in names];means=np.array([z['mean'] for z in vals])*100
 ax.bar(names,means,color=['#888888','#c95852','#3e799f','#c49e42','#3c865f','#a195b3','#907f5c']);ax.errorbar(names,means,yerr=[[100*(z['mean']-z['lo']) for z in vals],[100*(z['hi']-z['mean']) for z in vals]],fmt='none',ecolor='black',capsize=3);ax.set(ylabel='Scene mean percentage points',title=title);ax.axhline(0,color='gray',lw=.8)
fig.savefig(ROOT/'controls_summary.png',dpi=180);plt.close(fig)
print('diagnostic pass',passed)
for g in names:print(g,json.dumps(groups[g]))
print('top1',json.dumps(top))
print('detector',json.dumps(detectors))
