#!/usr/bin/env python3
"""Frozen-rollout tables, hierarchical paired bootstrap, and weight mechanism tables."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq

GRAPH='pathgraph_reward_v1_locked'
def seed_metrics(d,method,seed):
 x=d[(d.method==method)&(d.policy_seed==seed)]; val=lambda task,cond:float(x[(x.task_id==task)&(x.condition==cond)].success.mean())
 rec=np.mean([val('transport_recovery','drop_regrasp'),val('transport_recovery','gripper_reopen')])
 a,b=val('transport_dual_order','A_first'),val('transport_dual_order','B_first')
 allc=np.mean([val('transport_recovery',c) for c in ('natural','drop_regrasp','gripper_reopen')]+[a,b])
 fixed=np.mean([val('transport_recovery','natural'),a,b])
 return {'graph_task_success':allc,'recovery_success':rec,'worst_order_success':min(a,b),'fixed_order_success':fixed,'long_horizon_completion':allc,'order_gap':abs(a-b)}
def boot(d,base,resamples=5000):
 rng=np.random.default_rng(20261001);seeds=np.array(sorted(d.policy_seed.unique())); metrics=['graph_task_success','recovery_success','worst_order_success','fixed_order_success','long_horizon_completion','order_gap']; values={m:[] for m in metrics}
 # resample policy seed first; inside each seed resample the paired rollout index once per condition.
 for _ in range(resamples):
  picked=rng.choice(seeds,size=len(seeds),replace=True); per=[]
  for draw,seed in enumerate(picked):
   z=d[d.policy_seed==seed]; rows=[]
   for task,cond in z[['task_id','condition']].drop_duplicates().itertuples(index=False):
    q=z[(z.task_id==task)&(z.condition==cond)]; keys=rng.choice(q.rollout_id.unique(),size=len(q.rollout_id.unique()),replace=True);v=q.set_index('rollout_id').loc[keys].reset_index();v['bootstrap_draw']=draw;rows.append(v)
   per.append(pd.concat(rows,ignore_index=True))
  joined=pd.concat(per,ignore_index=True)
  # Preserve method pairing by computing both methods from exactly these selected keys.
  sm=[]
  for draw,seed in enumerate(picked):
   q=joined[joined.bootstrap_draw==draw];sm.append((seed_metrics(q,GRAPH,seed),seed_metrics(q,base,seed)))
  for m in metrics:values[m].append(float(np.mean([a[m]-b[m] for a,b in sm])))
 return {m:np.asarray(v) for m,v in values.items()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--rollouts',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--final',type=Path,required=True);p.add_argument('--comparator',default='linear_sarm_equiv');a=p.parse_args();d=pd.read_csv(a.rollouts);out=a.out;out.mkdir(parents=True,exist_ok=True);a.final.mkdir(parents=True,exist_ok=True)
 # Results by condition and method/seed.
 rc=d.groupby(['task_id','condition','method','policy_seed'],as_index=False).agg(n_rollouts=('success','size'),success_rate=('success','mean'),recovery_success_rate=('recovery_success','mean'),completion_mean=('success','mean'),median_steps=('steps','median'))
 rc.to_csv(out/'result_by_task_condition_method_seed.csv',index=False);rc.groupby(['task_id','method'],as_index=False).agg(success_rate=('success_rate','mean'),n=('n_rollouts','sum')).to_csv(out/'result_by_task_method.csv',index=False)
 methods=['bc_all','linear_sarm_equiv','sequential_transition']; comparisons=[]
 for m in methods:
  for seed in sorted(d.policy_seed.unique()):
   ga,ba=seed_metrics(d,GRAPH,seed),seed_metrics(d,m,seed)
   comparisons.append({'baseline':m,'policy_seed':seed,**{k:ga[k]-ba[k] for k in ga}})
 pd.DataFrame(comparisons).to_csv(out/'pathgraph_vs_each_baseline.csv',index=False);locked=pd.DataFrame(comparisons).query('baseline==@a.comparator');locked.to_csv(out/'pathgraph_vs_locked_comparator.csv',index=False)
 effects=[]
 for seed in sorted(d.policy_seed.unique()):
  g,b=seed_metrics(d,GRAPH,seed),seed_metrics(d,a.comparator,seed)
  for k in g:effects.append({'task_id':'all_graph_tasks','metric':k,'policy_seed':seed,'pathgraph_value':g[k],'comparator_value':b[k],'difference':g[k]-b[k],'improved':g[k]>b[k] if k!='order_gap' else g[k]<b[k]})
 seeddf=pd.DataFrame(effects);seeddf.to_csv(out/'seed_level_effects.csv',index=False)
 distributions=boot(d,a.comparator);bootrows=[]
 for metric,v in distributions.items():
  aa=seeddf[seeddf.metric==metric];diff=float(aa.difference.mean());bootrows.append({'task_id':'all_graph_tasks','condition_or_aggregate':'predefined_equal_condition_aggregate','metric':metric,'method_a':GRAPH,'method_b':a.comparator,'point_estimate_a':float(aa.pathgraph_value.mean()),'point_estimate_b':float(aa.comparator_value.mean()),'difference':diff,'ci95_low':float(np.quantile(v,.025)),'ci95_high':float(np.quantile(v,.975)),'prob_difference_gt_0':float(np.mean(v>0)),'resamples':len(v)})
 bootdf=pd.DataFrame(bootrows);bootdf.to_csv(out/'bootstrap_effects.csv',index=False)
 # Mechanism labels are metadata for analysis only, never reward-model features.
 w=pq.read_table(a.weights/'chunk_weights/pathgraph_reward_v1_locked.parquet').to_pandas(); semantic=np.where(w.scenario.isin(['drop_regrasp','gripper_reopen']),'recovery',np.where(w.scenario.eq('controlled_failure'),'failure_onset',np.where(w.order.eq('B_first'),'alternative_order_B_then_A',np.where(w.order.eq('A_first'),'alternative_order_A_then_B','forward_normal'))));w['semantic_type']=semantic
 mech=w.groupby('semantic_type',as_index=False).agg(chunk_count=('normalized_weight','size'),mean_weight=('normalized_weight','mean'),median_weight=('normalized_weight','median'),p90_weight=('normalized_weight',lambda x:np.quantile(x,.9)),positive_weight_rate=('normalized_weight',lambda x:np.mean(x>0)),total_weight_mass=('normalized_weight','sum'));mech.to_csv(out/'weight_by_semantic_type.csv',index=False)
 rec=w[w.semantic_type=='recovery'];fail=w[w.semantic_type=='failure_onset'];q25=float(w.normalized_weight.quantile(.25));pd.DataFrame([{'recovery_retention':float(np.mean(rec.normalized_weight>0)),'failure_suppression':float(np.mean(fail.normalized_weight<=q25)) if len(fail) else None,'all_weight_q25':q25}]).to_csv(out/'recovery_chunk_retention.csv',index=False)
 od=w[w.task_id=='transport_dual_order'].groupby('order').normalized_weight.mean();gap=float(abs(od.get('A_first',0)-od.get('B_first',0))/max(abs(od.get('A_first',0)),abs(od.get('B_first',0)),1e-8));pd.DataFrame([{'A_first_mean_weight':od.get('A_first',0),'B_first_mean_weight':od.get('B_first',0),'order_weight_gap':gap}]).to_csv(out/'alternative_order_balance.csv',index=False)
 top=w.sort_values('normalized_weight',ascending=False).head(20);top.to_csv(out/'top_weighted_chunks.csv',index=False);w[w.semantic_type=='failure_onset'].sort_values('normalized_weight').head(20).to_csv(out/'downweighted_failure_chunks.csv',index=False)
 reps=[]
 for task,cond,seed in d[['task_id','condition','policy_seed']].drop_duplicates().itertuples(index=False):
  x=d[(d.task_id==task)&(d.condition==cond)&(d.policy_seed==seed)];g=x[x.method==GRAPH].set_index('rollout_id');b=x[x.method==a.comparator].set_index('rollout_id')
  for rid in sorted(set(g.index)&set(b.index)):
   if g.loc[rid].success!=b.loc[rid].success:reps.append({'task_id':task,'condition':cond,'policy_seed':seed,'rollout_id':rid,'pathgraph_success':int(g.loc[rid].success),'comparator_success':int(b.loc[rid].success)})
 pd.DataFrame(reps).head(12).to_csv(out/'representative_rollouts.csv',index=False)
 primary={r.metric:r for r in bootdf.itertuples(index=False)};graph_gain=primary['graph_task_success'].difference;fixed_drop=max(0.,-primary['fixed_order_success'].difference);evidence={'locked_comparator_by_task':{'all_graph_tasks':a.comparator},'graph_task_success_gain':graph_gain,'graph_task_success_ci95':[primary['graph_task_success'].ci95_low,primary['graph_task_success'].ci95_high],'improved_policy_seed_count':int(seeddf[seeddf.metric=='graph_task_success'].improved.sum()),'recovery_success_gain':primary['recovery_success'].difference,'recovery_success_ci95':[primary['recovery_success'].ci95_low,primary['recovery_success'].ci95_high],'worst_order_success_gain':primary['worst_order_success'].difference,'worst_order_success_ci95':[primary['worst_order_success'].ci95_low,primary['worst_order_success'].ci95_high],'long_horizon_completion_gain':primary['long_horizon_completion'].difference,'fixed_order_drop':fixed_drop,'order_gap_pathgraph':primary['order_gap'].point_estimate_a,'order_gap_comparator':primary['order_gap'].point_estimate_b,'policy_protocol_locked':True,'reward_retuned_after_test':False,'paired_evaluation':True,'bootstrap_resamples':5000}
 (a.final/'metrics').mkdir(exist_ok=True);(a.final/'tables').mkdir(exist_ok=True);(a.final/'reports').mkdir(exist_ok=True);(a.final/'metrics/stage6_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
 for f in ('bootstrap_effects.csv','seed_level_effects.csv','weight_by_semantic_type.csv','result_by_task_condition_method_seed.csv'): (a.final/'tables'/f).write_bytes((out/f).read_bytes())
 (a.final/'reports/stage6_evidence_summary.md').write_text('# Stage 6 frozen evidence\n\n'+json.dumps(evidence,indent=2)+'\n')
 print(json.dumps(evidence,indent=2))
if __name__=='__main__':main()
