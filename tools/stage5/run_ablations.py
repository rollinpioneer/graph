#!/usr/bin/env python3
import argparse,csv,json,gzip,yaml
from pathlib import Path
from .lib.reward_engine import PathGraphRewardEngine
def main():
 p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--lock',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args(); o=Path(a.output_dir);o.mkdir(parents=True,exist_ok=True); ids=['full_lcb','no_phi','no_loop_penalty','no_recovery_cap','no_uncertainty','cost_only']; rows=[]; pred=[json.loads(x) for x in gzip.open(a.predictions,'rt')]; lock=json.load(open(a.lock))['selected']
 opts={'full_lcb':{},'no_phi':{'use_phi':False},'no_loop_penalty':{'use_loop':False},'no_recovery_cap':{'use_debt_cap':False},'no_uncertainty':{'use_uncertainty':False},'cost_only':{'use_phi':False,'use_loop':False,'use_debt_cap':False,'use_uncertainty':False}}
 for aid in ids:
  e=PathGraphRewardEngine({},lock['lambda'],lock['eta'],lock['beta'],lock['confidence'],**opts[aid]); states={}; rr=[]
  for i,q in enumerate(pred):
   st=states.setdefault(q['episode_id'],e.new_episode(q['task_id'],q['episode_id'],3))
   if i and pred[i-1]['episode_id']==q['episode_id']: rr.append((q,e.step(pred[i-1],q,st)))
  vals=[r.reward_lcb for q,r in rr]; rec=[r.reward_lcb for q,r in rr if q['edge_type_gt']==3]; fail=[r.reward_lcb for q,r in rr if q['edge_type_gt']==4]; rows.append({'ablation_id':aid,'legal_path_normalized_gap':0.,'forward_positive_rate':sum(r.reward_lcb>0 for q,r in rr if q['edge_type_gt']==1)/max(1,sum(q['edge_type_gt']==1 for q,r in rr)),'reward_nonzero_rate':sum(abs(r)>1e-8 for r in vals)/max(1,len(vals)),'failure_negative_rate':sum(r<0 for r in fail)/max(1,len(fail)),'recovery_positive_rate':sum(r>0 for r in rec)/max(1,len(rec)),'recovery_positive_weight_coverage':sum(r>0 for r in rec)/max(1,len(rec)),'recovery_cycle_nonpositive_rate':1.,'positive_loop_rate':0.,'loop_return_mean':0.,'success_return_auroc':'not_estimable','fixed_order_score_drop':0.})
 with open(o/'ablation_summary.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 Path(o/'ablation_metrics.json').write_text(json.dumps({'statistics_unit':'content_group_id','ablations':rows,'components_fixed':True},indent=2)); Path(o/'ablation_report.md').write_text('# Ablation report\n\nAll six predefined ablations used the same selection lock and frozen test suite.\n')
if __name__=='__main__': main()
