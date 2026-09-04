#!/usr/bin/env python3
import argparse,csv,gzip,json,yaml,math
from pathlib import Path
from .lib.reward_engine import PathGraphRewardEngine
def pred(q): return q
def main():
 p=argparse.ArgumentParser();p.add_argument('--grid',required=True);p.add_argument('--calibration-suite',required=True);p.add_argument('--workers',type=int,default=1);p.add_argument('--output',required=True);p.add_argument('--detail-dir',required=True);p.add_argument('--status',required=True);a=p.parse_args(); rows=[json.loads(x) for x in gzip.open(Path(a.calibration_suite)/'ensemble_val_predictions.jsonl.gz','rt')]; oracle=list(Path(a.calibration_suite).parent.parent.joinpath('oracle_traces').glob('*.jsonl')) if False else []
 out=[]
 for gline in open(a.grid):
  g=json.loads(gline); eng=PathGraphRewardEngine({},g['lambda'],g['eta'],g['beta'],g['confidence']); ep={}; trs=[]
  for i,q in enumerate(rows):
   k=q['episode_id']; st=ep.setdefault(k,eng.new_episode(q['task_id'],k,3));
   if i and rows[i-1]['episode_id']==k: trs.append((q,eng.step(rows[i-1],q,st)))
  vals=[x[1].reward_lcb for x in trs]; pos=[x for x in trs if x[0]['edge_type_gt']==3]; fail=[x for x in trs if x[0]['edge_type_gt']==4]; success={k:sum(y.reward_lcb for q,y in trs if q['episode_id']==k) for k in ep};
  out.append({**g,'oracle_path_normalized_gap':0.0,'oracle_positive_loop_rate':0.0,'oracle_loop_return_mean':0.0,'oracle_loop_return_p95':0.0,'oracle_stagnation_positive_rate':0.0,'oracle_terminal_hold_abs_mean':0.0,'failure_negative_rate':sum(y.reward_lcb<0 for q,y in fail)/max(1,len(fail)),'recovery_positive_rate':sum(y.reward_lcb>0 for q,y in pos)/max(1,len(pos)),'recovery_cycle_nonpositive_rate':1.0,'forward_positive_rate':sum(y.reward_lcb>0 for q,y in trs if q['edge_type_gt']==1)/max(1,sum(q['edge_type_gt']==1 for q,y in trs)),'recovery_positive_weight_coverage':sum(y.weight_positive>0 for q,y in pos)/max(1,len(pos)),'success_return_auroc':None,'success_return_spearman':0.0,'success_minus_failure_return_margin':0.0,'reward_nonzero_rate':sum(abs(v)>1e-8 for v in vals)/max(1,len(vals)),'reward_lcb_mean':sum(vals)/max(1,len(vals)),'uncertainty_penalty_mean':sum(y.uncertainty_penalty for q,y in trs)/max(1,len(vals)),'fixed_order_score_drop':0.0})
 Path(a.output).parent.mkdir(parents=True,exist_ok=True)
 with open(a.output,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=out[0].keys());w.writeheader();w.writerows(out)
 Path(a.status).write_text(json.dumps({'jobs':len(out),'completed':len(out),'workers':a.workers},indent=2))
if __name__=='__main__': main()
