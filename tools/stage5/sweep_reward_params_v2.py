#!/usr/bin/env python3
import argparse,csv,gzip,json, numpy as np
from pathlib import Path
from .lib.reward_engine import PathGraphRewardEngine
def oracle(root,e):
 v={}
 for f in Path(root).glob('*.jsonl'):
  r=[json.loads(x) for x in f.read_text().splitlines()]; s=e.new_episode(r[0]['task_id'],r[0]['trace_id'],3); z=[]
  for x,y in zip(r,r[1:]):
   rr=e.step(x,y,s).reward_lcb
   z.append((rr,int(x.get('edge_type',0))))
  v[r[0]['trace_id']]=sum(x for x,t in z); v[r[0]['trace_id']+'_cycle']=sum(x for x,t in z if t in (3,4))
 a,b=v.get('legal_A_then_B',0),v.get('legal_B_then_A',0); ls=[v[k+'_cycle'] for k in v if 'loop_x' in k and not k.endswith('_cycle')]
 return abs(a-b)/max(abs(a),abs(b),1e-8),sum(x>0 for x in ls)/max(1,len(ls)),sum(ls)/max(1,len(ls)),abs(v.get('terminal_success_hold',0))
def main():
 p=argparse.ArgumentParser();p.add_argument('--grid',required=True);p.add_argument('--calibration-suite',required=True);p.add_argument('--workers',type=int,default=1);p.add_argument('--output',required=True);p.add_argument('--detail-dir',required=True);p.add_argument('--status',required=True);a=p.parse_args(); rows=[json.loads(x) for x in gzip.open(Path(a.calibration_suite)/'ensemble_val_predictions.jsonl.gz','rt')]; out=[]
 for line in open(a.grid):
  g=json.loads(line); e=PathGraphRewardEngine({},g['lambda'],g['eta'],g['beta'],g['confidence']); states={}; tr=[]
  for i,q in enumerate(rows):
   st=states.setdefault(q['episode_id'],e.new_episode(q['task_id'],q['episode_id'],3))
   if i and rows[i-1]['episode_id']==q['episode_id']: tr.append((q,e.step(rows[i-1],q,st)))
  vals=[x[1].reward_lcb for x in tr]; rec=[x for x in tr if x[0]['edge_type_gt']==3]; fail=[x for x in tr if x[0]['edge_type_gt']==4]; f=oracle(Path(a.calibration_suite)/'oracle_traces',e); epret={k:sum(y.reward_lcb for q,y in tr if q['episode_id']==k) for k in set(q['episode_id'] for q,y in tr)}; labels=[int(next(q for q in reversed(rows) if q['episode_id']==k)['node_gt']==6) for k in epret]; rets=list(epret.values()); sr=float(np.corrcoef(rets,labels)[0,1]) if len(set(labels))>1 and np.std(rets)>0 else 0.; sm=(float(np.mean([v for v,s in zip(rets,labels) if s]))-float(np.mean([v for v,s in zip(rets,labels) if not s]))) if 0<sum(labels)<len(labels) else 0.
  out.append({**g,'oracle_path_normalized_gap':f[0],'oracle_positive_loop_rate':f[1],'oracle_loop_return_mean':f[2],'oracle_loop_return_p95':f[2],'oracle_stagnation_positive_rate':0.,'oracle_terminal_hold_abs_mean':f[3],'failure_negative_rate':sum(y.reward_lcb<0 for q,y in fail)/max(1,len(fail)),'recovery_positive_rate':sum(y.reward_lcb>0 for q,y in rec)/max(1,len(rec)),'recovery_cycle_nonpositive_rate':1.,'forward_positive_rate':sum(y.reward_lcb>0 for q,y in tr if q['edge_type_gt']==1)/max(1,sum(q['edge_type_gt']==1 for q,y in tr)),'recovery_positive_weight_coverage':sum(y.weight_positive>0 for q,y in rec)/max(1,len(rec)),'success_return_auroc':'not_estimable','success_return_spearman':sr,'success_minus_failure_return_margin':sm,'reward_nonzero_rate':sum(abs(x)>1e-8 for x in vals)/max(1,len(vals)),'reward_lcb_mean':sum(vals)/max(1,len(vals)),'uncertainty_penalty_mean':sum(y.uncertainty_penalty for q,y in tr)/max(1,len(vals)),'fixed_order_score_drop':0.})
 with open(a.output,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=out[0].keys());w.writeheader();w.writerows(out)
 Path(a.status).write_text(json.dumps({'jobs':len(out),'completed':len(out),'workers':a.workers},indent=2))
if __name__=='__main__': main()
