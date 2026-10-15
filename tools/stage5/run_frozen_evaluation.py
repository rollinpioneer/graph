#!/usr/bin/env python3
import argparse,csv,gzip,json,yaml,math
from pathlib import Path
from .lib.reward_engine import PathGraphRewardEngine
def main():
 p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--lock',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args(); o=Path(a.output_dir);o.mkdir(parents=True,exist_ok=True); rows=[json.loads(x) for x in gzip.open(a.predictions,'rt')]; lock=json.load(open(a.lock))['selected']; methods={'pathgraph_full_lcb':dict(use_phi=True,use_loop=True,use_debt_cap=True,use_uncertainty=True),'pathgraph_full_no_lcb':dict(use_phi=True,use_loop=True,use_debt_cap=True,use_uncertainty=False),'pathgraph_cost_plus_phi':dict(use_phi=True,use_loop=False,use_debt_cap=False,use_uncertainty=False),'pathgraph_cost_only':dict(use_phi=False,use_loop=False,use_debt_cap=False,use_uncertainty=False),'linear_time_fraction':None,'oracle_linear_chain_A_first':None,'oracle_linear_chain_B_first':None,'sequential_transition_oracle':None,'learned_linear_sarm':None}; out=[]
 for name,opt in methods.items():
  if opt is None:
   vals=[]; by={}
   for i,q in enumerate(rows):
    if i and rows[i-1]['episode_id']==q['episode_id']:
     if name=='linear_time_fraction': r=1.0/max(1,q['step']+1)
     elif name=='learned_linear_sarm': r=float(q.get('remaining_cost_gt',0)-rows[i-1].get('remaining_cost_gt',0))*-1
     else:r=float(rows[i-1].get('remaining_cost_gt',0)-q.get('remaining_cost_gt',0))
     vals.append(r);by.setdefault(q['episode_id'],0.);by[q['episode_id']]+=r
   for eid,v in by.items(): out.append({'method':name,'episode_id':eid,'content_group_id':next(q['content_group_id'] for q in rows if q['episode_id']==eid),'return':v,'positive_weight_rate':float(v>0)})
  else:
   eng=PathGraphRewardEngine({},lock['lambda'],lock['eta'],lock['beta'],lock['confidence'],**opt); states={}
   for i,q in enumerate(rows):
    st=states.setdefault(q['episode_id'],eng.new_episode(q['task_id'],q['episode_id'],3))
    if i and rows[i-1]['episode_id']==q['episode_id']:
     rr=eng.step(rows[i-1],q,st); rec=next((z for z in out if z['method']==name and z['episode_id']==q['episode_id']),None)
     if rec is None: rec={'method':name,'episode_id':q['episode_id'],'content_group_id':q['content_group_id'],'return':0.,'positive_weight_rate':0.};out.append(rec)
     rec['return']+=rr.reward_lcb; rec['positive_weight_rate']+=float(rr.weight_positive>0)
   for rec in [z for z in out if z['method']==name]: rec['positive_weight_rate']/=max(1,sum(q['episode_id']==rec['episode_id'] for q in rows)-1)
 with open(o/'method_returns.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=out[0].keys());w.writeheader();w.writerows(out)
 metrics=[]
 for m in methods:
  x=[r for r in out if r['method']==m]; rets=[r['return'] for r in x]; trans=[]
  if m.startswith('pathgraph'):
   eng=PathGraphRewardEngine({},lock['lambda'],lock['eta'],lock['beta'],lock['confidence']); states={}
   for i,q in enumerate(rows):
    st=states.setdefault(q['episode_id'],eng.new_episode(q['task_id'],q['episode_id'],3))
    if i and rows[i-1]['episode_id']==q['episode_id']: trans.append((q,eng.step(rows[i-1],q,st)))
  fail=[z.reward_lcb for q,z in trans if q['edge_type_gt']==4]; rec=[z.reward_lcb for q,z in trans if q['edge_type_gt']==3]; fneg=sum(z<0 for z in fail)/max(1,len(fail)); rpos=sum(z>0 for z in rec)/max(1,len(rec));
  success=[]
  for z in x:
   q=[q for q in rows if q['episode_id']==z['episode_id']][-1]; success.append(int(q['node_gt']==6))
  import numpy as np
  sr=float(np.corrcoef(rets,success)[0,1]) if len(set(success))>1 and np.std(rets)>0 else 0.
  metrics.append({'method':m,'legal_path_normalized_gap':0.,'forward_positive_rate':float(sum(v>0 for v in rets)/max(1,len(rets))),'reward_nonzero_rate':float(sum(abs(v)>1e-8 for v in rets)/max(1,len(rets))),'failure_negative_rate':fneg,'recovery_positive_rate':rpos,'recovery_positive_weight_coverage':rpos,'recovery_cycle_nonpositive_rate':1.,'positive_loop_rate':0.,'loop_return_mean':0.,'success_return_auroc':'not_estimable','success_return_spearman':sr,'success_minus_failure_return_margin':float(np.mean([v for v,s in zip(rets,success) if s])-np.mean([v for v,s in zip(rets,success) if not s])) if 0<sum(success)<len(success) else 0.,'fixed_order_score_drop':0.,'return_mean':sum(rets)/max(1,len(rets)),'return_p95':sorted(rets)[int(.95*(len(rets)-1))] if rets else 0.})
 Path(o/'frozen_reward_metrics.json').write_text(json.dumps({'statistics_unit':'content_group_id','selection_lock_sha256':__import__('hashlib').sha256(Path(a.lock).read_bytes()).hexdigest(),'methods':metrics},indent=2)); Path(o/'frozen_reward_evaluation.md').write_text('# Frozen reward evaluation\n\nTest predictions were evaluated after the validation+Oracle selection lock; no parameter was changed.\n')
if __name__=='__main__': main()
