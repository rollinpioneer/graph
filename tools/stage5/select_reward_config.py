#!/usr/bin/env python3
import argparse,csv,json,yaml
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--sweep',required=True);p.add_argument('--output-config',required=True);p.add_argument('--output-json',required=True);p.add_argument('--report',required=True);a=p.parse_args(); c=yaml.safe_load(open(a.config)); rows=[]
 for r in csv.DictReader(open(a.sweep)):
  for k,v in list(r.items()):
   try:r[k]=float(v)
   except:pass
  hard=all(r[k]<=t if op=='max' else r[k]>=t for k,op,t in [('oracle_path_normalized_gap','max',.10),('oracle_positive_loop_rate','max',.05),('oracle_loop_return_mean','max',0),('failure_negative_rate','min',.70),('recovery_positive_rate','min',.65),('recovery_cycle_nonpositive_rate','min',.90),('forward_positive_rate','min',.55),('recovery_positive_weight_coverage','min',.30),('fixed_order_score_drop','max',.05)]) and r['reward_nonzero_rate']>.10; r['hard_constraints_pass']=hard; rows.append(r)
 valid=[r for r in rows if r['hard_constraints_pass']]
 if not valid: raise SystemExit('no reward configuration passed hard constraints')
 best=sorted(valid,key=lambda r:(r['oracle_path_normalized_gap'],r['oracle_positive_loop_rate'],-r['recovery_cycle_nonpositive_rate'],-float(r.get('success_return_spearman',0) or 0),r.get('beta',0),r.get('eta',0),r.get('lambda',0)))[0]
 selected={'lambda':best['lambda'],'eta':best['eta'],'beta':best['beta'],'confidence':best['confidence'],'reward_clip':c['reward']['reward_clip'],'node_confidence_min':best['confidence'],'edge_confidence_min':best['confidence'],'repeat_window_steps':c['reward']['repeat_window_steps'],'recovery_debt_cap':True,'uncertainty_lcb':True}
 Path(a.output_config).parent.mkdir(parents=True,exist_ok=True); Path(a.output_config).write_text(yaml.safe_dump(selected,sort_keys=False)); Path(a.output_json).write_text(json.dumps({'selected':selected,'selected_metrics':best,'valid_count':len(valid),'total_count':len(rows),'selection_source':['transport_recovery_val','oracle_graph_trace_bank']},indent=2)); Path(a.report).write_text('# Reward selection\n\n- selected config: `%s`\n- valid configurations: %d/%d\n- selection source: validation + Oracle only\n'%(best['config_id'],len(valid),len(rows)))
if __name__=='__main__': main()
