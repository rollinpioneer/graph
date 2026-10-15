#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from tools.stage6_refine1.common import aggregate_seed
def main():
 p=argparse.ArgumentParser();p.add_argument('--rollouts',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();d=pd.read_csv(a.rollouts);a.out.mkdir(parents=True,exist_ok=True)
 cond=d.groupby(['task_id','condition','method','policy_seed'],as_index=False).agg(n_rollouts=('success','size'),success_rate=('success','mean'),completion_mean=('success','mean'),recovery_success_rate=('recovery_success','mean'),failed_rate=('failed','mean'),median_steps=('steps','median'));cond.to_csv(a.out/'result_by_task_condition_method_seed.csv',index=False)
 task=cond.groupby(['task_id','method'],as_index=False).agg(n_rollouts=('n_rollouts','sum'),success_rate=('success_rate','mean'),completion_mean=('completion_mean','mean'),recovery_success_rate=('recovery_success_rate','mean'));task.to_csv(a.out/'result_by_task_method.csv',index=False)
 rows=[]
 for seed in sorted(d.policy_seed.unique()):
  g=aggregate_seed(d,'pathgraph_reward_v1_locked',int(seed));b=aggregate_seed(d,'linear_sarm_equiv',int(seed));rows.append({'policy_seed':int(seed),'graph_task_success_pathgraph':g['graph_task_success'],'graph_task_success_linear':b['graph_task_success'],'graph_task_success_difference':g['graph_task_success']-b['graph_task_success'],'recovery_success_pathgraph':g['recovery_success'],'recovery_success_linear':b['recovery_success'],'recovery_success_difference':g['recovery_success']-b['recovery_success'],'worst_order_success_pathgraph':g['worst_order_success'],'worst_order_success_linear':b['worst_order_success'],'worst_order_success_difference':g['worst_order_success']-b['worst_order_success'],'fixed_order_success_pathgraph':g['fixed_order_success'],'fixed_order_success_linear':b['fixed_order_success'],'fixed_order_success_difference':g['fixed_order_success']-b['fixed_order_success'],'long_horizon_completion_pathgraph':g['long_horizon_completion'],'long_horizon_completion_linear':b['long_horizon_completion'],'long_horizon_completion_difference':g['long_horizon_completion']-b['long_horizon_completion'],'order_gap_pathgraph':g['order_gap'],'order_gap_linear':b['order_gap']})
 pd.DataFrame(rows).to_csv(a.out/'pathgraph_vs_locked_comparator.csv',index=False)
 print(f'REFINE1_RESULT_TABLES_OK rows={len(cond)} seeds={len(rows)}')
if __name__=='__main__':main()
