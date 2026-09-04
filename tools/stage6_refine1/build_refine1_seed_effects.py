#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from tools.stage6_refine1.common import aggregate_seed,OLD_SEEDS,NEW_SEEDS
def main():
 p=argparse.ArgumentParser();p.add_argument('--rollouts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=pd.read_csv(a.rollouts);rows=[]
 for seed in sorted(d.policy_seed.unique()):
  g=aggregate_seed(d,'pathgraph_reward_v1_locked',int(seed));b=aggregate_seed(d,'linear_sarm_equiv',int(seed));
  for metric in ('graph_task_success','recovery_success','worst_order_success','fixed_order_success','long_horizon_completion','order_gap'):
   diff=g[metric]-b[metric];rows.append({'policy_seed':int(seed),'metric':metric,'pathgraph_value':g[metric],'comparator_value':b[metric],'difference':diff,'improved':bool(diff>0 if metric!='order_gap' else diff<0),'ceiling_tie':bool(metric=='graph_task_success' and diff==0 and g[metric]>=.95 and b[metric]>=.95),'degraded':bool(diff<0 if metric!='order_gap' else diff>0),'seed_block':'new' if int(seed) in NEW_SEEDS else 'old'})
 pd.DataFrame(rows).to_csv(a.output,index=False);print(f'REFINE1_SEED_EFFECTS_OK seeds={d.policy_seed.nunique()}')
if __name__=='__main__':main()
