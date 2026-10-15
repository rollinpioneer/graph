#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv
from pathlib import Path
import pandas as pd
from tools.stage6_refine1.common import aggregate_seed
def main():
 p=argparse.ArgumentParser();p.add_argument('--rollouts',type=Path,required=True);p.add_argument('--by-seed',type=Path,required=True);p.add_argument('--aggregate',type=Path,required=True);p.add_argument('--condition',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();d=pd.read_csv(a.rollouts);rows=[]
 for seed in sorted(d.policy_seed.unique()):
  for method in ('linear_sarm_equiv','pathgraph_reward_v1_locked'):
   z=aggregate_seed(d,method,int(seed));rows.append({'policy_seed':int(seed),'method':method,**z})
 out=pd.DataFrame(rows);a.by_seed.parent.mkdir(parents=True,exist_ok=True);out.to_csv(a.by_seed,index=False);d.groupby(['task_id','condition','method','policy_seed'],as_index=False).agg(n=('success','size'),success_rate=('success','mean'),recovery_success_rate=('recovery_success','mean'),completion_mean=('success','mean'),median_steps=('steps','median')).to_csv(a.condition,index=False);out.groupby('method',as_index=False).mean(numeric_only=True).to_csv(a.aggregate,index=False);a.report.write_text('# R1 frozen evaluation summary\n\nNew seed metrics were computed from the 1,500 paired rollout rows after checkpoint selection lock.\n');print(out.to_string(index=False))
if __name__=='__main__':main()
