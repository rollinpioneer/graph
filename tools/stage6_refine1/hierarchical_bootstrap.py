#!/usr/bin/env python3
"""Six-seed, paired two-level bootstrap with equal-condition aggregates."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from tools.stage6_refine1.common import aggregate_seed
def main():
 p=argparse.ArgumentParser();p.add_argument('--rollouts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--distribution-dir',type=Path,required=True);p.add_argument('--resamples',type=int,default=5000);p.add_argument('--seed',type=int,default=20261021);a=p.parse_args();d=pd.read_csv(a.rollouts);seeds=np.array(sorted(d.policy_seed.unique()));conds=[('transport_recovery','natural'),('transport_recovery','drop_regrasp'),('transport_recovery','gripper_reopen'),('transport_dual_order','A_first'),('transport_dual_order','B_first')];A=np.empty((len(seeds),5,50),float);B=np.empty_like(A)
 for si,s in enumerate(seeds):
  for ci,(task,cond) in enumerate(conds):
   for mi,m in enumerate(('pathgraph_reward_v1_locked','linear_sarm_equiv')):
    q=d[(d.policy_seed==s)&(d.task_id==task)&(d.condition==cond)&(d.method==m)].sort_values('rollout_index');assert len(q)==50; (A if mi==0 else B)[si,ci]=q.success.to_numpy()
 rng=np.random.default_rng(a.seed);picked=rng.integers(0,len(seeds),size=(a.resamples,len(seeds)));idx=rng.integers(0,50,size=(a.resamples,len(seeds),5,50));
 aa=A[picked[:, :, None, None],np.arange(5)[None,None,:,None],idx];bb=B[picked[:, :, None, None],np.arange(5)[None,None,:,None],idx]; # [R,S,C,50]
 # Per-seed equal-condition metrics then equal-seed average; paired keys are shared in aa/bb.
 def metrics(x):
  nat,drop,reopen,af,bf=[x[:,:,i].mean(-1) for i in range(5)];return {'graph_task_success':np.mean([nat,drop,reopen,af,bf],axis=0),'recovery_success':np.mean([drop,reopen],axis=0),'worst_order_success':np.minimum(af,bf),'fixed_order_success':np.mean([nat,af,bf],axis=0),'long_horizon_completion':np.mean([nat,drop,reopen,af,bf],axis=0),'order_gap':np.abs(af-bf)}
 ma,mb=metrics(aa),metrics(bb);a.distribution_dir.mkdir(parents=True,exist_ok=True);rows=[]
 for metric in ma:
  v=np.mean(ma[metric]-mb[metric],axis=1);np.save(a.distribution_dir/f'{metric}.npy',v);point=float(np.mean(ma[metric][0]-mb[metric][0]))
  # Point estimate is recomputed directly from all fixed rows below, not bootstrap draw 0.
  real_a=[];real_b=[]
  for s in seeds:real_a.append(aggregate_seed(d,'pathgraph_reward_v1_locked',int(s))[metric]);real_b.append(aggregate_seed(d,'linear_sarm_equiv',int(s))[metric])
  rows.append({'task_id':'all_graph_tasks','condition_or_aggregate':'predefined_equal_condition_aggregate','metric':metric,'method_a':'pathgraph_reward_v1_locked','method_b':'linear_sarm_equiv','point_estimate_a':float(np.mean(real_a)),'point_estimate_b':float(np.mean(real_b)),'difference':float(np.mean(real_a)-np.mean(real_b)),'ci95_low':float(np.quantile(v,.025)),'ci95_high':float(np.quantile(v,.975)),'prob_difference_gt_0':float(np.mean(v>0)),'resamples':a.resamples,'seed':a.seed})
 pd.DataFrame(rows).to_csv(a.output,index=False);print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
