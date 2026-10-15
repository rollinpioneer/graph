#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--inits',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--root-output',type=Path,required=True);p.add_argument('--steps',type=int,required=True);p.add_argument('--seeds',required=True);a=p.parse_args();methods=['bc_all','linear_sarm_equiv','sequential_transition','pathgraph_reward_v1_locked'];tasks=['transport_recovery','transport_dual_order'];rows=[]
 for task in tasks:
  for seed in map(int,a.seeds.split(',')):
   for method in methods:
    jid=f'{task}__{method}__s{seed}';out=a.root_output/jid;cmd=f'PYTHONPATH="$PWD" /home/xushijie/.conda/envs/cupid/bin/python tools/stage6/train_weighted_policy.py --task {task} --method {method} --policy-seed {seed} --config {a.config} --dataset-root artifacts/pathgraph_sarm/stage6/policy_data_v1 --weight-file {a.weights}/chunk_weights/{method}.parquet --init-checkpoint {a.inits/task/f"seed_{seed}"/"init.pt"} --output-dir {out} --max-optimizer-steps {a.steps} --device cuda:0';rows.append({'job_id':jid,'task_id':task,'method':method,'policy_seed':seed,'output_dir':str(out),'command':cmd})
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
