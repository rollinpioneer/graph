#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv
from pathlib import Path
from tools.stage6_refine1.common import METHODS,TASKS,NEW_SEEDS
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--weights-root',type=Path,required=True);p.add_argument('--init-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
 for task in TASKS:
  for seed in NEW_SEEDS:
   init=a.init_root/task/f'seed_{seed}'/'init.pt'
   for method in METHODS:
    out=a.output_root/f'{task}__{method}__s{seed}';weight=a.weights_root/f'{method}.parquet';cmd=f'PYTHONPATH="$PWD" /home/xushijie/.conda/envs/cupid/bin/python tools/stage6/train_weighted_policy.py --task {task} --method {method} --policy-seed {seed} --config {a.config.resolve()} --dataset-root artifacts/pathgraph_sarm/stage6/policy_data_v1 --weight-file {weight.resolve()} --init-checkpoint {init.resolve()} --output-dir {out.resolve()} --max-optimizer-steps 2000 --device cuda:0';rows.append({'job_id':out.name,'task_id':task,'method':method,'policy_seed':seed,'model_seed':seed,'data_seed':seed+100000,'output_dir':str(out.resolve()),'weight_file':str(weight.resolve()),'init_checkpoint':str(init.resolve()),'init_sha256':'','total_optimizer_steps':2000,'effective_batch_size':32,'command':cmd})
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
 print(f'REFINE1_JOB_MATRIX_12_OK rows={len(rows)}')
if __name__=='__main__':main()
