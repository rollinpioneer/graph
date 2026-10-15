#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv
from pathlib import Path
import torch
from tools.stage6.policy_model import ChunkPolicy
from tools.stage6_refine1.common import NEW_SEEDS,TASKS,sha256
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args();rows=[]
 for task in TASKS:
  for seed in NEW_SEEDS:
   torch.manual_seed(seed);m=ChunkPolicy();d=a.output_root/task/f'seed_{seed}';d.mkdir(parents=True,exist_ok=True);f=d/'init.pt';torch.save({'model':m.state_dict(),'seed':seed,'task_id':task},f);torch.load(f,map_location='cpu',weights_only=False);rows.append({'task_id':task,'policy_seed':seed,'model_seed':seed,'data_seed':seed+100000,'augmentation_seed':seed+200000,'validation_seed_start':seed+300000,'checkpoint_path':str(f.resolve()),'checkpoint_sha256':sha256(f),'load_ok':True})
 a.manifest.parent.mkdir(parents=True,exist_ok=True)
 with a.manifest.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
 print(f'REFINE1_INITIALIZATIONS_6_OK rows={len(rows)}')
if __name__=='__main__':main()
