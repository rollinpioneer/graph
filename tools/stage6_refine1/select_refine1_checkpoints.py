#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import torch
from tools.stage6_refine1.common import sha256,METHODS,NEW_SEEDS,TASKS
def main():
 p=argparse.ArgumentParser();p.add_argument('--jobs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--lock',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args();jobs=list(csv.DictReader(a.jobs.open(),delimiter='\t'));rows=[]
 for j in jobs:
  d=Path(j['output_dir']);status=json.loads((d/'status.json').read_text());ck=d/'checkpoints/best_val.pt';assert ck.exists();torch.load(ck,map_location='cpu',weights_only=False);rows.append({'task_id':j['task_id'],'method':j['method'],'policy_seed':int(j['policy_seed']),'selected_step':int(status['optimizer_steps']),'selected_val_metric':float(status['best_val_action_loss']),'checkpoint_path':str(ck.resolve()),'checkpoint_size_bytes':ck.stat().st_size,'checkpoint_sha256':sha256(ck),'selection_split':'val','selection_metric':'val_action_loss_unweighted','test_used':False,'job_status':status['status'],'init_sha256':j['init_sha256'],'weight_sha256':status['weight_sha256']})
 assert len(rows)==12 and len({(r['task_id'],r['method'],r['policy_seed']) for r in rows})==12
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 a.lock.write_text(json.dumps({'locked':True,'selection_source':'validation_only','selection_metric':'val_action_loss_unweighted','selection_mode':'min','test_used':False,'selected_count':len(rows),'selection_csv_sha256':sha256(a.output)},indent=2)+'\n')
 a.manifest.parent.mkdir(parents=True,exist_ok=True)
 with a.manifest.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['task_id','method','policy_seed','checkpoint_path','checkpoint_size_bytes','checkpoint_sha256'],delimiter='\t');w.writeheader();w.writerows({k:r[k] for k in w.fieldnames} for r in rows)
 print(json.dumps({'selected':len(rows),'metric':'val_action_loss_unweighted'},indent=2))
if __name__=='__main__':main()
