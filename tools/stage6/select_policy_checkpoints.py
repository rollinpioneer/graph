#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--jobs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--lock',type=Path,required=True);a=p.parse_args();rows=[]
 for j in csv.DictReader(a.jobs.open()):
  d=Path(j['output_dir']);s=json.loads((d/'status.json').read_text());ck=d/'checkpoints/best_val.pt';rows.append({'task_id':j['task_id'],'method':j['method'],'policy_seed':j['policy_seed'],'checkpoint_path':str(ck.resolve()),'checkpoint_sha256':sha(ck),'checkpoint_size_bytes':ck.stat().st_size,'selection_metric':'val_action_loss','selection_value':s['best_val_action_loss'],'selection_mode':'min','test_used':False,'optimizer_steps':s['optimizer_steps']})
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 a.lock.write_text(json.dumps({'locked':True,'selection_source':'validation_action_loss_only','test_used':False,'selected_checkpoint_count':len(rows),'selection_csv_sha256':sha(a.out)},indent=2)+'\n');print(json.dumps({'selected':len(rows)},indent=2))
if __name__=='__main__':main()
