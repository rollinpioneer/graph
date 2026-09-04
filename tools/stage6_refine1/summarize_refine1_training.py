#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--jobs',type=Path,required=True);p.add_argument('--status',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--checkpoint-manifest',type=Path,required=True);a=p.parse_args();jobs=list(csv.DictReader(a.jobs.open(),delimiter='\t'));statuses=list(csv.DictReader(a.status.open(),delimiter='\t'));status={r['job_id']:r for r in statuses};records=[]
 for j in jobs:
  s=status.get(j['job_id'],{});sp=Path(j['output_dir'])/'status.json';obj=json.loads(sp.read_text()) if sp.exists() else {};records.append({'job_id':j['job_id'],'task_id':j['task_id'],'method':j['method'],'policy_seed':int(j['policy_seed']),'status':s.get('status','MISSING'),'cuda_used':bool(obj.get('cuda_used',False)),'optimizer_steps':int(obj.get('optimizer_steps',0)),'best_val_action_loss':obj.get('best_val_action_loss'),'init_sha256':obj.get('init_sha256'),'weight_sha256':obj.get('weight_sha256'),'checkpoint_exists':(Path(j['output_dir'])/'checkpoints/best_val.pt').exists()})
 checks={'job_count_12':len(records)==12,'all_pass':all(x['status']=='PASS' for x in records),'all_cuda':all(x['cuda_used'] for x in records),'all_steps_2000':all(x['optimizer_steps']==2000 for x in records),'all_checkpoint':all(x['checkpoint_exists'] for x in records),'test_rollout_count_zero':not any('test' in str(x).lower() for x in records)};decision='REFINE1_TRAINING_COMPLETE' if all(checks.values()) else 'RETRY_FAILED_JOBS';result={'decision':decision,'checks':checks,'records':records};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');a.summary.write_text('# R1 training summary\n\nDecision: `'+decision+'`\n\n'+json.dumps(checks,indent=2)+'\n');a.checkpoint_manifest.parent.mkdir(parents=True,exist_ok=True)
 with a.checkpoint_manifest.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['job_id','checkpoint_path','size_bytes','sha256'],delimiter='\t');w.writeheader();
 with a.checkpoint_manifest.open('a',newline='') as f:
  w=csv.writer(f,delimiter='\t')
  for j in jobs:
   ck=Path(j['output_dir'])/'checkpoints/best_val.pt'
   if ck.exists():
    import hashlib;h=hashlib.sha256();h.update(ck.read_bytes());w.writerow([j['job_id'],str(ck),ck.stat().st_size,h.hexdigest()])
 print(json.dumps(result,indent=2));return 0 if decision.endswith('COMPLETE') else 2
if __name__=='__main__':raise SystemExit(main())
