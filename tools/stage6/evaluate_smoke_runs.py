#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import torch
def main():
 p=argparse.ArgumentParser();p.add_argument('--status',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows=list(csv.DictReader(a.status.open()));checks=[]
 for r in rows:
  d=Path(r['output_dir']);s=json.loads((d/'status.json').read_text());ck=d/'checkpoints/best_val.pt';ok=r['status']=='PASS' and s['optimizer_steps']>=2000 and s['cuda_used'] and ck.exists();
  if ok:torch.load(ck,map_location='cpu',weights_only=False)
  checks.append({'job_id':r['job_id'],'pass':ok,'val_action_loss':s.get('best_val_action_loss'),'optimizer_steps':s.get('optimizer_steps')})
 decision='POLICY_PROTOCOL_LOCKED' if all(x['pass'] for x in checks) else 'FIX_TRAINING_ADAPTER';a.out.mkdir(parents=True,exist_ok=True);(a.out/'smoke_metrics.json').write_text(json.dumps(checks,indent=2)+'\n');(a.out/'smoke_gate.json').write_text(json.dumps({'decision':decision,'jobs':checks},indent=2)+'\n');print(json.dumps({'decision':decision,'passed':sum(x['pass'] for x in checks),'total':len(checks)},indent=2));return 0 if decision=='POLICY_PROTOCOL_LOCKED' else 2
if __name__=='__main__':raise SystemExit(main())
