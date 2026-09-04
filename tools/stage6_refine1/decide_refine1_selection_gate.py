#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import torch
def main():
 p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);p.add_argument('--selection-lock',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();rows=list(csv.DictReader(a.selection.open()));lock=json.loads(a.selection_lock.read_text());checks={'selected_count_12':len(rows)==12,'unique_triplets':len({(r['task_id'],r['method'],r['policy_seed']) for r in rows})==12,'new_seeds_exact':set(int(r['policy_seed']) for r in rows)=={20260912,20260913,20260914},'methods_exact':set(r['method'] for r in rows)=={'linear_sarm_equiv','pathgraph_reward_v1_locked'},'selection_val_only':all(r['selection_split']=='val' and r['test_used'].lower()=='false' and r['selection_metric']=='val_action_loss_unweighted' for r in rows),'lock_valid':lock.get('locked') is True and lock.get('test_used') is False,'checkpoints_loadable':True}
 for r in rows:
  try:torch.load(r['checkpoint_path'],map_location='cpu',weights_only=False)
  except Exception:checks['checkpoints_loadable']=False
 decision='REFINE1_SELECTION_LOCKED' if all(checks.values()) else 'CHECKPOINT_LOAD_FAILURE';result={'decision':decision,'checks':checks,'selected_count':len(rows)};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');a.report.write_text('# R1 checkpoint selection summary\n\nDecision: `'+decision+'`\n\nAll 12 checkpoints are selected on validation-only unweighted action loss; test was not read.\n');print(json.dumps(result,indent=2));return 0 if decision.endswith('LOCKED') else 2
if __name__=='__main__':raise SystemExit(main())
