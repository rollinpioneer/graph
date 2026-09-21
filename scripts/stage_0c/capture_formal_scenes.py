"""Fail-closed formal scene capture entrypoint; never fabricates images."""
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--task-id',required=True);p.add_argument('--scene-id',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.task_id not in {'D0','T_A','T_C'}: raise SystemExit('BLOCKED: task not in formal matrix')
 if a.output.exists(): raise SystemExit('BLOCKED: refusing overwrite of formal scene')
 task=a.root/'configs/tasks/resolved'/f'{a.task_id}.yaml'
 import yaml
 d=yaml.safe_load(task.read_text())
 if d.get('status')!='READY_FOR_FORMAL_CAPTURE': raise SystemExit('BLOCKED: resolved task lacks READY_FOR_FORMAL_CAPTURE and bound controller/verifier/evaluator')
 raise SystemExit('BLOCKED: platform adapter not bound; no image generated')
if __name__=='__main__':main()
