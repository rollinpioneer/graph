"""Pure receipt checks for terminal controller work; never starts/stops a process."""
from pathlib import Path
import json
EXPECTED={f'{model}_seed{seed}' for model in ['G0','G1','D0','D1'] for seed in [0,1]}
def read(path):
 try:return json.loads(Path(path).read_text())
 except (FileNotFoundError,json.JSONDecodeError):return None
def completed_work(name,current,root):
 root=Path(root);runs=current.get('runs',{})
 if set(runs)!=EXPECTED:return False
 for job,item in runs.items():
  status=item.get('status') or {}
  if status.get('status')!='COMPLETED' or status.get('updates')!=1000000 or status.get('exit_code')!=0:return False
  phase=status.get('phase')
  if phase not in ['T1','T3']:return False
  receipt=read(root/'runs'/job/(phase+'_exit.json')) or {}
  if receipt.get('status')!='PASS' or receipt.get('exit_code')!=0:return False
 if name=='manager_started.json':
  finish=read(root/'receipts/manager_finished.json') or {};queue=current.get('queue') or {}
  return set(finish.get('completed',[]))==EXPECTED and not finish.get('failed',[]) and set(queue.get('completed',[]))==EXPECTED and not queue.get('active') and not queue.get('failed',[])
 if name!='evaluation_manager_started.json':return False
 finish=read(root/'receipts/evaluation_manager_finished.json') or {};queue=current.get('evaluation_queue') or {}
 if set(finish.get('completed',[]))!=EXPECTED or finish.get('failed') or finish.get('skipped'):return False
 if set(queue.get('completed',[]))!=EXPECTED or queue.get('active') is not None or queue.get('failed') or queue.get('skipped'):return False
 for job in EXPECTED:
  done=read(root/'runs'/job/'matched_verified/completed.json') or {}
  exited=read(root/'runs'/job/'matched_verified_exit.json') or {}
  if done.get('status')!='PASS' or done.get('optimizer_updates')!=0 or exited.get('exit_code')!=0 or exited.get('optimizer_updates')!=0:return False
  for step in [800000,900000,1000000]:
   d=read(root/'runs'/job/'matched_verified'/(str(step)+'.json')) or {}
   if d.get('status')!='PASS' or d.get('optimizer_updates')!=0 or d.get('step')!=step or len(d.get('episodes',[]))!=100:return False
 return True
