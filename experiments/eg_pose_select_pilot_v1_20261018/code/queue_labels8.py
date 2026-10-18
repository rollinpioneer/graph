from core import *
import subprocess
resume=json.loads((R/'results/label_queue_handoff.json').read_text());active={x['scene']:(x['pid'],None) for x in resume['active']};pending=resume['pending'];done=resume['done'];start=resume['start'];failed=False
while active or pending:
 for s,(pid,p) in list(active.items()):
  if p is not None:rc=p.poll();ended=rc is not None
  else:
   path=Path(f'/proc/{pid}/stat');ended=not path.exists() or path.read_text().split()[2]=='Z';rc=None
  if ended:
   receipt=R/f'results/labels_{s:04d}_full.json';ok=receipt.exists() and json.loads(receipt.read_text())['status']=='PASS' and (rc is None or rc==0)
   done.append(dict(scene=s,exit=0 if ok else 1,completion_basis='receipt_and_exit' if p is not None else 'receipt_original_child_exit_unavailable',seconds=json.loads(receipt.read_text())['seconds'] if receipt.exists() else None));del active[s];failed=failed or not ok
 if failed and not active:write(R/'results/label_queue.json',dict(status='FAILED',done=done,pending=pending));sys.exit(1)
 if not failed:
  while pending and len(active)<8:
   s=pending.pop(0);log=open(R/f'logs/labels_{s}.log','a');p=subprocess.Popen([sys.executable,str(R/'code/labels.py'),'--scene',str(s)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();active[s]=(p.pid,p)
 write(R/'results/label_queue.json',dict(status='RUNNING' if active or pending else 'PASS',done=done,active=[dict(scene=s,pid=v[0]) for s,v in active.items()],pending=pending,worker_limit=8,wall_seconds=time.time()-start))
 if active or pending:time.sleep(10)
