"""Six authorized fresh runs, three resource queues, no retry or score-based choice."""
import concurrent.futures,hashlib,json,os,subprocess,time,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PY='/home/xushijie3/work/graph_saw_rd01_20261012/experiments/saw_rd01_20261012/vendor/venv/bin/python'
lock=threading.Lock();jobs=[]
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def queue(gpu,cores,spec):
 out=[]
 for kind,v,seed in spec:
  label=f'C_{kind}_{v}_s{seed}';group='C-S' if kind=='stitch' else 'C-N'
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda,cpu')
  cmd=[PY,str(ROOT/'guard.py'),'--label',label,'--group',group,'--seconds','9900','--','taskset','-c',cores,PY,str(ROOT/'train_native.py'),'--kind',kind,'--variant',v,'--seed',str(seed)]
  with (ROOT/'logs'/f'{label}_guard.log').open('x') as f:
   p=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
   entry=dict(label=label,kind=kind,variant=v,seed=seed,GPU=gpu,CPU_affinity=cores,guard_pid=p.pid,start=time.time(),status='RUNNING')
   with lock:jobs.append(entry);write(ROOT/'receipts/SCHEDULER_PROGRESS.json',jobs)
   rc=p.wait();entry.update(exit_code=rc,finish=time.time(),status='EXITED')
   with lock:write(ROOT/'receipts/SCHEDULER_PROGRESS.json',jobs)
   out.append(entry)
  print('QUEUE_JOB_EXIT',label,rc,flush=True)
  receipt=json.loads((ROOT/'receipts'/f'{label}_resource.json').read_text())
  reason=receipt['stop_reason'] or ''
  if reason in ('AGGREGATE_VRAM_LIMIT','DISK_RESERVE') or reason.startswith('MONITOR_FAILURE'):break
 return out
def main():
 assert json.loads((ROOT/'results/PREFLIGHT.json').read_text())['status']=='PASS'
 source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 write(ROOT/'receipts/TRAINING_SOURCE.json',dict(commit=source,official='3535a16c901e5de1233ac5a586401b3ddfa5e30d',files=[dict(path=n,sha256=hashlib.sha256((ROOT/n).read_bytes()).hexdigest()) for n in ('common.py','train_native.py','guard.py','scheduler.py')]))
 specs=[(0,'0-3',[('stitch','A',1),('stitch','A',2)]),(3,'4-7',[('stitch','B',1),('stitch','B',2)]),(6,'8-11',[('navigate','A',1),('navigate','A',2)])]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  futures=[pool.submit(queue,*s) for s in specs];results=[f.result() for f in futures]
 write(ROOT/'receipts/SCHEDULER_EXIT.json',dict(status='SIX_JOBS_EXITED' if sum(map(len,results))==6 else 'PARTIAL_QUEUE_RESOURCE_STOP',jobs=[x for xs in results for x in xs]))
 print('SCHEDULER_FINISHED',flush=True)
if __name__=='__main__':main()
