import os,sys,json,time,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];receipt=R/'receipts/evaluation_manager.json'
with receipt.open('x') as f:json.dump(dict(pid=os.getpid(),start=time.time()),f)
jobs=['ReBRAC_seed0','ReBRAC_seed1','GFP_seed0','GFP_seed1'];active={};done=set();failed={}
def read(p):return json.loads(p.read_text()) if p.exists() else {}
while len(done)+len(failed)<len(jobs):
 for job,(p,start,gpu,handle) in list(active.items()):
  if p.poll() is not None:
   handle.close();rc=p.returncode;(R/'runs'/job/'matched_exit.json').write_text(json.dumps(dict(exit_code=rc,wall_seconds=time.time()-start,gpu=gpu)))
   if rc==0:done.add(job)
   else:failed[job]='Matched evaluation exit '+str(rc)
   del active[job]
 for job in jobs:
  if job in done or job in failed or job in active:continue
  state=read(R/'runs'/job/'status.json')
  if state.get('status')=='FAILED_STOPPED':failed[job]='Training failed';continue
  if state.get('status') not in ['COMPLETED','COMPLETED_ADOPTED']:continue
  if len(active)>=2:break
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  candidates=[]
  for l in rows.splitlines():
   g,mem,util=map(int,l.split(','))
   if mem>=8000 and g not in [v[2] for v in active.values()]:candidates.append((util,-mem,g))
  if not candidates:continue
  gpu=min(candidates)[2];env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',HOME=str(R/'home'),PYTHONUNBUFFERED='1')
  log=(R/'runs'/job/'matched.log').open('x');start=time.time();p=subprocess.Popen([str(R/'envs/gfp/bin/python'),str(R/'code/matched.py'),job],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  active[job]=(p,start,gpu,log)
 (R/'receipts/evaluation_queue.json').write_text(json.dumps(dict(active={k:dict(pid=v[0].pid,gpu=v[2],start=v[1]) for k,v in active.items()},completed=sorted(done),failed=failed,time=time.time()),indent=2))
 time.sleep(30)
(R/'receipts/evaluation_finished.json').write_text(json.dumps(dict(completed=sorted(done),failed=failed,time=time.time())))
