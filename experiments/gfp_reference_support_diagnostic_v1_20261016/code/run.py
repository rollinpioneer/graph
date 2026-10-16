from util import *
import os,sys,subprocess,resource,traceback
t0=time.time();rows=[]
write('receipts/launch.json',dict(pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),started=time.time(),gfp_updates=0,environment_steps=0))
try:
 for script,args in [('prepare.py',[]),('density.py',['1701']),('density.py',['1702']),('infer.py',[])]:
  before=time.time()
  with open(ROOT/'logs'/f'{script}_{"_".join(args)}.log','w') as f:p=subprocess.run([sys.executable,str(ROOT/'code'/script)]+args,stdout=f,stderr=subprocess.STDOUT)
  rows.append(dict(script=script,args=args,exit_code=p.returncode,seconds=time.time()-before))
  write('receipts/run_progress.json',dict(phases=rows,seconds=time.time()-t0))
  if p.returncode:raise RuntimeError(str(rows[-1]))
 write('receipts/run_complete.json',dict(phases=rows,exit_code=0,seconds=time.time()-t0,gfp_updates=0,environment_steps=0,children_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime))
except Exception:
 write('receipts/run_failed.json',dict(phases=rows,seconds=time.time()-t0,error=traceback.format_exc()))
 raise
