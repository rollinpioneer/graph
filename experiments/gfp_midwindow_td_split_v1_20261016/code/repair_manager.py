import os,json,time,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PYTHON='/home/xushijie3/work/flow_baselines_preflight_20261013/envs/gfp312/bin/python'
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def main():
 with (ROOT/'receipts/repair_manager_started.json').open('x') as f:json.dump(dict(pid=os.getpid(),time=time.time()),f)
 jobs={};rows={};pending=[('L1a',25000,5),('L1b',25000,2),('L1a',50000,5),('L1b',50000,2)];blocked=set()
 while pending or jobs:
  for key,j in list(jobs.items()):
   rc=j['proc'].poll()
   if rc is None:continue
   j['log'].close();rows[key].update(exit_code=rc,ended=time.time(),phase='complete' if rc==0 else 'failed')
   write(j['out']/'exit.json',dict(exit_code=rc,pid=j['proc'].pid,time=time.time(),dispatch_time=rows[key]['started']))
   if rc:blocked.add(j['arm'])
   del jobs[key]
  for item in list(pending):
   arm,it,gpu=item
   if arm in blocked:pending.remove(item);rows[f'{arm}_{it}']={'phase':'not_started_after_failure'};continue
   if len(jobs)>=2 or any(j['arm']==arm for j in jobs.values()):continue
   out=ROOT/f'runs/s3_{arm}'/(f'eval_{it}_repair1' if it==25000 else f'eval_{it}')
   assert not out.exists(),str(out)
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',HOME='/home/xushijie3/work/gfp_eta0_ablation_20261015/home')
   log=open(ROOT/f'logs/{arm}_eval_{it}_authorized.log','x');cmd=[PYTHON,str(ROOT/'code/evaluate_repair_pending.py'),'3',arm,str(it)]+(['--repair'] if it==25000 else [])
   proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
   key=f'{arm}_{it}';jobs[key]=dict(proc=proc,log=log,arm=arm,out=out);rows[key]=dict(pid=proc.pid,started=time.time(),phase='running',gpu=gpu,egl_device=gpu,command=cmd);pending.remove(item)
  write(ROOT/'receipts/repair_queue.json',rows)
  raw=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
  with open(ROOT/'logs/repair_resources.jsonl','a') as f:f.write(json.dumps(dict(time=time.time(),gpu=raw.stdout))+'\n')
  if jobs or pending:time.sleep(10)
 write(ROOT/'receipts/repair_manager_complete.json',dict(time=time.time(),jobs=rows,auto_retries=0))
if __name__=='__main__':
 try:main()
 except Exception:write(ROOT/'receipts/repair_manager_exception.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
