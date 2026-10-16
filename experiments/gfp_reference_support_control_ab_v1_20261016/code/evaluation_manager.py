import os,json,time,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PYTHON='/home/xushijie3/work/flow_baselines_preflight_20261013/envs/gfp312/bin/python'
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def main():
 with (ROOT/'receipts/evaluation_manager_started.json').open('x') as f:json.dump(dict(pid=os.getpid(),time=time.time()),f)
 pending=[('A',25000),('B',25000),('A',50000),('B',50000)];jobs={};rows={};blocked=set()
 while pending or jobs:
  for key,j in list(jobs.items()):
   rc=j['proc'].poll()
   if rc is None:continue
   j['log'].close();rows[key].update(exit_code=rc,ended=time.time(),phase='complete' if rc==0 else 'failed')
   write(j['out']/'exit.json',dict(exit_code=rc,pid=j['proc'].pid,time=time.time()))
   if rc:blocked.add(j['arm'])
   del jobs[key]
  resource=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  with open(ROOT/'logs/evaluation_resources.jsonl','a') as f:f.write(json.dumps(dict(time=time.time(),gpu=resource))+'\n')
  candidates=[]
  for line in resource.splitlines():
   gpu,free,util=map(int,line.split(','))
   if gpu!=0 and free>=7000 and gpu not in [v['gpu'] for v in jobs.values()]:candidates.append((util,-free,gpu))
  candidates.sort()
  failedtraining=(ROOT/'receipts/training_exit.json').exists() and json.loads((ROOT/'receipts/training_exit.json').read_text())['exit_code']!=0
  for arm,it in list(pending):
   key=f'{arm}_{it}';checkpoint=ROOT/f'runs/{arm}/checkpoint_{it}.json'
   if arm in blocked or (failedtraining and not checkpoint.exists()):
    pending.remove((arm,it));rows[key]=dict(phase='not_started_after_failure');continue
   if len(jobs)>=2 or not candidates or not checkpoint.exists() or any(j['arm']==arm for j in jobs.values()):continue
   _,_,gpu=candidates.pop(0);out=ROOT/f'runs/{arm}/eval_{it}';assert not out.exists()
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
   log=open(ROOT/f'logs/evaluate_{arm}_{it}.log','x');cmd=[PYTHON,str(ROOT/'code/evaluate.py'),'3',arm,str(it)]
   proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
   jobs[key]=dict(proc=proc,log=log,arm=arm,gpu=gpu,out=out);rows[key]=dict(pid=proc.pid,phase='running',gpu=gpu,started=time.time(),command=cmd);pending.remove((arm,it))
  write(ROOT/'receipts/evaluation_queue.json',rows)
  if pending or jobs:time.sleep(10)
 write(ROOT/'receipts/evaluation_complete.json',dict(time=time.time(),conditions=rows,auto_retries=0))
if __name__=='__main__':
 try:main()
 except Exception:write(ROOT/'receipts/evaluation_manager_error.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
