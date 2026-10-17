"""Supervise the single read-only GPU process and record its real exit and costs."""
from pathlib import Path
import subprocess,os,time,json
R=Path(__file__).resolve().parent
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
env.pop('JAX_PLATFORMS',None)
py='/home/xushijie3/work/flow_baselines_preflight_20261013/envs/gfp312/bin/python'
prior=json.loads((R/'receipts/inference_attempt1_exit.json').read_text())['wall_seconds']+60
limit=7200-prior
t=time.monotonic();observed=[];timed_out=False
with (R/'receipts/inference.log').open('w') as log:
 p=subprocess.Popen([py,str(R/'infer.py')],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
 (R/'receipts/inference_launch.json').write_text(json.dumps(dict(pid=p.pid,gpu=0,command=[py,str(R/'infer.py')],gpu_wall_budget_seconds=7200),indent=2))
 while p.poll() is None:
  if time.monotonic()-t>=limit:
   timed_out=True;p.terminate()
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:p.kill();p.wait()
   break
  try:
   text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
   for line in text.splitlines():
    pid,mem=line.split(',')
    if pid.strip()==str(p.pid):observed.append(dict(seconds=time.monotonic()-t,own_gpu_mib=int(mem)))
  except Exception:pass
  try:p.wait(timeout=5)
  except subprocess.TimeoutExpired:pass
 code=p.wait()
result=dict(exit_code=code,wall_seconds=time.monotonic()-t,gpu_resident_hours_upper_bound=(time.monotonic()-t)/3600,total_gpu_hours_conservative_upper_bound=(time.monotonic()-t+prior)/3600,prior_failed_attempt_seconds=prior-60,audit_gpu_resident_seconds_upper_bound=60,remaining_wall_limit_seconds=limit,timeout=timed_out,own_gpu_samples=observed,own_gpu_peak_mib=max([v['own_gpu_mib'] for v in observed],default=None),resource_sampling_interval_seconds=5)
(R/'receipts/inference_exit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='own_gpu_samples'}),flush=True)
raise SystemExit(code if code>=0 else 1)
