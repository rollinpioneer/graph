import os,json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY='/home/xushijie3/work/flow_baselines_preflight_20261013/envs/gfp312/bin/python'
def write(p,v):p.write_text(json.dumps(v,indent=2))
def main():
 with (ROOT/'receipts/launch.json').open('x') as f:
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  candidates=[(int(x[2]),-int(x[1]),int(x[0])) for line in rows.splitlines() if (x:=line.split(',')) and int(x[1])>=10000]
  assert candidates,'No GPU with safe 10000 MiB headroom'
  gpu=min(candidates)[2];json.dump(dict(pid=os.getpid(),training_gpu=gpu,time=time.time(),resource_before=rows),f)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 with open(ROOT/'logs/admission.log','x') as log:
  p=subprocess.Popen([PY,str(ROOT/'code/admission.py')],env=env,stdout=log,stderr=subprocess.STDOUT);rc=p.wait()
 write(ROOT/'receipts/admission_exit.json',dict(pid=p.pid,exit_code=rc,time=time.time()))
 if rc or not json.loads((ROOT/'receipts/admission_result.json').read_text())['passed']:
  write(ROOT/'receipts/training_exit.json',dict(exit_code=rc or 2,phase='admission',time=time.time()));return
 el=open(ROOT/'logs/evaluation_manager.log','x')
 ev=subprocess.Popen(['/usr/bin/python3',str(ROOT/'code/evaluation_manager.py')],stdout=el,stderr=subprocess.STDOUT)
 with open(ROOT/'logs/train.log','x') as log:
  p=subprocess.Popen([PY,str(ROOT/'code/continue_train.py')],env=env,stdout=log,stderr=subprocess.STDOUT);rc=p.wait()
 write(ROOT/'receipts/training_exit.json',dict(pid=p.pid,exit_code=rc,time=time.time()))
 er=ev.wait();el.close();write(ROOT/'receipts/evaluation_manager_exit.json',dict(pid=ev.pid,exit_code=er,time=time.time()))
if __name__=='__main__':main()
