from pathlib import Path
import os,subprocess,time,json,datetime
R=Path(__file__).resolve().parent
assert json.loads((R/'receipts/freeze_git.json').read_text())['verified']
assert not (R/'receipts/launch.json').exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',MUJOCO_GL='egl')
start=time.monotonic();log=open(R/'receipts/infer.log','w');p=subprocess.Popen(['/home/xushijie3/work/flow_baselines_preflight_20261013/envs/gfp312/bin/python','-u',str(R/'infer.py')],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
(R/'receipts/launch.json').write_text(json.dumps(dict(pid=p.pid,wrapper_pid=os.getpid(),gpu=0,training=False,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
samples=[]
while p.poll() is None:
 try:
  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  own=[]
  for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines():
   pid,mem=line.split(',');status=Path('/proc')/pid.strip()/'status'
   if status.exists() and int(next(x for x in status.read_text().splitlines() if x.startswith('Uid:')).split()[1])==os.getuid():own.append(dict(pid=int(pid),gpu_memory_mib=mem.strip()))
  samples.append(dict(seconds=time.monotonic()-start,gpu_aggregate=gpu,own_gpu_processes=own))
 except Exception as e:samples.append(dict(error=str(e)))
 time.sleep(10)
rc=p.wait();log.close();(R/'receipts/resources.json').write_text(json.dumps(samples,indent=2));(R/'receipts/execution.json').write_text(json.dumps(dict(exit_code=rc,wall_seconds=time.monotonic()-start,wrapper_pid=os.getpid(),pid=p.pid,training_updates=0,environment_steps=0),indent=2))
