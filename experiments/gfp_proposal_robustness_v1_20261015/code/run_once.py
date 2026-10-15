import subprocess,os,time,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def save(name,x):(R/'receipts'/name).write_text(json.dumps(x,indent=2))
assert not (R/'receipts/dispatch.json').exists(),'NO_RESTART'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
py='/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/envs/gfp/bin/python'
gpu=subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();assert int(gpu.split(',')[0])>8192,gpu
start=time.time();log=open(R/'receipts/probe.log','xb');p=subprocess.Popen([py,str(R/'code/probe.py')],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
save('dispatch.json',dict(pid=p.pid,manager_pid=os.getpid(),gpu=4,start_unix=start,preflight_gpu=gpu,command=[py,str(R/'code/probe.py')],code_sha256=hashlib.sha256((R/'code/probe.py').read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((R/'configs/frozen_protocol.json').read_bytes()).hexdigest(),optimizer_updates=0))
samples=[]
while p.poll() is None:
 s=dict(unix=time.time(),pid=p.pid)
 try:
  txt=Path(f'/proc/{p.pid}/status').read_text();s.update({l.split(':')[0]:int(l.split()[1]) for l in txt.splitlines() if l.startswith(('VmRSS:','VmHWM:'))})
  rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
  s['own_gpu_mib']=sum(int(l.split(',')[1]) for l in rows.splitlines() if int(l.split(',')[0])==p.pid)
 except Exception as e:s['sample_error']=str(e)
 samples.append(s);save('resource_samples.json',samples);time.sleep(5)
rc=p.wait();log.close();save('exit.json',dict(pid=p.pid,exit_code=rc,start_unix=start,end_unix=time.time(),wall_seconds=time.time()-start,peak_sampled_gpu_mib=max((s.get('own_gpu_mib',0) for s in samples),default=0),peak_sampled_rss_kib=max((s.get('VmRSS',0) for s in samples),default=0),sample_interval_seconds=5,optimizer_updates=0));print('EXIT',rc,flush=True)
