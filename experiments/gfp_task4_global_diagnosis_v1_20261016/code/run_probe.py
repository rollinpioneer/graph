from pathlib import Path
import os,subprocess,time,json
R=Path(__file__).resolve().parents[1];os.chdir(R)
assert not (R/'receipts/probe_launch.json').exists(), 'Do not dispatch twice'
gpu=7
free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>10000
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
start=time.time();log=open(R/'logs/probe.log','w')
p=subprocess.Popen(['/home/xushijie3/work/gfp_task4_baseline_pilot_v1_20261015/envs/gfp/bin/python',str(R/'code/probe.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
(R/'receipts/probe_launch.json').write_text(json.dumps(dict(pid=p.pid,gpu=gpu,epoch=start,free_mib=free),indent=2))
peakmem=0;peakrss=0;samples=0
while p.poll() is None:
 try:
  lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines()
  own=[int(x.split(',')[1]) for x in lines if int(x.split(',')[0])==p.pid];peakmem=max([peakmem]+own)
  status=Path(f'/proc/{p.pid}/status').read_text();rss=[int(x.split()[1]) for x in status.splitlines() if x.startswith('VmRSS:')];peakrss=max([peakrss]+rss)
 except (OSError,ValueError,subprocess.SubprocessError):pass
 samples+=1;time.sleep(2)
log.close();elapsed=time.time()-start
(R/'receipts/probe_exit.json').write_text(json.dumps(dict(pid=p.pid,exit_code=p.returncode,wall_seconds=elapsed,gpu_residency_hours=elapsed/3600,peak_gpu_mib_sampled=peakmem,peak_rss_kb_sampled=peakrss,sampling_interval_seconds=2,samples=samples,shared_gpu=True),indent=2))
print('PROBE_EXIT',p.returncode,elapsed,flush=True)
