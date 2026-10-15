from pathlib import Path
import subprocess,sys,time,json,os
R=Path(__file__).resolve().parents[1];start=time.time()
with (R/'receipts/dispatched.json').open('x') as f:json.dump(dict(wrapper_pid=os.getpid(),start=start),f)
p=subprocess.Popen([sys.executable,str(R/'code/probe.py')],cwd=R)
samples=[]
while p.poll() is None:
 row=dict(time=time.time(),pid=p.pid)
 try:
  for line in Path(f'/proc/{p.pid}/status').read_text().splitlines():
   if line.startswith(('VmRSS:','VmHWM:')):row[line.split(':')[0]]=int(line.split()[1])
  out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=5)
  row['own_gpu_memory_mib']=[int(v.split(',')[1]) for v in out.splitlines() if v.split(',')[0].strip()==str(p.pid)]
 except Exception as e:row['sampling_error']=repr(e)
 samples.append(row);time.sleep(2)
rc=p.wait();end=time.time()
(R/'receipts/resource_samples.json').write_text(json.dumps(samples,indent=2))
(R/'receipts/exit.json').write_text(json.dumps(dict(pid=p.pid,wrapper_pid=os.getpid(),exit_code=rc,start=start,end=end,process_residence_seconds=end-start),indent=2));sys.exit(rc)
