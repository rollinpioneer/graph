from pathlib import Path
import subprocess,os,json,time,datetime
R=Path(__file__).resolve().parents[1]; os.chdir(R)
if (R/'receipts/probe_process.json').exists():raise RuntimeError('Already dispatched; no automatic retry')
py='/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/envs/gfp/bin/python'
pre=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader'],text=True)
print(pre,flush=True); gpu=1
row=[x for x in pre.splitlines() if x.split(',')[0].strip()==str(gpu)][0]
assert int(row.split(',')[2].split()[0])>=8000
E=os.environ.copy();E.update(CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
start=time.time()
with open(R/'receipts/probe.log','w') as f:
 p=subprocess.Popen([py,'-u','code/probe.py'],stdout=f,stderr=subprocess.STDOUT,env=E)
 (R/'receipts/probe_process.json').write_text(json.dumps(dict(pid=p.pid,gpu=gpu,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),resource_before=pre,command=[py,'-u','code/probe.py'],environment_overrides={k:E[k] for k in ['CUDA_VISIBLE_DEVICES','XLA_PYTHON_CLIENT_PREALLOCATE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}),indent=2))
 with open(R/'receipts/probe_resources.jsonl','w') as out:
  while p.poll() is None:
   q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
   own=[l for l in q.stdout.splitlines() if l.split(',')[0].strip()==str(p.pid)]
   try:rss=[x for x in Path(f'/proc/{p.pid}/status').read_text().splitlines() if x.startswith(('VmRSS:','VmHWM:'))]
   except FileNotFoundError:rss=[]
   out.write(json.dumps(dict(time=time.time(),gpu=own,memory=rss))+'\n');out.flush();time.sleep(5)
 (R/'receipts/probe_exit.json').write_text(json.dumps(dict(pid=p.pid,exit_code=p.returncode,wall_seconds=time.time()-start,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
print('PROBE_EXIT',p.returncode,flush=True)
