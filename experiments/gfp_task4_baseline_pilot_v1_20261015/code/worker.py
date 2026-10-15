import os,sys,json,time,subprocess,datetime,resource,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]; job=sys.argv[1];d=R/'runs'/job;spec=json.loads((R/'configs'/f'{job}.json').read_text());d.mkdir(exist_ok=True)
def write(name,v): (d/name).write_text(json.dumps(v,indent=2))
lock=os.open(d/'worker.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(lock,str(os.getpid()).encode());os.close(lock)
env=os.environ.copy();env.update(spec['environment']);cmd=spec['command'];t=time.time();peak=0
write('status.json',dict(status='STARTING',worker_pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu=spec['gpu']))
with (d/'console.log').open('xb') as f:
 p=subprocess.Popen(cmd,cwd=R/'workspaces/gfp',env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 write('status.json',dict(status='RUNNING',worker_pid=os.getpid(),pid=p.pid,start_unix=t,gpu=spec['gpu']))
 with (d/'resources.jsonl').open('x') as res:
  while p.poll() is None:
   row={'time':time.time(),'pid':p.pid}
   try:
    row['process']=subprocess.check_output(['ps','-p',str(p.pid),'-o','etimes=,time=,%cpu=,rss='],text=True).strip()
    row['gpu']=subprocess.check_output(['nvidia-smi',f'--id={spec["gpu"]}','--query-gpu=utilization.gpu,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip()
    row['own_gpu_processes']=subprocess.check_output(['nvidia-smi',f'--id={spec["gpu"]}','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip()
   except Exception as e: row['resource_error']=str(e)
   res.write(json.dumps(row)+'\n');res.flush();time.sleep(15)
 rc=p.returncode
ru=resource.getrusage(resource.RUSAGE_CHILDREN)
write('exit.json',dict(exit_code=rc,wall_seconds=time.time()-t,cpu_seconds=ru.ru_utime+ru.ru_stime,max_rss_kib=ru.ru_maxrss,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
write('status.json',dict(status='COMPLETED' if rc==0 else 'FAILED_STOPPED',exit_code=rc,worker_pid=os.getpid(),gpu=spec['gpu']))
sys.exit(rc)
