from common import *
import subprocess,psutil
os.nice(10);start=time.time()
z=np.load(RP/'prepared.npz');scenes=sorted(set(map(int,z['scene'])))
jobs=[('head',s,0,1) for s in scenes]+[('c3',s,k,2) for s in C3 for k in range(2)]
running=[];done=[];spent=0.;maximum=24
pilot=sum(json.load(open(p))['seconds'] for p in (R/'results/jobs').glob('*pilot.json'))
def diskbytes():return sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and 'archive_worktree' not in p.parts)
while jobs or running:
 elapsed=sum(time.time()-v['start'] for v in running)
 disk=diskbytes()
 if spent+elapsed+pilot>5.5*3600 or disk>950*1024**2:
  for v in running:v['p'].terminate()
  write(R/'results/dispatch_status.json',dict(status='BUDGET_STOP',worker_seconds=spent+elapsed+pilot,bytes=disk,finished=done,pending=jobs))
  raise SystemExit(3)
 while jobs and len(running)<maximum:
  pool,scene,shard,shards=jobs.pop(0);log=open(R/f'logs/{pool}_{scene}_{shard}.log','w')
  p=subprocess.Popen([sys.executable,str(R/'code/worker.py'),pool,str(scene),'--shard',str(shard),'--shards',str(shards)],stdout=log,stderr=subprocess.STDOUT)
  running.append(dict(p=p,log=log,start=time.time(),task=[pool,scene,shard,shards]))
 for v in running[:]:
  rc=v['p'].poll()
  if rc is not None:
   sec=time.time()-v['start'];spent+=sec;v['log'].close()
   done.append(dict(task=v['task'],seconds=sec,returncode=rc));running.remove(v)
   if rc:
    # Do not repeatedly dispatch a failing technical phase.
    for u in running:u['p'].terminate()
    write(R/'results/dispatch_status.json',dict(status='TECHNICAL_STOP',finished=done,worker_seconds=spent+elapsed+pilot,pending=jobs))
    raise SystemExit(rc)
 write(R/'results/dispatch_status.json',dict(status='RUNNING' if jobs or running else 'COMPLETE',finished=done,pending=len(jobs),active=[dict(pid=v['p'].pid,task=v['task'],seconds=time.time()-v['start']) for v in running],worker_seconds=spent+sum(time.time()-v['start'] for v in running)+pilot,bytes=disk,cpu_percent=psutil.cpu_percent(.2),memory_available=psutil.virtual_memory().available))
 time.sleep(2)
print('EXTRACTION_COMPLETE',spent+pilot,flush=True)
