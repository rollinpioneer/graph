"""Bounded task set, unlimited time, adaptive to failures; no training launch."""
import os,sys,time,json,subprocess
from pathlib import Path
R=Path(os.environ['TASK_ROOT'])
def write(p,v):q=p.with_suffix('.partial.json');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
profile=json.loads((R/'results/eval_0100_profile.json').read_text());assert profile['status']=='PASS' and len(profile['native_checks'])==6
cfg=json.loads((R/'frozen_config.json').read_text());tasks=[(s,k) for s in sum(cfg['C3'].values(),[]) for k in range(8)];active=[];complete=[];failed=[];peak=0;t=time.time();storage=0;last_storage=0
while tasks or active:
 while tasks and len(active)<80 and not failed:
  scene,shard=tasks.pop(0);log=open(R/f'logs/eval_{scene:04d}_{shard}.log','w');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';p=subprocess.Popen([sys.executable,str(R/'code/evaluate.py'),'--scene',str(scene),'--shard',str(shard),'--shards','8'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active.append(dict(scene=scene,shard=shard,p=p,log=log,start=time.time()))
 peak=max(peak,len(active));done=[]
 for v in active:
  rc=v['p'].poll()
  if rc is not None:
   v['log'].close();done.append(v);receipt=dict(scene=v['scene'],shard=v['shard'],pid=v['p'].pid,returncode=rc,start=v['start'],end=time.time(),seconds=time.time()-v['start']);complete.append(receipt)
   if rc:failed.append(receipt)
 for v in done:active.remove(v)
 if time.time()-last_storage>30:
  storage=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and 'archive_worktree' not in p.parts);last_storage=time.time()
 if storage>3.8*1024**3 and not failed:
  failed.append(dict(reason='Approaching fixed4GiB disk cap'))
  for v in active:v['p'].terminate()
 used=sum(v['seconds'] for v in complete)+sum(time.time()-v['start'] for v in active)
 status=dict(stage='EVALUATING' if active else 'TECHNICAL_STOP' if failed else 'COMPLETE',completed_tasks=len(complete),planned_tasks=96,active=len(active),pending=len(tasks),peak_running_processes_observed=peak,seconds=time.time()-t,CPU_worker_seconds_conservative=used,new_runtime_bytes=storage,failures=failed,active_assignments=[dict(scene=v['scene'],shard=v['shard'],pid=v['p'].pid) for v in active],completed_jobs=complete)
 write(R/'results/pipeline_status.json',status)
 if failed and not active:break
 time.sleep(5)
assert not failed,failed
assert len(complete)==96
write(R/'results/evaluation_complete.json',dict(status='PASS',tasks=complete,frames=3072,configurations=16,records=49152,seconds=time.time()-t,CPU_worker_seconds_conservative=used+profile['seconds'],peak_running_processes_observed=peak))
print('FULL_C3_COMPLETE',time.time()-t,flush=True)
