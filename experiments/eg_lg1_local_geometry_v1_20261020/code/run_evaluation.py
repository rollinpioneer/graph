from common import *
import subprocess
profile=json.load(open(R/'results/eval_0100_profile.json'));assert profile['status']=='PASS' and len(profile['native_checks'])==4 and all(all(v.values()) for v in profile['native_checks'])
cfg=json.load(open(R/'frozen_config.json'));tasks=[(s,k) for s in sum(cfg['C3'].values(),[]) for k in range(8)];active=[];complete=[];failed=[];peak=0;t=time.time();initial=sum(json.load(open(R/f'results/geometry_{k}.json'))['worker_seconds'] for k in ['pilot','head','C3'])+profile['seconds'];limit=cfg['resources']['CPU_worker_hours_max']*3600
while tasks or active:
 while tasks and len(active)<88 and not failed:
  scene,shard=tasks.pop(0);log=open(R/f'logs/eval_{scene:04d}_{shard}.log','w');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';p=subprocess.Popen([sys.executable,str(R/'code/evaluate.py'),'--scene',str(scene),'--shard',str(shard),'--shards','8'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active.append(dict(scene=scene,shard=shard,p=p,log=log,start=time.time()))
 peak=max(peak,len(active));done=[]
 for v in active:
  rc=v['p'].poll()
  if rc is not None:
   v['log'].close();done.append(v);receipt=dict(scene=v['scene'],shard=v['shard'],pid=v['p'].pid,returncode=rc,seconds=time.time()-v['start']);complete.append(receipt)
   if rc:failed.append(receipt)
 for v in done:active.remove(v)
 used=initial+sum(v['seconds'] for v in complete)+sum(time.time()-v['start'] for v in active)
 if used>=limit:
  for v in active:v['p'].terminate()
  failed.append(dict(reason='CPU worker budget exhausted'));tasks=[]
 status=dict(stage='EVALUATING' if active else 'COMPLETE' if not failed and not tasks else 'TECHNICAL_STOP',completed_tasks=len(complete),planned_tasks=96,active=len(active),pending=len(tasks),peak_running_processes_observed=peak,seconds=time.time()-t,CPU_worker_seconds_conservative=used,failures=failed,active_assignments=[dict(scene=v['scene'],shard=v['shard'],pid=v['p'].pid) for v in active],completed_jobs=complete)
 write(R/'results/pipeline_status.json',status)
 if failed and not active:break
 time.sleep(5)
assert not failed,failed
assert len(complete)==96
write(R/'results/evaluation_complete.json',dict(status='PASS',tasks=complete,frames=3072,configurations=20,records=61440,seconds=time.time()-t,CPU_worker_seconds_conservative=used,peak_running_processes_observed=peak))
print('FULL_C3_EVALUATION_COMPLETE',time.time()-t,flush=True)
