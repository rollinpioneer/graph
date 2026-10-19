"""Pipeline C1 exports/evaluation, at most8 compute processes including controller."""
import os,sys,json,time,subprocess,concurrent.futures,resource
from pathlib import Path
from scoring import sha256

R=Path(os.environ['LQR_ROOT']);G=Path(os.environ['GRASP_ROOT']);cfg=json.loads((R/'frozen_config.json').read_text());lock=json.loads((R/'method_lock.json').read_text());audit=json.loads((R/'results/C1_exposure_audit.json').read_text());assert audit['status']=='PASS_NO_KNOWN_EXPOSURE' and audit['method_lock_sha256']==sha256(R/'method_lock.json') and lock['config_sha256']==sha256(R/'frozen_config.json')
scenes=sum([cfg['C1'][s] for s in ['seen','similar','novel']],[]);data=G/'eg_sub25_v1_20261015/data/graspnet';missing=[]
for scene in scenes:
 for frame in range(256):
  for sub,ext in [('rgb','png'),('depth','png'),('label','png'),('meta','mat')]:
   p=data/f'scenes/scene_{scene:04d}/kinect/{sub}/{frame:04d}.{ext}'
   if not p.is_file():missing.append(str(p))
(R/'results/C1_raw_inventory.json').write_text(json.dumps(dict(expected_frames=3072,required_observation_files=12288,missing=missing),indent=2)+'\n');assert not missing,'Incomplete C1 observations'
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode().splitlines();options=[]
for line in gpus:
 index,uuid,total,used,util=[x.strip() for x in line.split(',')]
 if float(total)-float(used)>8192:options.append((float(util),float(used),int(index),uuid))
assert options,'No GPU with8GiB free for bounded inference';light=[x for x in options if x[1]<=2048];util,used,gpu,uuid=min(light or options);start=time.time();done=[];pending=list(scenes);active={};active_started={};cpu_start=time.process_time()
def status(state,**extra):
 p=R/'results/C1_status.json';tmp=p.with_suffix('.partial.json');tmp.write_text(json.dumps(dict(status=state,family=lock['family'],wall_seconds=time.time()-start,done=done,active_scenes=sorted(active.values()),pending_scenes=pending,gpu=gpu,gpu_uuid=uuid,initial_shared_gpu_memory_mib=used,initial_shared_gpu_utilization_percent=util,export_active=exporter.poll() is None,limits='export+6 evaluators+controller=8; after export7 evaluators+controller=8',new_trainings=0,**extra),indent=2)+'\n');tmp.replace(p)
export_env=os.environ.copy();export_env['CUDA_VISIBLE_DEVICES']=str(gpu);export_log=open(R/'logs/C1_export.log','w');exporter=subprocess.Popen([sys.executable,str(R/'code/export_confirmation.py')],stdout=export_log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=export_env)
def ready(scene):
 p=R/f'exports/subset/scene_{scene:04d}/COMPLETE.json'
 if not p.exists():return False
 try:return json.loads(p.read_text())['status']=='PASS'
 except json.JSONDecodeError:return False
def evaluate(scene):
 t=time.time();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
 with open(R/f'logs/C1_{scene:04d}.log','w') as log:result=subprocess.run([sys.executable,str(R/'code/evaluate.py'),'--stage','C1','--scene',str(scene)],stdout=log,stderr=subprocess.STDOUT,env=env)
 assert result.returncode==0,(scene,result.returncode)
 return dict(scene=scene,worker_wall_seconds=time.time()-t)
status('RUNNING')
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
  while pending or active or exporter.poll() is None:
   if exporter.poll() not in [None,0]:raise RuntimeError('C1 export technical failure; see log')
   if exporter.poll()==0:assert all(ready(s) for s in pending),'Export exited without complete scene receipts'
   cap=6 if exporter.poll() is None else 7
   for scene in list(pending):
    if len(active)>=cap:break
    if ready(scene):
     future=pool.submit(evaluate,scene);active[future]=scene;active_started[future]=time.time();pending.remove(scene)
   completed,_=concurrent.futures.wait(list(active),timeout=1,return_when=concurrent.futures.FIRST_COMPLETED) if active else ([],[])
   for future in completed:done.append(future.result());del active[future];del active_started[future];status('RUNNING')
   completed_wall=sum(x['worker_wall_seconds'] for x in done);export_status=R/'results/C1_export_status.json';export_wall=json.loads(export_status.read_text())['wall_seconds'] if export_status.exists() else time.time()-start
   assert completed_wall+export_wall+sum(time.time()-t for t in active_started.values())<24*3600,'P2 worker-wall24h budget exceeded'
   assert time.time()-start<24*3600,'P2 elapsed limit exceeded'
   if not active:time.sleep(1)
   status('RUNNING')
 subprocess.run([sys.executable,str(R/'code/report.py'),'--stage','C1'],check=True);subprocess.run([sys.executable,str(R/'code/dedup_sensitivity.py')],check=True)
 status('COMPLETE',worker_wall_seconds=sum(x['worker_wall_seconds'] for x in done),child_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,controller_cpu_seconds=time.process_time()-cpu_start,export_wall_seconds=json.loads((R/'results/C1_export_status.json').read_text())['wall_seconds'])
except Exception as e:status('TECHNICAL_STOP',error=repr(e));raise
finally:export_log.close()
