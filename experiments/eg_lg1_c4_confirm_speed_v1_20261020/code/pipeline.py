from common import *
import subprocess
cfg=json.load(open(R/'frozen_config.json'));t=time.time();jobs=[];gpu_seconds=0.;cpu_seconds=0.
def logjob(kind,args,env=None,meta=None):
 key=kind+'_'+str(len(jobs));f=open(R/f'logs/{key}.log','w');pp=subprocess.Popen([sys.executable,*map(str,args)],env=env or os.environ.copy(),stdout=f,stderr=subprocess.STDOUT,start_new_session=True);j=dict(kind=kind,p=pp,log=f,start=time.time(),meta=meta or {},pid=pp.pid);jobs.append(j);return j
def finish(j):
 global gpu_seconds,cpu_seconds
 rc=j['p'].wait();j['log'].close();elapsed=time.time()-j['start'];j.update(rc=rc,seconds=elapsed)
 if j['kind'].startswith('export'):gpu_seconds+=elapsed
 else:cpu_seconds+=elapsed
 assert rc==0,(j['kind'],j['meta'],rc)
def state(stage,active,**kw):
 write(R/'results/pipeline_status.json',dict(stage=stage,seconds=time.time()-t,GPU_process_seconds=gpu_seconds,CPU_worker_seconds=cpu_seconds,active=[dict(kind=j['kind'],pid=j['pid'],**j['meta']) for j in active],**kw))
try:
 # Pilot export was produced separately with the same frozen extractor before full deployment.
 assert (R/'results/export_0_pilot.json').exists()
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
 # Generate pilot geometry before the unmodified-official comparison.
 from geometry import job
 receipt=job(('C4',120,0));cpu_seconds+=receipt['seconds'];write(R/'results/pilot_geometry.json',receipt)
 j=logjob('profile',[R/'code/evaluate.py','--scene',120,'--profile'],env);finish(j)
 prof=json.load(open(R/'results/eval_0120_profile.json'));assert prof['status']=='PASS' and len(prof['native_checks'])==3
 # One small inference process per shared GPU; sixth uses GPU0 with ample memory.
 assignment=[0,2,3,5,6,0];resource=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader'],text=True)
 write(R/'results/GPU_dispatch.json',dict(time=time.time(),snapshot=resource,assignment=assignment,shared=True))
 active=[]
 for shard,gpu in enumerate(assignment):
  e=os.environ.copy();e['CUDA_VISIBLE_DEVICES']=str(gpu);active.append(logjob('export',[R/'code/export.py','--shard',shard,'--shards',6],e,dict(shard=shard,GPU=gpu)))
 while active:
  done=[j for j in active if j['p'].poll() is not None]
  for j in done:finish(j);active.remove(j)
  estimate=gpu_seconds+sum(time.time()-j['start'] for j in active)
  assert estimate<2*3600,'GPU budget reached'
  state('C4_EXPORT',active,peak_GPU_processes=6);time.sleep(3)
 assert len(list((R/'exports').glob('scene_*/*.npz')))==3072
 j=logjob('geometry',[R/'code/geometry.py','--mode','C4','--workers',72],env);finish(j)
 gr=json.load(open(R/'results/geometry_C4.json'));cpu_seconds+=gr['worker_seconds']-j['seconds']
 tasks=[(s,k) for s in sum(cfg['C4'].values(),[]) for k in range(8)];active=[];completed=[];peak=0
 while tasks or active:
  while tasks and len(active)<72:
   s,k=tasks.pop(0);active.append(logjob('eval',[R/'code/evaluate.py','--scene',s,'--shard',k,'--shards',8],env,dict(scene=s,shard=k)))
  peak=max(peak,len(active));done=[j for j in active if j['p'].poll() is not None]
  for j in done:finish(j);active.remove(j);completed.append(dict(**j['meta'],pid=j['pid'],seconds=j['seconds']))
  used=cpu_seconds+sum(time.time()-j['start'] for j in active);assert used<40*3600,'CPU budget reached'
  state('C4_EVALUATION',active,completed_tasks=len(completed),planned_tasks=96,peak_CPU_processes=peak,CPU_worker_seconds_conservative=used);time.sleep(5)
 write(R/'results/evaluation_complete.json',dict(status='PASS',frames=3072,configurations=14,records=43008,CPU_worker_seconds_conservative=cpu_seconds,GPU_process_seconds=gpu_seconds,peak_GPU_processes=6,peak_CPU_processes=peak,jobs=completed))
 state('C4_COMPUTATION_COMPLETE',[])
 print('C4_COMPUTATION_COMPLETE',time.time()-t,flush=True)
except Exception as e:
 state('TECHNICAL_STOP',[j for j in jobs if j['p'].poll() is None],reason=repr(e));raise
