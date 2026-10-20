from common import *
import subprocess,signal
cfg=json.load(open(R/'frozen_config.json'));oldpid=3314128;command=(Path(f'/proc/{oldpid}/cmdline').read_bytes().replace(b'\0',b' ').decode());assert str(R/'code/pipeline.py') in command;assert Path(f'/proc/{oldpid}').stat().st_uid==os.getuid()
os.kill(oldpid,signal.SIGSTOP)
oldstate=json.load(open(R/'results/pipeline_status.json'));assert oldstate['stage']=='C4_EVALUATION'
lines=subprocess.check_output(['ps','--ppid',str(oldpid),'-o','pid=,etimes=,args='],text=True).splitlines();active=[];now=time.time()
for line in lines:
 parts=line.split();pid=int(parts[0]);age=float(parts[1]);args=parts[2:];assert str(R/'code/evaluate.py') in args
 scene=int(args[args.index('--scene')+1]);shard=int(args[args.index('--shard')+1]);active.append(dict(pid=pid,scene=scene,shard=shard,start=now-age-2,p=None))
write(R/'results/controller_takeover.json',dict(time=now,old_controller=oldpid,new_controller=os.getpid(),old_status=oldstate,inherited_jobs=[{k:v for k,v in j.items() if k!='p'} for j in active],resource_amendment_sha256=sha(R/'RESOURCE_AMENDMENT.md'),no_workers_stopped=True,no_aggregate_metrics_opened=True))
os.kill(oldpid,signal.SIGTERM);os.kill(oldpid,signal.SIGCONT)
alltasks=[(s,k) for s in sum(cfg['C4'].values(),[]) for k in range(8)];base=json.load(open(R/'results/geometry_C4.json'))['worker_seconds']+json.load(open(R/'results/pilot_geometry.json'))['seconds']+json.load(open(R/'results/eval_0120_profile.json'))['seconds']+60;gpu=oldstate['GPU_process_seconds']+30;started=now;peak=72
def receipt(s,k):return R/f'results/eval_{s:04d}_shard{k}.json'
def alive(j):
 if j['p'] is not None:return j['p'].poll() is None
 p=Path(f"/proc/{j['pid']}/cmdline")
 return p.exists() and str(R/'code/evaluate.py').encode() in p.read_bytes()
tasks=[v for v in alltasks if not receipt(*v).exists() and v not in [(j['scene'],j['shard']) for j in active]]
while active or tasks:
 for j in active[:]:
  if not alive(j):
   p=receipt(j['scene'],j['shard']);assert p.exists(),j;v=json.load(open(p));assert v['status']=='PASS' and len(v['frames'])==32,(j,v)
   if j['p'] is not None:assert j['p'].returncode==0;j['log'].close()
   active.remove(j)
 while tasks and len(active)<72:
  s,k=tasks.pop(0);f=open(R/f'logs/resumed_eval_{s}_{k}.log','w');e=os.environ.copy();e['CUDA_VISIBLE_DEVICES']='';p=subprocess.Popen([sys.executable,str(R/'code/evaluate.py'),'--scene',str(s),'--shard',str(k),'--shards','8'],env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);active.append(dict(scene=s,shard=k,pid=p.pid,p=p,log=f,start=time.time()))
 done=[(s,k,json.load(open(receipt(s,k)))) for s,k in alltasks if receipt(s,k).exists() and (s,k) not in [(j['scene'],j['shard']) for j in active]]
 used=base+sum(v['seconds']+5 for s,k,v in done)+sum(time.time()-j['start'] for j in active)
 if used>=60*3600:
  for j in active:
   if alive(j):os.kill(j['pid'],signal.SIGTERM)
  write(R/'results/pipeline_status.json',dict(stage='TECHNICAL_STOP',reason='amended CPU budget reached',CPU_worker_seconds_conservative=used));raise RuntimeError('CPU budget reached')
 state=dict(stage='C4_EVALUATION',completed_tasks=len(done),planned_tasks=96,active=[{k:j[k] for k in ['pid','scene','shard']} for j in active],peak_CPU_processes=peak,GPU_process_seconds=gpu,CPU_worker_seconds_conservative=used,seconds=time.time()-started,resource_amendment='60 CPU worker hours; scientific lock unchanged')
 write(R/'results/pipeline_status.json',state);time.sleep(5)
assert len(done)==96
write(R/'results/evaluation_complete.json',dict(status='PASS',frames=3072,configurations=14,records=43008,CPU_worker_seconds_conservative=used,GPU_process_seconds=gpu,peak_GPU_processes=6,peak_CPU_processes=peak,jobs=[dict(scene=s,shard=k,seconds=v['seconds']) for s,k,v in done],resource_amendment_sha256=sha(R/'RESOURCE_AMENDMENT.md')))
write(R/'results/pipeline_status.json',dict(state,stage='C4_COMPUTATION_COMPLETE'))
print('C4_COMPUTATION_COMPLETE',flush=True)
