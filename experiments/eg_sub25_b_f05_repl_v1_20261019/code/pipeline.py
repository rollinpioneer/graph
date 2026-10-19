from repl_common import *
import subprocess,threading,concurrent.futures,traceback
start=time.time();mutex=threading.Lock();abort=threading.Event();reservations={'GPU':{},'CPU':{}}
def disk_bytes():
 total=0
 for parent,dirs,names in os.walk(R):
  dirs[:]=[d for d in dirs if d not in ['archive_worktree','__pycache__']]
  for n in names:
   p=Path(parent)/n
   if not p.is_symlink():total+=p.stat().st_size
 return total
def usage():
 rows=[json.loads(x) for x in (R/'results/resource_jobs.jsonl').read_text().splitlines()] if (R/'results/resource_jobs.jsonl').exists() else []
 return {kind:sum(x['seconds'] for x in rows if x['kind']==kind) for kind in ['GPU','CPU']}
def status(stage,**extra):write(R/'results/pipeline_status.json',dict(status='RUNNING',stage=stage,controller_pid=os.getpid(),wall_seconds=time.time()-start,usage=usage(),**extra))
def run(name,args,kind,extra_env=None):
 env=dict(os.environ);env.update(extra_env or {});env['CUDA_VISIBLE_DEVICES']=env.get('CUDA_VISIBLE_DEVICES',os.environ['REPL_GPU']) if kind=='GPU' else ''
 with mutex:
  assert not abort.is_set(),'Batch stopped after technical failure'
  u=usage();cap=8 if kind=='GPU' else 32;limit=P['budget']['gpu_seconds'] if kind=='GPU' else P['budget']['cpu_worker_seconds']
  available=limit-u[kind]-sum(reservations[kind].values())-60*cap
  remaining=available/max(1,cap-len(reservations[kind]))
  assert remaining>0,'Resource budget exhausted';assert disk_bytes()<P['budget']['new_bytes']-512*1024**2,'Storage budget reserve exhausted'
  reservations[kind][name]=remaining
 t=time.time();code=None
 with (R/f'logs/{name}.log').open('w') as f:
  p=subprocess.Popen(args,env=env,stdout=f,stderr=subprocess.STDOUT,cwd=R)
  try:code=p.wait(timeout=remaining)
  except subprocess.TimeoutExpired:
   p.terminate()
   try:p.wait(timeout=60)
   except subprocess.TimeoutExpired:p.kill();p.wait()
   code=p.returncode;abort.set();raise RuntimeError(f'{name}: fixed resource timeout; own subprocess stopped')
  finally:
   with mutex:
    with (R/'results/resource_jobs.jsonl').open('a') as ledger:ledger.write(json.dumps(dict(name=name,kind=kind,pid=p.pid,seconds=time.time()-t,returncode=code))+'\n')
    reservations[kind].pop(name)
 if code:abort.set();raise RuntimeError((name,code))
def py(name,script,args=None,kind='CPU',env=None):return run(name,[sys.executable,str(R/'code'/script)]+(args or []),kind,env)
def batch(tasks,n):
 with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
  pending=[pool.submit(py,*task) for task in tasks]
  try:
   for f in concurrent.futures.as_completed(pending):f.result()
  except Exception:
   abort.set()
   for f in pending:f.cancel()
   raise
def gpu_slots(stage,count=8):
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 rows=[tuple(map(int,line.split(','))) for line in raw.splitlines()]
 eligible=sorted([v for v in rows if v[1]>=3072 and v[2]<85],key=lambda v:(v[2],-v[1]))
 assert eligible,'No eligible shared GPU at stage launch'
 slots=([str(v[0]) for v in eligible]*2)[:count]
 write(R/f'results/GPU_assignment_{stage}.json',dict(snapshot=raw,slots=slots,rule='fresh free>=3072MiB and util<85%; max2 processes/GPU; max8 exporters',time=time.time()))
 return slots
def export_stage(mode,bone):
 slots=gpu_slots(f'{mode}_{bone}');tasks=[(f'export_{mode}_{bone}_shard{i}','export.py',['--mode',mode,'--bone',bone,'--shard-id',str(i),'--shards',str(len(slots))],'GPU',dict(CUDA_VISIBLE_DEVICES=gpu)) for i,gpu in enumerate(slots)]
 batch(tasks,len(slots));rows=[json.loads((R/f'results/export_{mode}_{bone}_shard{i}.json').read_text()) for i in range(len(slots))]
 write(R/f'results/export_{mode}_{bone}.json',dict(status='PASS',shards=len(slots),scenes=sum([v['scenes'] for v in rows],[]),GPU_process_seconds=sum(v['seconds'] for v in rows),peak_torch_gpu_bytes=max(v['peak_torch_gpu_bytes'] for v in rows)))
def cpu_workers(stage,requested):
 def sample():
  v=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]));return sum(v),v[3]+v[4]
 x=sample();time.sleep(1);y=sample();busy=1-(y[1]-x[1])/(y[0]-x[0]);free_cores=max(1,int((1-busy)*os.cpu_count()))
 mem=int(next(v for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')).split()[1])*1024
 n=max(1,min(requested,32,free_cores,int(max(0,mem-32*1024**3)/(6*1024**3))))
 write(R/f'results/CPU_assignment_{stage}.json',dict(workers=n,requested=requested,cpu_busy_percent=100*busy,free_cores_estimate=free_cores,available_bytes=mem,per_worker_memory_reserve=6*1024**3,time=time.time()))
 return n
try:
 status('WAITING_FOR_BACKBONE')
 # Dependent stage wait; no scheduled monitoring or C3 evaluation during training.
 while True:
  q=json.loads((R/'results/backbone_status.json').read_text())
  if q['status']=='PASS':break
  if q['status']=='TECHNICAL_STOP':raise RuntimeError('Backbone technical stop: '+q['error'])
  time.sleep(30)
 for p,h in json.loads((R/'results/execution_code_manifest.json').read_text()).items():assert sha256(Path(p))==h
 assert json.loads((R/'results/input_preparation_complete.json').read_text())['status']=='PASS'
 status('P2_B_HEAD_EXPORT',GPU_workers_max=8);export_stage('head','B')
 split=json.loads((R/'B_head_split.json').read_text());scenes=split['fit']+split['model_val']+split['gate_cal'];assert len(scenes)==32
 nl=cpu_workers('labels',32);status('P2_LABELS_PARALLEL',workers=nl);batch([(f'labels_{s:04d}','labels.py',['--scene',str(s)],'CPU',None) for s in scenes],nl)
 counts=[json.loads((R/f'results/labels_{s:04d}.json').read_text()) for s in scenes];assert sum(x['actions'] for x in counts)==196608
 write(R/'results/label_validation.json',dict(status='PASS',actions=196608,scenes=32,frames_per_scene=8,workers_initial=nl,workers_max=nl,workers_requested=32,original_quality_crosschecks=512,worker_seconds=sum(x['seconds'] for x in counts)))
 status('P2_FIT_THREE_HEADS',GPU_workers_max=3);slots=gpu_slots('fit_heads',3)
 batch([(f'fit_head_{seed}','fit_heads.py',['--seed',str(seed)],'GPU',dict(CUDA_VISIBLE_DEVICES=slots[i%len(slots)])) for i,seed in enumerate(SEEDS)],len(slots))
 hs=[json.loads((R/f'results/B_heads_{s}.json').read_text()) for s in SEEDS]
 write(R/'results/B_heads.json',dict(status='PASS',heads=sum([v['heads'] for v in hs],[]),actions=196608,GPU_process_seconds=sum(v['seconds'] for v in hs),peak_torch_gpu_bytes=max(v['peak_torch_gpu_bytes'] for v in hs),model_selection='fixed step1000'))
 status('P3_METHOD_LOCK');py('lock_C3','lock_C3.py')
 for bone in ['A','B']:
  status('P3_C3_EXPORT_'+bone,GPU_workers_max=8);export_stage('C3',bone)
 tasks=[(f'eval_{bone}_{s:04d}','evaluate.py',['--phase','confirmation','--scene',str(s)],'CPU',dict(REPL_EVAL_ROOT=str(R/bone))) for s in sum(P['C3'].values(),[]) for bone in ['A','B']]
 ne=cpu_workers('evaluation',24);status('P3_C3_EVALUATION',workers=ne);batch(tasks,ne)
 status('P4_FINALIZE');py('finalize','finalize.py')
 u=usage();size=disk_bytes();assert u['GPU']<=28800 and u['CPU']<=180000 and size<=25*1024**3
 write(R/'results/resource_summary.json',dict(status='PASS',usage=u,new_runtime_bytes=size,archive_worktree_excluded=True,cpu_workers_initial=nl,cpu_workers_max=max(nl,ne),CPU_workers_requested_max=32,evaluation_workers=ne,GPU_export_workers_max=8,gpu_training_replicates=1,head_replicates=3,wall_seconds=time.time()-start))
 write(R/'results/pipeline_status.json',dict(status='COMPLETED_PENDING_GIT_ARCHIVE',wall_seconds=time.time()-start,usage=u,final_report=str(R/'results/final_report.md'),all_experiment_subprocesses_exited=True))
except Exception as e:
 write(R/'results/pipeline_status.json',dict(status='TECHNICAL_STOP',error=repr(e),traceback=traceback.format_exc(),wall_seconds=time.time()-start,usage=usage()));raise
