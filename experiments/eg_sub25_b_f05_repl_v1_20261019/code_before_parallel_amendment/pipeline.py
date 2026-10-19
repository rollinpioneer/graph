from repl_common import *
import subprocess,threading,concurrent.futures,traceback
start=time.time();mutex=threading.Lock();abort=threading.Event()
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
 env=dict(os.environ);env.update(extra_env or {});env['CUDA_VISIBLE_DEVICES']=os.environ['REPL_GPU'] if kind=='GPU' else ''
 with mutex:
  assert not abort.is_set(),'Batch stopped after technical failure'
  u=usage();remaining=P['budget']['gpu_seconds']-u['GPU'] if kind=='GPU' else (P['budget']['cpu_worker_seconds']-u['CPU'])/8
  assert remaining>0,'Resource budget exhausted';assert disk_bytes()<P['budget']['new_bytes']-512*1024**2,'Storage budget reserve exhausted'
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
try:
 status('WAITING_FOR_BACKBONE')
 # Dependent stage wait; no scheduled monitoring or C3 evaluation during training.
 while True:
  q=json.loads((R/'results/backbone_status.json').read_text())
  if q['status']=='PASS':break
  if q['status']=='TECHNICAL_STOP':raise RuntimeError('Backbone technical stop: '+q['error'])
  time.sleep(30)
 for p,h in json.loads((R/'results/execution_code_manifest.json').read_text()).items():assert sha256(Path(p))==h
 status('P2_B_HEAD_EXPORT');py('B_head_export','export.py',['--mode','head','--bone','B'],'GPU')
 split=json.loads((R/'B_head_split.json').read_text());scenes=split['fit']+split['model_val']+split['gate_cal'];assert len(scenes)==32
 status('P2_LABELS_INITIAL_4',workers=4);batch([(f'labels_{s:04d}','labels.py',['--scene',str(s)],'CPU',None) for s in scenes[:4]],4)
 status('P2_LABELS_8',workers=8);batch([(f'labels_{s:04d}','labels.py',['--scene',str(s)],'CPU',None) for s in scenes[4:]],8)
 counts=[json.loads((R/f'results/labels_{s:04d}.json').read_text()) for s in scenes];assert sum(x['actions'] for x in counts)==196608
 write(R/'results/label_validation.json',dict(status='PASS',actions=196608,scenes=32,frames_per_scene=8,workers_initial=4,workers_max=8,original_quality_crosschecks=512,worker_seconds=sum(x['seconds'] for x in counts)))
 status('P2_FIT_THREE_HEADS');py('fit_heads','fit_heads.py',[],'GPU')
 status('P3_METHOD_LOCK');py('lock_C3','lock_C3.py')
 for bone in ['A','B']:
  status('P3_C3_EXPORT_'+bone);py('export_C3_'+bone,'export.py',['--mode','C3','--bone',bone],'GPU')
 tasks=[(f'eval_{bone}_{s:04d}','evaluate.py',['--phase','confirmation','--scene',str(s)],'CPU',dict(REPL_EVAL_ROOT=str(R/bone))) for s in sum(P['C3'].values(),[]) for bone in ['A','B']]
 status('P3_C3_EVALUATION',workers=8);batch(tasks,8)
 status('P4_FINALIZE');py('finalize','finalize.py')
 u=usage();size=disk_bytes();assert u['GPU']<=28800 and u['CPU']<=180000 and size<=25*1024**3
 write(R/'results/resource_summary.json',dict(status='PASS',usage=u,new_runtime_bytes=size,archive_worktree_excluded=True,cpu_workers_initial=4,cpu_workers_max=8,gpu_training_replicates=1,head_replicates=3,wall_seconds=time.time()-start))
 write(R/'results/pipeline_status.json',dict(status='COMPLETED_PENDING_GIT_ARCHIVE',wall_seconds=time.time()-start,usage=u,final_report=str(R/'results/final_report.md'),all_experiment_subprocesses_exited=True))
except Exception as e:
 write(R/'results/pipeline_status.json',dict(status='TECHNICAL_STOP',error=repr(e),traceback=traceback.format_exc(),wall_seconds=time.time()-start,usage=usage()));raise
