"""Continue the same run at an epoch checkpoint on a less busy shared GPU."""
from repl_common import *
import signal,subprocess,shutil,datetime
plan=json.loads((R/'GPU_migration_plan.json').read_text());target=str(plan['target_GPU']);boundary=plan['checkpoint_epoch'];path=R/f'runs/B/train/economicgrasp_epoch{boundary:02d}.tar';affected=False
def state(status,**kw):write(R/'results/GPU_migration_status.json',dict(status=status,time=time.time(),target_GPU=target,boundary_epoch=boundary,**kw))
def identity(pid):
 try:
  folder=Path(f'/proc/{pid}');parts=(folder/'stat').read_text().rsplit(')',1)[1].split()
  if parts[0]=='Z':return None
  return dict(start=parts[19],cmd=(folder/'cmdline').read_bytes().split(b'\0')[:-1])
 except FileNotFoundError:return None
def stop_own(pid,expected):
 assert identity(pid)==expected,('PID identity changed',pid)
 os.kill(pid,signal.SIGTERM)
def wait_exit(pid,seconds=30):
 end=time.time()+seconds
 while identity(pid) is not None and time.time()<end:time.sleep(.1)
 assert identity(pid) is None,('Own process did not exit',pid)
def gpu_sample():
 line=subprocess.check_output(['nvidia-smi',f'--id={target}','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();return list(map(int,line.split(',')))
def launch_pipeline():
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',REPL_GPU=target)
 f=(R/'logs/pipeline_after_GPU_migration.log').open('w')
 p=subprocess.Popen([sys.executable,'-u',str(R/'code/pipeline.py')],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);f.close();return p.pid
try:
 state('WAITING_FOR_EPOCH_CHECKPOINT',original_training_PID=plan['training_PID'])
 end=time.time()+1800
 while not path.exists():
  assert identity(plan['training_PID']) is not None,'Original training stopped before boundary'
  assert time.time()<end,'Checkpoint wait timeout; original training untouched'
  time.sleep(.2)
 import torch
 ck=torch.load(path,map_location='cpu',weights_only=False)
 assert ck['epoch']==boundary and ck['optimizer_updates']==boundary*1600 and ck['seed']==20261015
 assert set(['python','numpy','torch','cuda'])<=set(ck['rng_state']) and len(ck['rng_state']['cuda'])==1
 assert all(torch.isfinite(v).all() for v in ck['model_state_dict'].values())
 checkpoint_sha=sha256(path);del ck
 samples=[]
 for i in range(3):
  samples.append(gpu_sample())
  if i<2:time.sleep(2)
 if min(v[0] for v in samples)<6144 or np.mean([v[1] for v in samples])>60:
  state('SKIPPED_TARGET_NO_LONGER_ELIGIBLE',samples=samples,original_training_untouched=True);sys.exit(0)
 if (R/f'runs/B/train/economicgrasp_epoch{boundary+1:02d}.tar').exists():
  state('SKIPPED_BOUNDARY_TOO_OLD',original_training_untouched=True);sys.exit(0)
 oldpid=plan['training_PID'];oldparent=plan['backbone_controller_PID'];pipe=plan['pipeline_PID']
 old=identity(oldpid);parent=identity(oldparent);pip=identity(pipe)
 assert old and str(CODE/'train.py').encode() in old['cmd'] and str(R/'B_frozen_split.json').encode() in old['cmd']
 assert parent and str(R/'code/run_backbone.py').encode() in parent['cmd']
 assert pip and str(R/'code/pipeline.py').encode() in pip['cmd']
 assert json.loads((R/'results/pipeline_status.json').read_text())['stage']=='WAITING_FOR_BACKBONE'
 assert not Path(f'/proc/{pipe}/task/{pipe}/children').read_text().strip()
 # Freeze the dependency first; only stop the precisely identified own training job.
 stop_own(pipe,pip);wait_exit(pipe);affected=True
 workers={int(pid):identity(int(pid)) for pid in Path(f'/proc/{oldpid}/task/{oldpid}/children').read_text().split()}
 stop_own(oldpid,old);wait_exit(oldpid);wait_exit(oldparent)
 for pid,ident in workers.items():
  if identity(pid)==ident and ident:
   time.sleep(2)
   if identity(pid)==ident:stop_own(pid,ident);wait_exit(pid)
 backup=R/'runs/B/migration_original_GPU6';backup.mkdir(exist_ok=False)
 shutil.copy2(R/'results/backbone_status.json',backup/'original_controller_exit.json')
 metrics=R/'runs/B/train/batch_metrics.jsonl';raw=metrics.read_bytes();(backup/'batch_metrics_unmodified.jsonl').write_bytes(raw)
 rows=[];partial=[]
 for index,line in enumerate(raw.splitlines()):
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:
   assert index==len(raw.splitlines())-1;partial.append(line.decode(errors='replace'))
 assert all(x['global_step']==i+1 for i,x in enumerate(rows))
 keep=[x for x in rows if x['global_step']<=boundary*1600];assert len(keep)==boundary*1600
 assert [json.loads(x)['epoch'] for x in (R/'runs/B/train/epoch_metrics.jsonl').read_text().splitlines()]==list(range(1,boundary+1))
 metrics.write_text(''.join(json.dumps(x)+'\n' for x in keep))
 receipt=dict(status='MIGRATION_RESUME_READY',source_GPU=6,target_GPU=int(target),checkpoint=str(path),checkpoint_sha256=checkpoint_sha,retained_updates=len(keep),confirmed_uncheckpointed_updates_replayed=len(rows)-len(keep),possible_inflight_update_upper_bound=1,partial_trailing_records=partial,original_metrics_sha256=hashlib.sha256(raw).hexdigest(),original_metrics=str(backup/'batch_metrics_unmodified.jsonl'),samples=samples,optimizer_and_all_RNG_states_restored_by_original_train_code=True,no_new_seed=True,no_hyperparameter_change=True)
 write(R/'results/GPU_migration_receipt.json',receipt)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=target,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',EG_SEED='20261015')
 args=[x.decode() for x in old['cmd']]+['--resume','--checkpoint_path',str(path)];start=time.time()
 used=sum(json.loads(x)['seconds'] for x in (R/'results/resource_jobs.jsonl').read_text().splitlines() if json.loads(x)['kind']=='GPU');assert used<28800
 with (R/'logs/train_B_resume_GPU1.log').open('w') as log:
  child=subprocess.Popen(args,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
  write(R/'results/backbone_status.json',dict(status='RUNNING',stage='train_B_resumed',pid=child.pid,controller_pid=os.getpid(),GPU=target,started=start,gpu_seconds_used=used,resume_checkpoint=str(path),completed_epochs=boundary,retained_optimizer_updates=len(keep)))
  newpipe=launch_pipeline();receipt.update(status='MIGRATED_TRAINING',training_PID=child.pid,pipeline_PID=newpipe,controller_PID=os.getpid());write(R/'results/GPU_migration_receipt.json',receipt);state('MIGRATED_TRAINING',training_PID=child.pid,pipeline_PID=newpipe,retained_updates=len(keep))
  try:code=child.wait(timeout=28800-used)
  except subprocess.TimeoutExpired:
   child.terminate()
   try:child.wait(timeout=60)
   except subprocess.TimeoutExpired:child.kill();child.wait()
   code=child.returncode
  dt=time.time()-start
  with (R/'results/resource_jobs.jsonl').open('a') as f:f.write(json.dumps(dict(name='train_B_resume_GPU1',kind='GPU',pid=child.pid,seconds=dt,returncode=code))+'\n')
  assert code==0,('Resumed training failed',code)
 final=R/'runs/B/train/economicgrasp_epoch10.tar';ck=torch.load(final,map_location='cpu',weights_only=False)
 assert ck['epoch']==10 and ck['optimizer_updates']==16000 and ck['seed']==20261015
 assert all(torch.isfinite(v).all() for v in ck['model_state_dict'].values())
 rows=[json.loads(x) for x in metrics.read_text().splitlines()];assert len(rows)==16000 and all(x['global_step']==i+1 for i,x in enumerate(rows))
 write(R/'results/backbone_status.json',dict(status='PASS',epochs=10,optimizer_updates=16000,checkpoint=str(final),sha256=sha256(final),gpu_seconds_used=used+dt,resumed_same_experiment=True,GPU=target))
 state('PASS_TRAINING_COMPLETED',optimizer_updates=16000)
except Exception as e:
 state('TECHNICAL_STOP',error=repr(e),original_training_affected=affected)
 if affected:write(R/'results/backbone_status.json',dict(status='TECHNICAL_STOP',error=repr(e),stage='GPU_migration'))
 raise
