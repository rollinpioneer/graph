from common import *
import subprocess,concurrent.futures,threading,traceback
mutex=threading.Lock();ledger=R/'results/resource_jobs.jsonl';(R/'logs').mkdir(exist_ok=True)

def costs():
 rows=[json.loads(v) for v in ledger.read_text().splitlines()]
 return {k:sum(v['seconds'] for v in rows if v['kind']==k) for k in ['CPU','GPU']}

def add(row):
 with mutex:
  with ledger.open('a') as f:f.write(json.dumps(row)+'\n')

def free_workers(maximum):
 def sample():
  v=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]));return sum(v[:8]),v[3]+v[4]
 a=sample();time.sleep(1);b=sample();busy=1-(b[1]-a[1])/(b[0]-a[0]);mem=int(next(v for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')).split()[1])*1024
 return max(1,min(maximum,max(1,int((1-busy)*os.cpu_count())-4),max(1,int((mem-32*1024**3)/(4*1024**3))))),dict(cpu_busy_percent=busy*100,available_bytes=mem)

def run(name,script,args,timeout):
 t=time.time();env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
 with (R/f'logs/{name}.log').open('w') as log:
  p=subprocess.Popen([sys.executable,str(R/'code'/script)]+args,env=env,stdout=log,stderr=subprocess.STDOUT)
  try:rc=p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:p.terminate();p.wait();raise RuntimeError(name+' exceeded own budget timeout')
  finally:add(dict(name=name,kind='CPU',pid=p.pid,seconds=time.time()-t,returncode=p.returncode))
 assert rc==0,(name,rc)

try:
 assert json.loads((R/'results/training_complete.json').read_text())['status']=='PASS'
 lock=json.loads((R/'P2_method_lock.json').read_text())
 for p,h in lock['code'].items():assert sha(p)==h
 names={json.loads(v)['name'] for v in ledger.read_text().splitlines()}
 if 'P0_conservative_slot_accounting' not in names:
  p0=json.loads((R/'results/pair_coverage.json').read_text());add(dict(name='P0_conservative_slot_accounting',kind='CPU',seconds=p0['seconds']*p0['workers'],returncode=0,basis='peak worker slots times entire preflight wall; conservative upper bound'))
 if 'P2_native_profile' not in names:
  prof=json.loads((R/'results/eval_0100_profile.json').read_text());assert len(prof['native_checks'])==4
  add(dict(name='P2_native_profile',kind='CPU',seconds=prof['seconds']+5,returncode=0,basis='profile script wall plus5s startup allowance'))
 n,snapshot=free_workers(9);write(R/'results/CPU_assignment_offline.json',dict(workers=n,**snapshot));write(R/'results/pipeline_status.json',dict(status='RUNNING',stage='P1_OFFLINE_DIAGNOSTICS',workers=n,controller_pid=os.getpid()))
 tasks=[(f,s) for f in ['O','M','R'] for s in SEEDS]
 with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:list(pool.map(lambda v:run(f'offline_{v[0]}_{v[1]}','offline.py',['--family',v[0],'--seed',str(v[1])],240),tasks))
 prof=json.loads((R/'results/eval_0100_profile.json').read_text());old=json.loads((P/'results/resource_summary.json').read_text());parent_jobs=[json.loads(v) for v in (P/'results/resource_jobs.jsonl').read_text().splitlines()];mean=sum(v['seconds'] for v in parent_jobs if v['name'].startswith('eval_B_'))/3072
 projection=384*max(mean,prof['seconds']/5)*3+48*10;remaining=36000-costs()['CPU'];assert projection<remaining,(projection,remaining)
 write(R/'results/application_cost_check.json',dict(profile_seconds=prof['seconds'],profile_checks=4,profile_new_native_calls=prof['new_quality_calls'],historical_B_mean_worker_seconds_per_frame=mean,projection_seconds=projection,formula='384*max(historical_mean,profile_wall/5)*3+48*10; profile includes four extra complete native paths',remaining_CPU_seconds=remaining,status='PASS'))
 scenes=sum(json.loads((R/'frozen_config.json').read_text())['C3'].values(),[]);tasks=[(s,k) for s in scenes for k in range(4)];n,snapshot=free_workers(48);deadline=.90*remaining/len(tasks)
 write(R/'results/CPU_assignment_development.json',dict(workers=n,tasks=len(tasks),shards_per_scene=4,frames_per_shard=8,per_task_timeout_seconds=deadline,**snapshot));write(R/'results/pipeline_status.json',dict(status='RUNNING',stage='P2_DEVELOPMENT',workers=n,tasks=len(tasks),controller_pid=os.getpid()))
 with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:list(pool.map(lambda v:run(f'eval_{v[0]:04d}_shard{v[1]}','evaluate.py',['--scene',str(v[0]),'--shard',str(v[1]),'--shards','4'],deadline),tasks))
 assert costs()['CPU']<=36000 and costs()['GPU']<=3600
 write(R/'results/pipeline_status.json',dict(status='P2_EVALUATION_COMPLETE_PENDING_REPORT',usage=costs(),P3_executed=False))
 print('DEVELOPMENT_EVALUATION_COMPLETE',costs(),flush=True)
except Exception as e:
 write(R/'results/pipeline_status.json',dict(status='TECHNICAL_STOP',error=repr(e),traceback=traceback.format_exc(),usage=costs()));raise
