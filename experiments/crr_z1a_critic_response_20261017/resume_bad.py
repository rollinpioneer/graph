from pathlib import Path
import os,json,subprocess,time,datetime
R=Path(__file__).resolve().parent;assert not (R/'receipts/bad_zero_update_resume_dispatch.json').exists()
assert json.loads((R/'receipts/aligned_native_validation.json').read_text())['status']=='PASS'
ps=[];rows=[];start=time.monotonic();done={};stats=[]
for seed,gpu in [(0,7),(1,6),(2,0),(3,2)]:
 assert not (R/f'runs/critic_s{seed}').exists()
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 log=open(R/f'receipts/critic_s{seed}.log','w');p=subprocess.Popen(['/home/xushijie3/work/flow_baselines_preflight_20261013/envs/gfp312/bin/python','-u',str(R/'critic.py'),str(seed)],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
 row=dict(name=f'critic_s{seed}',seed=seed,gpu=gpu,pid=p.pid,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='zero_update_admission_correction');rows.append(row);ps.append((p,log,row));(R/'receipts/bad_zero_update_resume_dispatch.json').write_text(json.dumps(rows,indent=2))
while len(done)<len(ps):
 for p,log,row in ps:
  if row['name'] in done or p.poll() is None:continue
  rc=p.wait();log.close();x=dict(**row,exit_code=rc,round_elapsed_seconds=time.monotonic()-start);done[row['name']]=x;(R/f'runs/critic_s{row["seed"]}/exit.json').write_text(json.dumps(x,indent=2))
 try:
  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  own=[]
  for ln in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines():
   pid,mem=ln.split(',');status=Path('/proc')/pid.strip()/'status'
   if status.exists() and int(next(x for x in status.read_text().splitlines() if x.startswith('Uid:')).split()[1])==os.getuid():own.append(dict(pid=int(pid),memory_mib=mem.strip()))
  stats.append(dict(seconds=time.monotonic()-start,gpu_aggregate=gpu,own_gpu_processes=own));(R/'receipts/critic_resources.json').write_text(json.dumps(stats,indent=2))
 except Exception as e:stats.append(dict(error=str(e)))
 time.sleep(10)
(R/'receipts/bad_zero_update_resume_complete.json').write_text(json.dumps(dict(completed=True,seconds=time.monotonic()-start,actual_exits=done),indent=2))
