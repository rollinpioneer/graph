"""Bounded concurrent jobs; record actual child exits. No retries."""
from pathlib import Path
import os,json,subprocess,time,datetime
R=Path(__file__).resolve().parent;PY='/home/xushijie3/work/flow_baselines_preflight_20261013/envs/gfp312/bin/python'
assert not (R/'receipts/dispatch.json').exists()
jobs={};dispatch=[];stats=[];start=time.monotonic()
def write(p,x):p.write_text(json.dumps(x,indent=2))
def launch(kind,seed,gpu):
 name=f'{kind}_s{seed}';assert name not in jobs
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
 log=open(R/f'receipts/{name}.log','w');p=subprocess.Popen([PY,'-u',str(R/f'{kind}.py'),str(seed)],env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT)
 row=dict(name=name,kind=kind,seed=seed,gpu=gpu,pid=p.pid,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());dispatch.append(row);jobs[name]=dict(p=p,log=log,row=row);write(R/'receipts/dispatch.json',dispatch)
for kind,seed,gpu in [('healthy',1,6),('healthy',2,0),('critic',0,7),('critic',3,2)]:launch(kind,seed,gpu)
done={}
while True:
 for name,j in list(jobs.items()):
  p=j['p']
  if name in done or p.poll() is None:continue
  rc=p.wait();j['log'].close();done[name]=dict(**j['row'],exit_code=rc,elapsed_round_seconds=time.monotonic()-start)
  out=R/f'runs/{name}';out.mkdir(exist_ok=True);write(out/'exit.json',done[name]);print('EXIT',name,rc,flush=True)
  if j['row']['kind']=='healthy' and rc==0:
   # Reuse the freed safe placement after the sole authorized healthy generator exits.
   launch('critic',j['row']['seed'],j['row']['gpu'])
 try:
  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  own=[]
  for ln in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines():
   pid,mem=ln.split(',');p=Path('/proc')/pid.strip()/'status'
   if p.exists() and int(next(x for x in p.read_text().splitlines() if x.startswith('Uid:')).split()[1])==os.getuid():own.append(dict(pid=int(pid),memory_mib=mem.strip()))
  stats.append(dict(seconds=time.monotonic()-start,gpu_aggregate=gpu,own_gpu_processes=own));write(R/'receipts/resources.json',stats)
 except Exception as e:stats.append(dict(error=str(e)))
 write(R/'receipts/controller_progress.json',dict(completed=done,running=[n for n in jobs if n not in done],seconds=time.monotonic()-start))
 if len(done)==len(jobs):break
 time.sleep(15)
write(R/'receipts/controller_complete.json',dict(status='COMPLETE_OR_PARTIAL_FAILURE',jobs=done,seconds=time.monotonic()-start,no_retries=True,ordinary_GFP_limit=80000,critic_only_limit=60000))
