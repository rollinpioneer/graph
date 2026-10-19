import os,sys,json,time,subprocess,concurrent.futures,resource,argparse
from pathlib import Path
R=Path(os.environ['LQR_ROOT']);cfg=json.loads((R/'frozen_config.json').read_text());p=argparse.ArgumentParser();p.add_argument('--stage',choices=['P1','C1'],required=True);p.add_argument('--workers',type=int,default=4);args=p.parse_args();assert 1<=args.workers<=7;start=time.time();scenes=sum([cfg[args.stage][s] for s in ['seen','similar','novel']],[]);budget=8 if args.stage=='P1' else 24
micro=json.loads((R/'results/preflight_complete.json').read_text());assert micro['status']=='PASS';projected=micro['native_seconds_per_frame']*(384 if args.stage=='P1' else 3072)/3600
assert projected<budget,(projected,budget)
def save(status,**extra):(R/f'results/{args.stage}_status.json').write_text(json.dumps(dict(status=status,workers=args.workers,controller=1,wall_seconds=time.time()-start,projected_worker_hours=projected,new_trainings=0,**extra),indent=2)+'\n')
def one(scene):
 t=time.time()
 with open(R/f'logs/{args.stage}_{scene:04d}.log','w') as f:result=subprocess.run([sys.executable,str(R/'code/evaluate.py'),'--stage',args.stage,'--scene',str(scene)],stdout=f,stderr=subprocess.STDOUT)
 assert result.returncode==0,(scene,result.returncode)
 return dict(scene=scene,worker_wall_seconds=time.time()-t)
done=[];save('RUNNING')
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
  futures=[pool.submit(one,s) for s in scenes]
  for f in concurrent.futures.as_completed(futures):done.append(f.result());assert sum(x['worker_wall_seconds'] for x in done)<budget*3600;save('RUNNING',done=done)
 subprocess.run([sys.executable,str(R/'code/report.py'),'--stage',args.stage],check=True)
 save('COMPLETE',done=done,worker_wall_seconds=sum(x['worker_wall_seconds'] for x in done),child_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime)
except Exception as e:save('TECHNICAL_STOP',done=done,error=repr(e));raise
