import os,sys,json,time,subprocess,argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
R=Path(os.environ['R']);cfg=json.loads((R/'frozen_config.json').read_text());p=argparse.ArgumentParser();p.add_argument('--node',choices=['161','245','248'],required=True);p.add_argument('--workers',type=int,required=True);a=p.parse_args()
tasks=[(s,k) for ss in cfg['C3'].values() for s in ss for k in range(8)];owners=['161','161','161','245','245','248'];tasks=[t for i,t in enumerate(tasks) if owners[i%6]==a.node and not (R/f'results/eval_{t[0]:04d}_{t[1]}.json').exists()]
def run(task):
 s,k=task;start=time.time();path=R/f'logs/eval_{s:04d}_{k}.log'
 with path.open('w') as out:
  proc=subprocess.Popen([sys.executable,str(R/'code/eval_p1.py'),'--scene',str(s),'--shard',str(k),'--shards','8'],stdout=out,stderr=subprocess.STDOUT);code=proc.wait()
 return dict(scene=s,shard=k,pid=proc.pid,start=start,end=time.time(),returncode=code)
results=[];start=time.time()
with ThreadPoolExecutor(a.workers) as pool:
 for f in as_completed([pool.submit(run,t) for t in tasks]):
  v=f.result();results.append(v);print(json.dumps(v),flush=True)
out=dict(node=a.node,max_workers=a.workers,start=start,end=time.time(),tasks=results,status='PASS' if all(v['returncode']==0 for v in results) else 'FAILED')
(R/f'results/controller_{a.node}.json').write_text(json.dumps(out,indent=2))
