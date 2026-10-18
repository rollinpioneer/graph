import sys,json,time,subprocess,concurrent.futures
from pathlib import Path
n=Path('/home/xushijie3/work/eg_quality_validation_v1_20261018')
c=json.loads((n/'config.json').read_text());jobs=[(s,'subset','dev') for s in c['dev_scenes']]
for role in ['subset','official']:
 for part,key in [('diagnostic_seen','seen'),('diagnostic_novel','novel')]:
  jobs += [(s,role,part) for s in c[key]]
def run(job):
 s,r,p=job;t=time.time()
 with (n/f'logs/{r}_{p}_{s}.log').open('w') as f:
  z=subprocess.run([sys.executable,'-u',str(n/'code/validate.py'),str(s),r,p],stdout=f,stderr=subprocess.STDOUT)
 return dict(scene=s,role=r,partition=p,exit=z.returncode,elapsed=time.time()-t)
t=time.time();out=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for future in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
  row=future.result();out.append(row);print(json.dumps(row),flush=True)
  (n/'queue_progress.json').write_text(json.dumps(out,indent=2))
(n/'QUEUE_COMPLETE.json').write_text(json.dumps(dict(jobs=out,elapsed=time.time()-t,all_pass=all(x['exit']==0 for x in out)),indent=2))
