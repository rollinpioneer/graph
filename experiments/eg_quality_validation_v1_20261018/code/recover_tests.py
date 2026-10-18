import sys,json,time,subprocess,concurrent.futures,shutil
from pathlib import Path
n=Path('/home/xushijie3/work/eg_quality_validation_v1_20261018')
print('WAITING_FOR_INITIAL_DEV_AND_TEST_PILOT',flush=True)
while not (n/'QUEUE_COMPLETE.json').exists():time.sleep(5)
initial=json.loads((n/'QUEUE_COMPLETE.json').read_text())
assert all(x['exit']==0 for x in initial['jobs'] if x['partition']=='dev')
p=n/'results/subset_diagnostic_seen_0119'
for i in range(120):
 if (p/'COMPLETE.json').exists():break
 time.sleep(5)
assert (p/'COMPLETE.json').exists(),'test pilot did not complete'
pilot=json.loads((p/'COMPLETE.json').read_text());assert pilot['pilot'] and pilot['verified']>0
p.rename(n/'pilot_test_scene0119')
shutil.copy2(n/'QUEUE_COMPLETE.json',n/'QUEUE_INITIAL.json')
jobs=[(x['scene'],x['role'],x['partition']) for x in initial['jobs'] if x['exit']!=0]
assert len(jobs)==8 and all(p!='dev' for _,_,p in jobs)
def run(job):
 s,r,p=job;log=n/f'logs/{r}_{p}_{s}.log';log.rename(log.with_name(log.stem+'_initial_failure.log'));t=time.time()
 with log.open('w') as f:z=subprocess.run([sys.executable,'-u',str(n/'code/validate.py'),str(s),r,p],stdout=f,stderr=subprocess.STDOUT)
 return dict(scene=s,role=r,partition=p,exit=z.returncode,elapsed=time.time()-t)
t=time.time();out=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for fu in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
  x=fu.result();out.append(x);print(json.dumps(x),flush=True)
  (n/'recovery_progress.json').write_text(json.dumps(out,indent=2))
merged=[x for x in initial['jobs'] if x['partition']=='dev']+out
(n/'QUEUE_COMPLETE.json').write_text(json.dumps(dict(jobs=merged,initial_wall=initial['elapsed'],recovery_wall=time.time()-t,all_pass=all(x['exit']==0 for x in merged)),indent=2))
(n/'RECOVERY_COMPLETE.json').write_text(json.dumps(dict(jobs=out,elapsed=time.time()-t,all_pass=all(x['exit']==0 for x in out)),indent=2))
