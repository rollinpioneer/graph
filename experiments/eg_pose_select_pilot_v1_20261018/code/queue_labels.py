from core import *
import subprocess,concurrent.futures
assert json.loads((R/'results/S0.json').read_text())['status']=='PASS';start=time.time();done=[]
def run(s):
 t=time.time()
 with open(R/f'logs/labels_{s}.log','w') as f:code=subprocess.run([sys.executable,str(R/'code/labels.py'),'--scene',str(s)],stdout=f,stderr=subprocess.STDOUT).returncode
 return dict(scene=s,exit=code,seconds=time.time()-t)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 futures=[pool.submit(run,s) for s in CFG['fit']+CFG['model_val']+CFG['gate_cal']]
 for f in concurrent.futures.as_completed(futures):
  d=f.result();done.append(d);write(R/'results/label_queue.json',dict(status='RUNNING',done=done,wall_seconds=time.time()-start));assert d['exit']==0,d
  assert sum(x['seconds'] for x in done)/3600<CFG['explore_cpu_hours']
write(R/'results/label_queue.json',dict(status='PASS',done=done,wall_seconds=time.time()-start))
