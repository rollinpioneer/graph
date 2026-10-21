from p2_residual import *
import subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
gate=json.loads((R/'results/invariance_checks.json').read_text());assert gate['status']=='PASS'
def run(task):
 fam,seed=task
 with (R/f'logs/train_{fam}_{seed}.log').open('w') as out:
  result=subprocess.run([sys.executable,str(R/'code/train_p2r.py'),'--family',fam,'--seed',str(seed)],stdout=out,stderr=subprocess.STDOUT)
 return dict(family=fam,seed=seed,returncode=result.returncode)
results=[]
with ThreadPoolExecutor(9) as pool:
 for v in as_completed([pool.submit(run,(f,s)) for f in FAMILIES for s in SEEDS]):results.append(v.result());print(json.dumps(results[-1]),flush=True)
write(R/'results/training_controller.json',dict(status='PASS' if all(v['returncode']==0 for v in results) else 'FAIL',jobs=results))
