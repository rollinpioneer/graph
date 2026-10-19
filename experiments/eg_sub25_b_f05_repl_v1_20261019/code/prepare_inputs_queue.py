from repl_common import *
import subprocess,concurrent.futures,threading
start=time.time();mutex=threading.Lock()
def run(s):
 t=time.time();env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
 with (R/f'logs/prepare_input_{s:04d}.log').open('w') as f:p=subprocess.run([sys.executable,str(R/'code/prepare_inputs.py'),'--scene',str(s)],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
 with mutex:
  with (R/'results/resource_jobs.jsonl').open('a') as f:f.write(json.dumps(dict(name=f'prepare_input_{s:04d}',kind='CPU',seconds=time.time()-t,returncode=p.returncode))+'\n')
 assert p.returncode==0,s
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:list(pool.map(run,sum(P['C3'].values(),[])))
write(R/'results/input_preparation_complete.json',dict(status='PASS',CPU_workers=12,scenes=12,frames=3072,seconds=time.time()-start,performance_computed=False))
