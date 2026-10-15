from pathlib import Path
import os,json,time,subprocess
R=Path(__file__).resolve().parents[1]
with (R/'receipts/z2_manager.json').open('x') as f:json.dump(dict(pid=os.getpid(),time=time.time()),f)
while True:
 states=[json.loads((R/'runs'/f'GFP_seed{s}'/'status.json').read_text()) for s in [0,1]]
 if any(x.get('status')=='FAILED_STOPPED' for x in states):raise RuntimeError('Training failed; Z2 not dispatched')
 if all(x.get('status')=='COMPLETED' for x in states):break
 time.sleep(30)
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
available=[(int(v[2]),-int(v[1]),int(v[0])) for l in rows.splitlines() if (v:=l.split(',')) and int(v[1])>=8000]
if not available:raise RuntimeError('No safe resource for Z2; preserve assets and report')
gpu=min(available)[2];env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');start=time.time()
for script in ['probe.py','analyze.py','plots.py','spatial.py','topology.py']:
 with (R/'logs'/('z2_'+script+'.log')).open('x') as f:
  p=subprocess.run([str(R/'envs/gfp/bin/python'),str(R/'code'/script)],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
 (R/'receipts'/('z2_'+script+'.exit.json')).write_text(json.dumps(dict(exit_code=p.returncode,time=time.time(),gpu=gpu)))
 if p.returncode:raise RuntimeError('Z2 stopped: '+script)
(R/'receipts/z2_completed.json').write_text(json.dumps(dict(status='PASS',time=time.time(),gpu=gpu,wall_seconds=time.time()-start,optimizer_updates=0,environment_steps=0)))
