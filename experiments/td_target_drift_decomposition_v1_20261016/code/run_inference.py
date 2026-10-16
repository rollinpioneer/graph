from pathlib import Path
import os,subprocess,json,time
r=Path(__file__).resolve().parents[1]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
cmd=['taskset','-c','104-111','/home/xushijie3/work/flow_baselines_preflight_20261013/envs/gfp312/bin/python',str(r/'code/decompose.py')]
t=time.time()
with open(r/'receipts/inference.log','x') as f:
 p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 (r/'receipts/launch.json').write_text(json.dumps(dict(pid=p.pid,started=t,command=cmd,gpu=4,optimizer_updates_allowed=0,environment_steps_allowed=0),indent=2))
 rc=p.wait()
(r/'receipts/exit.json').write_text(json.dumps(dict(exit_code=rc,started=t,ended=time.time()),indent=2))
