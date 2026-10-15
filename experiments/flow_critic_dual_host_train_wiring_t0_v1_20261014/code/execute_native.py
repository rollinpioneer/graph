"""Retain native process exit status; no restart policy."""
import argparse,json,os,subprocess,time
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--launch',required=True);args=parser.parse_args()
p=Path(args.launch);job=json.loads(p.read_text())
env=os.environ.copy();env.update(job['environment'])
for key in ['LD_LIBRARY_PATH','JAX_PLATFORMS','JAX_DEFAULT_MATMUL_PRECISION','JAX_ENABLE_X64']:env.pop(key,None)
env.update(PYTHONUNBUFFERED='1',HYDRA_FULL_ERROR='1')
start=time.time()
with open(job['log'],'w') as log:
 proc=subprocess.Popen(job['command'],cwd=job['cwd'],env=env,stdout=log,stderr=log)
 job['native_pid']=proc.pid;p.write_text(json.dumps(job,indent=2))
 rc=proc.wait()
receipt=dict(host=job['host'],native_pid=proc.pid,returncode=rc,started_epoch=start,ended_epoch=time.time(),
             launch_path=str(p),automatic_restart=False)
p.with_name(p.stem+'_exit.json').write_text(json.dumps(receipt,indent=2))
raise SystemExit(rc)
