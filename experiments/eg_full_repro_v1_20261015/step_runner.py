import argparse,json,os,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--cwd');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
r=Path(__file__).parent; cmd=a.command
if cmd and cmd[0]=='--':cmd=cmd[1:]
start=time.time();receipt={'name':a.name,'command':cmd,'cwd':a.cwd,'started_utc':datetime.now(timezone.utc).isoformat(),'runner_pid':os.getpid(),'status':'RUNNING'}
target=r/'receipts'/f'{a.name}.json'
def save():
 tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(receipt,indent=2));tmp.replace(target)
save()
try:
 with (r/'logs'/f'{a.name}.log').open('a') as f:
  q=subprocess.Popen(cmd,cwd=a.cwd,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy());receipt['child_pid']=q.pid;save();rc=q.wait()
 receipt.update(status='PASSED' if rc==0 else 'FAILED',returncode=rc)
except Exception as ex:receipt.update(status='FAILED',error=repr(ex),returncode=-1)
receipt.update(ended_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.time()-start);save()
