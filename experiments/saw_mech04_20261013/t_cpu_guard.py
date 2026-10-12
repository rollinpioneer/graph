"""Independent own-job guard, package-specific residency and aggregate VRAM caps."""
import argparse,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--group',required=True,choices=['T-EVAL']);p.add_argument('--seconds',type=float,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
root=Path(__file__).resolve().parent;r=root/'receipts';active=r/f'{a.label}_active.json';receipt=r/f'{a.label}_resource.json'
cmd=a.command[1:] if a.command[0]=='--' else a.command;start=time.time();peak=0;samples=[];reason=None
limits={'C-S':39600,'C-N':19800,'T':9000,'R':14400,'C-EVAL':14400,'T-EVAL':14400};isgpu=a.group in ('C-S','C-N','T')
def read(p):return json.loads(p.read_text())
def registry(pattern):
 ans=[]
 for f in r.glob(pattern):
  try:ans.append(read(f))
  except FileNotFoundError:pass
 return ans
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
assert not receipt.exists() and not active.exists()
with (root/'logs'/f'{a.label}.log').open('x') as log:
 child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 write(active,dict(pid=child.pid,start=start,label=a.label,group=a.group,GPU=isgpu))
 while child.poll() is None:
  now=time.time();elapsed=now-start
  try:
   live=registry('*_active.json');done=registry('*_resource.json')
   own=[v for v in live if v['group']==a.group];total=sum(v['residency_seconds'] for v in done if v['group']==a.group)+sum(now-v['start'] for v in own)
   if isgpu:
    pids={v['pid'] for v in live if v['GPU']};lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=5).splitlines();mem=sum(int(l.split(',')[1]) for l in lines if int(l.split(',')[0]) in pids)
   else:mem=0
   peak=max(peak,mem);disk=shutil.disk_usage(root).free;samples.append([round(elapsed,3),mem,round(total,3),disk])
   if elapsed>=a.seconds-90:reason='JOB_TIME_LIMIT'
   elif total>=limits[a.group]-90*len(own):reason='PACKAGE_TIME_LIMIT'
   elif isgpu and mem>=12288:reason='AGGREGATE_VRAM_LIMIT'
   elif disk<10*1024**3:reason='DISK_RESERVE'
  except Exception as e:reason='MONITOR_FAILURE:'+str(e)
  if reason:
   os.killpg(child.pid,signal.SIGUSR1 if isgpu else signal.SIGTERM)
   try:child.wait(timeout=75)
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   break
  time.sleep(2)
 child.wait()
elapsed=time.time()-start;write(receipt,dict(label=a.label,group=a.group,GPU=isgpu,pid=child.pid,command=cmd,exit_code=child.returncode,
 residency_seconds=elapsed,GPU_hours=elapsed/3600 if isgpu else 0,peak_aggregate_mib=peak,stop_reason=reason,samples=samples));active.unlink(missing_ok=True)
sys.exit(child.returncode if child.returncode>=0 else 1)
