"""Project-owned child jobs, fresh idle-GPU checks, process-group monitoring."""
import csv,fcntl,json,os,subprocess,time,threading
from pathlib import Path
R=Path(__file__).resolve().parents[1];guard=threading.Lock()
def snapshot():
 text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 g=[]
 for row in csv.reader(text.splitlines()):
  i,uuid,name,used,total,util=[v.strip() for v in row];g.append({'index':int(i),'uuid':uuid,'name':name,'memory_used_mib':int(used),'memory_total_mib':int(total),'utilization_percent':int(util)})
 proc=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
 ps=[]
 for row in csv.reader(proc.splitlines()):
  if row:
   uuid,pid,used=[v.strip() for v in row];ps.append({'uuid':uuid,'pid':int(pid),'memory_mib':int(used) if used.isdigit() else None})
 return {'unix':time.time(),'gpus':g,'compute_processes':ps}
def acquire_gpu(name,preferred=None):
 while True:
  with guard:
   s=snapshot()
   for gpu in s['gpus']:
    if preferred is not None and gpu['index']!=preferred:continue
    if gpu['memory_used_mib']>100 or gpu['utilization_percent']>1 or any(p['uuid']==gpu['uuid'] for p in s['compute_processes']):continue
    f=(R/'data_recovery'/f'gpu_{gpu["index"]}.lock').open('a')
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:f.close();continue
    return gpu,f,s
  (R/'receipts'/f'{name}.waiting.json').write_text(json.dumps({'status':'WAITING_FOR_CURRENTLY_IDLE_GPU','snapshot':s,'preferred':preferred},indent=2))
  time.sleep(20)
def own_processes(pgid):
 out=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   raw=(p/'stat').read_text();rest=raw[raw.rfind(')')+2:].split()
   if int(rest[2])!=pgid:continue
   out.append({'pid':int(p.name),'rss_bytes':int(rest[21])*os.sysconf('SC_PAGE_SIZE'),'cpu_seconds':(int(rest[11])+int(rest[12]))/os.sysconf('SC_CLK_TCK')})
  except (FileNotFoundError,PermissionError,ProcessLookupError,ValueError):pass
 return out
def run(name,command,cwd=None,gpu=False,preferred=None):
 token=None;before=None;env=os.environ.copy();env['PYTHONUNBUFFERED']='1'
 if gpu:
  chosen,token,before=acquire_gpu(name,preferred);env['CUDA_VISIBLE_DEVICES']=str(chosen['index'])
 else:env['CUDA_VISIBLE_DEVICES']=''
 started=time.time();receipt={'name':name,'command':list(map(str,command)),'cwd':str(cwd or R),'status':'RUNNING','started_unix':started,'gpu_before':before,'selected_gpu':chosen if gpu else None,'peak_own_rss_bytes':0,'peak_own_gpu_memory_mib':0,'resource_sample_interval_seconds':15}
 target=R/'receipts'/f'{name}.json';log=R/'logs'/f'{name}.log';samples=R/'logs'/f'{name}.resources.jsonl'
 def save():
  tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(receipt,indent=2));tmp.replace(target)
 try:
  if target.exists():raise FileExistsError(f'Preserve existing receipt; use a new attempt name: {target}')
  save()
  with log.open('a') as f,samples.open('a') as sf:
   child=subprocess.Popen(list(map(str,command)),cwd=cwd or R,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   receipt['child_pid']=child.pid;receipt['process_group']=child.pid;save()
   while True:
    ps=own_processes(child.pid);row={'unix':time.time(),'processes':ps,'rss_bytes':sum(p['rss_bytes'] for p in ps)}
    receipt['peak_own_rss_bytes']=max(receipt['peak_own_rss_bytes'],row['rss_bytes'])
    if gpu:
     s=snapshot();ids={p['pid'] for p in ps};own=[p for p in s['compute_processes'] if p['pid'] in ids]
     row['own_gpu_processes']=own;row['foreign_compute_on_selected_gpu']=[p for p in s['compute_processes'] if p['uuid']==chosen['uuid'] and p['pid'] not in ids]
     used=sum(p['memory_mib'] or 0 for p in own);receipt['peak_own_gpu_memory_mib']=max(receipt['peak_own_gpu_memory_mib'],used)
    sf.write(json.dumps(row)+'\n');sf.flush();save()
    rc=child.poll()
    if rc is not None:break
    time.sleep(15)
  receipt.update(status='PASS' if rc==0 else 'FAIL',returncode=rc,ended_unix=time.time(),wall_seconds=time.time()-started);save()
  if rc!=0:raise RuntimeError(f'{name} failed rc={rc}; {log}')
  return receipt
 except Exception as e:
  receipt.update(status='FAIL',error=repr(e),ended_unix=time.time(),wall_seconds=time.time()-started)
  if not target.exists() or receipt.get('child_pid'):save()
  raise
 finally:
  if token is not None:fcntl.flock(token,fcntl.LOCK_UN);token.close()
