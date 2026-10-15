import os,json,time,hashlib,urllib.request,concurrent.futures,threading,fcntl,argparse,shutil
from pathlib import Path
P=argparse.ArgumentParser();P.add_argument('name');P.add_argument('--workers',type=int,default=3);A=P.parse_args()
R=Path(__file__).parent;M=json.loads((R/'official_verified_metadata.json').read_text());I=M['files'][A.name];SIZE=I['size'];EXPECTED=I['sha256'];CHUNK=64*1024*1024
D=R/'archives';D.mkdir(exist_ok=True);PART=D/(A.name+'.part');FINAL=D/A.name;STATE=D/(A.name+'.progress.json');LOCK=(D/(A.name+'.lock')).open('a');fcntl.flock(LOCK,fcntl.LOCK_EX|fcntl.LOCK_NB)
lock=threading.Lock();started=time.time();s={'file':A.name,'revision':M['revision'],'bytes':SIZE,'sha256_expected':EXPECTED,'chunk_bytes':CHUNK,'done':{},'status':'DOWNLOADING','started_unix':started,'workers':A.workers}
def save():
 t=STATE.with_suffix('.tmp');t.write_text(json.dumps(s));t.replace(STATE)
def fullhash(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
if FINAL.exists():
 assert FINAL.stat().st_size==SIZE and fullhash(FINAL)==EXPECTED
 print(json.dumps({'file':A.name,'status':'ALREADY_VERIFIED'}));raise SystemExit(0)
if PART.exists():
 assert STATE.exists(),'unregistered partial file';s=json.loads(STATE.read_text());assert s['revision']==M['revision'] and s['sha256_expected']==EXPECTED
 fd=os.open(PART,os.O_RDWR)
 for key,h in list(s['done'].items()):
  n=int(key);b=os.pread(fd,min(CHUNK,SIZE-n*CHUNK),n*CHUNK)
  if hashlib.sha256(b).hexdigest()!=h:del s['done'][key]
else:
 assert shutil.disk_usage(D).free>SIZE+24*1024**3,'insufficient disk reserve'
 fd=os.open(PART,os.O_RDWR|os.O_CREAT|os.O_EXCL,0o600);os.posix_fallocate(fd,0,SIZE)
save()
def get(n):
 start=n*CHUNK;end=min(SIZE,start+CHUNK)-1;count=end-start+1
 for attempt in range(5):
  try:
   u=f"https://hf-mirror.com/datasets/{M['repository']}/resolve/{M['revision']}/{A.name}?download=true&egchunk={n}&attempt={attempt}&nonce={time.time_ns()}"
   req=urllib.request.Request(u,headers={'Range':f'bytes={start}-{end}','User-Agent':'Mozilla/5.0','Accept-Encoding':'identity'})
   h=hashlib.sha256();written=0
   with urllib.request.urlopen(req,timeout=60) as q:
    assert q.status==206 and q.headers.get('Content-Range')==f'bytes {start}-{end}/{SIZE}','bad HTTP range'
    while written<count:
     b=q.read(min(1024*1024,count-written))
     if not b:raise EOFError('incomplete range')
     pos=0
     while pos<len(b):pos+=os.pwrite(fd,b[pos:],start+written+pos)
     h.update(b);written+=len(b)
   os.fsync(fd)
   with lock:
    s['done'][str(n)]=h.hexdigest();s['downloaded_bytes']=sum(min(CHUNK,SIZE-int(k)*CHUNK) for k in s['done']);s['updated_unix']=time.time();save()
   return n
  except Exception as ex:
   with lock:s['last_error']=str(ex);s['last_error_chunk']=n;s['retry_attempt']=attempt;save()
   if attempt==4:raise
   time.sleep(min(20,2**attempt))
try:
 todo=[i for i in range((SIZE+CHUNK-1)//CHUNK) if str(i) not in s['done']]
 with concurrent.futures.ThreadPoolExecutor(max_workers=A.workers) as pool:
  for n in pool.map(get,todo):
   if n%8==0:print(json.dumps({'file':A.name,'downloaded':s.get('downloaded_bytes',0),'total':SIZE}),flush=True)
 os.close(fd);s['status']='HASHING';save();actual=fullhash(PART);s['sha256_actual']=actual;assert actual==EXPECTED,'full SHA256 mismatch'
 PART.rename(FINAL);s.update(status='VERIFIED',ended_unix=time.time(),wall_seconds=time.time()-started);save();print(json.dumps({'file':A.name,'status':'VERIFIED','sha256':actual,'seconds':s['wall_seconds']}),flush=True)
except Exception as ex:
 s.update(status='FAILED',error=repr(ex),ended_unix=time.time());save();raise
