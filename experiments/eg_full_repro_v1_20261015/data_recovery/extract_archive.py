import json,tarfile,hashlib,os,shutil,argparse,time,fcntl,zlib
from pathlib import Path,PurePosixPath
P=argparse.ArgumentParser();P.add_argument('name');P.add_argument('--remove-verified-archive',action='store_true');A=P.parse_args();R=Path(__file__).parent;D=R/'archives';ROOT=R.parent/'data/graspnet';ROOT.mkdir(parents=True,exist_ok=True)
meta=json.loads((R/'official_verified_metadata.json').read_text());proof=json.loads((D/(A.name+'.progress.json')).read_text());assert proof['status']=='VERIFIED' and proof['sha256_actual']==meta['files'][A.name]['sha256'];archive=D/A.name
lock=(D/(A.name+'.extract.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.time();receipt=R/(A.name+'.extraction.json');ledger=R/(A.name+'.files.jsonl');manifest={}
if ledger.exists():
 for line in ledger.read_text().splitlines():
  a=json.loads(line);manifest[a['path']]=a
def selected(m):
 p=PurePosixPath(m.name);parts=p.parts
 if p.is_absolute() or '..' in parts:raise ValueError('unsafe archive path')
 if m.issym() or m.islnk() or not (m.isdir() or m.isfile()):raise ValueError('nonregular archive member')
 if not m.isfile():return False
 if A.name=='scenes.tar':
  if len(parts)<3 or parts[0]!='scenes' or not parts[1].startswith('scene_'):return False
  sid=int(parts[1][6:]);assert 0<=sid<190
  return len(parts)==3 or (parts[2]=='kinect' and (len(parts)<4 or parts[3]!='rect'))
 if A.name=='collision_label.tar':return len(parts)>=2 and parts[0]=='collision_label' and parts[1].startswith('scene_') and int(parts[1][6:])<100
 if A.name=='grasp_label.tar':return parts[0]=='grasp_label' and p.suffix=='.npz'
 if A.name=='models.tar':return parts[0]=='models' and len(parts)>1 and parts[1].isdigit() and 0<=int(parts[1])<88
 if A.name=='dex_models.tar':return parts[0]=='dex_models' and p.suffix=='.pkl'
 raise ValueError(A.name)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(4*1024*1024):h.update(b)
 return h.hexdigest()
def save(status,count,total,**extra):
 a={'archive':A.name,'archive_sha256':proof['sha256_actual'],'revision':meta['revision'],'status':status,'files_completed':count,'selected_bytes':total,'root':str(ROOT),'wall_seconds':time.time()-started,**extra};t=receipt.with_suffix('.tmp');t.write_text(json.dumps(a,indent=2));t.replace(receipt)
with tarfile.open(archive,'r:') as tar:
 members=[m for m in tar if selected(m)];total=sum(m.size for m in members);needed=sum(m.size for m in members if m.name not in manifest);assert shutil.disk_usage(ROOT).free>needed+24*1024**3,'insufficient extraction reserve';save('EXTRACTING',0,total,total_selected_files=len(members))
 with ledger.open('a') as log:
  for i,m in enumerate(members):
   dst=ROOT/m.name;dst.resolve().relative_to(ROOT.resolve())
   if m.name in manifest:
    a=manifest[m.name];assert dst.stat().st_size==a['bytes'] and digest(dst)==a['sha256'];continue
   assert not dst.exists(),'unregistered existing target '+str(dst)
   dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_name(dst.name+'.egpartial')
   if tmp.exists():tmp.resolve().relative_to(ROOT.resolve());tmp.unlink()
   h=hashlib.sha256();n=0;crc=0
   with tar.extractfile(m) as f,tmp.open('xb') as out:
    while b:=f.read(4*1024*1024):out.write(b);h.update(b);crc=zlib.crc32(b,crc);n+=len(b)
    out.flush();os.fsync(out.fileno())
   assert n==m.size;tmp.rename(dst);a={'path':m.name,'bytes':n,'sha256':h.hexdigest(),'crc32':f'{crc:08x}'};log.write(json.dumps(a)+'\n');log.flush()
   if i%256==0:save('EXTRACTING',i+1,total,total_selected_files=len(members))
save('EXTRACTED',len(members),total,total_selected_files=len(members),file_manifest_sha256=digest(ledger));print(receipt.read_text(),flush=True)
if A.remove_verified_archive:
 archive.resolve().relative_to(D.resolve());archive.unlink()
 proof['archive_removed_after_verified_extraction']=True;(D/(A.name+'.progress.json')).write_text(json.dumps(proof))
