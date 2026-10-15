from pathlib import Path
import os,json,time,subprocess,fcntl,shutil
R=Path(__file__).parent;LOCK=(R/'data_driver.lock').open('a');fcntl.flock(LOCK,fcntl.LOCK_EX|fcntl.LOCK_NB)
M=json.loads((R/'official_verified_metadata.json').read_text());names=['models.tar','grasp_label.tar','collision_label.tar','scenes.tar'];children={};started=time.time();status_file=R/'DATA_ACQUISITION_STATUS.json'
def read(p):return json.loads(p.read_text()) if p.exists() else None
def start(tag,cmd):
 log=(R.parent/'logs'/f'{tag}.log').open('a');p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children[tag]=p
 (R/(tag+'.process.json')).write_text(json.dumps({'pid':p.pid,'command':cmd}));return p
try:
 while True:
  progress={n:read(R/'archives'/(n+'.progress.json')) for n in names};extracted={n:read(R/(n+'.extraction.json')) for n in names}
  for tag,p in children.items():
   if p.poll() not in [None,0]:raise RuntimeError(tag+' exited '+str(p.returncode))
  for n,s in progress.items():
   if s and s['status']=='FAILED':raise RuntimeError(n+' download failed: '+str(s.get('error')))
   done=extracted[n] and extracted[n]['status']=='EXTRACTED'
   if s and s['status']=='VERIFIED' and not done and 'extract_'+n not in children:
    if n!='scenes.tar' or all(extracted[x] and extracted[x]['status']=='EXTRACTED' for x in names[:3]):start('extract_'+n,['python3',str(R/'extract_archive.py'),n,'--remove-verified-archive'])
  if progress['scenes.tar'] is None and 'fetch_scenes.tar' not in children:
   if extracted['models.tar'] and extracted['models.tar']['status']=='EXTRACTED' and shutil.disk_usage(R).free>M['files']['scenes.tar']['size']+24*1024**3:start('fetch_scenes.tar',['python3',str(R/'fetch_archive.py'),'scenes.tar','--workers','8'])
  summary={'status':'ACQUIRING','started_unix':started,'updated_unix':time.time(),'downloads':{n:{k:s.get(k) for k in ['status','downloaded_bytes','bytes','sha256_actual','last_error']} if s else None for n,s in progress.items()},'extractions':extracted,'child_pids':{n:p.pid for n,p in children.items()},'raw_data_root':str(R.parent/'data/graspnet')}
  if all(extracted[n] and extracted[n]['status']=='EXTRACTED' for n in names):summary['status']='RAW_ARCHIVES_EXTRACTED_PENDING_VALIDATION'
  tmp=status_file.with_suffix('.tmp');tmp.write_text(json.dumps(summary,indent=2));tmp.replace(status_file)
  if summary['status']!='ACQUIRING':break
  time.sleep(15)
except Exception as e:status_file.write_text(json.dumps({'status':'FAILED','error':repr(e),'updated_unix':time.time()},indent=2));raise
print(status_file.read_text(),flush=True)
