"""Conservative scene-level exclusion from existing fit and performance records."""
import os,re,json,csv,datetime
from common import *

def sid(v):
 m=re.fullmatch(r'(?:scene_)?(\d{1,4})(?:\.0)?',str(v).strip())
 return int(m[1]) if m and 100<=int(m[1])<=189 else None

hits={};checked=[]
def add(s,p,why):
 if s is not None:hits.setdefault(s,set()).add((str(p.relative_to(G)),why))

def walk(x,p):
 if isinstance(x,dict):
  for k in ['scene','scene_id','scenename']:
   if k in x:add(sid(x[k]),p,'explicit_scene_record')
  for k,v in x.items():
   # A metadata-only candidate availability pool is not an executed split.
   if k.lower() in ['available','raw_available','raw_inventory','candidate_pool','eligible_pool']:continue
   s=sid(k)
   if s is not None and isinstance(v,(dict,list)):add(s,p,'scene_keyed_record')
   key=k.lower()
   if isinstance(v,list) and ('scene' in key or key in ['seen','similar','novel','train','fit','calibration','development','confirmation','probe_fit','probe_calibration','probe_development']):
    for a in v:add(sid(a),p,'registered_scene_or_fit_split')
   walk(v,p)
 elif isinstance(x,list):
  for v in x:walk(v,p)

roots=[p for p in sorted(G.glob('eg_*')) if p.is_dir() and p!=R]
for root in roots:
 files=set(root.glob('results/*.json'))|set(root.glob('results/*.csv'))
 files|={p for p in root.glob('*.json') if any(k in p.name.lower() for k in ['split','manifest','config','lock'])}
 files|=set(root.glob('runs/*/evaluation/per_scene*.csv'))
 for p in sorted(files):
  checked.append({'path':str(p.relative_to(G)),'sha256':sha256(p)})
  try:
   if p.suffix=='.json' and p.stat().st_size<20*1024**2:walk(json.loads(p.read_text()),p)
   elif p.suffix=='.csv':
    with p.open(newline='') as f:
     rd=csv.DictReader(f);key=next((k for k in ['scene','scene_id','scenename'] if k in (rd.fieldnames or [])),None)
     if key:
      for s in {sid(a[key]) for a in rd}:add(s,p,'performance_csv_scene')
  except (ValueError,UnicodeError):continue
 for base in [root/'evaluation',root/'labels']:
  if not base.exists():continue
  for parent,dirs,names in os.walk(base):
   p=Path(parent);parts=p.relative_to(base).parts
   if len(parts)>6 or any(k in parts for k in ['predictions','inputs','exports','archive_worktree']):dirs[:]=[];continue
   s=sid(p.name)
   if s is not None and any(Path(n).suffix in ['.json','.npz','.npy'] for n in names):add(s,p,'computed_evaluation_or_label_cache')
   for n in names:
    m=re.match(r'^(\d{4})_\d{4}\.(?:npz|npy|json)$',n)
    if m:add(sid(m[1]),p/n,'computed_frame_cache')

available={};missing=[]
for split,lo in zip(SPLITS,[100,130,160]):
 pool=[s for s in range(lo,lo+30) if s not in hits]
 pool.sort(key=lambda s:hashlib.sha256(f'EG-LQR-CD-DIAGNOSIS-V4|{s}'.encode()).hexdigest())
 selected=[]
 for s in pool:
  folder=DATA/f'scenes/scene_{s:04d}/kinect'
  absent=[str(folder/f'{sub}/{f:04d}.{ext}') for sub,ext in [('rgb','png'),('depth','png'),('label','png'),('meta','mat')] for f in range(256) if not (folder/f'{sub}/{f:04d}.{ext}').is_file()]
  absent += [str(folder/n) for n in ['camera_poses.npy','cam0_wrt_table.npy'] if not (folder/n).is_file()]
  if absent:missing.append({'scene':s,'missing_count':len(absent),'examples':absent[:4]});continue
  selected.append(s)
  if len(selected)==4:break
 available[split]=selected
passed=all(len(v)==4 for v in available.values())
audit={'status':'PASS_NO_KNOWN_EXPOSURE_AND_RAW_FILES_PRESENT' if passed else 'BLOCKED_INSUFFICIENT_UNEXPOSED_COMPLETE_SCENES','excluded_scenes':{str(k):[{'path':p,'reason':w} for p,w in sorted(v)] for k,v in sorted(hits.items())},'checked_files':checked,'selected':available,'missing_raw_files':missing,'scope':'Known migrated project fit/calibration/development/confirmation records and computed quality/performance caches; unknown external history cannot be excluded. Raw feature exports alone are not performance exposure.','selection':'First four per split sorted by SHA256(EG-LQR-CD-DIAGNOSIS-V4|scene_id) among unexposed complete raw scenes','C2_performance_opened':False,'time':datetime.datetime.now().astimezone().isoformat()}
write('results/C2_exposure_audit.json',audit)
reservation={'status':audit['status'],'selected':available,'audit_sha256':sha256(R/'results/C2_exposure_audit.json'),'method':'fixed F05 only if exploration gate passes','C2_performance_opened':False}
assert not (R/'C2_reservation.json').exists();write('C2_reservation.json',reservation)
print(json.dumps({'status':audit['status'],'selected':available,'excluded_count':len(hits),'checked_files':len(checked)}),flush=True)
