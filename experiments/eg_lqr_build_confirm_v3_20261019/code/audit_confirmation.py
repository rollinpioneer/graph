"""After locking P1 only: audit known project performance exposure by identity."""
import os,json,csv,re,datetime
from pathlib import Path
from scoring import sha256

G=Path(os.environ['GRASP_ROOT']);R=Path(os.environ['LQR_ROOT']);cfg=json.loads((R/'frozen_config.json').read_text());lock=R/'method_lock.json';assert lock.is_file(),'Do not audit/open confirmation performance before method lock'
targets=set(sum([cfg['C1'][s] for s in ['seen','similar','novel']],[]));roots=sorted(p for p in G.glob('eg_*') if p.is_dir() and p!=R);hits=[];checked=[]
def scene_id(value):
 match=re.fullmatch(r'(?:scene_)?(\d{1,4})(?:\.0)?',str(value).strip());return int(match.group(1)) if match else None
def metric_key(key):
 k=str(key).lower();return k in ['ap','ap04','ap08','mean_ap','top1_valid','top1_collision','quality','mu','net','damage'] or k.startswith('ap_') or k.endswith('_ap')
def inspect_json(value,path):
 if isinstance(value,dict):
  sid=next((scene_id(value[k]) for k in ['scene','scene_id','scenename'] if k in value),None)
  if sid in targets and any(metric_key(k) for k in value):hits.append(dict(path=path,scene=sid,kind='performance_json_record'))
  for key,child in value.items():
   sid=scene_id(key)
   if sid in targets and isinstance(child,dict) and any(metric_key(k) for k in child):hits.append(dict(path=path,scene=sid,kind='performance_json_scene_mapping'))
   inspect_json(child,path)
 elif isinstance(value,list):
  for child in value:inspect_json(child,path)
for root in roots:
 files=set(root.glob('results/*.csv'))|set(root.glob('results/*.json'))|set(root.glob('runs/*/evaluation/per_scene*.csv'))|set(root.glob('evaluation/per_scene*.csv'))
 for path in sorted(files):
  relative=str(path.relative_to(G));checked.append(dict(path=relative,sha256=sha256(path)))
  if path.suffix=='.csv':
   with path.open(newline='') as file:
    reader=csv.DictReader(file);fields=reader.fieldnames or [];key=next((k for k in ['scene','scene_id','scenename'] if k in fields),None)
    if key and any(metric_key(k) for k in fields):
     observed={sid for row in reader if (sid:=scene_id(row[key])) in targets}
     hits.extend(dict(path=relative,scene=sid,kind='performance_csv_record') for sid in sorted(observed))
  elif path.stat().st_size<16*1024**2:
   try:inspect_json(json.loads(path.read_text()),relative)
   except (ValueError,UnicodeError):pass
 for folder in [root/'evaluation',root/'labels']:
  if not folder.exists():continue
  for parent,dirs,files in os.walk(folder):
   p=Path(parent);parts=p.relative_to(folder).parts
   if len(parts)>5:dirs[:]=[];continue
   if any(x in ['predictions','inputs','exports','archive_worktree'] for x in parts):dirs[:]=[];continue
   sid=scene_id(p.name)
   if sid in targets and any(Path(name).suffix in ['.json','.npz','.npy'] for name in files):hits.append(dict(path=str(p.relative_to(G)),scene=sid,kind='computed_evaluation_or_label_cache'))
   for name in files:
    match=re.match(r'^(\d{4})_\d{4}\.npz$',name)
    if match and int(match.group(1)) in targets:hits.append(dict(path=str((p/name).relative_to(G)),scene=int(match.group(1)),kind='computed_pose_label_cache'))
manifest=G/'eg_pose_select_pilot_v1_20261018/confirmation_manifest.json';previous=json.loads(manifest.read_text());assert set(sum(previous['selected'].values(),[]))==targets
unique={json.dumps(x,sort_keys=True):x for x in hits};result=dict(status='PAUSE_KNOWN_PERFORMANCE_EXPOSURE' if unique else 'PASS_NO_KNOWN_EXPOSURE',method_lock_sha256=sha256(lock),frozen_manifest_sha256=sha256(manifest),target_scenes=sorted(targets),roots=[str(p.relative_to(G)) for p in roots],checked_performance_files=checked,hits=list(unique.values()),raw_exports_are_not_performance_evidence=True,scope='Known current migrated project records only; no claim about unknown external history',time=datetime.datetime.now().astimezone().isoformat(),C1_opened=False)
(R/'results/C1_exposure_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],hits=result['hits'],checked_files=len(checked))),flush=True)
