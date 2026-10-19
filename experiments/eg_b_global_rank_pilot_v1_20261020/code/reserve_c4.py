from common import *
import re,math
audit_path=P/'results/C3_exposure_audit.json';audit=json.loads(audit_path.read_text());evidence={};sources=[]
metric_keys={'ap','ap_percent','ap04','ap08','top1_valid','top1_collision','recovered','recoverable','failures'}

def add(scene,path):
 try:s=int(str(scene).replace('scene_',''))
 except (TypeError,ValueError):return
 if 100<=s<190:evidence.setdefault(s,set()).add(str(path))

def visit(v,path):
 if isinstance(v,dict):
  if any(k.lower() in metric_keys and isinstance(x,(int,float)) for k,x in v.items()):
   add(v.get('scene',v.get('scene_id')),path)
  for k,x in v.items():
   if str(k).isdigit() and isinstance(x,dict) and any(t.lower() in metric_keys for t in x):add(k,path)
   visit(x,path)
 elif isinstance(v,list):
  for x in v:visit(x,path)

paths=[G/v['path'] for v in audit['checked_files'] if 'metadata_pool_misclassified' not in v['path']]
paths += [P/f'B/results/confirmation_per_scene.csv',P/f'A/results/confirmation_per_scene.csv']
paths += [G/'eg_c3_b_top1_rank_diag_v1_20261020/results/by_scene.csv']
for path in sorted(set(paths)):
 if not path.exists() or path.suffix not in ['.csv','.json']:continue
 if path.suffix=='.csv':
  with path.open() as f:
   reader=csv.DictReader(f)
   if not set(reader.fieldnames or []) & metric_keys:continue
   for v in reader:
    if any(k in v and v[k] not in ['',None,'null','None','nan'] for k in metric_keys):add(v.get('scene',v.get('scene_id')),path)
 else:visit(json.loads(path.read_text()),path)
 sources.append(dict(path=str(path),sha256=sha(path)))
selected={};available={};checks=[]
for split,lo,hi in [('seen',100,130),('similar',130,160),('novel',160,190)]:
 remaining=[s for s in range(lo,hi) if s not in evidence]
 ordered=sorted(remaining,key=lambda s:hashlib.sha256(f'EG-B-GLOBAL-RANK-PILOT-V1|C4|kinect|scene_{s:04d}'.encode()).hexdigest())
 available[split]=ordered;selected[split]=ordered[:4]
 for s in selected[split]:
  folder=DATA/f'scenes/scene_{s:04d}/kinect';missing=[]
  for sub,ext in [('rgb','png'),('depth','png'),('label','png'),('meta','mat')]:
   missing += [str(folder/sub/f'{f:04d}.{ext}') for f in range(256) if not (folder/sub/f'{f:04d}.{ext}').is_file()]
  checks.append(dict(scene=s,missing_count=len(missing),missing_sample=missing[:3]))
ok=all(len(v)==4 for v in selected.values()) and not any(v['missing_count'] for v in checks)
write(R/'C4_reservation.json',dict(status='RESERVED_KNOWN_HISTORY_ONLY' if ok else 'CONDITIONAL_P3_BLOCKED_DATA_OR_HISTORY',authorized_to_evaluate=False,selected=selected,remaining_hash_order=available,excluded_scenes=sorted(evidence),performance_evidence={str(k):sorted(v) for k,v in sorted(evidence.items())},source_audit=dict(path=str(audit_path),sha256=sha(audit_path)),source_files=sources,raw_checks=checks,no_C4_predictions_or_performance_opened=True,scope='Reuses prior checked-file inventory and adds completed C3/Top1 records; only actual numeric scene performance is exposure. Metadata-only available pools are not performance. Unknown external history cannot be excluded.',time=time.time()))
print('C4_RESERVED',ok,selected,flush=True)
