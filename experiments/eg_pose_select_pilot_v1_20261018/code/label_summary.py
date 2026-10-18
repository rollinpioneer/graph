from core import *
import csv,inspect,importlib.metadata
rows=[];manifest={}
for group in ['fit','model_val','gate_cal']:
 for s in CFG[group]:
  counts=dict(scene=s,group=group,candidates=0,actions=0,positive=0,collision=0,empty=0,all_failed_groups=0,different_utility_groups=0,technical_failures=0)
  for f in CFG['fit_frames']:
   p=R/f'labels/{s:04d}_{f:04d}.npz';z=np.load(p);assert len(z['ids'])==128 and len(np.unique(z['ids']))==128 and z['q'].shape==(128,6) and np.isfinite(z['q']).all();u=utility(z['q']);counts['candidates']+=128;counts['actions']+=768;counts['positive']+=int((u>0).sum());counts['collision']+=int(z['collision'].sum());counts['empty']+=int(z['empty'].sum());counts['all_failed_groups']+=int((u.max(1)==0).sum());counts['different_utility_groups']+=int((np.ptp(u,axis=1)>0).sum());manifest[str(p.relative_to(R))]=dict(sha256=sha(p),bytes=p.stat().st_size,ids=z['ids'].tolist(),source_export_sha256=str(z['export_sha256']))
  rows.append(counts)
assert sum(x['actions'] for x in rows)==196608
with open(R/'results/label_coverage.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write(R/'results/label_manifest.json',manifest);write(R/'results/label_validation.json',dict(status='PASS',candidate_groups=32768,actions=196608,source='native actual geometry, no nearest-neighbor mask',technical_failures=0,all_failed_included=True,total_bytes=sum(x['bytes'] for x in manifest.values())))
print('LABEL_SUMMARY_PASS')
