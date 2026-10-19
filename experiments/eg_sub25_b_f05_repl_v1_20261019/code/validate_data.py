import os,json,time,hashlib
from pathlib import Path
import numpy as np
R=Path(os.environ['REPL_ROOT']);G=R.parent;D=G/'eg_sub25_v1_20261015/data/graspnet';a=json.loads((R/'B_frozen_split.json').read_text());start=time.time();records=[]
full=G/'eg_full_repro_v1_20261015/data_recovery/SUPERVISION_VALIDATION.json'
print('FULL_VALIDATION',full.exists(),flush=True)
if full.exists():
 old=json.loads(full.read_text());print('FULL_STATUS',old.get('status'),flush=True);assert old['status']=='PASS'
for s in a['train_scene_ids']:
 for f in a['train_frame_ids']:
  p=D/f'graspness/scene_{s:04d}/kinect/{f:04d}.npy';raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();side=Path(str(p)+'.sha256.json');assert side.exists() and json.loads(side.read_text())['sha256']==h
  v=np.load(p);assert v.size and np.isfinite(v).all() and v.min()>=0 and v.max()<=1
  records.append((str(p.relative_to(D)),h))
 p=D/f'economic_grasp_label_300views/scene_{s:04d}_labels.npz';h=hashlib.sha256(p.read_bytes()).hexdigest();side=Path(str(p)+'.sha256.json');assert side.exists() and json.loads(side.read_text())['sha256']==h
 with np.load(p) as z:
  assert set(['points','rotations','depth','scores','widths','topview','vgraspness','pointid'])<=set(z.files)
  n=len(z['points']);assert n>0
  for k in z.files:
   v=z[k];assert np.isfinite(v).all(),(s,k);assert len(v)==n,(s,k,v.shape,n)
  assert z['rotations'].min()>=0 and z['rotations'].max()<12
  assert z['depth'].min()>=0 and z['depth'].max()<4
  assert z['topview'].min()>=0 and z['topview'].max()<300
 records.append((str(p.relative_to(D)),h));print('VALIDATED',s,flush=True)
text=''.join(f'{p}\t{h}\n' for p,h in records);(R/'results/selected_labels.sha256.tsv').write_text(text)
out=dict(status='PASS',training_frames=6400,economic_scenes=50,labels_checked=len(records),manifest_sha256=hashlib.sha256(text.encode()).hexdigest(),seconds=time.time()-start,full_semantic_validation=str(full),full_validation_sha256=hashlib.sha256(full.read_bytes()).hexdigest())
(R/'results/data_validation.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
