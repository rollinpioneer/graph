"""Official GraspNetAPI parser and pickle cache; verified models only."""
import hashlib,json,os,pickle,time
from pathlib import Path
from graspnetAPI.utils.eval_utils import load_dexnet_model
R=Path(__file__).resolve().parents[1];DATA=R/'data/graspnet'; OUT=DATA/'dex_models'
assert json.loads((R/'data_recovery/OFFICIAL_CRC_VERIFICATION_models.json').read_text())['status']=='PASS'
OUT.mkdir(exist_ok=True); manifest=R/'data_recovery/dex_cache_manifest.json'
rows=json.loads(manifest.read_text()).get('files',{}) if manifest.exists() else {}
started=time.time()
for obj in range(88):
 name=f'{obj:03d}.pkl';p=OUT/name
 if p.exists():
  if name not in rows or hashlib.sha256(p.read_bytes()).hexdigest()!=rows[name]['sha256']:
   raise RuntimeError(f'Unregistered or changed existing cache: {p}')
 else:
  model=load_dexnet_model(str(DATA/'models'/f'{obj:03d}'/'textured'))
  tmp=OUT/(name+'.egpartial')
  with tmp.open('wb') as f: pickle.dump(model,f); f.flush(); os.fsync(f.fileno())
  del model
  tmp.replace(p)
  with p.open('rb') as f: digest=hashlib.sha256(f.read()).hexdigest()
  rows[name]={'bytes':p.stat().st_size,'sha256':digest}
  temp=manifest.with_suffix('.tmp');temp.write_text(json.dumps({'status':'BUILDING','files':rows},indent=2));temp.replace(manifest)
 print(f'cached {obj:03d} bytes={rows[name]["bytes"]}',flush=True)
manifest.write_text(json.dumps({'status':'PASS','count':len(rows),'wall_seconds':time.time()-started,'files':rows,'method':'Official load_dexnet_model, pickle default protocol, independent model bytes verified against official ZIP CRC'},indent=2))
print('COMPLETE',len(rows),flush=True)
