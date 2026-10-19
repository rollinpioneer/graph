"""Conditional C1 only, original EG actions, one frozen sample shared by all heads."""
import os,json,time,hashlib,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import scipy.io as sio
from scoring import sha256
from infer import EconomicExtractor

G=Path(os.environ['GRASP_ROOT']);R=Path(os.environ['LQR_ROOT']);cfg=json.loads((R/'frozen_config.json').read_text());lock=R/'method_lock.json';audit=R/'results/C1_exposure_audit.json'
assert lock.is_file() and json.loads(audit.read_text())['status']=='PASS_NO_KNOWN_EXPOSURE'
assert json.loads(audit.read_text())['method_lock_sha256']==sha256(lock) and json.loads(lock.read_text())['config_sha256']==sha256(R/'frozen_config.json')
old=G/'eg_sub25_v1_20261015';DATA=old/'data/graspnet';N=G/'eg_nqs_pilot_v2_20261018';start=time.time();cpu=time.process_time();initial_bytes=int(subprocess.check_output(['du','-sb',str(R)]).split()[0]);added=0;scenes=sum([cfg['C1'][s] for s in ['seen','similar','novel']],[]);done=[]
def save(status,**extra):
 path=R/'results/C1_export_status.json';tmp=path.with_suffix('.partial.json');tmp.write_text(json.dumps(dict(status=status,wall_seconds=time.time()-start,cpu_seconds=time.process_time()-cpu,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),conservative_prior_gpu_seconds_reserve=300,initial_bytes=initial_bytes,added_export_input_bytes=added,done_scenes=done,**extra),indent=2)+'\n');tmp.replace(path)
def observation_hash(scene,frame):
 folder=DATA/f'scenes/scene_{scene:04d}/kinect';h=hashlib.sha256()
 for key,relative in [('rgb',f'rgb/{frame:04d}.png'),('depth',f'depth/{frame:04d}.png')]:
  with Image.open(folder/relative) as image:value=np.ascontiguousarray(np.asarray(image))
  h.update(key.encode());h.update(str((value.dtype.str,value.shape)).encode());h.update(value.tobytes())
 K=np.ascontiguousarray(sio.loadmat(folder/f'meta/{frame:04d}.mat')['intrinsic_matrix']);h.update(K.dtype.str.encode());h.update(K.tobytes());return h.hexdigest()
save('RUNNING')
try:
 ex=EconomicExtractor(old/'runtime/EconomicGrasp',DATA,old/'runs/EG_SUB25_V1/train/economicgrasp_epoch10.tar',cfg['backbone_sha256'],'cuda:0')
 for scene in scenes:
  out=R/f'exports/subset/scene_{scene:04d}';out.mkdir(parents=True,exist_ok=True);inputs=R/f'inputs/scene_{scene:04d}';inputs.mkdir(parents=True,exist_ok=True);identities=[]
  for frame in range(256):
   assert time.time()-start<6900,'GPU2h budget including prior300s reserve exceeded'
   assert initial_bytes+added<cfg['budget']['new_bytes']-512*1024**2,'10GiB artifact budget including evaluation reserve exceeded'
   source=inputs/f'{frame:04d}.npz';target=out/f'{frame:04d}.npz'
   if target.exists():
    with np.load(target) as previous:assert str(previous['input_sha256'])==sha256(source) and str(previous['backbone_sha256'])==cfg['backbone_sha256']
    identities.append(dict(frame=frame,export_sha256=sha256(target),input_sha256=sha256(source),resumed=True));continue
   frozen=N/f'inputs/scene_{scene:04d}/{frame:04d}.npz';sample=ex.sample(scene,frame,source if source.exists() else frozen if frozen.exists() else None)
   if not source.exists():np.savez_compressed(source,**sample);added+=source.stat().st_size
   pred,feature,action,seconds=ex.extract(sample,verify_hook=frame==0);ih=sha256(source);obs=observation_hash(scene,frame);geometry=hashlib.sha256(np.ascontiguousarray(pred[:,1:]).tobytes()).hexdigest()
   np.savez_compressed(target,pred=pred,feature320=feature,original_action=action,candidate_ids=np.arange(1024),input_sha256=np.array(ih),raw_observation_sha256=np.array(obs),geometry_sha256=np.array(geometry),backbone_sha256=np.array(cfg['backbone_sha256']));added+=target.stat().st_size;identities.append(dict(frame=frame,export_sha256=sha256(target),input_sha256=ih,observation_sha256=obs,EG_seconds=seconds))
   if frame%32==0:print('EXPORT',scene,frame,time.time()-start,flush=True)
  (out/'COMPLETE.json').write_text(json.dumps(dict(status='PASS',scene=scene,frames=256,identity=identities,method_lock_sha256=sha256(lock)),indent=2)+'\n');done.append(scene);save('RUNNING')
 save('PASS',gpu_peak_torch_bytes=ex.torch.cuda.max_memory_allocated());print('EXPORT_COMPLETE',time.time()-start,flush=True)
except Exception as e:save('TECHNICAL_STOP',error=repr(e));raise
