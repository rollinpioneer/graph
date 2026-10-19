import time,subprocess
from common import *
from PIL import Image
from scipy.io import loadmat
from infer import EconomicExtractor
lock=json.loads((R/'C2_method_lock.json').read_text());assert lock['config_sha256']==sha256(R/'frozen_config.json') and lock['reservation_sha256']==sha256(R/'C2_reservation.json');start=time.time();done=[];added=0
initial=int(subprocess.check_output(['du','-sb',str(R)]).split()[0])
def save(status,**extra):write('results/C2_export_status.json',dict(status=status,wall_seconds=time.time()-start,cpu_seconds=time.process_time(),GPU=os.environ['CUDA_VISIBLE_DEVICES'],done_scenes=done,added_bytes=added,**extra))
def observation_hash(scene,frame):
 folder=DATA/f'scenes/scene_{scene:04d}/kinect';h=hashlib.sha256()
 for key,sub in [('rgb','rgb'),('depth','depth')]:
  with Image.open(folder/f'{sub}/{frame:04d}.png') as im:value=np.ascontiguousarray(np.asarray(im))
  h.update(key.encode());h.update(str((value.dtype.str,value.shape)).encode());h.update(value.tobytes())
 K=np.ascontiguousarray(loadmat(folder/f'meta/{frame:04d}.mat')['intrinsic_matrix']);h.update(K.dtype.str.encode());h.update(K.tobytes());return h.hexdigest()
save('RUNNING')
try:
 old=G/'eg_sub25_v1_20261015';ex=EconomicExtractor(CODE,DATA,old/'runs/EG_SUB25_V1/train/economicgrasp_epoch10.tar',CFG['backbone_sha256'],'cuda:0')
 for split in SPLITS:
  for scene in lock['selected'][split]:
   folder=R/f'exports/subset/scene_{scene:04d}';folder.mkdir(parents=True,exist_ok=True);inputs=R/f'inputs/scene_{scene:04d}';inputs.mkdir(parents=True,exist_ok=True);identities=[]
   for frame in range(256):
    assert time.time()-start<2600,'GPU0.75-process-hour bound';assert initial+added<CFG['budget']['new_bytes']-512*1024**2,'Artifact budget reserve'
    source=inputs/f'{frame:04d}.npz';target=folder/f'{frame:04d}.npz'
    if target.exists():
     with np.load(target) as previous:assert str(previous['input_sha256'])==sha256(source) and str(previous['backbone_sha256'])==CFG['backbone_sha256']
     identities.append(dict(frame=frame,export_sha256=sha256(target),input_sha256=sha256(source)));continue
    sample=ex.sample(scene,frame,source if source.exists() else None)
    if not source.exists():np.savez_compressed(source,**sample);added+=source.stat().st_size
    pred,feature,action,seconds=ex.extract(sample,verify_hook=frame==0);ih=sha256(source);obs=observation_hash(scene,frame)
    np.savez_compressed(target,pred=pred,feature320=feature,original_action=action,candidate_ids=np.arange(1024),input_sha256=np.array(ih),raw_observation_sha256=np.array(obs),geometry_sha256=np.array(hashlib.sha256(np.ascontiguousarray(pred[:,1:]).tobytes()).hexdigest()),backbone_sha256=np.array(CFG['backbone_sha256']));added+=target.stat().st_size;identities.append(dict(frame=frame,export_sha256=sha256(target),input_sha256=ih,observation_sha256=obs,EG_seconds=seconds))
    if frame%32==0:print('EXPORT_C2',scene,frame,time.time()-start,flush=True)
   write(str((folder/'COMPLETE.json').relative_to(R)),dict(status='PASS',scene=scene,frames=256,source_identities=identities,method_lock_sha256=sha256(R/'C2_method_lock.json')));done.append(scene);save('RUNNING')
 save('PASS',peak_torch_gpu_bytes=ex.torch.cuda.max_memory_allocated())
except Exception as e:save('TECHNICAL_STOP',error=repr(e));raise
