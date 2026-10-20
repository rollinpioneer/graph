from common import *
import argparse,resource
p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args();os.nice(10);t=time.time()
lock=json.load(open(R/'method_lock.json'));cfg=json.load(open(R/'frozen_config.json'));assert sha(R/'frozen_config.json')==lock['frozen_config_sha256']
from infer import EconomicExtractor
from PIL import Image
from scipy.io import loadmat
ex=EconomicExtractor(CODE,DATA,Path(lock['backbone_path']),lock['backbone_sha256'],'cuda:0')
scenes=sum(cfg['C4'].values(),[])[a.shard::a.shards];identities=[]
for scene in scenes:
 for frame in ([0] if a.pilot else range(256)):
  out=R/f'exports/scene_{scene:04d}/{frame:04d}.npz';out.parent.mkdir(parents=True,exist_ok=True)
  if out.exists():
   with np.load(out) as z:assert str(z['backbone_sha256'])==lock['backbone_sha256']
   continue
  sample=ex.sample(scene,frame);hh=hashlib.sha256()
  for k in ['point_clouds','cloud_colors','coordinates_for_voxel','seg']:
   x=np.ascontiguousarray(sample[k]);hh.update(k.encode());hh.update(str((x.dtype.str,x.shape)).encode());hh.update(x.tobytes())
  pred,feat,action,seconds=ex.extract(sample,verify_hook=frame==0)
  assert np.isfinite(pred).all() and np.isfinite(feat).all() and np.isfinite(action).all()
  obs=hashlib.sha256();raw=DATA/f'scenes/scene_{scene:04d}/kinect'
  for k in ['rgb','depth']:
   x=np.ascontiguousarray(np.asarray(Image.open(raw/f'{k}/{frame:04d}.png')));obs.update(k.encode());obs.update(str((x.dtype.str,x.shape)).encode());obs.update(x.tobytes())
  K=np.ascontiguousarray(loadmat(raw/f'meta/{frame:04d}.mat')['intrinsic_matrix']);obs.update(K.dtype.str.encode());obs.update(K.tobytes())
  np.savez_compressed(out,pred=pred,feature320=feat,original_action=action,candidate_ids=np.arange(1024),input_sha256=np.array(hh.hexdigest()),input_identity_format=np.array('canonical_sample_arrays_v1'),geometry_sha256=np.array(hashlib.sha256(np.ascontiguousarray(pred[:,1:]).tobytes()).hexdigest()),backbone_sha256=np.array(lock['backbone_sha256']),raw_observation_sha256=np.array(obs.hexdigest()))
  identities.append(dict(scene=scene,frame=frame,sha256=sha(out),inference_seconds=seconds))
  if frame%32==0:print('EXPORT',scene,frame,time.time()-t,flush=True)
write(R/('results/export_'+str(a.shard)+'_'+('pilot.json' if a.pilot else 'complete.json')),dict(status='PASS',seconds=time.time()-t,peak_torch_gpu_bytes=ex.torch.cuda.max_memory_allocated(),sources=identities,scenes=scenes,pilot=a.pilot))
