import argparse,subprocess
from repl_common import *
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['head','C3'],required=True);p.add_argument('--bone',choices=['A','B'],required=True);p.add_argument('--shard-id',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();start=time.time();os.nice(10)
from infer import EconomicExtractor
ck,h=backbone(a.bone);ex=EconomicExtractor(CODE,DATA,ck,h,'cuda:0');ep={}
ex.net.register_forward_hook(lambda m,i,o:ep.update(value=o))
if a.mode=='head':
 assert a.bone=='B';split=json.loads((R/'B_head_split.json').read_text());scenes=split['fit']+split['model_val']+split['gate_cal'];frames=split['frame_ids']
else:
 lock=json.loads((R/'C3_method_lock.json').read_text());assert lock['protocol_sha256']==sha256(R/'protocol.json');scenes=sum(lock['selected'].values(),[]);frames=list(range(256));assert lock['backbones'][a.bone]==h
scenes=scenes[a.shard_id::a.shards];assert scenes
from models.economicgrasp import batch_viewpoint_params_to_matrix
from PIL import Image
from scipy.io import loadmat
for scene in scenes:
 folder=R/f'{a.bone}/exports/subset/scene_{scene:04d}';folder.mkdir(parents=True,exist_ok=True);identities=[]
 for frame in frames:
  target=folder/f'{frame:04d}.npz';assert not target.exists(),target
  inp=R/f'inputs/scene_{scene:04d}/{frame:04d}.npz';inp.parent.mkdir(parents=True,exist_ok=True);sample=ex.sample(scene,frame,inp if inp.exists() else None)
  if not inp.exists():np.savez_compressed(inp,**sample)
  pred,feature,action,seconds=ex.extract(sample,verify_hook=frame==frames[0]);assert np.isfinite(pred).all() and np.isfinite(feature).all() and np.isfinite(action).all()
  data=dict(pred=pred,feature320=feature,original_action=action,candidate_ids=np.arange(1024),input_sha256=np.array(sha256(inp)),geometry_sha256=np.array(hashlib.sha256(np.ascontiguousarray(pred[:,1:]).tobytes()).hexdigest()),backbone_sha256=np.array(h))
  if a.mode=='head':
   end=ep['value'];torch=ex.torch;al=end['grasp_angle_pred'][0,:12].detach().cpu().numpy().T;dl=end['grasp_depth_pred'][0,:4].detach().cpu().numpy().T;ar=np.argsort(-al,axis=1,kind='stable')[:,:3];dr=np.argsort(-dl,axis=1,kind='stable')[:,:2]
   ai=np.stack([ar[:,i] for j in range(2) for i in range(3)],1);di=np.stack([dr[:,j] for j in range(2) for i in range(3)],1);hyp=np.repeat(pred[:,None,:],6,1)
   for k in range(6):
    angles=torch.from_numpy(ai[:,k].copy()).to(ex.device)*np.pi/12
    hyp[:,k,4:13]=batch_viewpoint_params_to_matrix(-end['grasp_top_view_xyz'][0].float(),angles).detach().cpu().numpy().reshape(1024,9)
    hyp[:,k,3]=(di[:,k]+1)*.01
   assert np.array_equal(hyp[:,0],pred),'Original action must exactly equal first hypothesis'
   if frame==frames[0] and scene==scenes[0]:
    for k in range(6):
     forced=dict(end)
     for key,ix in [('grasp_angle_pred',ai[:,k]),('grasp_depth_pred',di[:,k])]:
      value=torch.zeros_like(end[key]);index=torch.from_numpy(ix.copy()).to(ex.device);value[0].scatter_(0,index[None,:],1.);forced[key]=value
     official=ex.decode(forced)[0].detach().cpu().numpy();assert np.array_equal(hyp[:,k],official),('Hypothesis decoder mismatch',k)
   acts=np.stack([np.sin(2*ai*np.pi/12),np.cos(2*ai*np.pi/12),(di+1)/4,hyp[:,:,1]/.1],-1).astype(np.float32)
   data.update(hypotheses=hyp,actions=acts,action_ids=ai*4+di,angle_logits=al,depth_logits=dl)
  obs=hashlib.sha256();folderraw=DATA/f'scenes/scene_{scene:04d}/kinect'
  for key in ['rgb','depth']:
   value=np.ascontiguousarray(np.asarray(Image.open(folderraw/f'{key}/{frame:04d}.png')));obs.update(key.encode());obs.update(str((value.dtype.str,value.shape)).encode());obs.update(value.tobytes())
  K=np.ascontiguousarray(loadmat(folderraw/f'meta/{frame:04d}.mat')['intrinsic_matrix']);obs.update(K.dtype.str.encode());obs.update(K.tobytes());data['raw_observation_sha256']=np.array(obs.hexdigest())
  np.savez_compressed(target,**data);identities.append(dict(frame=frame,export_sha256=sha256(target),input_sha256=sha256(inp),raw_observation_sha256=obs.hexdigest(),inference_seconds=seconds))
  if frame%32==0:print('EXPORT',a.mode,a.bone,scene,frame,time.time()-start,flush=True)
 write(folder/'COMPLETE.json',dict(status='PASS',source_identities=identities,frames=frames,backbone_sha256=h))
write(R/f'results/export_{a.mode}_{a.bone}_shard{a.shard_id}.json',dict(status='PASS',seconds=time.time()-start,peak_torch_gpu_bytes=ex.torch.cuda.max_memory_allocated(),scenes=scenes,frames=frames,backbone_sha256=h))
