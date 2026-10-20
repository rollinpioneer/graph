from compact import *
import importlib.util,subprocess
t=time.time();torch.set_num_threads(1)
old=G/'eg_geometry_compact_v1_20261020'
spec=importlib.util.spec_from_file_location('protected_compact',old/'code/compact.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
cfg=json.load(open(R/'frozen_config.json'));d=dict(np.load(RP/'prepared.npz'))
checks=[];tokens_example=None
for scene,frame,training in [(int(d['scene'][0]),int(d['frame'][0]),True),(100,0,False)]:
 with np.load(P/f'B/exports/subset/scene_{scene:04d}/{frame:04d}.npz') as z:
  pred=z['pred'];poses=z['hypotheses'][d['ids'][0][:4]].reshape(-1,17) if training else pred[:24]
 cloud,_=visible_fast(scene,frame,pred);tree=CachedTree(cKDTree(cloud),poses)
 kept=0;before=[];after=[]
 for pose in poses:
  local=(cloud[tree.query_ball_point(pose[13:16],.16)]-pose[13:16])@pose[4:13].reshape(3,3);masks=regions(local,pose)
  orig,om=base.point_values(local,pose,masks);points,mask=point_values(local,pose,masks)
  again,am=point_values(local,pose,masks)
  assert np.array_equal(points[om],orig[om]) and mask[om].all()
  assert np.array_equal(points,again) and np.array_equal(mask,am)
  assert np.array_equal(compact_values(local,pose,masks),base.compact_fast(local,pose),equal_nan=True)
  before.append(int(om.sum()));after.append(int(mask.sum()));tokens_example=(points,mask)
 checks.append(dict(scene=scene,frame=frame,poses=len(poses),training_six_pose=training,old_valid_mean=float(np.mean(before)),filled_valid_mean=float(np.mean(after)),original_points_preserved=True,C_unchanged=True,deterministic=True))
# Empty input mask produces finite encoding, no padded observations contribute.
pose=poses[0];pp,mm=point_values(np.empty((0,3)),pose,[np.zeros(0,bool)]*5)
assert not mm.any() and not pp.any()
sf=initialize('SF',20261018);h=initialize('H',20261018)
for n,v in sf.state_dict().items():
 if n.startswith(('point.','compress.')):assert torch.equal(v,h.state_dict()[n])
assert sum(p.numel() for p in sf.parameters())==60577
assert sum(p.numel() for p in h.parameters())==65953
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
gpu_t=time.time();torch.cuda.reset_peak_memory_stats()
models={}
for family in ['SF','H']:
 net=initialize(family,20261018).cuda()
 # Same real point tensor, duplicated only for engineering batch-capacity validation.
 points,mask=tokens_example
 xx=torch.zeros((768,324),device='cuda');pt=torch.from_numpy(np.broadcast_to(points,(768,128,8)).copy()).cuda();ma=torch.from_numpy(np.broadcast_to(mask,(768,128)).copy()).cuda()
 args=[xx,pt,ma]
 if family=='H':args.append(torch.zeros((768,42),device='cuda'))
 q=net(*args);assert torch.isfinite(q).all()
 loss=q.square().mean();loss.backward()
 assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
 models[family]=dict(parameters=sum(p.numel() for p in net.parameters()),finite_forward_backward=True,formal_updates=0)
 del net,args,xx,pt,ma,q,loss
gpu_seconds=time.time()-gpu_t
result=dict(status='PASS',checks=checks,models=models,initial_encoder_identical=True,formal_updates=0,peak_gpu_bytes=torch.cuda.max_memory_allocated(),GPU_process_seconds_upper=gpu_seconds,CPU_worker_seconds_upper=time.time()-t,GPU_slot=os.environ.get('CUDA_VISIBLE_DEVICES'),time=time.time())
write(R/'results/P0_validation.json',result)
cfg['P0']={k:result[k] for k in ['GPU_process_seconds_upper','CPU_worker_seconds_upper']}
cfg['code_sha256']={p.name:sha(p) for p in sorted((R/'code').glob('*')) if p.suffix in ['.py','.cpp','.so']}
write(R/'frozen_config.json',cfg)
write(R/'method_lock.json',dict(experiment=cfg['experiment'],config_sha256=sha(R/'frozen_config.json'),code_sha256=cfg['code_sha256'],protected_G_C_heads=cfg['protected_G_C_heads'],training_steps=1000,C4=False,freeze_time=time.time()))
print('P0_PASS',json.dumps(result),flush=True)
