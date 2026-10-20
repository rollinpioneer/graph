from compact import *
import argparse,resource

p=argparse.ArgumentParser();p.add_argument('--scene',type=int,required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=2);a=p.parse_args();start=time.time();os.nice(10);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
cfg=json.load(open(R/'frozen_config.json'));assert a.scene in sum(cfg['evaluation']['C3'].values(),[]);heads={};norm=dict(np.load(R/'C_normalization.npz'));identities={}
for family in ['C','S']:
 for seed in SEEDS:
  receipt=json.load(open(R/f'results/train_{family}_{seed}.json'));path=Path(receipt['checkpoint']);assert sha(path)==receipt['checkpoint_sha256'];ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['step']==1000
  net=initialize(family,seed);net.load_state_dict(ck['model']);heads[f'{family}_{seed}']=net.cuda().eval();identities[f'{family}_{seed}']=receipt['checkpoint_sha256']
out=R/f'scores/{a.scene:04d}';out.mkdir(parents=True,exist_ok=True);frames=list(range(256))[a.shard::a.shards];rows=[];torch.cuda.reset_peak_memory_stats()
for frame in frames:
 dest=out/f'{frame:04d}.npz';assert not dest.exists()
 ep=P/f'B/exports/subset/scene_{a.scene:04d}/{frame:04d}.npz'
 with np.load(ep) as z:pred=z['pred'];feat=z['feature320'];action=z['original_action']
 raw,points,mask,identity=extract(a.scene,frame,pred,mode='both')
 # Full-dataset C input check against protected G, including missing indicators.
 with np.load(G/f'eg_lg1_local_geometry_v1_20261020/geometry/C3/{a.scene:04d}/{frame:04d}.npz') as z:assert np.array_equal(raw,z['raw'][:,0,COLUMNS],equal_nan=True)
 x=np.concatenate([(feat-ck['mean'])/ck['std'],action],-1).astype(np.float32)
 with torch.no_grad():
  xx=torch.from_numpy(x).cuda();cc=torch.from_numpy(geom_input(raw,norm)).cuda();pp=torch.from_numpy(points).cuda();mm=torch.from_numpy(mask).cuda()
  q={name:(net(xx,cc) if name.startswith('C_') else net(xx,pp,mm)).cpu().numpy() for name,net in heads.items()}
 assert all(v.shape==(1024,) and np.isfinite(v).all() for v in q.values())
 np.savez_compressed(dest,**q,export_sha256=np.array(sha(ep)),config_sha256=np.array(sha(R/'frozen_config.json')),head_hashes_json=np.array(json.dumps(identities,sort_keys=True)),C_raw_sha256=np.array(hashlib.sha256(raw.tobytes()).hexdigest()),S_input_sha256=np.array(hashlib.sha256(points.tobytes()+mask.tobytes()).hexdigest()))
 rows.append(dict(frame=frame,path=str(dest),sha256=sha(dest),**identity))
 if len(rows)%16==0:print('SCORE',a.scene,a.shard,len(rows),len(frames),round(time.time()-start,2),flush=True)
write(R/f'results/score_{a.scene:04d}_{a.shard}.json',dict(status='PASS',scene=a.scene,shard=a.shard,shards=a.shards,frames=len(frames),six_heads=True,EG_inference=0,seconds=time.time()-start,cpu_seconds=time.process_time(),peak_gpu_bytes=torch.cuda.max_memory_allocated(),GPU_slot=os.environ['CUDA_VISIBLE_DEVICES'],peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,receipts=rows))
print('SCORE_DONE',a.scene,a.shard,time.time()-start,flush=True)
