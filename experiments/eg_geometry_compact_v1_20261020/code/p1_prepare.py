from compact import *
from concurrent.futures import ProcessPoolExecutor, as_completed
import resource

def job(task):
 i,scene,frame,ids=task;os.nice(10);t=time.time();ct=time.process_time()
 ep=P/f'B/exports/subset/scene_{scene:04d}/{frame:04d}.npz'
 with np.load(ep) as z:pred=z['pred'];hyp=z['hypotheses'];actions=z['actions'];feat=z['feature320']
 with np.load(P/f'B/labels/{scene:04d}_{frame:04d}.npz') as z:
  assert np.array_equal(z['ids'],ids);assert str(z['export_sha256'])==sha(ep)
 assert np.array_equal(hyp[:,0],pred)
 raw,tokens,mask,identity=extract(scene,frame,pred,hyp[ids].reshape(-1,17),'both')
 with np.load(G/f'eg_lg1_local_geometry_v1_20261020/geometry/head/{scene:04d}/{frame:04d}.npz') as z:
  assert np.array_equal(raw.reshape(128,6,21),z['raw'][...,COLUMNS],equal_nan=True)
 norm=dict(np.load(R/'C_normalization.npz'));cn=geom_input(raw,norm)
 for key,value in [('compact',cn.reshape(128,6,42)),('tokens',tokens.reshape(128,6,128,8)),('mask',mask.reshape(128,6,128))]:
  mmap=np.load(R/f'cache/{key}.npy',mmap_mode='r+');mmap[i]=value;mmap.flush();del mmap
 return dict(index=i,scene=scene,frame=frame,seconds=time.time()-t,cpu_seconds=time.process_time()-ct,export_sha256=sha(ep),tokens_sha256=hashlib.sha256(tokens.tobytes()+mask.tobytes()).hexdigest(),C_exact_G_subset=True,padding_fraction=float((~mask).mean()),empty_pose_fraction=float((~mask.any(-1)).mean()),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**identity)

if __name__=='__main__':
 start=time.time();cfg=json.load(open(R/'frozen_config.json'));assert json.load(open(R/'results/approval_P1_P3.json'))['approved'];assert not (R/'results/preparation.json').exists()
 d=dict(np.load(RP/'prepared.npz'));assert sha(RP/'prepared.npz')==cfg['training']['prepared']['sha256']
 (R/'cache').mkdir(exist_ok=True)
 for name,shape,dtype in [('compact',(256,128,6,42),'float32'),('tokens',(256,128,6,128,8),'float32'),('mask',(256,128,6,128),'bool')]:
  p=R/f'cache/{name}.npy';assert not p.exists();m=np.lib.format.open_memmap(p,mode='w+',dtype=dtype,shape=shape);del m
 workers=72;receipts=[]
 with ProcessPoolExecutor(max_workers=workers) as pool:
  fut=[pool.submit(job,(i,int(s),int(f),d['ids'][i])) for i,(s,f) in enumerate(zip(d['scene'],d['frame']))]
  for fu in as_completed(fut):
   receipts.append(fu.result());write(R/'results/preparation_progress.json',dict(completed=len(receipts),total=256,workers=workers,seconds=time.time()-start))
   if len(receipts)%16==0:print('PREPARE',len(receipts),256,round(time.time()-start,2),flush=True)
 receipts.sort(key=lambda v:v['index']);streams=cfg['training']['sampler_streams']
 fit=d['role']=='fit';groups=np.column_stack([np.repeat(d['scene'][fit],128),np.repeat(d['frame'][fit],128),d['ids'][fit].reshape(-1)]).astype(np.int64)
 assert hashlib.sha256(groups.tobytes()).hexdigest()==cfg['training']['fit_group_identity_sha256']
 for spec in streams:
  rng=np.random.default_rng(spec['seed']);order=rng.permutation(len(groups));cursor=0;ixh=hashlib.sha256();idh=hashlib.sha256()
  for step in range(1000):
   if cursor+128>len(order):order=rng.permutation(len(groups));cursor=0
   ix=order[cursor:cursor+128];cursor+=128;ixh.update(ix.tobytes());idh.update(groups[ix].tobytes())
  assert ixh.hexdigest()==spec['index_stream_sha256'];assert idh.hexdigest()==spec['scene_frame_candidate_stream_sha256']
 result=dict(status='PASS',frames=256,poses=196608,new_training_GT=0,C_exact_G_subset=True,prepared_sha256=sha(RP/'prepared.npz'),sampler_streams=streams,seconds=time.time()-start,CPU_worker_seconds_upper=workers*(time.time()-start),receipts=receipts,cache=[dict(path=str(R/f'cache/{k}.npy'),sha256=sha(R/f'cache/{k}.npy'),bytes=(R/f'cache/{k}.npy').stat().st_size) for k in ['compact','tokens','mask']])
 write(R/'results/preparation.json',result);print('PREPARATION_PASS',result['seconds'],flush=True)
