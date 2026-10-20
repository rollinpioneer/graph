from common import *
from geometry_core import geom,visible,NAMES
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
def job(task):
 mode,scene,frame=task;os.nice(10);t=time.time();cpu=time.process_time();ep=P/f'B/exports/subset/scene_{scene:04d}/{frame:04d}.npz'
 with np.load(ep) as z:
  pred=z['pred'];hyp=z['hypotheses'] if mode=='head' else pred[:,None,:]
  if mode=='head':
   with np.load(P/f'B/labels/{scene:04d}_{frame:04d}.npz') as y:ids=y['ids'];assert str(y['export_sha256'])==sha(ep)
   assert np.array_equal(hyp[:,0],pred);hyp=hyp[ids]
 cloud,identity=visible(scene,frame,pred);tree=cKDTree(cloud);raw=np.array([[geom(cloud,tree,g) for g in group] for group in hyp],np.float32);assert not np.isinf(raw).any()
 out=R/f'geometry/{mode}/{scene:04d}/{frame:04d}.npz';out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists():
  with np.load(out) as z:assert np.array_equal(z['raw'],raw,equal_nan=True)
 else:np.savez_compressed(out,raw=raw,export_sha256=np.array(sha(ep)),depth_sha256=np.array(identity['depth_sha256']),meta_sha256=np.array(identity['meta_sha256']))
 return dict(mode=mode,scene=scene,frame=frame,seconds=time.time()-t,cpu_seconds=time.process_time()-cpu,shape=list(raw.shape),missing_fraction=float((~np.isfinite(raw)).mean()),path=str(out),sha256=sha(out))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--mode',choices=['pilot','head','C3'],required=True);a.add_argument('--workers',type=int,default=88);args=a.parse_args();start=time.time();cfg=json.load(open(R/'frozen_config.json'))
 if args.mode=='pilot':tasks=[('pilot',s,f) for s,f in cfg['P0']['frames']];workers=1
 elif args.mode=='head':
  split=json.load(open(P/'B_head_split.json'));tasks=[('head',s,f) for role in ['fit','model_val','gate_cal'] for s in split[role] for f in split['frame_ids']];workers=min(args.workers,len(tasks))
 else:tasks=[('C3',s,f) for s in sum(cfg['C3'].values(),[]) for f in range(256)];workers=args.workers
 receipts=[]
 with ProcessPoolExecutor(max_workers=workers) as pool:
  futs=[pool.submit(job,t) for t in tasks]
  for fu in as_completed(futs):
   receipts.append(fu.result())
   if len(receipts)%32==0:print('GEOMETRY',args.mode,len(receipts),'/',len(tasks),flush=True)
 if args.mode=='pilot':
  lat=np.array([v['seconds'] for v in receipts]);passed=np.median(lat)<=cfg['P0']['latency_1024_single_process_median_cap_seconds'] and np.quantile(lat,.95)<=cfg['P0']['latency_p95_cap_seconds'];write(R/'results/P0_cost.json',dict(status='PASS' if passed else 'COST_STOP',latency_seconds=lat.tolist(),median=float(np.median(lat)),p95=float(np.quantile(lat,.95)),names=NAMES,source='visible depth only, identical training/inference geometry functions',single_CPU_worker=True));assert passed,'P0 geometry cost cap failed'
 write(R/f'results/geometry_{args.mode}.json',dict(status='PASS',frames=len(receipts),workers=workers,worker_seconds=sum(v['seconds'] for v in receipts),cpu_seconds=sum(v['cpu_seconds'] for v in receipts),wall_seconds=time.time()-start,receipts=receipts));print('GEOMETRY_DONE',args.mode,len(receipts),time.time()-start,flush=True)
