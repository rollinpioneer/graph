from visibility import *
import argparse,resource

def frame_job(pool,scene,frame,contact_native=None,dv0=None,pilot=False):
 start=time.time();cpu=time.process_time();ep=export(scene,frame)
 with np.load(ep) as z:
  pred=z['pred'];feature=z['feature320'];action=z['original_action']
  hy=z['hypotheses'] if pool=='head' else None
 obs=Observation(scene,frame);source={'export':sha(ep),'depth':sha(obs.dp),'meta':sha(obs.mp)}
 if pool=='head':
  with np.load(P/f'B/labels/{scene:04d}_{frame:04d}.npz') as z:
   ids=z['ids'];mu=z['q'][:,0];co=z['collision'][:,0];em=z['empty'][:,0]
   assert str(z['export_sha256'])==source['export'];allmu=z['q'];allco=z['collision']
  rawpath=LG/f'geometry/head/{scene:04d}/{frame:04d}.npz'
  with np.load(rawpath) as z:
   raw=z['raw'][:,0];assert str(z['export_sha256'])==source['export'];assert str(z['depth_sha256'])==source['depth']
  # CD for original 128 labeled actions; inference-only visible cloud, no GT.
  sys.path.insert(0,str(B/'eg_sub25_v1_20261015/runtime/EconomicGrasp'))
  from utils.collision_detector import ModelFreeCollisionDetector
  from graspnetAPI import GraspGroup
  from utils.data_utils import get_workspace_mask
  folder=DATA/f'scenes/scene_{scene:04d}/kinect'
  seg=np.asarray(Image.open(folder/f'label/{frame:04d}.png'))
  trans=np.load(folder/'cam0_wrt_table.npy')@np.load(folder/'camera_poses.npy')[frame]
  # Match protected evaluation CD. Segmentation is used only for this
  # pre-existing evaluation selection convention, never in V/K/align inputs.
  mask=(obs.depth>0)&get_workspace_mask(obs.points,seg,trans=trans,organized=True,outlier=.02)
  cloud=obs.points[mask]
  cd=ModelFreeCollisionDetector(cloud,voxel_size=.005).detect(GraspGroup(pred[ids].copy()),approach_dist=.05,collision_thresh=.01)
  top=np.empty((3,0),int);q=np.empty((3,len(ids)))
 else:
  ids,mu0,co0,em0,cd0,top,q0=frame_sources(scene,frame)
  mu,co,em,cd=mu0[ids],co0[ids],em0[ids],cd0[ids];q=q0[:,ids]
  rawpath=LG/f'geometry/C3/{scene:04d}/{frame:04d}.npz'
  with np.load(rawpath) as z:
   raw=z['raw'][ids,0];assert str(z['export_sha256'])==source['export'];assert str(z['depth_sha256'])==source['depth']
 source['geometry']=sha(rawpath)
 vf,kf,_,_=obs.inputs(pred[ids]);aligned=official_aligned(obs,pred[ids])
 assert vf.shape==(len(ids),40) and kf.shape==(len(ids),88) and aligned.shape==(len(ids),21)
 assert np.isfinite(vf).all()
 arrays=dict(ids=ids,V=vf,K=kf,align=aligned,mu=mu,collision=co,empty=em,cd=cd,top=top,q=q)
 stats=dict(pool=pool,scene=scene,frame=frame,candidates=len(ids),source=source)
 if contact_native is not None:
  from contacts import contact,collision_evidence
  contact_native.frame(frame)
  if pool=='head':
   selected=dv0[str(scene)][str(frame)] if str(frame) in dv0[str(scene)] else []
   if pilot:selected=selected[:32]
   rows=np.array([hy[ids[i],j] for i,j in selected]);truth=np.array([allmu[i,j] for i,j in selected])
   records=[contact(contact_native,p,obs) for p in rows]
   arrays.update(dv0_id=np.array(selected,int).reshape(-1,2),dv0_mu=truth,dv0=np.array(records).reshape(-1,22))
  else:
   selected=np.unique(top);records=[];collisions=[]
   for idx in selected:
    # Contact failures and actual collisions both retained; GT contact is diagnostic only.
    records.append(contact(contact_native,pred[idx],obs))
    collisions.append(collision_evidence(contact_native,pred[idx],obs))
   arrays.update(diag_ids=selected,contacts=np.array(records),collision_points=np.array(collisions))
   pos=np.searchsorted(ids,selected)
   assert np.array_equal(ids[pos],selected)
   computed=np.array(collisions);exact_empty=computed[:,1]<10;exact_col=(computed[:,0]>0)|exact_empty
   assert np.array_equal(exact_col,co[pos]) and np.array_equal(exact_empty,em[pos]),'official collision reconstruction mismatch'
 stats.update(seconds=time.time()-start,cpu_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 out=R/f'cache/{pool}/{scene:04d}/{frame:04d}.npz';out.parent.mkdir(parents=True,exist_ok=True)
 assert not out.exists();np.savez_compressed(out,**arrays);stats['bytes']=out.stat().st_size
 write(R/f'results/frames/{pool}_{scene:04d}_{frame:04d}.json',stats)
 print('FRAME',pool,scene,frame,len(ids),round(stats['seconds'],2),flush=True)
 return stats

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('pool');p.add_argument('scene',type=int);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--pilot',action='store_true');a=p.parse_args()
 os.nice(10);start=time.time();dv0=json.load(open(R/'dv0_sample.json'));n=native(a.scene)
 frames=[0] if a.pilot else list(range(0,256,32)) if a.pool=='head' else list(range(256))
 frames=frames[a.shard::a.shards]
 for frame in frames:
  if (R/f'cache/{a.pool}/{a.scene:04d}/{frame:04d}.npz').exists():continue
  frame_job(a.pool,a.scene,frame,n,dv0,a.pilot)
 write(R/f'results/jobs/{a.pool}_{a.scene}_{a.shard}{"_pilot" if a.pilot else ""}.json',dict(seconds=time.time()-start,pool=a.pool,scene=a.scene,shard=a.shard,frames=len(frames)))
