"""Direct candidate evaluation with official collision and Dex-Net scoring; no training."""
import os,sys,json,time,hashlib,inspect,concurrent.futures
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from graspnetAPI import GraspNetEval,GraspGroup
import graspnetAPI.graspnet_eval as GE
import graspnetAPI.utils.eval_utils as EU
from graspnetAPI.utils.dexnet.grasping.grasp_quality_config import GraspQualityConfigFactory
B=Path.home()/'work/grasp';ROOT=B/'eg_pose_quality_diagnosis_v1_20261022';D=B/'eg_sub25_v1_20261015/data/graspnet'
FR=np.array([.2,.4,.6,.8,1.,1.2]);ge=GraspNetEval(str(D),camera='kinect',split='train')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
class Direct:
 def __init__(self,s):
  self.scene=s;models,self.dex,self.ids=ge.get_scene_models(s,0)
  assert all(x.sdf is not None and x.mesh is not None for x in self.dex)
  self.models=[GE.voxel_sample_points(m,.008) for m in models]
  self.table=GE.create_table_points(1,1,.05,dx=-.5,dy=-.5,dz=-.05,grid_size=.008)
  conf=GE.get_config();self.fc={}
  for f in FR:
   conf['metrics']['force_closure']['friction_coef']=round(float(f),2)
   self.fc[round(float(f),2)]=GraspQualityConfigFactory.create_config(conf['metrics']['force_closure'])
 def frame(self,f,observed=None):
  ids,self.poses,cam,align=ge.get_model_poses(self.scene,f);assert list(ids)==list(self.ids)
  self.mt=[GE.transform_points(m,p) for m,p in zip(self.models,self.poses)]
  self.objpts=np.concatenate(self.mt);self.seg=np.concatenate([np.full(len(m),i,int) for i,m in enumerate(self.mt)])
  self.table_trans=GE.transform_points(self.table,np.linalg.inv(align@cam));self.allpts=np.concatenate([self.objpts,self.table_trans])
  self.observed=observed;self.tree=None if observed is None else cKDTree(observed)
 def owners(self,rows):
  return np.concatenate([self.seg[EU.compute_closest_points(rows[i:i+128,13:16],self.objpts)] for i in range(0,len(rows),128)]) if len(rows) else np.zeros(0,int)
 def score(self,rows,contacts=False):
  rows=rows.astype(np.float64).copy();rows[:,1]=rows[:,1].clip(0,.1);n=len(rows)
  assert np.isfinite(rows).all()
  owner=self.owners(rows);mu=np.full(n,np.nan);collision=np.zeros(n,bool);empty=np.zeros(n,bool);reason=np.full(n,'',dtype='<U100');support=np.full((n,2),np.nan)
  for o in np.unique(owner):
   allix=np.flatnonzero(owner==o)
   for start in range(0,len(allix),64):
    ix=allix[start:start+64];gl=[np.zeros((0,17)) for _ in self.models];gl[o]=rows[ix]
    cs,es,ds=EU.collision_detection(gl,self.mt,self.dex,self.poses,self.allpts,outlier=.05,empty_thresh=10,return_dexgrasps=True)
    for j,k in enumerate(ix):
     c=bool(cs[o][j]);e=bool(es[o][j]);g=ds[o][j];collision[k]=c;empty[k]=e
     if c:mu[k]=-1;continue
     if g is None:reason[k]='dexgrasp_missing';continue
     try:mu[k]=float(EU.get_grasp_score(g,self.dex[o],FR[::-1],self.fc))
     except Exception as ex:reason[k]='force_closure_exception:'+type(ex).__name__;continue
     if contacts and 0<mu[k]<=.4000001 and self.tree is not None:
      try:
       ok,ct=g.close_fingers(self.dex[o],check_approach=False)
       if ok and len(ct)==2:
        pts=np.stack([x.point for x in ct]);pts=GE.transform_points(pts,self.poses[o]);support[k]=self.tree.query(pts)[0]
      except Exception:pass
  return dict(mu=mu,collision=collision,empty=empty,owner=owner,object_id=np.array(self.ids)[owner],unknown_reason=reason,contact_observed_distance=support)

def postprocess(pred,cd,owner,apply_cd=True):
 ids=np.flatnonzero(~cd) if apply_cd else np.arange(len(pred))
 track=pred[ids].astype(float).copy();track[:,16]=ids
 kept=GraspGroup(track).nms(.03,np.pi/6).grasp_group_array
 ni=kept[:,16].astype(int);assert np.array_equal(kept[:,:16],pred[ni,:16])
 oi=[]
 for o in np.unique(owner[ni]):
  pool=ni[owner[ni]==o];pool=pool[np.argsort(pred[pool,0])[::-1]];oi.extend(pool[:10])
 oi=np.array(oi,int);ranked=oi[np.argsort(pred[oi,0])[::-1]]
 return ni,oi,ranked[:50]
def rx(a):
 r=np.zeros((len(a),3,3));r[:,0,0]=1;r[:,1,1]=np.cos(a);r[:,2,2]=np.cos(a);r[:,2,1]=np.sin(a);r[:,1,2]=-np.sin(a);return r
def variants(a):
 pred=a['pred'].astype(float);v=np.flatnonzero(a['batch_valid_mask']);p=pred[v].copy();angle=a['batch_grasp_rotations'][v]*np.pi/12;depth=(a['batch_grasp_depth'][v]+1)*.01;width=a['batch_grasp_width'][v]
 rb=a['view_rotation'][v].astype(float);rlabel=a['top_grasp_views_rot'][v].astype(float)@rx(angle)
 rr=rb@rx(angle);data={}
 for name,aa,dd,ww in [('P1',True,False,False),('P2',False,True,False),('P3',False,False,True),('P4',True,True,False),('P5',True,True,True)]:
  z=p.copy()
  if aa:z[:,4:13]=rr.reshape(-1,9)
  if dd:z[:,3]=depth
  if ww:z[:,1]=width
  assert np.array_equal(z[:,13:16],p[:,13:16])
  assert np.allclose(z[:,4:13].reshape(-1,3,3)[:,:,0],p[:,4:13].reshape(-1,3,3)[:,:,0],atol=2e-6)
  data[name]=(v,z)
 # Match physical roll to the inference frame while preserving prediction approach.
 theta=np.arctan2((rlabel[:,:,1]*rb[:,:,2]).sum(1),(rlabel[:,:,1]*rb[:,:,1]).sum(1))
 z=data['P5'][1].copy();z[:,4:13]=(rb@rx(theta)).reshape(-1,9);data['P5_aligned']=(v,z)
 z=data['P5'][1].copy();z[:,4:13]=rlabel.reshape(-1,9);data['R_predpoint']=(v,z)
 z=z.copy();z[:,13:16]=a['batch_grasp_point'][v];data['R_labelpoint']=(v,z)
 z=pred.copy();z[:,1]=np.clip(a['raw_width']/10,0,.1);data['width_no_scale']=(np.arange(1024),z)
 return data
def save_result(out,name,ids,r,n=1024):
 for k,val in r.items():
  shape=(n,)+val.shape[1:]
  arr=np.full(shape,np.nan) if val.dtype.kind=='f' else np.zeros(shape,bool) if val.dtype.kind=='b' else np.full(shape,-1,dtype=val.dtype) if val.dtype.kind in 'iu' else np.full(shape,'not_evaluated',dtype=val.dtype)
  arr[ids]=val;out[name+'_'+k]=arr

def frame_job(task):
 split,s,f=task;dest=ROOT/f'evaluation/{split}_{s:04d}_{f:04d}.npz'
 if dest.exists():return {'scene':s,'frame':f,'cached':True}
 ts=time.time();ct=time.process_time();path=ROOT/f'cache/{split}_{s:04d}_{f:04d}.npz'
 with np.load(path) as z:a={k:z[k].copy() for k in z.files}
 direct=Direct(s);direct.frame(f,a['input_points']);pred=a['pred'];owner=direct.owners(pred)
 nms,obj10,final=postprocess(pred,a['cd_mask'],owner);_,_,noCD=postprocess(pred,a['cd_mask'],owner,False)
 out=dict(final_ids=final,final_noCD=noCD,nms_ids=nms,obj10_ids=obj10,owner=owner,object_id=np.array(direct.ids)[owner],scene_object_ids=np.array(direct.ids),candidate_id=np.arange(1024),geometry_sha256=a['geometry_sha256'])
 ids=np.arange(1024) if split=='train' else np.union1d(final,noCD)
 save_result(out,'P0',ids,direct.score(pred[ids],contacts=split=='train'))
 if split=='train':
  for name,(vi,poses) in variants(a).items():
   save_result(out,name,vi,direct.score(poses,contacts=name in ['P5','P5_aligned','R_labelpoint']))
   print('VARIANT',s,f,name,len(vi),flush=True)
 else:
  poses=pred[ids].copy();poses[:,1]=np.clip(a['raw_width'][ids]/10,0,.1);save_result(out,'width_no_scale',ids,direct.score(poses))
 dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest,**out)
 meta=dict(scene=s,frame=f,split=split,cache_sha256=sha(path),result_sha256=sha(dest),wall_seconds=time.time()-ts,cpu_seconds=time.process_time()-ct,final_count=len(final),raw_evaluated=len(ids),unknowns={k:int(np.isnan(v).sum()) for k,v in out.items() if k.endswith('_mu')},source_hashes={fn.__name__:hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() for fn in [EU.collision_detection,EU.get_grasp_score,EU.compute_closest_points,GE.voxel_sample_points]})
 dest.with_suffix('.json').write_text(json.dumps(meta,indent=2));print('FRAME '+json.dumps(meta),flush=True);return meta

def pilot():
 direct=Direct(100);direct.frame(0)
 P=B/'eg_sub25_b_f05_repl_v1_20261019';E=B/'eg_tf32_residual_p2_v1_20261021'
 with np.load(P/'B/exports/subset/scene_0100/0000.npz') as z:p=z['pred']
 with np.load(E/'evaluation/development/0100/0000.npz') as z:q=z['quality'];c=z['collision'];e=z['empty']
 groups=[np.flatnonzero(np.isfinite(q)&e),np.flatnonzero(np.isfinite(q)&c&~e),np.flatnonzero((q>0)&(q<=.4)),np.flatnonzero((q>.4)),np.flatnonzero((q<0)&~c)]
 ids=np.unique(np.concatenate([g[:2] for g in groups]));r=direct.score(p[ids]);assert np.array_equal(r['mu'],q[ids]) and np.array_equal(r['collision'],c[ids]) and np.array_equal(r['empty'],e[ids])
 path=ROOT/'cache/C3_0100_0000.npz'
 with np.load(path) as z:a={k:z[k] for k in z.files}
 pred=a['pred'];owner=direct.owners(pred);_,_,final=postprocess(pred,a['cd_mask'],owner)
 gl,ql,cl=EU.eval_grasp(GraspGroup(pred[~a['cd_mask']].copy()),direct.models,direct.dex,direct.poses,GE.get_config(),table=direct.table_trans,TOP_K=50)
 rows=np.concatenate(gl);qq=np.concatenate(ql);cc=np.concatenate(cl);new=direct.score(rows)
 assert np.array_equal(new['mu'],qq) and np.array_equal(new['collision'],cc)
 keys={tuple(p[:16]):i for i,p in enumerate(pred)};officialids=np.array([keys[tuple(p[:16])] for p in rows]);assert set(officialids)==set(final)
 receipt=dict(historical_verified=len(ids),official_full_model_verified=len(rows),postprocess_id_set_equal=True,quality_direction='mu lower is better; 0<mu<=0.4',no_nearest_GT=True,source_hashes={fn.__name__:hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() for fn in [EU.collision_detection,EU.get_grasp_score]})
 (ROOT/'evaluation_preflight.json').write_text(json.dumps(receipt,indent=2));print('PREFLIGHT '+json.dumps(receipt),flush=True)

if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='pilot':pilot()
 else:
  plan=json.loads((ROOT/'study_plan.json').read_text());tasks=[('C3',s,f) for s in plan['C3_scenes'] for f in plan['C3_frames']]+[('train',s,f) for s in plan['training_scenes'] for f in plan['training_frames']]
  if mode=='single':tasks=[('train',plan['training_scenes'][0],0)]
  else:
   if mode in ['A','B']:tasks=[t for t in tasks if (t[0]=='C3')==(mode=='A')]
   shard=int(sys.argv[2]);nshard=int(sys.argv[3]);tasks=[t for i,t in enumerate(tasks) if i%nshard==shard]
  workers=1 if mode=='single' else int(sys.argv[4])
  with concurrent.futures.ProcessPoolExecutor(workers) as pool:
   for r in pool.map(frame_job,tasks):print('DONE '+json.dumps(r),flush=True)
  print('COMPLETE',mode,flush=True)
