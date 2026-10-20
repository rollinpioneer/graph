from common import *
import argparse,inspect,importlib.util,resource

p=argparse.ArgumentParser();p.add_argument('--scene',type=int,required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=6);p.add_argument('--profile',action='store_true');a=p.parse_args();os.nice(10);start=time.time()
cfg=json.load(open(R/'frozen_config.json'));assert a.scene in sum(cfg['evaluation']['C3'].values(),[])
sys.path[:0]=[str(CODE),str(CODE/'libs/pointnet2')];sys.argv=[sys.argv[0],'--dataset_root',str(DATA),'--camera','kinect','--model','economicgrasp','--batch_size','1']
from graspnetAPI import GraspGroup
import graspnetAPI.utils.eval_utils as EU
from dataset.graspnet_dataset import GraspNetDataset
from utils.collision_detector import ModelFreeCollisionDetector
spec=importlib.util.spec_from_file_location('unchanged_native',G/'eg_nqs_pilot_v2_20261018/code/common.py');nm=importlib.util.module_from_spec(spec);spec.loader.exec_module(nm);native=nm.Native(a.scene);ap=nm.ap
source=inspect.getsource(EU.eval_grasp);namespace=EU.__dict__.copy();exec(source[:source.index('    ## collision detection')].replace('def eval_grasp(','def select_only(')+'\n    return grasp_list\n',namespace);select_only=namespace['select_only']
dataset=GraspNetDataset(str(DATA),split='train',camera='kinect',num_points=20000,voxel_size=.005,remove_outlier=True,augment=False,load_label=False)
oldroot=G/'eg_geometry_compact_v1_20261020';manifest={v['path']:v['sha256'] for v in json.load(open(R/'results/reference_evaluation_manifest.json'))}
frames=[0] if a.profile else list(range(256))[a.shard::a.shards];section='engineering_validation' if a.profile else 'development';out=R/f'evaluation/{section}/{a.scene:04d}';out.mkdir(parents=True,exist_ok=True);calls=0;reused=0;checks=[];completed=[]
for frame in frames:
 dest=out/f'{frame:04d}.json';assert not dest.exists()
 ep=P/f'B/exports/subset/scene_{a.scene:04d}/{frame:04d}.npz';jp=oldroot/f'evaluation/development/{a.scene:04d}/{frame:04d}.json';zp=jp.with_suffix('.npz')
 assert sha(jp)==manifest[str(jp)] and sha(zp)==manifest[str(zp)];old=json.load(open(jp));reference={v['configuration']:v for v in old}
 with np.load(ep) as z:pred=z['pred'];input_hash=str(z['input_sha256']);observation=str(z['raw_observation_sha256'])
 assert pred.shape==(1024,17) and sha(ep)==reference['G_20261018']['export_sha256'];geometry_hash=hashlib.sha256(np.ascontiguousarray(pred[:,1:]).tobytes()).hexdigest();assert geometry_hash==reference['G_20261018']['geometry_sha256']
 with np.load(zp) as z:mu=z['quality'].copy();collision=z['collision'].copy();empty=z['empty'].copy();oldids={v['configuration']:z['selected_'+v['configuration']] for v in old if v['family']=='C'}
 with np.load(G/f'eg_lg1_local_geometry_v1_20261020/evaluation/development/{a.scene:04d}/{frame:04d}.npz') as z:
  oldids.update({v['configuration']:z['selected_'+v['configuration']] for v in old if v['family']=='G'})
 sp=R/f'scores/{a.scene:04d}/{frame:04d}.npz'
 with np.load(sp) as z:
  assert str(z['export_sha256'])==sha(ep);assert str(z['config_sha256'])==sha(R/'frozen_config.json');q={f'{f}_{s}':z[f'{f}_{s}'] for f in ['SF','H'] for s in SEEDS}
 rows={};fallback={}
 for name,v in q.items():
  score,info=fused_score(pred[:,0],v);rows[name]=replace_score(pred,score);fallback[name]=info['fallback']
 native.frame(frame);owners=EU.compute_closest_points(pred[:,13:16].astype(np.float64),native.objpts);lookup_owner={p.tobytes():int(j) for p,j in zip(pred[:,13:16].astype(np.float64),owners)}
 def cached_closest(A,B):
  assert np.array_equal(B,native.objpts);return np.array([lookup_owner[np.asarray(v,dtype=np.float64).tobytes()] for v in A])
 namespace['compute_closest_points']=cached_closest
 folder=DATA/f'scenes/scene_{a.scene:04d}/kinect'
 for attr,sub,ext in [('colorpath','rgb','png'),('depthpath','depth','png'),('labelpath','label','png'),('metapath','meta','mat')]:getattr(dataset,attr)[0]=str(folder/f'{sub}/{frame:04d}.{ext}')
 dataset.scenename[0]=f'scene_{a.scene:04d}';dataset.frameid[0]=frame;cloud,_=dataset.get_data(0,return_raw_cloud=True);detector=ModelFreeCollisionDetector(cloud,voxel_size=.005);cd=np.concatenate([detector.detect(GraspGroup(pred[i:i+64].copy()),approach_dist=.05,collision_thresh=.01) for i in range(0,1024,64)])
 assert float(cd.mean())==reference['G_20261018']['cd_rejection_rate']
 selected={};submitted={};native_rows={}
 for name,raw in rows.items():
  assert np.array_equal(raw[:,1:],pred[:,1:]);clipped=raw.copy();clipped[:,1]=clipped[:,1].clip(0,.1);lookup={}
  for i,g in enumerate(clipped):lookup.setdefault(g.tobytes(),[]).append(i)
  for use_cd in [False,True]:
   tag=name+('_CD' if use_cd else '');inp=clipped[~cd] if use_cd else clipped
   gs=np.concatenate(select_only(GraspGroup(inp.copy()),native.models,native.dex,native.poses,native.config,table=native.table_trans,voxel_size=.008,TOP_K=50)) if len(inp) else np.zeros((0,17))
   gs=gs[np.argsort(gs[:,0])[::-1]];ids=np.array([min(i for i in lookup[g.tobytes()] if not use_cd or not cd[i]) for g in gs],int);selected[tag]=ids;submitted[tag]=inp;native_rows[tag]=gs
 needed=np.unique(np.concatenate(list(selected.values())));missing=needed[~np.isfinite(mu[needed])];reused+=len(needed)-len(missing)
 if len(missing):mu[missing],collision[missing],empty[missing],_=native.score(pred[missing]);calls+=len(missing)
 if a.profile:
  assert np.array_equal(cd,detector.detect(GraspGroup(pred.copy()),approach_dist=.05,collision_thresh=.01))
  for name in ['SF_20261018','SF_20261018_CD','H_20261018','H_20261018_CD']:
   gs,qq,cc=native.official(submitted[name]);ids=selected[name];v=dict(selection=np.array_equal(gs,native_rows[name]),quality=np.array_equal(qq,mu[ids]),collision=np.array_equal(cc,collision[ids]),AP=np.array_equal(ap(qq),ap(mu[ids])));assert all(v.values()),(name,v);checks.append(dict(configuration=name,checks=v))
 values=[]
 for v in old:
  if v['family'] not in ['G','C']:continue
  vv=v.copy();ids=oldids[vv['configuration']];assert np.isfinite(mu[ids]).all();assert float(ap(mu[ids]).mean()*100)==v['ap'];vv['source_method_lock_sha256']=v['method_lock_sha256'];vv['method_lock_sha256']=sha(R/'method_lock.json');vv['reused_protected_result']=True;values.append(vv)
 for name,ids in selected.items():
  base=name.removesuffix('_CD');family,seed=base.split('_');matrix=ap(mu[ids]);n=min(50,len(ids));i=int(ids[0]) if len(ids) else None
  cat='no_candidate' if i is None else 'empty' if empty[i] else 'nonempty_collision' if collision[i] else 'valid' if 0<mu[i]<=.4000001 else 'weak_positive_quality' if mu[i]>0 else 'no_force_closure_in_evaluated_range'
  values.append(dict(scene=a.scene,frame=frame,configuration=name,family=family,seed=int(seed),use_cd=name.endswith('_CD'),ap=float(matrix.mean()*100),ap04=float(matrix[:,1].mean()*100),ap08=float(matrix[:,3].mean()*100),top1_valid=cat=='valid',top1_collision=bool(i is not None and collision[i]),top1_nonempty_collision=cat=='nonempty_collision',top1_empty=cat=='empty',top1_category=cat,top_candidate=i,top1_mu=float(mu[i]) if i is not None else None,top1_utility=float(utility(mu[i])) if i is not None else 0.,selected_top50_count=n,selected_top50_nonempty_collisions=int((collision[ids[:n]]&~empty[ids[:n]]).sum()),cd_rejection_rate=float(cd.mean()),F05_fallback=fallback[base],input_sha256=input_hash,raw_observation_sha256=observation,export_sha256=reference['G_20261018']['export_sha256'],geometry_sha256=geometry_hash,backbone_sha256=reference['G_20261018']['backbone_sha256'],method_lock_sha256=sha(R/'method_lock.json'),score_file_sha256=sha(sp),reused_protected_result=False))
 assert len(values)==24 and len({v['configuration'] for v in values})==24
 np.savez_compressed(out/f'{frame:04d}.npz',quality=mu,collision=collision,empty=empty,cd_mask=cd,**{'selected_'+k:v for k,v in selected.items()});write(dest,values);completed.append(frame)
 if len(completed)%8==0 or a.profile:print('EVAL',a.scene,a.shard,len(completed),len(frames),'new_quality',calls,'seconds',round(time.time()-start,2),flush=True)
receipt=dict(status='PASS',scene=a.scene,shard=a.shard,shards=a.shards,frames=completed,profile=a.profile,seconds=time.time()-start,cpu_seconds=time.process_time(),new_test_quality_calls=calls,reused_quality_calls=reused,native_checks=checks,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,poses_unchanged=True,protected_results_reused=True)
write(R/(f'results/eval_{a.scene:04d}_'+('profile.json' if a.profile else f'{a.shard}.json')),receipt)
print('EVAL_DONE',a.scene,a.shard,receipt['seconds'],flush=True)
