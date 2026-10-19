from common import *
import argparse,inspect,importlib.util,resource
p=argparse.ArgumentParser();p.add_argument('--scene',type=int,required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--profile',action='store_true');a=p.parse_args();os.nice(10);start=time.time()
cfg=json.loads((R/'frozen_config.json').read_text());lock=json.loads((R/'P2_method_lock.json').read_text());assert lock['frozen_config_sha256']==sha(R/'frozen_config.json');assert a.scene in sum(cfg['C3'].values(),[])
sys.path[:0]=[str(CODE),str(CODE/'libs/pointnet2')];sys.argv=[sys.argv[0],'--dataset_root',str(DATA),'--camera','kinect','--model','economicgrasp','--batch_size','1']
from graspnetAPI import GraspGroup
import graspnetAPI.utils.eval_utils as EU
from dataset.graspnet_dataset import GraspNetDataset
from utils.collision_detector import ModelFreeCollisionDetector
spec=importlib.util.spec_from_file_location('unchanged_native',G/'eg_nqs_pilot_v2_20261018/code/common.py');native_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(native_module);native=native_module.Native(a.scene);ap=native_module.ap
source=inspect.getsource(EU.eval_grasp);namespace=EU.__dict__.copy();exec(source[:source.index('    ## collision detection')].replace('def eval_grasp(','def select_only(')+'\n    return grasp_list\n',namespace);select_only=namespace['select_only']
dataset=GraspNetDataset(str(DATA),split='train',camera='kinect',num_points=20000,voxel_size=.005,remove_outlier=True,augment=False,load_label=False)
heads={(v['family'],v['seed']):FrozenScorer(Path(v['path']),v['sha256']) for v in lock['new_heads']}
source_hashes={v['path']:v['sha256'] for v in json.loads((P/'results/B_artifact_manifest.json').read_text())}
frames=[0] if a.profile else cfg['development_frames'][a.shard::a.shards];out=R/f'evaluation/development/{a.scene:04d}';out.mkdir(parents=True,exist_ok=True);calls=0;reused=0;checks=[];completed=[]
for frame in frames:
 target=out/f'{frame:04d}.json'
 if target.exists():
  vv=json.loads(target.read_text());assert len(vv)==20 and all(v['method_lock_sha256']==sha(R/'P2_method_lock.json') for v in vv);completed.append(frame);continue
 ep=P/f'B/exports/subset/scene_{a.scene:04d}/{frame:04d}.npz';jp=P/f'B/evaluation/confirmation/{a.scene:04d}/{frame:04d}.json';zp=jp.with_suffix('.npz');old={v['configuration']:v for v in json.loads(jp.read_text())}
 assert sha(jp)==source_hashes[str(jp)] and sha(zp)==source_hashes[str(zp)]
 with np.load(ep) as z:pred=z['pred'];feat=z['feature320'];action=z['original_action'];input_hash=str(z['input_sha256']);observation=str(z['raw_observation_sha256'])
 assert pred.shape==(1024,17) and sha(ep)==old['R0']['export_sha256'];assert input_hash==old['R0']['input_sha256'] and observation==old['R0']['raw_observation_sha256'];assert hashlib.sha256(source.encode()).hexdigest()==old['R0']['native_eval_source_sha256']
 with np.load(zp) as z:
  mu=z['quality'].copy();collision=z['collision'].copy();empty=z['empty'].copy();old_selected={k.removeprefix('selected_'):z[k] for k in z.files if k.startswith('selected_')};oq={s:z[f'Q_{s}'] for s in SEEDS}
 geometry_hash=hashlib.sha256(np.ascontiguousarray(pred[:,1:]).tobytes()).hexdigest();assert geometry_hash==old['R0']['geometry_sha256']
 rows={'EG':pred.astype(np.float64).copy()};fallback={};Q={}
 for s,q in oq.items():
  name=f'O_{s}';score,info=fused_score(pred[:,0],q);rows[name]=replace_score(pred,score);fallback[name]=info['fallback'];Q[name]=q
 for (family,s),head in heads.items():
  name=f'{family}_{s}';q=head.quality(feat,action);score,info=fused_score(pred[:,0],q);rows[name]=replace_score(pred,score);fallback[name]=info['fallback'];Q[name]=q
 native.frame(frame);folder=DATA/f'scenes/scene_{a.scene:04d}/kinect'
 for attr,sub,ext in [('colorpath','rgb','png'),('depthpath','depth','png'),('labelpath','label','png'),('metapath','meta','mat')]:getattr(dataset,attr)[0]=str(folder/f'{sub}/{frame:04d}.{ext}')
 dataset.scenename[0]=f'scene_{a.scene:04d}';dataset.frameid[0]=frame;cloud,_=dataset.get_data(0,return_raw_cloud=True);cd=ModelFreeCollisionDetector(cloud,voxel_size=.005).detect(GraspGroup(pred.copy()),approach_dist=.05,collision_thresh=.01);assert float(cd.mean())==old['R0']['cd_rejection_rate']
 selected={};submitted={};native_rows={}
 for name,raw in rows.items():
  assert np.array_equal(raw[:,1:],pred[:,1:]);clipped=raw.copy();clipped[:,1]=clipped[:,1].clip(0,.1);lookup={}
  for i,g in enumerate(clipped):lookup.setdefault(g.tobytes(),[]).append(i)
  for use_cd in [False,True]:
   tag=name+('_CD' if use_cd else '');inp=clipped[~cd] if use_cd else clipped
   if len(inp):
    gs=np.concatenate(select_only(GraspGroup(inp.copy()),native.models,native.dex,native.poses,native.config,table=native.table_trans,voxel_size=.008,TOP_K=50));gs=gs[np.argsort(gs[:,0])[::-1]]
   else:gs=np.zeros((0,17))
   ids=np.array([min(i for i in lookup[g.tobytes()] if not use_cd or not cd[i]) for g in gs],int);selected[tag]=ids;submitted[tag]=inp;native_rows[tag]=gs
   if name=='EG' or name.startswith('O_'):
    old_name=('R0' if name=='EG' else name.replace('O_','F05_'))+('_CD' if use_cd else '');assert np.array_equal(ids,old_selected[old_name])
 needed=np.unique(np.concatenate(list(selected.values())));missing=needed[~np.isfinite(mu[needed])];reused+=len(needed)-len(missing)
 if len(missing):mu[missing],collision[missing],empty[missing],_=native.score(pred[missing]);calls+=len(missing)
 if a.profile:
  for name in ['EG_CD','O_20261018_CD','M_20261018_CD','R_20261018_CD']:
   gs,q,c=native.official(submitted[name]);ids=selected[name];check=dict(selection=np.array_equal(gs,native_rows[name]),quality=np.array_equal(q,mu[ids]),collision=np.array_equal(c,collision[ids]),AP=np.array_equal(ap(q),ap(mu[ids])));assert all(check.values()),(name,check);checks.append(dict(configuration=name,**check))
 values=[]
 for name,ids in selected.items():
  base=name.removesuffix('_CD');family=base.split('_')[0];seed=int(base.split('_')[1]) if '_' in base else 0;matrix=ap(mu[ids]);n=min(50,len(ids));i=int(ids[0]) if len(ids) else None
  cat='no_candidate' if i is None else 'empty' if empty[i] else 'nonempty_collision' if collision[i] else 'valid' if 0<mu[i]<=.4000001 else 'weak_positive_quality' if mu[i]>0 else 'no_force_closure_in_evaluated_range'
  v=dict(scene=a.scene,frame=frame,configuration=name,family=family,seed=seed,use_cd=name.endswith('_CD'),ap=float(matrix.mean()*100),ap04=float(matrix[:,1].mean()*100),ap08=float(matrix[:,3].mean()*100),top1_valid=cat=='valid',top1_collision=bool(i is not None and collision[i]),top1_nonempty_collision=cat=='nonempty_collision',top1_empty=cat=='empty',top1_category=cat,top_candidate=i,top1_mu=float(mu[i]) if i is not None else None,selected_top50_count=n,selected_top50_quality_mean_utility=float(utility(mu[ids[:n]]).mean()) if n else 0.,selected_top50_collisions=int(collision[ids[:n]].sum()),selected_top50_nonempty_collisions=int((collision[ids[:n]]&~empty[ids[:n]]).sum()),cd_rejection_rate=float(cd.mean()),F05_fallback=fallback.get(base,False),input_sha256=input_hash,raw_observation_sha256=observation,export_sha256=old['R0']['export_sha256'],geometry_sha256=geometry_hash,backbone_sha256=old['R0']['backbone_sha256'],method_lock_sha256=sha(R/'P2_method_lock.json'))
  if family in ['EG','O']:
   old_name=('R0' if family=='EG' else name.replace('O_','F05_'))+('_CD' if family=='EG' and name.endswith('_CD') else '');reference=old[old_name]
   for key in ['ap','ap04','ap08','top1_valid','top1_collision','top1_empty','top1_nonempty_collision','top_candidate']:assert v[key]==reference[key],(name,key)
  values.append(v)
 assert len(values)==20 and all(np.isfinite(mu[ids]).all() for ids in selected.values())
 np.savez_compressed(out/f'{frame:04d}.npz',quality=mu,collision=collision,empty=empty,**{'selected_'+k:v for k,v in selected.items()},**{'Q_'+k:v for k,v in Q.items()});write(target,values);completed.append(frame);print('FRAME_DONE',a.scene,frame,'new_native',calls,'seconds',time.time()-start,flush=True)
receipt=dict(status='PASS',scene=a.scene,frames=completed,profile=a.profile,shard=a.shard,shards=a.shards,seconds=time.time()-start,cpu_seconds=time.process_time(),new_quality_calls=calls,reused_quality_calls=reused,native_checks=checks,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,geometry_unchanged=True,old_all_selection_and_metrics_exact=True)
write(R/(f'results/eval_{a.scene:04d}_'+('profile.json' if a.profile else f'shard{a.shard}.json')),receipt);print('EVAL_DONE',receipt,flush=True)
