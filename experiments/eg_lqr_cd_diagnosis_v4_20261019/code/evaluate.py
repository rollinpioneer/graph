"""Frozen F05, unchanged native selection and full-geometry quality; no training."""
import argparse,time,inspect,importlib.util,resource
from common import *
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['exploration','confirmation'],required=True);p.add_argument('--scene',type=int,required=True);a=p.parse_args();scene=a.scene;start=time.time();os.nice(10)
explore=a.phase=='exploration'
if explore:assert json.loads((R/'results/diagnosis_gate.json').read_text())['execute_fixed_F05'];groups=CFG['exploration']
else:
 lock=json.loads((R/'C2_method_lock.json').read_text());assert lock['config_sha256']==sha256(R/'frozen_config.json');assert lock['reservation_sha256']==sha256(R/'C2_reservation.json');groups=lock['selected']
assert scene in sum([groups[s] for s in SPLITS],[])
sys.path[:0]=[str(CODE),str(CODE/'libs/pointnet2')];sys.argv=[sys.argv[0],'--dataset_root',str(DATA),'--camera','kinect','--model','economicgrasp','--batch_size','1']
from graspnetAPI import GraspGroup
import graspnetAPI.utils.eval_utils as EU
from dataset.graspnet_dataset import GraspNetDataset
from utils.collision_detector import ModelFreeCollisionDetector
from scoring import FrozenScorer,cache_inputs
spec=importlib.util.spec_from_file_location('v4_native_reference',G/'eg_nqs_pilot_v2_20261018/code/common.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);Native,ap=m.Native,m.ap
source=inspect.getsource(EU.eval_grasp);namespace=EU.__dict__.copy();exec(source[:source.index('    ## collision detection')].replace('def eval_grasp(','def select_only(')+'\n    return grasp_list\n',namespace);select_only=namespace['select_only'];native=Native(scene)
data=GraspNetDataset(str(DATA),split='train',camera='kinect',num_points=20000,voxel_size=.005,remove_outlier=True,augment=False,load_label=False)
heads={} if explore else {h['seed']:FrozenScorer(G/f'eg_pose_select_pilot_v1_20261018/heads/{h["seed"]}/step1000.pt',h['sha256']) for h in CFG['heads']}
out=R/f'evaluation/{a.phase}/{scene:04d}';out.mkdir(parents=True,exist_ok=True);newcalls=reused=0;checks=[];identities=[]
for frame in range(256):
 root=V if explore else R;export=root/f'exports/subset/scene_{scene:04d}/{frame:04d}.npz';pred,features,action,input_hash=cache_inputs(export,'lqr_v3')
 assert pred.shape==(1024,17)
 with np.load(export) as z:observation=str(z['raw_observation_sha256'])
 identity=dict(backbone_sha256=CFG['backbone_sha256'],scene=scene,frame=frame,input_sha256=input_hash,export_sha256=sha256(export),geometry_sha256=hashlib.sha256(np.ascontiguousarray(pred[:,1:]).tobytes()).hexdigest(),native_eval_source_sha256=hashlib.sha256(source.encode()).hexdigest(),open3d_version=__import__('open3d').__version__,raw_observation_sha256=observation)
 oldrecords=[];stored={}
 if explore:
  oldrecords=json.loads((V/f'evaluation/C1/{scene:04d}/{frame:04d}.json').read_text());assert all(all(row[k]==v for k,v in identity.items()) for row in oldrecords)
  with np.load(V/f'evaluation/C1/{scene:04d}/{frame:04d}.npz') as z:stored={k:z[k].copy() for k in z.files}
 qheads={s:stored[f'Q_{s}'] if explore else heads[s].quality(features,action) for s in SEEDS};rows={'R0':pred.astype(np.float64).copy()};fallback={}
 for seed,q in qheads.items():
  score,info=fused_score(pred[:,0],q);fallback[seed]=info;rows[f'F05_{seed}']=replace_score(pred,score);assert np.array_equal(rows[f'F05_{seed}'][:,1:],pred[:,1:])
 native.frame(frame)
 folder=DATA/f'scenes/scene_{scene:04d}/kinect'
 for attr,sub,ext in [('colorpath','rgb','png'),('depthpath','depth','png'),('labelpath','label','png'),('metapath','meta','mat')]:getattr(data,attr)[0]=str(folder/f'{sub}/{frame:04d}.{ext}')
 data.scenename[0]=f'scene_{scene:04d}';data.frameid[0]=frame;cloud,_=data.get_data(0,return_raw_cloud=True);cd=ModelFreeCollisionDetector(cloud,voxel_size=.005).detect(GraspGroup(pred.copy()),approach_dist=.05,collision_thresh=.01)
 selected={};submitted={};native_rows={}
 for name,raw in rows.items():
  clipped=raw.copy();clipped[:,1]=clipped[:,1].clip(0,.1);lookup={}
  for i,g in enumerate(clipped):lookup.setdefault(g.tobytes(),[]).append(i)
  for use_cd in [False,True]:
   namecd=name+('_CD' if use_cd else '');inp=clipped[~cd] if use_cd else clipped
   if len(inp):
    gs=np.concatenate(select_only(GraspGroup(inp.copy()),native.models,native.dex,native.poses,native.config,table=native.table_trans,voxel_size=.008,TOP_K=50));gs=gs[np.argsort(gs[:,0])[::-1]]
   else:gs=np.zeros((0,17))
   selected[namecd]=np.array([min(i for i in lookup[g.tobytes()] if not use_cd or not cd[i]) for g in gs],int);submitted[namecd]=inp;native_rows[namecd]=gs
   if explore and name=='R0':assert np.array_equal(selected[namecd],stored['selected_'+namecd])
 quality=stored.get('quality',np.full(1024,np.nan)).copy();collision=stored.get('collision',np.zeros(1024,bool)).copy();empty=stored.get('empty',np.zeros(1024,bool)).copy();cache=out/f'{frame:04d}.npz'
 if cache.exists():
  with np.load(cache) as z:
   assert json.loads(str(z['identity']))==identity;quality=z['quality'].copy();collision=z['collision'].copy();empty=z['empty'].copy()
 requested=np.unique(np.concatenate(list(selected.values())));missing=requested[~np.isfinite(quality[requested])];reused+=len(requested)-len(missing)
 if len(missing):quality[missing],collision[missing],empty[missing],_=native.score(pred[missing]);newcalls+=len(missing)
 if explore and (scene,frame) in [(136,0),(102,0)]:
  for name in selected:
   gs,q,c=native.official(submitted[name]);ids=selected[name];check={'selection':np.array_equal(gs,native_rows[name]),'quality':np.array_equal(q,quality[ids]),'collision':np.array_equal(c,collision[ids]),'AP':np.array_equal(ap(q),ap(quality[ids]))};assert all(check.values()),(name,check);checks.append(dict(scene=scene,frame=frame,configuration=name,checks=check))
 records=[x for x in oldrecords if x['family']=='S']
 for name,ids in selected.items():
  matrix=ap(quality[ids]);base=name.removesuffix('_CD');seed=int(base.split('_')[1]) if '_' in base else 0;family=base.split('_')[0];n=min(50,len(ids))
  record=dict(configuration=name,family=family,seed=seed,use_cd=name.endswith('_CD'),ap=float(matrix.mean()*100),ap04=float(matrix[:,1].mean()*100),ap08=float(matrix[:,3].mean()*100),top1_valid=bool(len(ids) and 0<quality[ids[0]]<=.4000001),top1_collision=bool(len(ids) and collision[ids[0]]),top1_empty=bool(len(ids) and empty[ids[0]]),selected_top50_count=n,selected_top50_collisions=int(collision[ids[:n]].sum()),top_candidate=int(ids[0]) if len(ids) else None,cd_rejection_rate=float(cd.mean()),F05_fallback=bool(fallback.get(seed,{}).get('fallback',False)),**identity)
  if explore and family=='R0':
   original=next(x for x in oldrecords if x['configuration']==name)
   for key in ['ap','ap04','ap08','top1_valid','top1_collision','top1_empty','top_candidate','cd_rejection_rate']:assert record[key]==original[key],(scene,frame,name,key)
  records.append(record)
 assert len(records)==(14 if explore else 8)
 write(str((out/f'{frame:04d}.json').relative_to(R)),records)
 saved={k:v for k,v in stored.items() if k.startswith('selected_')};saved.update({'selected_'+k:v for k,v in selected.items()})
 np.savez_compressed(cache,quality=quality,collision=collision,empty=empty,identity=np.array(json.dumps(identity,sort_keys=True)),**saved,**{f'Q_{s}':q for s,q in qheads.items()});identities.append(identity)
 if frame%32==0:print('EVAL',a.phase,scene,frame,'new_calls',newcalls,'wall',time.time()-start,flush=True)
write(str((out/'COMPLETE.json').relative_to(R)),dict(status='PASS',phase=a.phase,scene=scene,frames=list(range(256)),configurations_per_frame=14 if explore else 8,seconds=time.time()-start,cpu_seconds=time.process_time(),new_quality_calls=newcalls,reused_quality_calls=reused,native_exact_checks=checks,source_identities=identities,geometry_unchanged=True,rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print('EVAL_COMPLETE',a.phase,scene,time.time()-start,flush=True)
