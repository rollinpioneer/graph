from common import *
import argparse,inspect,importlib.util,resource
import torch
torch.set_num_threads(1)
p=argparse.ArgumentParser();p.add_argument('--scene',type=int,required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=8);p.add_argument('--pilot',action='store_true');a=p.parse_args();start=time.time();os.nice(10)
P0=G/'eg_role_p0_audit_v1_20261021';L=G/'eg_lg1_local_geometry_v1_20261020';cfg=json.loads((R/'frozen_config.json').read_text());assert a.scene in sum(cfg['C3'].values(),[])
lock=json.loads((R/'method_lock.json').read_text());assert lock['config_sha256']==sha(R/'frozen_config.json')
assert lock['evaluation_code_sha256']==sha(Path(__file__))
assert lock['P0_union_manifest_sha256']==sha(R/'results/P0_union_manifest.json')
union_manifest={v['path']:v['sha256'] for v in json.loads((R/'results/P0_union_manifest.json').read_text())}
manifest={v['path']:v['sha256'] for v in json.loads((P0/'P0_input_manifest.json').read_text())};assert sha(P0/'P0_input_manifest.json')==cfg['P0_input_manifest_sha256']
def checked(p):
 assert str(p) in manifest,('not_in_manifest',str(p));assert sha(p)==manifest[str(p)],str(p)
heads={f"{h['family']}_{h['seed']}":GeoScorer(Path(h['path']),h['sha256']) for h in lock['heads']}
sys.path[:0]=[str(CODE),str(CODE/'libs/pointnet2')];sys.argv=[sys.argv[0],'--dataset_root',str(DATA),'--camera','kinect','--model','economicgrasp','--batch_size','1']
from graspnetAPI import GraspGroup
import graspnetAPI.utils.eval_utils as EU
spec=importlib.util.spec_from_file_location('native_reference',G/'eg_nqs_pilot_v2_20261018/code/common.py');nm=importlib.util.module_from_spec(spec);spec.loader.exec_module(nm);native=nm.Native(a.scene);ap=nm.ap
source=inspect.getsource(EU.eval_grasp);namespace=EU.__dict__.copy();exec(source[:source.index('    ## collision detection')].replace('def eval_grasp(','def select_only(')+'\n    return grasp_list\n',namespace);select_only=namespace['select_only']
frames=[0] if a.pilot else list(range(256))[a.shard::a.shards];out=R/f"evaluation/{'pilot' if a.pilot else 'development'}/{a.scene:04d}";out.mkdir(parents=True,exist_ok=True);calls=0;checks=[];done=[]
for frame in frames:
 target=out/f'{frame:04d}.json';assert not target.exists(),target
 ep=P/f'B/exports/subset/scene_{a.scene:04d}/{frame:04d}.npz';checked(ep)
 with np.load(ep) as z:pred=z['pred'];feat=z['feature320'];act=z['original_action'];gh=str(z['geometry_sha256']);assert gh==hashlib.sha256(np.ascontiguousarray(pred[:,1:]).tobytes()).hexdigest()
 jp=L/f'evaluation/development/{a.scene:04d}/{frame:04d}.json';zp=jp.with_suffix('.npz');checked(jp);checked(zp);old={v['configuration']:v for v in json.loads(jp.read_text())};assert all(v['export_sha256']==manifest[str(ep)] and v['geometry_sha256']==gh for v in old.values())
 with np.load(zp) as z:qs={f'{f}_{s}':z[f'Q_{f}_{s}'] for f in ['O','G'] for s in SEEDS};oldids={k.removeprefix('selected_'):z[k] for k in z.files if k.startswith('selected_')}
 up=P0/f'union/{a.scene:04d}/{frame:04d}.npz';assert sha(up)==union_manifest[str(up)]
 with np.load(up) as z:mu=z['quality'].copy();collision=z['collision'].copy();empty=z['empty'].copy();cd=z['cd_mask']
 gp=L/f'geometry/C3/{a.scene:04d}/{frame:04d}.npz';checked(gp)
 with np.load(gp) as z:raw=z['raw'][:,0];assert str(z['export_sha256'])==manifest[str(ep)]
 for name,head in heads.items():qs[name]=head.quality(feat,act,raw,a.scene,frame)
 scores={'EG':pred[:,0].astype(float)}
 for name,q in qs.items():scores[name]=fused_score(pred[:,0],q)[0]
 native.frame(frame);owners=np.concatenate([EU.compute_closest_points(pred[k:k+64,13:16].astype(float),native.objpts) for k in range(0,1024,64)]);owner_lookup={p.tobytes():int(j) for p,j in zip(pred[:,13:16].astype(float),owners)}
 namespace['compute_closest_points']=lambda A,B:np.array([owner_lookup[np.asarray(v,dtype=float).tobytes()] for v in A])
 selected={};submitted={};native_rows={}
 for name,score in scores.items():
  rows=replace_score(pred,score);rows[:,1]=rows[:,1].clip(0,.1);lookup={}
  for i,g in enumerate(rows):lookup.setdefault(g.tobytes(),[]).append(i)
  for use_cd in [False,True]:
   tag=name+('_CD' if use_cd else '');inp=rows[~cd] if use_cd else rows
   gs=np.concatenate(select_only(GraspGroup(inp.copy()),native.models,native.dex,native.poses,native.config,table=native.table_trans,voxel_size=.008,TOP_K=50)) if len(inp) else np.zeros((0,17));gs=gs[np.argsort(gs[:,0])[::-1]]
   ids=np.array([min(i for i in lookup[g.tobytes()] if not use_cd or not cd[i]) for g in gs],int);selected[tag]=ids;submitted[tag]=inp;native_rows[tag]=gs
   if tag in old and name.split('_')[0] in ['EG','O','G']:assert np.array_equal(ids,oldids[tag]),('selection',a.scene,frame,tag)
 needed=np.unique(np.concatenate(list(selected.values())));missing=needed[~np.isfinite(mu[needed])]
 if len(missing):mu[missing],collision[missing],empty[missing],_=native.score(pred[missing]);calls+=len(missing)
 if a.pilot:
  for name in ['EG_CD','O_20261018_CD','G_20261018_CD','GW_20261018_CD','GW_20261018']:
   gs,q,c=native.official(submitted[name]);ids=selected[name];v=dict(selection=np.array_equal(gs,native_rows[name]),quality=np.array_equal(q,mu[ids]),collision=np.array_equal(c,collision[ids]),AP=np.array_equal(ap(q),ap(mu[ids])));assert all(v.values()),(name,v);checks.append(dict(configuration=name,**v))
 values=[]
 for name,ids in selected.items():
  base=name.removesuffix('_CD');family=base.split('_')[0];seed=int(base.split('_')[1]) if '_' in base else 0;mat=ap(mu[ids]);i=int(ids[0]) if len(ids) else None
  cat='no_candidate' if i is None else 'empty' if empty[i] else 'nonempty_collision' if collision[i] else 'valid' if 0<mu[i]<=.4000001 else 'weak_positive_quality' if mu[i]>0 else 'no_force_closure_in_evaluated_range'
  v=dict(scene=a.scene,frame=frame,configuration=name,family=family,seed=seed,use_cd=name.endswith('_CD'),ap=float(mat.mean()*100),ap04=float(mat[:,1].mean()*100),ap08=float(mat[:,3].mean()*100),top1_valid=cat=='valid',top1_nonempty_collision=cat=='nonempty_collision',top1_empty=cat=='empty',no_output=i is None,top1_category=cat,top_candidate=i,top1_mu=float(mu[i]) if i is not None else None,export_sha256=manifest[str(ep)],geometry_sha256=gh,method_lock_sha256=sha(R/'method_lock.json'))
  if family in ['EG','O','G']:
   for key in ['ap','ap04','ap08','top1_valid','top1_nonempty_collision','top1_empty','top_candidate']:assert v[key]==old[name][key],(name,key)
  values.append(v)
 np.savez_compressed(out/f'{frame:04d}.npz',quality=mu,collision=collision,empty=empty,**{'selected_'+k:v for k,v in selected.items()});write(target,values);done.append(frame)
 if a.pilot or len(done)%8==0:print('FRAME',a.scene,a.shard,len(done),'new_labels',calls,'seconds',time.time()-start,flush=True)
receipt=dict(status='PASS',scene=a.scene,shard=a.shard,pilot=a.pilot,frames=done,seconds=time.time()-start,cpu_seconds=time.process_time(),start=start,end=time.time(),pid=os.getpid(),host=os.uname().nodename,new_evaluation_labels=calls,official_checks=checks,protected_selection_metrics_exact=True,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
write(R/f"results/eval_{a.scene:04d}_{'pilot' if a.pilot else a.shard}.json",receipt);print('DONE',json.dumps(receipt),flush=True)
