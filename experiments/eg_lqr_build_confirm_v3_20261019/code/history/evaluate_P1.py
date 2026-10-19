"""Original-geometry R0/S/F05, full native selection, cached exact qualities."""
import os,sys,json,time,inspect,argparse,importlib.util,hashlib,resource
from pathlib import Path
import numpy as np
from scoring import FrozenScorer,cache_inputs,sha256,replace_score,fused_score

G=Path(os.environ['GRASP_ROOT']);R=Path(os.environ['LQR_ROOT']);N=G/'eg_nqs_pilot_v2_20261018';OLD=G/'eg_sub25_v1_20261015';DATA=OLD/'data/graspnet';CODE=OLD/'runtime/EconomicGrasp'
cfg=json.loads((R/'frozen_config.json').read_text());p=argparse.ArgumentParser();p.add_argument('--stage',choices=['P0','P1','C1'],required=True);p.add_argument('--scene',type=int,required=True);p.add_argument('--frames');args=p.parse_args();os.nice(10);start=time.time();cpu=time.process_time()
if args.stage=='P0':frames=[f for s,f in cfg['p0_frames'] if s==args.scene];families=['S','F05']
elif args.stage=='P1':
    assert json.loads((R/'results/preflight_complete.json').read_text())['status']=='PASS'
    assert args.scene in sum([cfg['P1'][x] for x in ['seen','similar','novel']],[]);frames=list(range(0,256,8));families=['S','F05']
else:
    lock=json.loads((R/'method_lock.json').read_text());audit=json.loads((R/'results/C1_exposure_audit.json').read_text());assert audit['status']=='PASS_NO_KNOWN_EXPOSURE';assert args.scene in sum([cfg['C1'][x] for x in ['seen','similar','novel']],[]);families=[lock['family']];frames=list(range(256))
if args.frames:
    assert args.stage=='P0';frames=list(map(int,args.frames.split(',')))
sys.path[:0]=[str(CODE),str(CODE/'libs/pointnet2')];sys.argv=[sys.argv[0],'--dataset_root',str(DATA),'--camera','kinect','--model','economicgrasp','--batch_size','1']
from graspnetAPI import GraspGroup
import graspnetAPI.utils.eval_utils as EU
from dataset.graspnet_dataset import GraspNetDataset
from utils.collision_detector import ModelFreeCollisionDetector
spec=importlib.util.spec_from_file_location('lqr_native_reference',N/'code/common.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);Native,ap=module.Native,module.ap
source=inspect.getsource(EU.eval_grasp);native_hash=hashlib.sha256(source.encode()).hexdigest();namespace=EU.__dict__.copy();exec(source[:source.index('    ## collision detection')].replace('def eval_grasp(','def select_only(')+'\n    return grasp_list\n',namespace);select_only=namespace['select_only'];native=Native(args.scene)
data=GraspNetDataset(str(DATA),split='train',camera='kinect',num_points=20000,voxel_size=.005,remove_outlier=True,augment=False,load_label=False)
heads={h['seed']:FrozenScorer(G/f'eg_pose_select_pilot_v1_20261018/heads/{h["seed"]}/step1000.pt',h['sha256']) for h in cfg['heads']};folder=R/f'evaluation/{args.stage}/{args.scene:04d}';folder.mkdir(parents=True,exist_ok=True);new_count=reused_count=0;micro=[];identities=[];startup_seconds=time.time()-start;verification_seconds=0
for frame in frames:
    source_path=(R if args.stage=='C1' else N)/f'exports/subset/scene_{args.scene:04d}/{frame:04d}.npz';layout='lqr_v3' if args.stage=='C1' else 'nqs_v1';pred,feature,action,input_hash=cache_inputs(source_path,layout)
    assert pred.shape==(1024,17) and input_hash,'Missing complete frozen input identity'
    export_hash=sha256(source_path);geometry_hash=hashlib.sha256(np.ascontiguousarray(pred[:,1:]).tobytes()).hexdigest()
    identity=dict(backbone_sha256=cfg['backbone_sha256'],scene=args.scene,frame=frame,input_sha256=input_hash,export_sha256=export_hash,geometry_sha256=geometry_hash,native_eval_source_sha256=native_hash,open3d_version=__import__('open3d').__version__)
    rows={'R0':pred.astype(np.float64).copy()};q_by_seed={};fallback={}
    for seed,head in heads.items():
        q=head.quality(feature,action);q_by_seed[seed]=q
        for family in families:
            if family=='S':score=q
            else:score,info=fused_score(pred[:,0],q);fallback[seed]=info
            name=f'{family}_{seed}';rows[name]=replace_score(pred,score);assert np.array_equal(rows[name][:,1:],pred[:,1:])
    native.frame(frame)
    for attr,sub,ext in [('colorpath','rgb','png'),('depthpath','depth','png'),('labelpath','label','png'),('metapath','meta','mat')]:getattr(data,attr)[0]=str(DATA/f'scenes/scene_{args.scene:04d}/kinect/{sub}/{frame:04d}.{ext}')
    data.scenename[0]=f'scene_{args.scene:04d}';data.frameid[0]=frame;cloud,_=data.get_data(0,return_raw_cloud=True);detector=ModelFreeCollisionDetector(cloud,voxel_size=.005);cd=detector.detect(GraspGroup(pred.copy()),approach_dist=.05,collision_thresh=.01)
    selected={};submitted={};native_rows={}
    for name,raw_rows in rows.items():
        assert np.array_equal(raw_rows[:,1:],pred[:,1:]);assert np.array_equal(cd,detector.detect(GraspGroup(raw_rows.copy()),approach_dist=.05,collision_thresh=.01)) if args.stage=='P0' else True
        clipped=raw_rows.copy();clipped[:,1]=clipped[:,1].clip(0,.1);lookup={}
        for i,g in enumerate(clipped):lookup.setdefault(g.tobytes(),[]).append(i)
        for use_cd in [False,True]:
            name_cd=name+('_CD' if use_cd else '');inp=clipped[~cd] if use_cd else clipped
            if len(inp):
                groups=select_only(GraspGroup(inp.copy()),native.models,native.dex,native.poses,native.config,table=native.table_trans,voxel_size=.008,TOP_K=50);gs=np.concatenate(groups);gs=gs[np.argsort(gs[:,0])[::-1]]
            else:gs=np.zeros((0,17))
            selected[name_cd]=np.array([min(i for i in lookup[g.tobytes()] if not use_cd or not cd[i]) for g in gs],int);submitted[name_cd]=inp;native_rows[name_cd]=gs
    quality=np.full(1024,np.nan);collision=np.zeros(1024,bool);empty=collision.copy();cache_path=folder/f'{frame:04d}.npz'
    if cache_path.exists():
        with np.load(cache_path) as old:
            assert json.loads(str(old['identity']))==identity;quality=old['quality'].copy();collision=old['collision'].copy();empty=old['empty'].copy()
    requested=np.unique(np.concatenate(list(selected.values())));missing=requested[~np.isfinite(quality[requested])];reused_count+=len(requested)-len(missing)
    if len(missing):quality[missing],collision[missing],empty[missing],_=native.score(pred[missing]);new_count+=len(missing)
    if args.stage=='P0':
        verify_start=time.perf_counter()
        for name in selected:
            gs,q,c=native.official(submitted[name]);ids=selected[name];check=dict(selection=np.array_equal(gs,native_rows[name]),quality=np.array_equal(q,quality[ids]),collision=np.array_equal(c,collision[ids]),AP=np.array_equal(ap(q),ap(quality[ids])));assert all(check.values()),(name,check);micro.append(dict(frame=frame,configuration=name,checks=check))
        verification_seconds+=time.perf_counter()-verify_start
    records=[]
    for name,ids in selected.items():
        q=quality[ids];matrix=ap(q);m=min(50,len(ids));base=name.removesuffix('_CD');seed=int(base.split('_')[1]) if '_' in base else 0;family=base.split('_')[0]
        record=dict(configuration=name,family=family,seed=seed,use_cd=name.endswith('_CD'),ap=float(matrix.mean()*100),ap04=float(matrix[:,1].mean()*100),ap08=float(matrix[:,3].mean()*100),top1_valid=bool(len(q) and 0<q[0]<=.4000001),top1_collision=bool(len(ids) and collision[ids[0]]),top1_empty=bool(len(ids) and empty[ids[0]]),selected_top50_count=m,selected_top50_collisions=int(collision[ids[:m]].sum()),top_candidate=int(ids[0]) if len(ids) else None,cd_rejection_rate=float(cd.mean()),F05_fallback=bool(fallback.get(seed,{}).get('fallback',False)) if family=='F05' else False,**identity)
        if args.stage=='P0' and family in ['R0','S']:
            previous=json.loads((G/f'eg_score_pose_2x2_v1_20261019/evaluation/development/{args.scene:04d}/{frame:04d}.json').read_text());old=next(x for x in previous if x['configuration']==name)
            for key in ['ap','ap04','ap08','top1_valid','top1_collision','top1_empty','top_candidate']:assert record[key]==old[key],(args.scene,frame,name,key,record[key],old[key])
        records.append(record)
    (folder/f'{frame:04d}.json').write_text(json.dumps(records,indent=2)+'\n');np.savez_compressed(cache_path,quality=quality,collision=collision,empty=empty,identity=np.array(json.dumps(identity,sort_keys=True)),**{f'selected_{k}':v for k,v in selected.items()},**{f'Q_{s}':q for s,q in q_by_seed.items()});identities.append(identity)
    print('FRAME',args.stage,args.scene,frame,'new',len(missing),'wall',time.time()-start,flush=True)
result=dict(status='PASS',stage=args.stage,scene=args.scene,frames=frames,configuration_count=len(rows)*2,seconds=time.time()-start,startup_seconds=startup_seconds,reference_verification_seconds=verification_seconds,production_seconds=time.time()-start-startup_seconds-verification_seconds,cpu_seconds=time.process_time()-cpu,new_quality_calls=new_count,reused_quality_calls=reused_count,source_identities=identities,native_exact_checks=micro,geometry_unchanged=True,CD_common_geometry=True,rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(folder/'COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',args.stage,args.scene,result['seconds'],flush=True)
