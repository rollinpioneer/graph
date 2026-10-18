"""Native selection for each ranking; cache quality only after native-path checks."""
import os,sys,argparse,time,json,inspect,resource
from pathlib import Path
import numpy as np
from common import *
p=argparse.ArgumentParser();p.add_argument('--scene',type=int,required=True);p.add_argument('--role',default='subset');p.add_argument('--validate-only',action='store_true');a=p.parse_args()
sys.path.insert(0,str(CODE));sys.path.insert(0,str(CODE/'libs/pointnet2'))
sys.argv=[sys.argv[0],'--dataset_root',str(DATA),'--camera','kinect','--model','economicgrasp','--batch_size','1']
import torch
from torch import nn
from graspnetAPI import GraspGroup
import graspnetAPI.utils.eval_utils as EU
from dataset.graspnet_dataset import GraspNetDataset
from utils.collision_detector import ModelFreeCollisionDetector
from scipy.stats import spearmanr
os.nice(10);torch.set_num_threads(1);start=time.time()
source=inspect.getsource(EU.eval_grasp);prefix=source[:source.index('    ## collision detection')]
ns=EU.__dict__.copy();exec(prefix.replace('def eval_grasp(', 'def select_only(')+'\n    return grasp_list\n',ns)
select_only=ns['select_only'];native=Native(a.scene)
data=GraspNetDataset(str(DATA),split='train',camera='kinect',num_points=20000,voxel_size=.005,remove_outlier=True,augment=False,load_label=False)
models={}
for arm in ['E','B6','A']:
    for seed_ in [20261018,20261019,20261020]:
        checkpoint=torch.load(R/f'heads/{a.role}/{arm}_{seed_}/step1000.pt',map_location='cpu',weights_only=False)
        net=nn.Sequential(nn.Linear(324,128),nn.ReLU(),nn.Linear(128,1),nn.Sigmoid()).eval();net.load_state_dict(checkpoint['model'])
        models[f'{arm}_{seed_}']=(net,checkpoint['mean'],checkpoint['std'])
validated=R/f'receipts/eval_cache_validation_{a.role}.json'
if not a.validate_only:assert validated.exists() and json.loads(validated.read_text())['status']=='PASS'
all_records=[];validation=[];calls=0
fr=frames(a.scene)[:4] if a.validate_only else frames(a.scene)
outdir=R/f'evaluation/{a.role}/scene_{a.scene:04d}';outdir.mkdir(parents=True,exist_ok=True)
for frame in fr:
    out=outdir/f'{frame:04d}.npz'
    if out.exists() and not a.validate_only:continue
    with np.load(R/f'exports/{a.role}/scene_{a.scene:04d}/{frame:04d}.npz') as z:pred=z['pred'].astype(np.float64);features=z['feature']
    pred[:,1]=pred[:,1].clip(0,.1);native.frame(frame)
    scores={'R0':pred[:,0]}
    for name,(model,mean,std) in models.items():
        x=features.copy();x[:,:320]=(x[:,:320]-mean)/std
        with torch.no_grad():scores[name]=model(torch.from_numpy(x)).squeeze(-1).numpy().astype(np.float64)
    if a.scene<100:cloud,_=data.get_data(a.scene*256+frame,return_raw_cloud=True)
    else:
        for attr,sub,ext in [('colorpath','rgb','png'),('depthpath','depth','png'),('labelpath','label','png'),('metapath','meta','mat')]:getattr(data,attr)[0]=str(DATA/f'scenes/scene_{a.scene:04d}/kinect/{sub}/{frame:04d}.{ext}')
        data.scenename[0]=f'scene_{a.scene:04d}';data.frameid[0]=frame;cloud,_=data.get_data(0,return_raw_cloud=True)
    detector=ModelFreeCollisionDetector(cloud,voxel_size=.005)
    cd=detector.detect(GraspGroup(pred.copy()),approach_dist=.05,collision_thresh=.01)
    # Scores may differ for identical geometry; native ties remain native ties.
    geom={}
    for i,g in enumerate(pred):geom.setdefault(g[1:].tobytes(),[]).append(i)
    selections={};scored={}
    for name,score_ in scores.items():
        for use_cd in [False,True]:
            key=name+('_CD' if use_cd else '');rows=pred.copy();rows[:,0]=score_;rows=rows[~cd] if use_cd else rows
            scored[key]=rows
            if len(rows):
                groups=select_only(GraspGroup(rows.copy()),native.models,native.dex,native.poses,native.config,table=native.table_trans,voxel_size=.008,TOP_K=50)
                gs=np.concatenate(groups);order=np.argsort(gs[:,0])[::-1];gs=gs[order]
            else:gs=np.zeros((0,17))
            ids=[]
            for g in gs:
                candidates=geom[g[1:].astype(np.float64).tobytes()]
                match=[i for i in candidates if score_[i]==g[0] and (not use_cd or not cd[i])]
                assert match;ids.append(min(match))
            selections[key]=np.asarray(ids,int)
    union=np.unique(np.concatenate(list(selections.values())))
    qs,cs,es,owners=native.score(pred[union]);calls+=len(qs)
    quality=np.full(1024,np.nan);collision=np.zeros(1024,bool);empty=collision.copy();quality[union]=qs;collision[union]=cs;empty[union]=es
    if a.validate_only:
        for name in ['R0','E_20261018','B6_20261018','A_20261018']:
            for suffix in ['','_CD']:
                key=name+suffix;gs,q,c=native.official(scored[key]);ix=selections[key]
                expected=pred[ix].copy();expected[:,0]=scores[name][ix]
                assert np.array_equal(gs,expected),(frame,key,'selection')
                assert np.array_equal(q,quality[ix]),(frame,key,'quality')
                assert np.array_equal(c,collision[ix]),(frame,key,'collision')
                assert np.array_equal(ap(q),ap(quality[ix]))
                validation.append(dict(frame=frame,configuration=key,candidates=len(ix),exact=True))
    packed={};rec=[]
    for key,ix in selections.items():
        name=key.removesuffix('_CD');q=quality[ix];mat=ap(q);pool=selections['R0_CD' if key.endswith('_CD') else 'R0']
        order=np.argsort(-scores[name][pool],kind='stable');fixed=pool[order];fixedmat=ap(quality[fixed])
        corr=float(spearmanr(scores[name][ix],utility(q)).statistic) if len(ix)>1 and np.ptp(utility(q))>0 and np.ptp(scores[name][ix])>0 else None
        record=dict(role=a.role,scene=a.scene,frame=frame,configuration=key,ap=float(mat.mean()*100),ap04=float(mat[:,1].mean()*100),ap08=float(mat[:,3].mean()*100),fixed_pool_ap=float(fixedmat.mean()*100),n=len(ix),top1_valid=bool(len(q) and 0<q[0]<=.4),top1_collision=bool(len(q) and collision[ix[0]]),top1_empty=bool(len(q) and empty[ix[0]]),spearman=corr)
        rec.append(record);packed[key+'_ids']=ix;packed[key+'_ap']=mat;packed[key+'_fixed_ap']=fixedmat
    save_npz(out,quality=quality,collision=collision,empty=empty,quality_evaluated_ids=union,cd_mask=cd,**packed,**{k+'_scores':v for k,v in scores.items()})
    write_json(out.with_suffix('.json'),rec);all_records.extend(rec)
    print(json.dumps(dict(scene=a.scene,frame=frame,quality_calls=len(union),R0_ap=rec[0]['ap'],elapsed=time.time()-start)),flush=True)
if a.validate_only:write_json(validated,dict(status='PASS',scene=a.scene,frames=fr,checks=validation,selection_quality_collision_AP_exact=True,native_source_sha256=hashlib.sha256(source.encode()).hexdigest(),wall_seconds=time.time()-start))
else:write_json(outdir/'COMPLETE.json',dict(status='PASS',scene=a.scene,frames=frames(a.scene),wall_seconds=time.time()-start,quality_calls=calls,native_source_sha256=hashlib.sha256(source.encode()).hexdigest(),rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
