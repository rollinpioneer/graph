from common_v2 import *
import argparse,inspect
from scipy.special import expit
p=argparse.ArgumentParser();p.add_argument('--scene',type=int,required=True);p.add_argument('--stage',default='development');p.add_argument('--micro',action='store_true');args=p.parse_args();os.nice(10);start=time.time();cpu=time.process_time()
sys.path[:0]=[str(CODE),str(CODE/'libs/pointnet2')];sys.argv=[sys.argv[0],'--dataset_root',str(DATA),'--camera','kinect','--model','economicgrasp','--batch_size','1']
from graspnetAPI import GraspGroup
import graspnetAPI.utils.eval_utils as EU
from dataset.graspnet_dataset import GraspNetDataset
from utils.collision_detector import ModelFreeCollisionDetector
from core import sample_ids
source=inspect.getsource(EU.eval_grasp);prefix=source[:source.index('    ## collision detection')];ns=EU.__dict__.copy();exec(prefix.replace('def eval_grasp(','def select_only(')+'\n    return grasp_list\n',ns);select_only=ns['select_only'];native=Native(args.scene)
dataset=GraspNetDataset(str(DATA),split='train',camera='kinect',num_points=20000,voxel_size=.005,remove_outlier=True,augment=False,load_label=False)
locked=json.loads((R/'results/locked_method.json').read_text());heads={}
for h in locked['heads']:
 path=P/f'heads/{h["seed"]}/step{h["step"]}.pt';assert sha(path)==h['sha256'];ck=torch.load(path,map_location='cpu',weights_only=False);m=old_model().eval();m.load_state_dict(ck['model']);heads[h['seed']]=(m,ck['mean'],ck['std'],h['tau'])
frames=CFG['dev_frames'][:1] if args.micro else CFG['dev_frames'];folder=R/f'evaluation/{args.stage}/{args.scene:04d}';folder.mkdir(parents=True,exist_ok=True);calls=0;equiv=[];io_seconds=0;processing=[];completed=[]
for f in frames:
 path=folder/f'{f:04d}.json'
 if path.exists() and not args.micro:completed.append(f);continue
 t=time.perf_counter();pred,hyp,features,actions,action_ids=export(args.scene,f);source_export=N/f'exports/subset/scene_{args.scene:04d}/{f:04d}.npz';export_hash=sha(source_export);io_seconds+=time.perf_counter()-t
 pred=pred.astype(np.float64);hyp=hyp.astype(np.float64);native.frame(f);base_score=expit(locked['score_calibration']['a']*pred[:,0]+locked['score_calibration']['b']);assert len(np.unique(base_score))==len(np.unique(pred[:,0]));assert np.array_equal(np.argsort(base_score),np.argsort(pred[:,0]))
 choices={'R0':np.zeros(1024,int)};rows_by_name={'R0':pred.copy()};qpred={}
 for seed,(m,mean,std,tau) in heads.items():
  t=time.perf_counter();x=np.concatenate([np.broadcast_to(((features-mean)/std)[:,None,:],(1024,6,320)),actions],-1).copy()
  with torch.no_grad():q=m(torch.from_numpy(x)).squeeze(-1).numpy()
  ch=select(q,action_ids,'S4',tau);delta=q[np.arange(1024),ch]-q[:,0];delta[ch==0]=0;new=hyp[np.arange(1024),ch].copy();assert np.array_equal(new[:,[1,2,13,14,15,16]],pred[:,[1,2,13,14,15,16]])
  for method,score in [('G0',pred[:,0]),('Gabs',q[np.arange(1024),ch]),('Gdelta',base_score+delta.astype(np.float64))]:
   name=f'{method}_{seed}';rows=new.copy();rows[:,0]=score;rows_by_name[name]=rows;choices[name]=ch.copy();assert np.array_equal(rows[:,1:],new[:,1:])
  qpred[seed]=q;processing.append(dict(frame=f,seed=seed,seconds=time.perf_counter()-t))
 if args.scene<100:cloud,_=dataset.get_data(args.scene*256+f,return_raw_cloud=True)
 else:
  for attr,sub,ext in [('colorpath','rgb','png'),('depthpath','depth','png'),('labelpath','label','png'),('metapath','meta','mat')]:getattr(dataset,attr)[0]=str(DATA/f'scenes/scene_{args.scene:04d}/kinect/{sub}/{f:04d}.{ext}')
  dataset.scenename[0]=f'scene_{args.scene:04d}';dataset.frameid[0]=f;cloud,_=dataset.get_data(0,return_raw_cloud=True)
 detector=ModelFreeCollisionDetector(cloud,voxel_size=.005);cd_cache={};selected={};submitted={};selected_rows={}
 def native_select(rows,cd,usecd):
  inp=rows[~cd] if usecd else rows
  if not len(inp):return np.zeros((0,17)),inp
  groups=select_only(GraspGroup(inp.copy()),native.models,native.dex,native.poses,native.config,table=native.table_trans,voxel_size=.008,TOP_K=50);gs=np.concatenate(groups);return gs[np.argsort(gs[:,0])[::-1]],inp
 for name,rows in rows_by_name.items():
  ch=choices[name];signature=ch.tobytes()
  if signature not in cd_cache:cd_cache[signature]=detector.detect(GraspGroup(rows.copy()),approach_dist=.05,collision_thresh=.01)
  cd=cd_cache[signature];rows=rows.copy();rows[:,1]=rows[:,1].clip(0,.1);lookup={}
  for i,g in enumerate(rows):lookup.setdefault(g.tobytes(),[]).append(i)
  for usecd in [False,True]:
   gs,inp=native_select(rows,cd,usecd);ids=np.array([min(i for i in lookup[g.tobytes()] if not usecd or not cd[i]) for g in gs],int);name_cd=name+('_CD' if usecd else '');selected[name_cd]=(ids,ch[ids]);submitted[name_cd]=inp;selected_rows[name_cd]=gs
   if name=='R0':
    calibrated=rows.copy();calibrated[:,0]=base_score;cgs,_=native_select(calibrated,cd,usecd);assert np.array_equal(cgs[:,1:],gs[:,1:]),'Monotonic C changed complete native selection'
 topid=int(selected['R0'][0][0]) if len(selected['R0'][0]) else None;sampled=sample_ids(pred,args.scene,f);probe=np.unique(np.r_[sampled[:8],sampled[64:72],[] if topid is None else [topid]]).astype(int)
 requested=set(i*6+j for ids,js in selected.values() for i,j in zip(ids,js));requested.update(i*6+j for i in probe for j in range(6));uid=np.array(sorted(requested),int);mu=np.full(6144,np.nan);collision=np.zeros(6144,bool);empty=collision.copy();mu[uid],collision[uid],empty[uid],_=native.score(hyp.reshape(-1,17)[uid]);calls+=len(uid)
 if args.micro:
  for name in ['R0_CD',f'G0_{SEEDS[0]}_CD',f'Gabs_{SEEDS[0]}_CD',f'Gdelta_{SEEDS[0]}_CD']:
   gs,q,c=native.official(submitted[name]);ids,js=selected[name];checks=dict(selection=np.array_equal(gs,selected_rows[name]),quality=np.array_equal(q,mu[ids*6+js]),collision=np.array_equal(c,collision[ids*6+js]),AP=np.array_equal(ap(q),ap(mu[ids*6+js])));print('MICRO_CHECK',name,checks,flush=True)
   if not all(checks.values()):
    np.savez_compressed(folder/'micro_mismatch.npz',official_grasps=gs,selected_grasps=selected_rows[name],official_mu=q,direct_mu=mu[ids*6+js],official_collision=c,direct_collision=collision[ids*6+js],ids=ids,actions=js,submitted=submitted[name]);write(folder/'micro_mismatch.json',dict(configuration=name,checks={k:bool(v) for k,v in checks.items()}));raise AssertionError((name,checks))
   equiv.append(name)
 records=[]
 for name,(ids,js) in selected.items():
  rawname=name.removesuffix('_CD');ch=choices[rawname];quality=mu[ids*6+js];matrix=ap(quality);u0=utility(mu[probe*6]);un=utility(mu[probe*6+ch[probe]]);grid=utility(mu.reshape(1024,6)[probe]);changed=ch[probe]!=0
  rec=dict(scene=args.scene,frame=f,configuration=name,ap=float(matrix.mean()*100),ap04=float(matrix[:,1].mean()*100),ap08=float(matrix[:,3].mean()*100),top1_valid=bool(len(quality) and 0<quality[0]<=.4000001),top1_collision=bool(len(ids) and collision[ids[0]*6+js[0]]),top1_empty=bool(len(ids) and empty[ids[0]*6+js[0]]),modified=int((ch!=0).sum()),candidates=1024,probe_n=len(probe),probe_improved=int((un>u0).sum()),probe_degraded=int((un<u0).sum()),probe_regret=float((grid[:,[0,1,2,3]].max(1)-un).mean()),probe_utility_gain=float((un-u0).mean()),top_identity=topid,source_export_sha256=export_hash)
  if topid is not None:
   i=topid;j=int(ch[i]);og=bool(0<mu[i*6]<=.4000001);ng=bool(0<mu[i*6+j]<=.4000001);rec.update(original_top_success=og,R=int(not og and ng),D=int(og and not ng),original_top_mu=float(mu[i*6]),selected_top_mu=float(mu[i*6+j]),selected_action=j,selected_top_collision=bool(collision[i*6+j]),selected_top_empty=bool(empty[i*6+j]))
  for st,ix in [('high',sampled[:8]),('low',sampled[64:72])]:
   b=utility(mu[ix*6]);v=utility(mu[ix*6+ch[ix]]);rec[st]=dict(n=len(ix),modified=int((ch[ix]!=0).sum()),utility_gain=float((v-b).mean()),improved=int((v>b).sum()),degraded=int((v<b).sum()))
  records.append(rec)
 write(path,records);np.savez_compressed(folder/f'{f:04d}.npz',mu=mu.reshape(1024,6),collision=collision.reshape(1024,6),empty=empty.reshape(1024,6),probe_ids=probe,source_export_sha256=export_hash,**{k:v.astype(np.int8) for k,v in choices.items()},**{f'Q_{s}':q for s,q in qpred.items()});completed.append(f);print('FRAME',args.scene,f,len(uid),time.time()-start,flush=True)
write(folder/('MICRO.json' if args.micro else 'COMPLETE.json'),dict(status='PASS',scene=args.scene,frames=completed,configurations=20,new_native_calls=calls,seconds=time.time()-start,cpu_seconds=time.process_time()-cpu,export_io_seconds=io_seconds,head_processing=processing,reference_exact_checks=equiv,monotonic_C_full_native_selection_exact_all_frames=True,native_eval_source_sha256=hashlib.sha256(source.encode()).hexdigest(),quality_cache_reuse='V1 application quality cache not reused: complete per-frame source export identity was not recorded in that cache. New cache records export and native evaluator identity; training labels reused.',rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
