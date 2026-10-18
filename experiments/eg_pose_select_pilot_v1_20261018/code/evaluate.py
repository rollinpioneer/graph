from core import *
import argparse,inspect,torch,resource
p=argparse.ArgumentParser();p.add_argument('--scene',type=int,required=True);p.add_argument('--stage',choices=['development','transfer','confirmation'],default='development');p.add_argument('--micro',action='store_true');a=p.parse_args();os.nice(10);torch.set_num_threads(1);start=time.time();cpu=time.process_time()
sys.path[:0]=[str(CODE),str(CODE/'libs/pointnet2')];sys.argv=[sys.argv[0],'--dataset_root',str(DATA),'--camera','kinect','--model','economicgrasp','--batch_size','1']
from graspnetAPI import GraspGroup
import graspnetAPI.utils.eval_utils as EU
from dataset.graspnet_dataset import GraspNetDataset
from utils.collision_detector import ModelFreeCollisionDetector
source=inspect.getsource(EU.eval_grasp);prefix=source[:source.index('    ## collision detection')];ns=EU.__dict__.copy();exec(prefix.replace('def eval_grasp(','def select_only(')+'\n    return grasp_list\n',ns);select_only=ns['select_only'];native=Native(a.scene)
data=GraspNetDataset(str(DATA),split='train',camera='kinect',num_points=20000,voxel_size=.005,remove_outlier=True,augment=False,load_label=False)
heads=json.loads((R/'results/selected_heads.json').read_text());gate=json.loads((R/'results/gate_calibration.json').read_text());tau=float('inf') if gate['tau'] is None else gate['tau'];models={}
for h in heads:
 ck=torch.load(R/f'heads/{h["seed"]}/step{h["selected_step"]}.pt',map_location='cpu',weights_only=False);m=make_model().eval();m.load_state_dict(ck['model']);models[h['seed']]=(m,ck['mean'],ck['std'])
policies=list(SETS) if a.stage=='development' else ['S4'] if a.stage=='transfer' else ['S4','S4-zero']
fr=CFG['dev_frames'] if a.stage!='confirmation' else CFG['confirmation_frames'];fr=fr[:1] if a.micro else fr
folder=R/f'evaluation/{a.stage}/{a.scene:04d}';folder.mkdir(parents=True,exist_ok=True);calls=0;equiv=[];allrec=[]
for f in fr:
 path=folder/f'{f:04d}.json'
 if path.exists() and not a.micro:allrec+=json.loads(path.read_text());continue
 pred,hyp,features,act,actions=export(a.scene,f);pred=pred.astype(np.float64);hyp=hyp.astype(np.float64);native.frame(f);choices={'R0':np.zeros(1024,int)};timings=[]
 for seed,(m,mean,std) in models.items():
  t=time.perf_counter();x=np.concatenate([np.broadcast_to(((features-mean)/std)[:,None,:],(1024,6,320)),act],-1).copy()
  with torch.no_grad():qp=m(torch.from_numpy(x)).squeeze(-1).numpy()
  for policy in policies:choices[f'{policy}_{seed}']=select(qp,actions,policy,tau)
  timings.append(time.perf_counter()-t)
 if a.scene<100:cloud,_=data.get_data(a.scene*256+f,return_raw_cloud=True)
 else:
  for attr,sub,ext in [('colorpath','rgb','png'),('depthpath','depth','png'),('labelpath','label','png'),('metapath','meta','mat')]:getattr(data,attr)[0]=str(DATA/f'scenes/scene_{a.scene:04d}/kinect/{sub}/{f:04d}.{ext}')
  data.scenename[0]=f'scene_{a.scene:04d}';data.frameid[0]=f;cloud,_=data.get_data(0,return_raw_cloud=True)
 detector=ModelFreeCollisionDetector(cloud,voxel_size=.005);selections={};submitted={};dedup={}
 for name,ch in choices.items():
  rows=hyp[np.arange(1024),ch].copy();assert np.array_equal(rows[:,[0,1,2,13,14,15,16]],pred[:,[0,1,2,13,14,15,16]])
  sig=ch.tobytes()
  if sig not in dedup:
   cd=detector.detect(GraspGroup(rows.copy()),approach_dist=.05,collision_thresh=.01);found={}
   lookup={}
   for i,g in enumerate(rows):lookup.setdefault(g.tobytes(),[]).append(i)
   for usecd in [False,True]:
    inp=rows[~cd] if usecd else rows
    if len(inp):
     groups=select_only(GraspGroup(inp.copy()),native.models,native.dex,native.poses,native.config,table=native.table_trans,voxel_size=.008,TOP_K=50);gs=np.concatenate(groups);gs=gs[np.argsort(gs[:,0])[::-1]]
    else:gs=np.zeros((0,17))
    ids=np.array([min(i for i in lookup[g.tobytes()] if not usecd or not cd[i]) for g in gs],int);found[usecd]=(ids,inp)
   dedup[sig]=found
  for usecd in [False,True]:
   key=name+('_CD' if usecd else '');ids,inp=dedup[sig][usecd];selections[key]=(ids,ch[ids]);submitted[key]=inp
 # Fixed original R0 Top-1 plus a bounded quality-independent high/low sample.
 topid=int(selections['R0'][0][0]) if len(selections['R0'][0]) else None
 sampled=sample_ids(pred,a.scene,f);probe=np.unique(np.r_[sampled[:8],sampled[64:72],[] if topid is None else [topid]]).astype(int)
 requested=set(i*6+j for ids,js in selections.values() for i,j in zip(ids,js));requested.update(i*6+j for i in probe for j in range(6));uid=np.array(sorted(requested),int)
 q=np.full(6144,np.nan);c=np.zeros(6144,bool);e=c.copy();q[uid],c[uid],e[uid],_=native.score(hyp.reshape(-1,17)[uid]);calls+=len(uid)
 if a.micro:
  for key in ['R0','R0_CD',f'S4-zero_{SEEDS[0]}',f'S4-zero_{SEEDS[0]}_CD']:
   if key not in selections:continue
   gs,oq,oc=native.official(submitted[key]);ids,js=selections[key];assert np.array_equal(gs,hyp[ids,js]) and np.array_equal(oq,q[ids*6+js]) and np.array_equal(oc,c[ids*6+js]);assert np.array_equal(ap(oq),ap(q[ids*6+js]));equiv.append(key)
 rec=[]
 for key,(ids,js) in selections.items():
  name=key.removesuffix('_CD');ch=choices[name];qual=q[ids*6+js];mat=ap(qual);u0=utility(q[probe*6]);u1=utility(q[probe*6+ch[probe]]);changed=ch[probe]!=0
  policy=name.rsplit('_',1)[0] if name!='R0' else 'R0';allowed=[0] if policy=='R0' else SETS[policy];ugrid=utility(q.reshape(1024,6)[probe]);reg=ugrid[:,allowed].max(1)-u1;good0=(q[probe*6]>0)&(q[probe*6]<=.4000001)
  item=dict(scene=a.scene,frame=f,configuration=key,ap=float(mat.mean()*100),ap04=float(mat[:,1].mean()*100),ap08=float(mat[:,3].mean()*100),top1_valid=bool(len(qual) and 0<qual[0]<=.4000001),top1_collision=bool(len(ids) and c[ids[0]*6+js[0]]),top1_empty=bool(len(ids) and e[ids[0]*6+js[0]]),modified=int((ch!=0).sum()),candidates=1024,probe_n=len(probe),probe_changed=int(changed.sum()),probe_improved=int((changed&(u1>u0)).sum()),probe_degraded=int((changed&(u1<u0)).sum()),probe_success_degraded=int((good0&(u1<u0)).sum()),probe_original_success=int(good0.sum()),probe_regret=float(reg.mean()),probe_oracle_coverage=float((ugrid[:,allowed].max(1)>=np.float32(5/6)).mean()),top_identity=topid)
  if topid is not None:
   i=topid;j=ch[i];og=bool(0<q[i*6]<=.4000001);ng=bool(0<q[i*6+j]<=.4000001);item.update(original_top_success=og,R=int(not og and ng),D=int(og and not ng),original_top_mu=float(q[i*6]),selected_top_mu=float(q[i*6+j]),selected_top_collision=bool(c[i*6+j]),selected_top_empty=bool(e[i*6+j]),selected_action=int(j))
  for stratum,pids in [('high',sampled[:8]),('low',sampled[64:72])]:
   b=utility(q[pids*6]);v=utility(q[pids*6+ch[pids]]);item[stratum]=dict(n=len(pids),modified=int((ch[pids]!=0).sum()),improved=int((v>b).sum()),degraded=int((v<b).sum()),oracle=float((utility(q.reshape(1024,6)[pids])[:,allowed].max(1)>=np.float32(5/6)).mean()),regret=float((utility(q.reshape(1024,6)[pids])[:,allowed].max(1)-v).mean()))
  rec.append(item)
 write(path,rec);np.savez_compressed(folder/f'{f:04d}.npz',q=q.reshape(1024,6),collision=c.reshape(1024,6),empty=e.reshape(1024,6),probe_ids=probe,**{k:v.astype(np.int8) for k,v in choices.items()});allrec+=rec
 print('FRAME',a.scene,f,len(uid),time.time()-start,flush=True)
write(folder/('MICRO.json' if a.micro else 'COMPLETE.json'),dict(status='PASS',scene=a.scene,stage=a.stage,frames=fr,configuration_count=len(rec),seconds=time.time()-start,cpu_seconds=time.process_time()-cpu,new_native_calls=calls,reference_exact_checks=equiv,source_sha256=hashlib.sha256(source.encode()).hexdigest(),rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
