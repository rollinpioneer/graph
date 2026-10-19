"""Cached-quality diagnosis; unknown candidate quality stays unknown."""
import argparse,time,resource
from common import *
from scipy.stats import spearmanr

p=argparse.ArgumentParser();p.add_argument('--scene',type=int);a=p.parse_args()
if a.scene is None:
 with (V/'results/C1_per_frame.csv').open() as f:old=list(csv.DictReader(f))
 table={(int(x['scene']),int(x['frame']),x['configuration']):x for x in old}
 rows=[];loss_gate={}
 for split in SPLITS:
  for scene in CFG['exploration'][split]:
   for seed in SEEDS:
    means={n:np.mean([float(table[scene,f,n]['ap']) for f in range(256)]) for n in ['R0','R0_CD',f'S_{seed}',f'S_{seed}_CD']}
    r=means['R0_CD']-means['R0'];s=means[f'S_{seed}_CD']-means[f'S_{seed}'];gain0=means[f'S_{seed}']-means['R0'];gain1=means[f'S_{seed}_CD']-means['R0_CD']
    assert np.isclose(gain1-gain0,s-r)
    rows.append(dict(split=split,scene=scene,seed=seed,EG_noCD_AP=means['R0'],EG_CD_AP=means['R0_CD'],S_noCD_AP=means[f'S_{seed}'],S_CD_AP=means[f'S_{seed}_CD'],delta_noCD=gain0,delta_CD=gain1,EG_benefit_CD=r,S_benefit_CD=s,CD_change_in_rescoring_gain=s-r))
    if scene==136:
     losses=sum(float(table[scene,f,'R0_CD']['ap'])>float(table[scene,f,f'S_{seed}_CD']['ap']) and table[scene,f,'R0_CD']['top_candidate']!=table[scene,f,f'S_{seed}_CD']['top_candidate'] for f in range(256))
     loss_gate[str(seed)]=losses
 csv_write('results/CD_AP_decomposition.csv',rows)
 write('results/diagnosis_gate.json',{'execute_fixed_F05':sum(v>0 for v in loss_gate.values())>=2,'scene_0136_frames_EG_better_and_different_top1':loss_gate,'rule':'At least two heads have EG better AP and a different actual selected Top1 in scene0136; explore only','C1_is_exploratory':True})
 print('DIAGNOSIS_GATE',json.dumps(loss_gate),flush=True);sys.exit(0)

scene=a.scene;start=time.time();os.nice(10)
sys.path[:0]=[str(CODE),str(CODE/'libs/pointnet2')];sys.argv=[sys.argv[0],'--dataset_root',str(DATA),'--camera','kinect','--model','economicgrasp','--batch_size','1']
from graspnetAPI import GraspGroup
from dataset.graspnet_dataset import GraspNetDataset
from utils.collision_detector import ModelFreeCollisionDetector
from PIL import Image
from scipy.io import loadmat
data=GraspNetDataset(str(DATA),split='train',camera='kinect',num_points=20000,voxel_size=.005,remove_outlier=True,augment=False,load_label=False)
frames=list(range(256)) if scene==136 else list(range(0,256,8));rows=[];bins=[];top=[]
def corr(x,y):
 return float(spearmanr(x,y).statistic) if len(x)>2 and np.ptp(x)>0 and np.ptp(y)>0 else None
for frame in frames:
 export=V/f'exports/subset/scene_{scene:04d}/{frame:04d}.npz';cache=V/f'evaluation/C1/{scene:04d}/{frame:04d}.npz'
 with np.load(export) as z:original_pred=z['pred'].copy();pred=original_pred.astype(np.float64)
 with np.load(cache) as z:stored={k:z[k].copy() for k in z.files}
 records=json.loads((cache.with_suffix('.json')).read_text());base={x['configuration']:x for x in records};quality=stored['quality'];coll=stored['collision'];empty=stored['empty'];known=np.isfinite(quality);u=np.zeros(1024);u[known]=utility(quality[known]);valid=known&(quality>0)&(quality<=.4000001)
 folder=DATA/f'scenes/scene_{scene:04d}/kinect'
 for attr,sub,ext in [('colorpath','rgb','png'),('depthpath','depth','png'),('labelpath','label','png'),('metapath','meta','mat')]:getattr(data,attr)[0]=str(folder/f'{sub}/{frame:04d}.{ext}')
 data.scenename[0]=f'scene_{scene:04d}';data.frameid[0]=frame;cloud,_=data.get_data(0,return_raw_cloud=True)
 cd=ModelFreeCollisionDetector(cloud,voxel_size=.005).detect(GraspGroup(original_pred.copy()),approach_dist=.05,collision_thresh=.01)
 assert float(cd.mean())==base['R0']['cd_rejection_rate']
 with Image.open(folder/f'label/{frame:04d}.png') as im:seg=np.asarray(im)
 K=loadmat(folder/f'meta/{frame:04d}.mat')['intrinsic_matrix'];uv=pred[:,13:16]@K.T;uv=np.rint(uv[:,:2]/uv[:,2:]).astype(int);ok=(uv[:,0]>=0)&(uv[:,0]<seg.shape[1])&(uv[:,1]>=0)&(uv[:,1]<seg.shape[0]);obj=np.zeros(1024,int);obj[ok]=seg[uv[ok,1],uv[ok,0]];labels,counts=np.unique(seg,return_counts=True);area=dict(zip(labels,counts));vis=np.array([area.get(i,0)/seg.size if i>0 else 0 for i in obj])
 for seed in SEEDS:
  q=stored[f'Q_{seed}'];eg=pred[:,0];pool=known&~cd
  rows.append(dict(scene=scene,frame=frame,seed=seed,known_candidates=int(known.sum()),total_candidates=1024,CD_removed=int(cd.sum()),known_removed=int((known&cd).sum()),known_survivors=int(pool.sum()),known_true_collisions_removed=int((known&cd&coll).sum()),known_true_collisions_kept=int((pool&coll).sum()),known_good_removed=int((valid&cd).sum()),known_good_kept=int((valid&~cd).sum()),rho_EG_quality=corr(eg[known],u[known]),rho_Q_quality=corr(q[known],u[known]),rho_EG_quality_after_CD=corr(eg[pool],u[pool]),rho_Q_quality_after_CD=corr(q[pool],u[pool])))
  for use_cd in [False,True]:
   suffix='_CD' if use_cd else '';ri=stored['selected_R0'+suffix][:50];si=stored[f'selected_S_{seed}'+suffix][:50];r=int(ri[0]) if len(ri) else -1;s=int(si[0]) if len(si) else -1
   for model,ids,score in [('R0',ri,eg),('S',si,q)]:
    for i in ids:
     assert known[i]
     width=min(max(pred[i,1],0),.1);depth=pred[i,3]
     bins.append(dict(scene=scene,frame=frame,seed=seed,use_cd=use_cd,model=model,candidate_id=int(i),width_bin=int(np.digitize(width,[.02,.04,.06,.08])),depth_mm=round(float(depth*1000),3),camera_approach_alignment_bin=int(np.digitize(abs(pred[i,10]),[.25,.5,.75])),visible_area_bin=int(np.digitize(vis[i],[.01,.05,.15])),valid=int(valid[i]),collision=int(coll[i]),empty=int(empty[i]),quality=float(quality[i]),score=float(score[i])))
   eligible=np.flatnonzero(~cd) if use_cd else np.arange(1024)
   top.append(dict(scene=scene,frame=frame,seed=seed,use_cd=use_cd,EG_AP=base['R0'+suffix]['ap'],S_AP=base[f'S_{seed}'+suffix]['ap'],EG_top1=r,S_top1=s,EG_valid=bool(r>=0 and valid[r]),S_valid=bool(s>=0 and valid[s]),EG_collision=bool(r>=0 and coll[r]),S_collision=bool(s>=0 and coll[s]),EG_top1_rank_in_Q_survivors=int(1+(q[eligible]>q[r]).sum()) if r>=0 else None,EG_top1_in_S_selected_top50=bool(r in si),S_top1_rank_in_EG_survivors=int(1+(eg[eligible]>eg[s]).sum()) if s>=0 else None,S_top1_width=float(pred[s,1]) if s>=0 else None,S_top1_depth=float(pred[s,3]) if s>=0 else None,S_top1_visible_area=float(vis[s]) if s>=0 else None))
 if frame%32==0:print('DIAG',scene,frame,time.time()-start,flush=True)
csv_write(f'diagnosis/{scene:04d}/candidate_context.csv',bins);csv_write(f'diagnosis/{scene:04d}/frame_metrics.csv',rows);csv_write(f'diagnosis/{scene:04d}/top1_transitions.csv',top)
write(f'diagnosis/{scene:04d}/COMPLETE.json',dict(status='PASS',scene=scene,frames=frames,seconds=time.time()-start,cpu_seconds=time.process_time(),rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,CD_prediction_dtype=str(original_pred.dtype),known_quality_scope='Union of candidates actually selected by prior V3 official R0/S paths; unknown is not failure',source_root=str(V)))
print('DIAG_COMPLETE',scene,time.time()-start,flush=True)
