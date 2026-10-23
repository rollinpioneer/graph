"""R1.3 actual train-path target/gradient audit. No optimizer and no parameter steps."""
import sys, json, random, hashlib, time, csv, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
B=Path.home()/'work/grasp'; ROOT=B/'eg_causal_mechanism_v2_20261023'; OUT=ROOT/'training_path'
SRC=B/'eg_full_repro_v1_20261015/runtime/EconomicGrasp'; D=B/'eg_sub25_v1_20261015/data/graspnet'
mode=sys.argv[1] if len(sys.argv)>1 else 'all'
sys.path[:0]=[str(SRC),str(SRC/'libs/pointnet2')]
sys.argv=['training_path','--dataset_root',str(D),'--camera','kinect','--batch_size','1']
from dataset.graspnet_dataset import GraspNetDataset,collate_fn
from models.economicgrasp import economicgrasp
from models import loss_economicgrasp as LOSS
import utils.label_generation as LG
from utils.arguments import cfgs
P=json.loads((ROOT/'protocol_v2_frozen.json').read_text())
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def cpu(x):return x.detach().cpu().numpy()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def cuda(x):
 if torch.is_tensor(x):return x.cuda()
 if isinstance(x,dict):return {k:cuda(v) for k,v in x.items()}
 if isinstance(x,list):return [cuda(v) for v in x]
 return x
def dataset(s,f):
 x=object.__new__(GraspNetDataset);x.root=str(D);x.camera='kinect';x.num_points=20000;x.voxel_size=.005;x.remove_outlier=True;x.augment=True;x.load_label=True
 p=D/f'scenes/scene_{s:04d}/kinect';x.colorpath=[str(p/f'rgb/{f:04d}.png')];x.depthpath=[str(p/f'depth/{f:04d}.png')];x.labelpath=[str(p/f'label/{f:04d}.png')];x.metapath=[str(p/f'meta/{f:04d}.mat')];x.scenename=[f'scene_{s:04d}'];x.frameid=[f]
 x.graspnesspath=[str(D/f'graspness/scene_{s:04d}/kinect/{f:04d}.npy')];x.grasp_labels={f'scene_{s:04d}':str(D/f'economic_grasp_label_300views/scene_{s:04d}_labels.npz')}
 return x
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(r):random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])
ck=B/'eg_full_repro_v1_20261015/downloads/economicgrasp_kinect.tar';assert sha(ck)==P['checkpoint_sha256']
net=economicgrasp(seed_feat_dim=512,is_training=True).cuda()
net.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True)['model_state_dict'],strict=True)
initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
names=list(net.named_parameters()); par=[p for _,p in names]
def block(n):
 if n.startswith('backbone.'):return 'backbone'
 if n.startswith('cy_group.'):return 'neighborhood'
 if n.startswith('grasp_head.global_interaction_module.'):return 'head_attention'
 return '.'.join(n.split('.')[:2])
groups={}
for i,(n,p) in enumerate(names):groups.setdefault(block(n),[]).append(i)
capt={}; knns=[]
net.cy_group.register_forward_pre_hook(lambda m,args:capt.update(cy_args=args))
original_knn=LG.knn
def knn(*a,**kw):
 result=original_knn(*a,**kw);knns.append(cpu(result).astype(np.int64)-1);return result
LG.knn=knn
def ambiguity(data,end,knns):
 # Preserve the exact label-generation KNN calls. No alternative mapping.
 maps=[x.reshape(-1) for x in knns[:-1]];nn=knns[-1].reshape(-1)
 lengths=[len(x) for x in data['grasp_points_list'][0]]
 oi=np.searchsorted(np.cumsum(lengths),nn,side='right');v=cpu(end['grasp_top_view_inds'])[0]
 amb=np.zeros(len(nn),bool)
 for j,vi in enumerate(maps):
  ix=oi==j;amb[ix]=np.bincount(vi,minlength=300)[vi[v[ix]]]>1
 return amb,oi,nn
def gradients(loss):
 return [None if g is None else g.detach().cpu().float() for g in torch.autograd.grad(loss,par,retain_graph=True,allow_unused=True)]
def norm(g,ids):return math.sqrt(sum(float(g[i].double().square().sum()) for i in ids if g[i] is not None))
def writecsv(p,rows):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
pairs=[(s,f) for s in P['stage1_scope']['train_scenes'] for f in P['stage1_scope']['train_frames']]
if mode=='pilot':pairs=pairs[:1]
for s,f in pairs:
 tag=f'train_{s:04d}_{f:04d}';rp=OUT/f'{tag}.json'
 if rp.exists():continue
 start=time.time();seed=20261023+s*1000+f
 net.load_state_dict(initial);net.train();net.is_training=True;net.view.is_training=True
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 initial_rng=rng();ds=dataset(s,f);flip=[];origaug=ds.augment_data
 def aug(cloud,poses):
  before=cloud.copy();ret=origaug(cloud,poses);flip.append(bool(not np.array_equal(before,ret[0])));return ret
 ds.augment_data=aug
 sample=ds.get_data_label(0);data=cuda(collate_fn([sample]));forward_rng=rng()
 torch.save({'before_data':initial_rng,'before_forward':forward_rng},OUT/f'{tag}_rng.pt')
 knns.clear();capt.clear();end=net(data);knn_actual=[a.copy() for a in knns]
 amb,oi,nn=ambiguity(data,end,knn_actual)
 valid=end['batch_valid_mask'];q=end['batch_grasp_score'];w=end['batch_grasp_width'];zero=valid&(q==0);pos=valid&(q>0);den=valid.sum().clamp_min(1)
 raw=end['grasp_width_pred'].squeeze(1);diff=raw-10*w;rawgrad=torch.clamp(diff,-1,1)
 perwidth=F.smooth_l1_loss(raw,10*w,reduction='none')
 loss={};weights={n:float(getattr(cfgs,n+'_loss_weight')) for n in ['objectness','graspness','view','angle','depth','width','score']}
 total,_=LOSS.get_loss(end)
 width=(perwidth*valid).sum()/den
 loss['width_zero']=(perwidth*zero).sum()/den
 loss['width_positive']=(perwidth*pos).sum()/den
 loss['width_all']=width
 loss['other_weighted_losses']=total-weights['width']*width
 loss['angle_zero']=(F.cross_entropy(end['grasp_angle_pred'],end['batch_grasp_rotations'].long(),reduction='none')*zero).sum()/den
 loss['depth_zero']=(F.cross_entropy(end['grasp_depth_pred'],end['batch_grasp_depth'].long(),reduction='none')*zero).sum()/den
 for label,m in [('zero_width0',zero&(w==0)),('zero_width_gt100mm',zero&(w>.1)),('zero_width_other',zero&(w>0)&(w<=.1))]:loss[label]=(perwidth*m).sum()/den
 routes=[];zg=None;cosines=[]
 for name,l in loss.items():
  g=gradients(l)
  if name=='width_zero':zg=g
  for group,ids in groups.items():
   weight=1 if name=='other_weighted_losses' or name.startswith(('angle','depth')) else weights['width']
   normg=norm(g,ids);routes.append(dict(batch=tag,scene=s,frame=f,loss=name,block=group,loss_value=float(l.detach()),norm=normg,weight=weight,weighted_norm=weight*normg))
   if name in ['width_positive','other_weighted_losses']:
    nz=norm(zg,ids);dot=sum(float((g[i].double()*zg[i].double()).sum()) for i in ids if g[i] is not None and zg[i] is not None)
    cosines.append(dict(batch=tag,scene=s,loss1='width_zero',loss2=name,block=group,cosine=dot/(nz*normg) if nz*normg>1e-30 else '',norm_zero=nz,norm_other=normg))
  if name!='width_zero':del g
 del zg
 fields=['batch_valid_mask','batch_grasp_score','batch_grasp_width','batch_grasp_rotations','batch_grasp_depth','grasp_top_view_inds','xyz_graspable','batch_grasp_point','grasp_angle_pred','grasp_depth_pred','grasp_width_pred','grasp_score_pred']
 frozen={k:cpu(end[k]).copy() for k in fields}
 cyargs=tuple(x.detach().clone() for x in capt['cy_args']);state_after={k:v.detach().cpu().clone() for k,v in net.named_buffers()}
 bn_changed=sum(not torch.equal(v,initial[k]) for k,v in state_after.items())
 # Paired downstream mode control: identical input/augmentation/sampled view/
 # grouping inputs/targets, restore original buffers and disable BN/dropout only.
 for n,b in net.named_buffers():b.copy_(initial[n].to(b))
 net.cy_group.eval();net.grasp_head.eval()
 with torch.no_grad():
  paired=net.grasp_head(net.cy_group(*cyargs),{})
  evalraw=cpu(paired['grasp_width_pred']).reshape(-1)
 paired_counts={'angle_changed':int((cpu(paired['grasp_angle_pred']).argmax(1)!=frozen['grasp_angle_pred'].argmax(1)).sum()),'depth_changed':int((cpu(paired['grasp_depth_pred']).argmax(1)!=frozen['grasp_depth_pred'].argmax(1)).sum()),'raw_width_abs_mean':float(abs(evalraw-frozen['grasp_width_pred'].reshape(-1)).mean())}
 rows=[]
 va=cpu(valid)[0];qa=cpu(q)[0];wa=cpu(w)[0];ra=cpu(raw)[0];ga=cpu(rawgrad)[0]
 for i in range(1024):rows.append(dict(batch=tag,scene=s,frame=f,candidate_id=i,valid=bool(va[i]),q_ref=float(qa[i]),w_ref_m=float(wa[i]),raw_width=float(ra[i]),scaled_width_m=float(1.2*ra[i]/10),decoded_width_m=float(np.clip(1.2*ra[i]/10,0,.1)),upper_saturated=bool(1.2*ra[i]/10>=.1),lower_saturated=bool(ra[i]<=0),raw_smoothl1_derivative=float(ga[i]),weighted_denormalized_gradient=float(10*ga[i]*va[i]/int(den)),saturated_pushed_further=bool(1.2*ra[i]/10>=.1 and ga[i]<0),duplicate_view_ambiguous=bool(amb[i]),matched_object_slot=int(oi[i]),merged_label_point=int(nn[i]),sampled_view=int(frozen['grasp_top_view_inds'][0,i]),angle_target=int(frozen['batch_grasp_rotations'][0,i]),depth_target=int(frozen['batch_grasp_depth'][0,i]),paired_eval_raw_width=float(evalraw[i])))
 replay={'performed':False}
 if (s,f)==pairs[0]:
  # Rebuild from identical input & complete RNG/buffer state once; official
  # duplicate view scatter ambiguity is retained, never repaired implicitly.
  net.load_state_dict(initial);net.train();net.is_training=True;net.view.is_training=True;restore_rng(forward_rng)
  knns.clear()
  with torch.no_grad():repeat=net(cuda(collate_fn([sample])))
  replay={'performed':True,'query_centers_exact':np.array_equal(cpu(repeat['xyz_graspable']),frozen['xyz_graspable']),'sampled_views_exact':np.array_equal(cpu(repeat['grasp_top_view_inds']),frozen['grasp_top_view_inds']),'label_differences':{}}
  assert replay['query_centers_exact'] and replay['sampled_views_exact']
  for k in ['batch_valid_mask','batch_grasp_score','batch_grasp_width','batch_grasp_rotations','batch_grasp_depth']:
   changed=(cpu(repeat[k])!=frozen[k])[0];assert np.all(~changed|amb),(k,'unexplained_label_change')
   replay['label_differences'][k]=int(changed.sum())
  replay['max_raw_width_difference']=float(abs(cpu(repeat['grasp_width_pred'])-frozen['grasp_width_pred']).max())
  del repeat
 for n,p in net.named_parameters():assert torch.equal(p.detach().cpu(),initial[n]),n
 net.load_state_dict(initial)
 for n,b in net.state_dict().items():assert torch.equal(b.detach().cpu(),initial[n]),n
 writecsv(OUT/f'{tag}_population.csv',rows);writecsv(OUT/f'{tag}_routes.csv',routes);writecsv(OUT/f'{tag}_cosines.csv',cosines)
 np.savez_compressed(OUT/f'{tag}_frozen.npz',input_points=sample['point_clouds'],ambiguous=amb,matched_object_slot=oi,nearest_label_index=nn,**frozen)
 rec=dict(batch=tag,scene=s,frame=f,seed=seed,batch_size=1,augmentation_flip=flip,valid=int(va.sum()),zero=int((va&(qa==0)).sum()),zero_unambiguous=int((va&(qa==0)&~amb).sum()),positive=int((va&(qa>0)).sum()),ambiguous=int(amb.sum()),modes=dict(net_is_training=True,view_is_training=True,torch_train=True,BN_train=True,dropout_train=True),precision='FP32 TF32 disabled',input_hash=arrsha(sample['point_clouds']),rng_file=f'{tag}_rng.pt',checkpoint_sha256=P['checkpoint_sha256'],parameter_updates=0,parameters_unchanged=True,buffers_changed_in_training_forward=bn_changed,buffers_restored=True,optimizer=None,paired_eval_control=paired_counts,replay=replay,seconds=time.time()-start,source_label_sha256=sha(ds.grasp_labels[ds.scenename[0]]))
 rp.write_text(json.dumps(rec,indent=2));print('TRAIN_PATH_DONE',tag,json.dumps(rec),flush=True)
 del data,end,raw,loss,total,perwidth,valid,zero,pos,cyargs,capt,paired
 capt={};torch.cuda.empty_cache()
(OUT/f'{mode}_complete.json').write_text(json.dumps(dict(frames=len(pairs),parameter_updates=0,parameters_unchanged=True,buffers_restored=True)))
print('TRAIN_PATH_COMPLETE',mode,flush=True)
