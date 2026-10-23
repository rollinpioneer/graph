"""Independent frozen-weight probes. Never creates an optimizer or updates parameters."""
import os,sys,json,hashlib,random,math,time,types,csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
B=Path.home()/'work/grasp';OLD=B/'eg_pose_quality_diagnosis_v1_20261022';ROOT=B/'eg_module_controlled_diagnosis_20261023';SRC=B/'eg_full_repro_v1_20261015/runtime/EconomicGrasp';D=B/'eg_sub25_v1_20261015/data/graspnet'
MODE=sys.argv[1];sys.path[:0]=[str(SRC),str(SRC/'libs/pointnet2')];sys.argv=['probe','--dataset_root',str(D),'--camera','kinect','--batch_size','1']
from dataset.graspnet_dataset import GraspNetDataset,collate_fn
from models.economicgrasp import economicgrasp,pred_decode
from utils.label_generation import process_grasp_labels
from utils.arguments import cfgs
from utils.loss_utils import generate_grasp_views,transform_point_cloud
from libs.knn.knn_modules import knn
import libs.pointnet2.pointnet2_utils as PU
from models import loss_economicgrasp as LOSS
P=json.loads((ROOT/'protocol.json').read_text());torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
def cpu(x):return x.detach().cpu().numpy()
def cuda(x):
 if torch.is_tensor(x):return x.cuda()
 if isinstance(x,dict):return {k:cuda(v) for k,v in x.items()}
 if isinstance(x,list):return [cuda(v) for v in x]
 return x
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rx(a):
 r=np.zeros((len(a),3,3));r[:,0,0]=1;r[:,1,1]=r[:,2,2]=np.cos(a);r[:,2,1]=np.sin(a);r[:,1,2]=-np.sin(a);return r
def ds(s,f):
 x=object.__new__(GraspNetDataset);x.root=str(D);x.camera='kinect';x.num_points=20000;x.voxel_size=.005;x.remove_outlier=True;x.augment=False;x.load_label=False
 p=D/f'scenes/scene_{s:04d}/kinect';x.colorpath=[str(p/f'rgb/{f:04d}.png')];x.depthpath=[str(p/f'depth/{f:04d}.png')];x.labelpath=[str(p/f'label/{f:04d}.png')];x.metapath=[str(p/f'meta/{f:04d}.mat')];x.scenename=[f'scene_{s:04d}'];x.frameid=[f]
 x.graspnesspath=[str(D/f'graspness/scene_{s:04d}/kinect/{f:04d}.npy')];x.grasp_labels={f'scene_{s:04d}':str(D/f'economic_grasp_label_300views/scene_{s:04d}_labels.npz')};return x
ck=B/'eg_full_repro_v1_20261015/downloads/economicgrasp_kinect.tar';assert sha(ck)==P['checkpoint_sha256']
net=economicgrasp(seed_feat_dim=512,is_training=False).cuda();net.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True)['model_state_dict'],strict=True);net.eval()
assert not any([net.is_training,net.view.is_training,net.training])
initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
head=net.grasp_head;cy=net.cy_group;am=head.global_interaction_module;nm=cy.local_interaction_module
capt={}
def before(key):
 def h(m,args):capt[key]=args
 return h
def after(key):
 def h(m,args,out):capt[key]=out
 return h
for m,k in [(cy,'cy_args'),(head,'head_args'),(am,'head_tokens_pre'),(nm,'neighbor_pre')]:m.register_forward_pre_hook(before(k))
for m,k in [(cy.grouper,'grouped'),(cy,'vp_features'),(am,'head_tokens_post'),(nm,'neighbor_post')]:m.register_forward_hook(after(k))
origquery=PU.cylinder_query
def query(*a,**kw):
 out=origquery(*a,**kw);capt['neighbors']=out;return out
PU.cylinder_query=query
origatt=am.msa.attention;origam=am.forward;orignm=nm.forward
def att_control(kind,strength):
 def fun(q,k,v,mask=None,dropout=None):
  logits=q@k.transpose(-1,-2)/math.sqrt(q.size(-1));bias=torch.zeros((4,4),device=q.device,dtype=q.dtype)
  if kind=='H1':bias[3,:3]=-1e9 if strength==0 else math.log(strength)
  elif kind=='H2':bias[:3,3]=-1e9 if strength==0 else math.log(strength)
  elif kind=='H3':bias[~torch.eye(4,device=q.device,dtype=torch.bool)]=-1e9
  return torch.softmax(logits+bias,dim=-1)@v
 return fun
def pack(end,rb=None):
 p=cpu(pred_decode(end)[0]);a=cpu(end['grasp_angle_pred'])[0];d=cpu(end['grasp_depth_pred'])[0];q=cpu(end['grasp_score_pred'])[0]
 if rb is not None:p[:,4:13]=(rb@rx(a[:12].argmax(0)*np.pi/12)).reshape(-1,9)
 return dict(pred=p,angle_logits=a,depth_logits=d,quality_logits=q,raw_width=cpu(end['grasp_width_pred'])[0])
def writecsv(path,rows):
 if not rows:return
 with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def block(name):
 if name.startswith('backbone.'):return 'backbone'
 if name.startswith('cy_group.'):return 'neighborhood'
 if name.startswith('grasp_head.global_interaction_module.msa.linears.'):
  return 'head_attention_'+['Q','K','V','output'][int(name.split('.')[4])]
 if name.startswith('grasp_head.global_interaction_module.norm1'):return 'head_attention_LN'
 return '.'.join(name.split('.')[:2])
params=list(net.named_parameters());groups={}
for i,(n,p) in enumerate(params):groups.setdefault(block(n),[]).append(i)
def grads(loss):return [None if g is None else g.detach().cpu().float() for g in torch.autograd.grad(loss,[p for _,p in params],retain_graph=True,allow_unused=True)]
def normdot(a,b,ids):
 aa=bb=ab=0.
 for i in ids:
  if a[i] is not None:aa+=float((a[i].double()**2).sum())
  if b[i] is not None:bb+=float((b[i].double()**2).sum())
  if a[i] is not None and b[i] is not None:ab+=float((a[i].double()*b[i].double()).sum())
 return math.sqrt(aa),math.sqrt(bb),ab/math.sqrt(aa*bb) if aa>1e-24 and bb>1e-24 else None
def gradient_probe(end,a,e,tag):
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 valid=torch.from_numpy(a['batch_valid_mask']).cuda()[None];qref=torch.from_numpy(a['batch_grasp_score']).cuda()[None];zero=valid&(qref==0);positive=valid&~zero
 names=['angle','depth','width','score','view','graspness','objectness'];loss={};weights={n:float(getattr(cfgs,n+'_loss_weight')) for n in names}
 for n in names:
  fun=LOSS.compute_view_graspness_loss if n=='view' else getattr(LOSS,'compute_'+n+'_loss'+('_cls' if n=='score' else ''))
  loss[n]=fun(end)[0]
 pop=[];raw={
 'angle':F.cross_entropy(end['grasp_angle_pred'],end['batch_grasp_rotations'].long(),reduction='none'),
 'depth':F.cross_entropy(end['grasp_depth_pred'],end['batch_grasp_depth'].long(),reduction='none'),
 'width':F.smooth_l1_loss(end['grasp_width_pred'].squeeze(1),end['batch_grasp_width']*10,reduction='none'),
 'score':F.cross_entropy(end['grasp_score_pred'],(end['batch_grasp_score']*10/2).long(),reduction='none')}
 for lab,m in [('zero',zero),('positive',positive),('invalid',~valid)]:
  for n,x in raw.items():
   contribution=(x*(m&valid)).sum()/valid.sum();pop.append(dict(frame=tag,population=lab,loss=n,n=int(m.sum()),valid_denominator=int(valid.sum()),raw_contribution=float(contribution.detach()),weighted_contribution=float(contribution.detach())*weights[n]))
   if lab=='zero':loss[n+'_zero']=contribution
 for key,mu in [('Q1',e['R_labelpoint_mu']),('Q2',e['P0_mu'])]:
  assert np.isfinite(mu[a['batch_valid_mask']]).all();target=np.where(mu>0,np.clip(np.round(1.1-mu,6),0,1),0).astype(np.float32)
  tt=(torch.from_numpy(target).cuda()[None]*10/2).long();loss[key]=F.cross_entropy(end['grasp_score_pred'],tt,reduction='none')[valid].mean()
 gs={};routes=[]
 for key,l in loss.items():
  gg=grads(l);gs[key]=gg;wt=weights.get(key.split('_')[0],1)
  for b,ids in groups.items():
   n=sum(params[i][1].numel() for i in ids);gn=normdot(gg,gg,ids)[0]
   routes.append(dict(frame=tag,loss=key,block=b,parameters=n,reachable_parameters=sum(params[i][1].numel() for i in ids if gg[i] is not None),norm=gn,rms=gn/math.sqrt(n),weight=wt,weighted_norm=wt*gn))
 total=sum(weights[n]*loss[n] for n in names);net.zero_grad(set_to_none=True);total.backward(retain_graph=True)
 ss=dd=0.
 for i,(_,p) in enumerate(params):
  g=sum((gs[n][i]*weights[n] for n in names if gs[n][i] is not None),torch.zeros_like(p,device='cpu'))
  if p.grad is not None:dd+=float(((g-p.grad.detach().cpu()).double()**2).sum());ss+=float((p.grad.detach().cpu().double()**2).sum())
 rel=math.sqrt(dd/max(ss,1e-30));assert rel<=P['tolerances']['gradient_sum_relative_L2'],rel
 angles=[]
 for x,y in [('angle','score'),('depth','score'),('width','score'),('angle','Q1'),('angle','Q2'),('depth','Q1'),('depth','Q2'),('score','Q1'),('Q1','Q2'),('view','score'),('graspness','score'),('objectness','score'),('angle_zero','angle'),('depth_zero','depth'),('width_zero','width')]:
  for b,ids in groups.items():
   na,nb,c=normdot(gs[x],gs[y],ids);angles.append(dict(frame=tag,loss1=x,loss2=y,block=b,norm1=na,norm2=nb,cosine=c))
 writecsv(ROOT/'gradient'/f'{tag}_routes.csv',routes);writecsv(ROOT/'gradient'/f'{tag}_population.csv',pop);writecsv(ROOT/'gradient'/f'{tag}_cosines.csv',angles)
 net.zero_grad(set_to_none=True);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 return dict(gradient_sum_relative_L2=rel,losses={k:float(v.detach()) for k,v in loss.items()},weights=weights,valid=int(valid.sum()),zero=int(zero.sum()),valid_angle13=int(((end['batch_grasp_rotations']==12)&valid).sum()),valid_depth5=int(((end['batch_grasp_depth']==4)&valid).sum()),angle13_argmax=int((end['grasp_angle_pred'].argmax(1)==12).sum()),depth5_argmax=int((end['grasp_depth_pred'].argmax(1)==4).sum()))

for d in ['cache','gradient','receipts']: (ROOT/d).mkdir(exist_ok=True)
pairs=[('train',s,f) for s in P['train_probe_scenes'] for f in P['train_probe_frames']]+[('C3',s,f) for s in P['c3_scenes'] for f in P['c3_frames']]
if MODE=='pilot':pairs=pairs[:2]
for split,s,f in pairs:
 tag=f'{split}_{s:04d}_{f:04d}';dest=ROOT/f'cache/{tag}.npz'
 if dest.exists():continue
 start=time.time();a=dict(np.load(OLD/f'cache/{tag}.npz'));e=dict(np.load(OLD/f'evaluation/{tag}.npz'));seed=int(a['seed']);np.random.seed(seed);random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 sample=ds(s,f).get_data_label(0) if split=='train' else ds(s,f).get_data(0)
 assert np.array_equal(sample['point_clouds'],a['input_points']);data=cuda(collate_fn([sample]));end=net(data);base=pack(end)
 assert np.array_equal(base['pred'][:,1:],a['pred'][:,1:]),float(abs(base['pred'][:,1:]-a['pred'][:,1:]).max())
 scoreerr=float(abs(base['pred'][:,0]-a['pred'][:,0]).max());assert scoreerr<=3e-5,scoreerr
 assert np.array_equal(cpu(end['grasp_top_view_inds'])[0],a['view_index'])
 c={k:v for k,v in capt.items()};F0=c['vp_features'].detach();cyargs=tuple(x.detach() for x in c['cy_args']);rb=a['view_rotation'].astype(float)
 out={k+'_'+n:v for k,z in [('H0',base)] for n,v in z.items()};out['candidate_id']=np.arange(1024);out['neighbors']=cpu(c['neighbors']);out['group_rotation']=a['view_rotation'];out['group_relative_xyz']=cpu(c['grouped'][:,:3]);out['vp_features']=cpu(F0);out['tokens_pre']=cpu(c['head_tokens_pre'][0]);out['tokens_post']=cpu(c['head_tokens_post']);out['neighbor_pre']=cpu(c['neighbor_pre'][0]).astype(np.float16);out['neighbor_post']=cpu(c['neighbor_post']).astype(np.float16)
 with torch.no_grad():
  q,k,v=[layer(c['head_tokens_pre'][j]).view(1024,-1,1,64).transpose(1,2) for j,layer in enumerate(am.msa.linears[:3])];out['head_attention']=cpu(torch.softmax(q@k.transpose(-1,-2)/8,dim=-1))
  qa,ka=[layer(c['neighbor_pre'][j]).view(1024,-1,1,259).transpose(1,2) for j,layer in enumerate(nm.msa.linears[:2])];out['neighbor_attention']=cpu(torch.softmax(qa@ka.transpose(-1,-2)/math.sqrt(259),dim=-1)).astype(np.float16)
  check=pack(head(F0,dict(end)));assert np.array_equal(base['pred'],check['pred'])
  for name,kind,strength in [('H1_half','H1',.5),('H1_off','H1',0),('H2_half','H2',.5),('H2_off','H2',0),('H3','H3',0),('H4','H4',0)]:
   if kind=='H4':am.forward=lambda q,k,v,mask:am.norm1(v)
   else:am.msa.attention=att_control(kind,strength)
   z=pack(head(F0,dict(end)))
   if kind=='H1':assert np.array_equal(z['pred'][:,1:],base['pred'][:,1:])
   if kind=='H2':assert np.array_equal(z['quality_logits'],base['quality_logits'])
   out.update({name+'_'+k:v for k,v in z.items()});am.msa.attention=origatt;am.forward=origam
  nm.forward=lambda q,k,v,mask:nm.norm1(v);newF=cy(*cyargs);nm.forward=orignm;out['N_F']=cpu(newF);out.update({'N_'+k:v for k,v in pack(head(newF,dict(end))).items()})
  assert np.array_equal(out['neighbors'],cpu(capt['neighbors']))
  if split=='train':
   ref=a['top_grasp_views_rot'].astype(float)@rx(a['batch_grasp_rotations']*np.pi/12);phi=np.arctan2((ref[:,:,1]*rb[:,:,2]).sum(1),(ref[:,:,1]*rb[:,:,1]).sum(1))%np.pi;T=rx(phi);newrb=rb@T
   originalR=base['pred'][:,4:13].reshape(-1,3,3);reencoded=newrb.transpose(0,2,1)@originalR;rounderr=float(abs(newrb@reencoded-originalR).max());assert rounderr<2e-6
   # Reuse exact baseline neighbor/features; transform only local xyz, retaining graph-independent fixed inputs.
   group=c['grouped'].detach().clone();group[:,:3]=torch.from_numpy(np.einsum('bnsi,nij->bnsj',cpu(group[:,:3]).transpose(0,2,3,1),T).transpose(0,3,1,2)).to(group)
   hook=cy.grouper.register_forward_hook(lambda m,args,o:group)
   newF=cy(*cyargs);hook.remove();out['E1_F']=cpu(newF);out['E1_phi']=phi;out['E1_basis']=newrb;out.update({'E1_'+k:v for k,v in pack(head(newF,dict(end)),newrb).items()})
   assert np.array_equal(out['neighbors'],cpu(capt['neighbors']))
 receipt=dict(frame=tag,historical_geometry_exact=True,historical_score_max_error=scoreerr,full_open_exact=True,H1_pose_exact=True,H2_score_exact=True,modes={'net_is_training':net.is_training,'view_is_training':net.view.is_training,'torch_training':net.training},TF32_inference=True,TF32_label_probe=False,updates=0)
 if split=='train':
  probe={k:v for k,v in data.items() if k.endswith('_list')};probe.update(xyz_graspable=end['xyz_graspable'].detach(),grasp_top_view_inds=end['grasp_top_view_inds'].detach())
  torch.backends.cuda.matmul.allow_tf32=False
  with torch.no_grad():_,probe=process_grasp_labels(probe)
  torch.backends.cuda.matmul.allow_tf32=True
  assert np.array_equal(cpu(probe['batch_grasp_point'])[0],a['batch_grasp_point'])
  # Official scatter has duplicate destination indices when transformed views collide.
  # Preserve the pre-registered historical ADWQ targets/mask; verify every replay
  # difference belongs to that duplicate-view ambiguity, never relax a threshold.
  ambiguous=np.zeros(1024,bool);torch.backends.cuda.matmul.allow_tf32=False
  for oi,pose in enumerate(data['object_poses_list'][0]):
   vv=generate_grasp_views(cfgs.num_view).cuda();tv=transform_point_cloud(vv,pose[:3,:3],'3x3');vi=cpu(knn(tv.T.contiguous()[None],vv.T.contiguous()[None],k=1).squeeze()-1).astype(int)
   mult=np.bincount(vi,minlength=300)[vi]>1;sel=a['matched_object_id']==a['object_ids_meta'][oi];ambiguous[sel]=mult[a['view_index'][sel]]
  torch.backends.cuda.matmul.allow_tf32=True
  diffs={}
  for k in ['batch_valid_mask','batch_grasp_score','batch_grasp_rotations','batch_grasp_depth','batch_grasp_width']:
   diff=cpu(probe[k])[0]!=a[k];assert np.all(~diff|ambiguous),(k,'unexplained_label_difference');diffs[k]=int(diff.sum())
   probe[k]=torch.from_numpy(a[k].copy()).cuda()[None]
  receipt['label_replay_differences']=diffs;receipt['duplicate_view_ambiguous_queries']=int(ambiguous.sum());receipt['ADWQ_target_source']='frozen historical FP32 labels; duplicate-destination view scatter replay differences audited'
  print('LABEL_REPLAY',tag,diffs,'ambiguous',int(ambiguous.sum()),flush=True)
  end.update({k:v for k,v in probe.items() if k.startswith('batch_')});receipt['gradient']=gradient_probe(end,a,e,tag);receipt['geometry_roundtrip_max_error']=rounderr
  ranks=np.argsort(-a['pred'][:,0],kind='stable');chosen=list(e['final_ids']);rest=[int(x) for x in ranks if x not in chosen];chosen+=list(np.array(rest)[np.linspace(0,len(rest)-1,128-len(chosen),dtype=int)]);out['mechanism_ids']=np.array(chosen)
  ids=[int(ranks[0])]+list(map(int,ranks[np.linspace(1,1023,7,dtype=int)]));rem=np.array([i for i in range(1024) if i not in ids]);ids+=list(map(int,np.random.default_rng(P['seed']+s*1000+f).choice(rem,8,replace=False)));assert len(set(ids))==16;out['grid_ids']=np.array(ids)
 else:out['mechanism_ids']=e['final_ids']
 np.savez_compressed(dest,**out);receipt['seconds']=time.time()-start;receipt['cache_sha256']=sha(dest);(ROOT/f'receipts/{tag}.json').write_text(json.dumps(receipt,indent=2));print('DONE',tag,receipt['seconds'],flush=True)
 del end,data,c,capt,out,F0;capt={};torch.cuda.empty_cache()
for k,v in net.state_dict().items():assert torch.equal(v.detach().cpu(),initial[k]),k
(ROOT/f'{MODE}_complete.json').write_text(json.dumps({'mode':MODE,'model_unchanged':True,'updates':0}));print('COMPLETE',MODE,flush=True)
