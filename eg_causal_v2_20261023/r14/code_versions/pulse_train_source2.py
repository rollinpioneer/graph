"""Authorized R1.4 sidecar: 8 independent batches x 7 arms x 5 Adam steps.

Official sources and previous assets are read-only. Query sampling is frozen,
but the entire backbone is recomputed with gradients on every update.
"""
import os, sys, json, copy, random, hashlib, time, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['preflight','source','trace']);ap.add_argument('--scene',type=int);args=ap.parse_args()
B=Path.home()/'work/grasp';ROOT=B/'eg_causal_mechanism_v2_20261023';OUT=ROOT/'r14'
OLD=B/'eg_pose_quality_diagnosis_v1_20261022';MODULE=B/'eg_module_controlled_diagnosis_20261023'
SRC=B/'eg_full_repro_v1_20261015/runtime/EconomicGrasp';D=B/'eg_sub25_v1_20261015/data/graspnet'
sys.path[:0]=[str(SRC),str(SRC/'libs/pointnet2')]
sys.argv=['pulse_train','--dataset_root',str(D),'--camera','kinect','--batch_size','1']
import MinkowskiEngine as ME
import models.economicgrasp as EG
from models.economicgrasp import economicgrasp,pred_decode
from dataset.graspnet_dataset import GraspNetDataset,collate_fn
from models import loss_economicgrasp as LOSS
from utils.arguments import cfgs
from libs.pointnet2.pointnet2_utils import gather_operation

P=json.loads((OUT/'protocol_R14_frozen.json').read_text())
SCENES=[int(b['scene']) for b in P['update_batches']];assert set(SCENES)=={2,9,31,33,68,64,76,82}
assert len(P['arms'])==7 and P['steps']==[1,5] and P['maximum_adam_steps_all_independent_branches']==280
torch.set_num_threads(4)
for n in ['baseline','specs','receipts','predictions','journals']:(OUT/n).mkdir(parents=True,exist_ok=True)
def cpu(x):return x.detach().cpu().numpy()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def jsonsave(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def tsave(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp');torch.save(x,tmp);tmp.replace(p)
def cuda(x):
 if torch.is_tensor(x):return x.cuda()
 if isinstance(x,dict):return {k:cuda(v) for k,v in x.items()}
 if isinstance(x,list):return [cuda(v) for v in x]
 return x
def detach(x):
 if torch.is_tensor(x):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:detach(v) for k,v in x.items()}
 if isinstance(x,list):return [detach(v) for v in x]
 return x
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def rr(r):random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])
def rnghash():
 h=hashlib.sha256(torch.get_rng_state().numpy().tobytes())
 for z in torch.cuda.get_rng_state_all():h.update(z.cpu().numpy().tobytes())
 return h.hexdigest()
def precision(train):torch.backends.cuda.matmul.allow_tf32=not train;torch.backends.cudnn.allow_tf32=not train
def mode(train):net.train(train);net.is_training=train;net.view.is_training=train;precision(train)
def ds(s,f,training):
 assert s in SCENES and f in [0,128]
 x=object.__new__(GraspNetDataset);x.root=str(D);x.camera='kinect';x.num_points=20000;x.voxel_size=.005;x.remove_outlier=True;x.augment=training;x.load_label=training
 p=D/f'scenes/scene_{s:04d}/kinect';x.colorpath=[str(p/f'rgb/{f:04d}.png')];x.depthpath=[str(p/f'depth/{f:04d}.png')];x.labelpath=[str(p/f'label/{f:04d}.png')];x.metapath=[str(p/f'meta/{f:04d}.mat')];x.scenename=[f'scene_{s:04d}'];x.frameid=[f]
 x.graspnesspath=[str(D/f'graspness/scene_{s:04d}/kinect/{f:04d}.npy')];x.grasp_labels={f'scene_{s:04d}':str(D/f'economic_grasp_label_300views/scene_{s:04d}_labels.npz')};return x

ck=B/'eg_full_repro_v1_20261015/downloads/economicgrasp_kinect.tar';assert sha(ck)==P['checkpoint']
checkpoint=torch.load(ck,map_location='cpu',weights_only=True)
net=economicgrasp(seed_feat_dim=512,is_training=True).cuda();net.load_state_dict(checkpoint['model_state_dict'],strict=True)
initial=detach(net.state_dict());named=list(net.named_parameters());params=[p for n,p in named];names=[n for n,p in named]
def statehash(state):
 h=hashlib.sha256()
 for k,v in state.items():h.update(k.encode());h.update(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes())
 return h.hexdigest()
initial_hash=statehash(initial)
savedopt=checkpoint['optimizer_state_dict'];ids=[i for g in savedopt['param_groups'] for i in g['params']]
assert len(ids)==len(params)==len(savedopt['state'])==152 and len(set(ids))==152
optimizer_rows=[]
for i,(n,p) in zip(ids,named):
 st=savedopt['state'][i]
 assert st['exp_avg'].shape==p.shape and st['exp_avg_sq'].shape==p.shape,(i,n)
 assert torch.isfinite(st['exp_avg']).all() and torch.isfinite(st['exp_avg_sq']).all()
 optimizer_rows.append(dict(index=i,name=n,shape=list(p.shape),step=float(st['step'])))
assert savedopt['param_groups'][0]['lr']==P['learning_rate']
jsonsave(OUT/'optimizer_mapping.json',dict(checkpoint_sha256=P['checkpoint'],parameter_order='official Adam(net.parameters()) order',parameters=optimizer_rows,groups=[{k:v for k,v in g.items() if k!='params'} for g in savedopt['param_groups']]))
def optimizer():
 opt=torch.optim.Adam(net.parameters(),lr=P['learning_rate']);opt.load_state_dict(copy.deepcopy(savedopt))
 for (n,p),i in zip(named,ids):
  for k in ['exp_avg','exp_avg_sq']:assert torch.equal(opt.state[p][k].cpu(),savedopt['state'][i][k]),(n,k)
 return opt

FIELDS=['batch_valid_mask','batch_grasp_score','batch_grasp_width','batch_grasp_rotations','batch_grasp_depth','batch_grasp_view_graspness','batch_grasp_point']
LOGITS=['grasp_angle_pred','grasp_depth_pred','grasp_width_pred','grasp_score_pred']
def capture(data,training):
 cap={};oldfps=EG.furthest_point_sample
 def fps(x,n):
  z=oldfps(x,n);cap['fps']=z.detach().cpu().clone();cap['fps_input']=cpu(x);return z
 def gh(m,a,o):
  cap['mask']=((o['objectness_score'].argmax(1)==1)&(o['graspness_score'].squeeze(1)>cfgs.graspness_threshold)).detach().cpu()
 def ch(m,a):cap['rotation']=a[2].detach().cpu().clone()
 EG.furthest_point_sample=fps;h1=net.graspable.register_forward_hook(gh);h2=net.cy_group.register_forward_pre_hook(ch)
 try:end=net(data)
 finally:EG.furthest_point_sample=oldfps;h1.remove();h2.remove()
 spec={k:cap[k] for k in ['mask','fps','rotation']};spec['xyz']=detach(end['xyz_graspable']);spec['view_inds']=detach(end['grasp_top_view_inds'])
 spec['labels']={k:detach(end[k]) for k in FIELDS} if training else {}
 if not training:
  spec['view_xyz']=detach(end['grasp_top_view_xyz']);spec['view_rot']=detach(end['grasp_top_view_rot'])
 return end,spec

def frozen_forward(data,spec,training,diagnostic_backbone=None):
 # Same official sparse construction and same gather kernels; no cached features.
 end=dict(data);xyz=end['point_clouds'];bs,nn,_=xyz.shape
 co,fe=ME.utils.sparse_collate([c for c in end['coordinates_for_voxel']],[f for f in np.ones_like(xyz.cpu()).astype(np.float32)])
 co,fe,_,inv=ME.utils.sparse_quantize(co,fe,return_index=True,return_inverse=True)
 co=co.cuda();fe=fe.cuda();end.update(coors=co,feats=fe,quantize2original=inv)
 features=net.backbone(ME.SparseTensor(fe,coordinates=co)).F
 if diagnostic_backbone is not None:features=diagnostic_backbone.to(features)
 features=features[inv].view(bs,nn,-1).transpose(1,2);end=net.graspable(features,end)
 mask=spec['mask'].cuda();fps=spec['fps'].cuda();ff=features.transpose(1,2)
 qxyz=gather_operation(xyz[0][mask[0]][None].transpose(1,2).contiguous(),fps).transpose(1,2).contiguous()
 qfeat=gather_operation(ff[0][mask[0]][None].transpose(1,2).contiguous(),fps)
 assert torch.equal(qxyz,spec['xyz'].cuda())
 end['xyz_graspable']=qxyz;end['D: Graspable Points']=mask.sum().float()
 end,res=net.view(qfeat,end);end['grasp_top_view_inds']=spec['view_inds'].cuda();qfeat=qfeat+res
 if training:end.update(cuda(spec['labels']))
 else:
  end['grasp_top_view_xyz']=spec['view_xyz'].cuda();end['grasp_top_view_rot']=spec['view_rot'].cuda()
 return net.grasp_head(net.cy_group(qxyz.contiguous(),qfeat.contiguous(),spec['rotation'].cuda()),end)

def pack(end):
 return dict(pred=cpu(pred_decode(end)[0]),angle_logits=cpu(end['grasp_angle_pred'])[0],depth_logits=cpu(end['grasp_depth_pred'])[0],quality_logits=cpu(end['grasp_score_pred'])[0],raw_width=cpu(end['grasp_width_pred'])[0],candidate_id=np.arange(1024))
def maxerr(a,b):return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
def traced(fun):
 trace={};hooks=[]
 def save(name,x):trace[name]=cpu(x).copy()
 hooks.append(net.backbone.register_forward_hook(lambda m,a,o:save('01_backbone',o.F)))
 hooks.append(net.graspable.register_forward_pre_hook(lambda m,a:save('02_graspable_input',a[0])))
 hooks.append(net.view.register_forward_pre_hook(lambda m,a:save('03_view_input',a[0])))
 hooks.append(net.view.register_forward_hook(lambda m,a,o:save('04_view_residual',o[1])))
 hooks.append(net.cy_group.register_forward_pre_hook(lambda m,a:(save('05_group_features',a[1]),save('06_group_rotation',a[2])) and None))
 hooks.append(net.grasp_head.register_forward_pre_hook(lambda m,a:save('07_head_input',a[0])))
 try:result=fun()
 finally:
  for h in hooks:h.remove()
 return result,trace
def trace_diff(a,b):
 return {k:dict(exact=np.array_equal(a[k],b[k]),max_abs=maxerr(a[k],b[k]),rms=float(np.sqrt(np.mean((a[k].astype(float)-b[k])**2))),shape=list(a[k].shape),hash_a=ah(a[k]),hash_b=ah(b[k])) for k in a}
def trace_train(s):
 """Three independent zero-update forwards plus same-backbone isolation."""
 tag=f'train_{s:04d}_0000';old=dict(np.load(ROOT/'training_path'/f'{tag}_frozen.npz'));r=torch.load(ROOT/'training_path'/f'{tag}_rng.pt',map_location='cpu',weights_only=False)
 rr(r['before_data']);sample=ds(s,0,True).get_data_label(0);assert np.array_equal(sample['point_clouds'],old['input_points']);data=cuda(collate_fn([sample]))
 traces={};values={};controls={};spec=None
 for kind in ['native1','native2','fixed','fixed_shared_backbone']:
  net.load_state_dict(initial);mode(True);rr(r['before_forward']);before=rnghash()
  if kind.startswith('native'):
   (end,current),tr=traced(lambda:capture(data,True))
   if spec is None:
    spec=current
    for k in FIELDS:
     if k in old:spec['labels'][k]=torch.from_numpy(old[k].copy())
  else:
   override=torch.from_numpy(traces['native1']['01_backbone']) if kind=='fixed_shared_backbone' else None
   end,tr=traced(lambda:frozen_forward(data,spec,True,override))
  traces[kind]=tr;values[kind]={k:cpu(end[k]).copy() for k in LOGITS}
  controls[kind]=dict(before_rng_hash=before,after_rng_hash=rnghash(),queries_exact=np.array_equal(cpu(end['xyz_graspable']),old['xyz_graspable']),views_exact=np.array_equal(cpu(end['grasp_top_view_inds']),old['grasp_top_view_inds']),FP32=True,autograd_enabled=torch.is_grad_enabled(),BN_train=net.training)
  del end
 def err(a,b):return {k:maxerr(a[k],b[k]) for k in LOGITS}
 nn=err(values['native1'],values['native2']);nf=err(values['native1'],values['fixed'])
 rec=dict(source_scene=s,parameter_updates=0,controls=controls,native_to_native_logits=nn,native_to_fixed_logits=nf,native_to_shared_backbone_logits=err(values['native1'],values['fixed_shared_backbone']),native_to_native_layers=trace_diff(traces['native1'],traces['native2']),native_to_fixed_layers=trace_diff(traces['native1'],traces['fixed']),native_to_shared_backbone_layers=trace_diff(traces['native1'],traces['fixed_shared_backbone']),historical_logits={name:err(v,old) for name,v in values.items()},tolerance_unchanged=3e-5,native_repeat_pass=max(nn.values())<=3e-5,fixed_path_pass=max(nf.values())<=3e-5)
 net.load_state_dict(initial);assert statehash(net.state_dict())==initial_hash
 jsonsave(OUT/'receipts'/f'train_{s:04d}_native_vs_fixed_trace.json',rec);print('NATIVE_VS_FIXED_TRACE',json.dumps(rec),flush=True)
 assert rec['native_repeat_pass'] and rec['fixed_path_pass'],'Trace retained: at least one unchanged 3e-5 criterion failed; zero updates'
def prepare_train(s):
 tag=f'train_{s:04d}_0000';fp=OUT/'specs'/f'{tag}.pt'
 if fp.exists():
  blob=torch.load(fp,map_location='cpu',weights_only=False);assert 'shared_step1' in blob,'Older independent-forward spec cannot be used for updates';return blob
 mode(True);net.load_state_dict(initial);old=dict(np.load(ROOT/'training_path'/f'{tag}_frozen.npz'));r=torch.load(ROOT/'training_path'/f'{tag}_rng.pt',map_location='cpu',weights_only=False)
 rr(r['before_data']);sample=ds(s,0,True).get_data_label(0);assert np.array_equal(sample['point_clouds'],old['input_points'])
 data=cuda(collate_fn([sample]));rr(r['before_forward'])
 # Stage1 captured its train-mode forward with autograd enabled. Keep that
 # same numerical path here, although this preparation performs no backward
 # or optimizer step (no_grad may select different CUDA forward kernels).
 (end,spec),native_trace=traced(lambda:capture(data,True))
 assert np.array_equal(cpu(end['xyz_graspable']),old['xyz_graspable']);assert np.array_equal(cpu(end['grasp_top_view_inds']),old['grasp_top_view_inds'])
 assert np.array_equal(cpu(end['batch_grasp_point']),old['batch_grasp_point'])
 diffs={}
 for k in FIELDS:
  if k in old:
   diff=cpu(end[k])!=old[k]
   if diff.ndim==2:assert np.all(~diff[0]|old['ambiguous']),(s,k,'unexpected_label_change')
   diffs[k]=int(diff.sum());spec['labels'][k]=torch.from_numpy(old[k].copy())
 orig={k:cpu(end[k]).copy() for k in LOGITS};stageerr={k:maxerr(orig[k],old[k]) for k in LOGITS}
 jsonsave(OUT/'receipts'/f'train_{s:04d}_initial_history_check.json',dict(source_scene=s,parameter_updates=0,logits_max_abs=stageerr,tolerance=3e-5,passed=max(stageerr.values())<=3e-5))
 assert max(stageerr.values())<=3e-5,(s,'stage1_logits',stageerr)
 # One original full-network graph supplies step1 to every arm. The backward
 # genuinely crosses the backbone; only its resulting gradients are reused.
 end.update(cuda(spec['labels']));postrng=rng();postbuffers=detach(dict(net.named_buffers()))
 valid=cpu(end['batch_valid_mask'])[0];q=cpu(end['batch_grasp_score'])[0];zero=valid&(q==0)
 score=(orig['grasp_score_pred'][0]*np.linspace(0,1,6)[:,None]).sum(0)
 quart=np.full(1024,-1,int);rank=np.argsort(score[valid],kind='stable');vi=np.flatnonzero(valid);quart[vi[rank]]=np.minimum(3,np.arange(len(vi))*4//len(vi))
 objectslot=old['matched_object_slot'];randommask=np.zeros(1024,bool);rows=[];gen=np.random.default_rng(20261023+s*1000+714)
 for ob in np.unique(objectslot[valid]):
  for qu in range(4):
   candidates=np.flatnonzero(valid&(objectslot==ob)&(quart==qu));count=int((zero&(objectslot==ob)&(quart==qu)).sum())
   chosen=gen.choice(candidates,count,replace=False);randommask[chosen]=True;rows.append(dict(object_slot=int(ob),score_quartile=qu,valid=len(candidates),q0=count,random_count=len(chosen),q0_overlap=int(zero[chosen].sum())))
 assert randommask.sum()==zero.sum()
 total,_=LOSS.get_loss(end);pw=F.smooth_l1_loss(end['grasp_width_pred'].squeeze(1),end['batch_grasp_width']*10,reduction='none');den=int(valid.sum())
 zl=cfgs.width_loss_weight*(pw*torch.from_numpy(zero).cuda()[None]).sum()/den
 rl=cfgs.width_loss_weight*(pw*torch.from_numpy(randommask).cuda()[None]).sum()/den
 initial_metrics=metrics(end,orig,valid,q);grads={};backbone_norms={}
 for key,ll in [('total',total),('zero',zl),('random',rl)]:
  gg=torch.autograd.grad(ll,params,retain_graph=key!='random',allow_unused=True)
  grads[key]=[None if x is None else x.detach().cpu().clone() for x in gg]
  backbone_norms[key]=sum(float(g.double().square().sum()) for (n,p),g in zip(named,gg) if n.startswith('backbone.') and g is not None)**.5
  assert backbone_norms[key]>0,(s,key,'backbone gradient missing')
  if key!='total':
   for (n,p),g in zip(named,gg):
    if n in ['grasp_head.conv_angle.weight','grasp_head.conv_angle.bias','grasp_head.conv_depth.weight','grasp_head.conv_depth.bias','grasp_head.conv_score.weight','grasp_head.conv_score.bias']:assert g is None or torch.count_nonzero(g)==0,(key,n)
  del gg
 loss=float(total.detach());shared=dict(gradients=grads,loss=loss,zero_width_loss=float(zl.detach()),random_width_loss=float(rl.detach()),metrics=initial_metrics,post_forward_buffers=postbuffers,post_forward_rng=postrng,backbone_gradient_norms=backbone_norms)
 del end,total,pw,zl,rl
 # Shared-backbone is ONLY a zero-update equivalence check. Its detached
 # representation never supplies any update gradient or later forward.
 net.load_state_dict(initial);rr(r['before_forward'])
 repeat,fixed_trace=traced(lambda:frozen_forward(data,spec,True,torch.from_numpy(native_trace['01_backbone'])))
 replayerr={k:maxerr(cpu(repeat[k]),orig[k]) for k in LOGITS};layercheck=trace_diff(native_trace,fixed_trace)
 assert max(replayerr.values())==0,(s,'same_backbone_fixed_path',replayerr)
 assert float(LOSS.get_loss(repeat)[0].detach())==loss
 for k in ['grasp_angle_pred','grasp_depth_pred']:assert np.array_equal(cpu(repeat[k]).argmax(1),orig[k].argmax(1)),(s,k)
 del repeat
 rec=dict(source_scene=s,input_hash=ah(sample['point_clouds']),query_hash=ah(old['xyz_graspable']),label_replay_differences=diffs,stage1_logits_max_abs=stageerr,fixed_path_logits_max_abs=replayerr,layertrace=layercheck,shared_step1_graph=True,step1_backbone_gradient_norms=backbone_norms,AD_classes_exact=True,points_views_exact=True,loss=loss,valid=int(valid.sum()),zero=int(zero.sum()),positive=int((valid&~zero).sum()),random_overlap=int((zero&randommask).sum()),random_mask_rows=rows,logit_tolerance_source='existing module protocol historical_logits_max_abs=3e-5; same-backbone path equality exact',backbone_recomputed=True,parameter_updates=0)
 blob=dict(data=detach(collate_fn([sample])),spec=spec,rng=r['before_forward'],baseline_logits=orig,random_mask=randommask,receipt=rec,shared_step1=shared)
 tsave(fp,blob);jsonsave(OUT/'receipts'/f'train_{s:04d}_preflight.json',rec)
 net.load_state_dict(initial);print('TRAIN_PREFLIGHT',s,json.dumps(rec),flush=True);return blob

def prepare_eval(s):
 tag=f'train_{s:04d}_0128';fp=OUT/'specs'/f'{tag}.pt'
 if fp.exists():return torch.load(fp,map_location='cpu',weights_only=False)
 old=dict(np.load(OLD/'cache'/f'{tag}.npz'));module=np.load(MODULE/'cache'/f'{tag}.npz');seed=int(old['seed'])
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 mode(False);net.load_state_dict(initial);sample=ds(s,128,False).get_data(0);assert np.array_equal(sample['point_clouds'],old['input_points'])
 data=cuda(collate_fn([sample]))
 with torch.no_grad():end,spec=capture(data,False);base=pack(end);repeat=pack(frozen_forward(data,spec,False))
 assert np.array_equal(base['pred'][:,1:],old['pred'][:,1:]),(s,'historical_pose',maxerr(base['pred'][:,1:],old['pred'][:,1:]))
 assert maxerr(base['pred'][:,0],old['pred'][:,0])<=3e-5
 assert np.array_equal(base['pred'],repeat['pred']),(s,'frozen_eval_pose',maxerr(base['pred'],repeat['pred']))
 for k in ['angle_logits','depth_logits','quality_logits','raw_width']:assert np.array_equal(base[k],repeat[k]),(s,k)
 mech=module['mechanism_ids'];assert len(mech)==128 and len(set(mech))==128
 base['mechanism_ids']=mech;np.savez_compressed(OUT/'baseline'/f'{tag}.npz',**base)
 rec=dict(scene=s,frame=128,historical_geometry_exact=True,historical_score_max_abs=maxerr(base['pred'][:,0],old['pred'][:,0]),fixed_path_exact=True,TF32=True,train_mode=False,mechanism_ids_hash=ah(mech),updates=0)
 jsonsave(OUT/'baseline'/f'{tag}.json',rec)
 blob=dict(data=detach(collate_fn([sample])),spec=spec,mechanism_ids=mech,receipt=rec);tsave(fp,blob);print('EVAL_PREFLIGHT',s,json.dumps(rec),flush=True);return blob

def metrics(end,baseline,valid,q):
 pos=valid&(q>0);width=cpu(end['grasp_width_pred'])[0,0]
 return dict(qpositive_score_logit_abs_delta_mean=float(abs(cpu(end['grasp_score_pred'])-baseline['grasp_score_pred'])[:,:,pos].mean()),width_upper_saturation_rate=float((1.2*width/10>=.1).mean()),width_lower_saturation_rate=float((width<=0).mean()),angle_changed=int((cpu(end['grasp_angle_pred']).argmax(1)!=baseline['grasp_angle_pred'].argmax(1)).sum()),depth_changed=int((cpu(end['grasp_depth_pred']).argmax(1)!=baseline['grasp_depth_pred'].argmax(1)).sum()))

def evaluate(s,arm,step,holdouts):
 savedrng=rng();mode(False)
 for hs,h in holdouts.items():
  tag=f'train_{hs:04d}_0128';dest=OUT/'predictions'/f'src_{s:04d}'/arm/f'step{step}'/f'{tag}.npz'
  if dest.exists():continue
  with torch.no_grad():end=frozen_forward(cuda(h['data']),h['spec'],False);new=pack(end)
  base=np.load(OUT/'baseline'/f'{tag}.npz');assert np.array_equal(new['pred'][:,13:16],base['pred'][:,13:16]);assert np.array_equal(new['pred'][:,4:13].reshape(-1,3,3)[:,:,0],base['pred'][:,4:13].reshape(-1,3,3)[:,:,0])
  new.update(mechanism_ids=h['mechanism_ids'],source_scene=np.array(s),target_scene=np.array(hs),target_frame=np.array(128),step=np.array(step),arm=np.array(arm))
  dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.tmp.npz');np.savez_compressed(tmp,**new);tmp.replace(dest)
  jsonsave(dest.with_suffix('.json'),dict(source_scene=s,target_scene=hs,target_frame=128,arm=arm,step=step,points_approaches_exact=True,TF32=True,train_mode=False,sha256=sha(dest),mechanism_ids_hash=ah(h['mechanism_ids'])))
  print('PREDICTION',str(dest.relative_to(OUT)),flush=True)
 mode(True);rr(savedrng)

def run_source(s):
 assert s in SCENES
 tr=prepare_train(s);opposite=next(fold for fold in P['folds'] if s not in fold);holdouts={h:prepare_eval(h) for h in opposite}
 data=cuda(tr['data']);spec=tr['spec'];valid=cpu(spec['labels']['batch_valid_mask'])[0];q=cpu(spec['labels']['batch_grasp_score'])[0];zero=torch.from_numpy(valid&(q==0)).cuda()[None]
 rm=torch.from_numpy(tr['random_mask']).cuda()[None];den=int(valid.sum());assert den>0
 r0=[];r0rng=[]
 for arm in P['arms']:
  receiptpath=OUT/'receipts'/f'src_{s:04d}'/f'{arm}.json';journal=OUT/'journals'/f'src_{s:04d}_{arm}.json'
  if receiptpath.exists():
   done=json.loads(receiptpath.read_text());assert done['completed_steps']==5
   if arm=='R0_original':r0=[x['applied_delta_l2'] for x in done['steps']];r0rng=[x['forward_rng_hash'] for x in done['steps']]
   print('SKIP_COMPLETED',s,arm,flush=True);continue
  recovery=OUT/'journals'/f'src_{s:04d}_{arm}_recovery.pt'
  net.load_state_dict(initial);mode(True);rr(tr['rng']);opt=optimizer();rows=[];firststep=1
  restored_hash=statehash(net.state_dict());assert restored_hash==initial_hash
  if journal.exists():
   j=json.loads(journal.read_text());completed=j['completed_adam_steps']
   if completed:
    assert recovery.exists(),f'No recovery state for completed updates; never repeat: {journal}'
    saved=torch.load(recovery,map_location='cpu',weights_only=False)
    assert saved['step']==completed and saved['source_scene']==s and saved['arm']==arm,'Recovery/journal mismatch; never repeat an update'
    net.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);rr(saved['rng']);rows=saved['rows'];firststep=completed+1
    if arm=='R0_original':r0=[x['applied_delta_l2'] for x in rows];r0rng=[x['forward_rng_hash'] for x in rows]
    if completed in [1,5]:evaluate(s,arm,completed,holdouts)
    del saved;print('RESUME_NO_REPEAT',s,arm,completed,flush=True)
  else:jsonsave(journal,dict(source_scene=s,arm=arm,started=True,completed_adam_steps=0))
  for step in range(firststep,6):
   opt.zero_grad(set_to_none=True)
   intervention='local' if arm.startswith('RW_local') else ('random' if arm.startswith('Rrandom') else ('remove' if arm.startswith('RW_') else 'original'))
   gradcheck=None
   if step==1:
    shared=tr['shared_step1'];gs=shared['gradients'];removed_key='random' if intervention=='random' else 'zero'
    # Same native full-network graph, same buffers, same stochastic draw.
    # These are genuine full-network gradients, not feature-cache gradients.
    for name,b in net.named_buffers():b.copy_(shared['post_forward_buffers'][name].to(b))
    rr(shared['post_forward_rng']);diff2=ref2=0.
    for pi,(name,p) in enumerate(named):
     gref=gs['total'][pi]
     if gref is None:assert gs[removed_key][pi] is None;continue
     gref=gref.to(p);gremove=gs[removed_key][pi]
     if gremove is not None:gremove=gremove.to(p)
     keep=intervention=='local' and name in ['grasp_head.conv_width.weight','grasp_head.conv_width.bias']
     expected=gref if intervention=='original' or gremove is None or keep else gref-gremove
     p.grad=expected.clone()
     diff2+=float((p.grad-expected).double().square().sum());ref2+=float(expected.double().square().sum())
     if name in ['grasp_head.conv_score.weight','grasp_head.conv_score.bias','grasp_head.conv_angle.weight','grasp_head.conv_angle.bias','grasp_head.conv_depth.weight','grasp_head.conv_depth.bias']:
      assert torch.equal(p.grad,gref),(arm,name,'unexpected_direct_output_gradient_change')
     if keep:assert torch.equal(p.grad,gref),(arm,name,'local final width retention failed')
    gradcheck=dict(assignment_relative_L2=(diff2/max(ref2,1e-30))**.5,ADQ_final_output_gradients_exact=True,local_final_width_retained=intervention=='local',shared_native_full_network_graph=True)
    assert gradcheck['assignment_relative_L2']==0
    errors={k:0. for k in LOGITS};lossvalue=shared['loss'];zerovalue=shared['zero_width_loss'];removedvalue=shared['random_width_loss'] if intervention=='random' else zerovalue;monitor=shared['metrics']
   else:
    end=frozen_forward(data,spec,True)
    total,_=LOSS.get_loss(end);pw=F.smooth_l1_loss(end['grasp_width_pred'].squeeze(1),end['batch_grasp_width']*10,reduction='none')
    zloss=cfgs.width_loss_weight*(pw*zero).sum()/den;rmloss=cfgs.width_loss_weight*(pw*rm).sum()/den;removed=rmloss if intervention=='random' else zloss
    correction=torch.autograd.grad(zloss,params,retain_graph=True,allow_unused=True) if intervention=='local' else None
    if intervention in ['remove','random']:(total-removed).backward()
    else:total.backward()
    if correction is not None:
     for (name,p),g in zip(named,correction):
      if g is not None and name not in ['grasp_head.conv_width.weight','grasp_head.conv_width.bias']:p.grad.sub_(g)
    lossvalue=float(total.detach());zerovalue=float(zloss.detach());removedvalue=float(removed.detach());monitor=metrics(end,tr['baseline_logits'],valid,q)
    del end,total,pw,zloss,rmloss,removed,correction
   forward_rng_hash=rnghash()
   if arm=='R0_original':r0rng.append(forward_rng_hash)
   else:assert forward_rng_hash==r0rng[step-1],(s,arm,step,'RNG consumption mismatch')
   assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
   gn=sum(float(p.grad.double().square().sum()) for p in params if p.grad is not None)**.5
   before=[p.detach().clone() for p in params];opt.step()
   # Record the completed update before any optional evaluation or file work.
   jsonsave(journal,dict(source_scene=s,arm=arm,started=True,completed_adam_steps=step))
   delta=[p.detach()-b for p,b in zip(params,before)];rawnorm=sum(float(v.double().square().sum()) for v in delta)**.5
   assert rawnorm>0 and np.isfinite(rawnorm);matched='actual_step_norm_matched' in arm;scale=r0[step-1]/rawnorm if matched else 1.
   if matched:
    with torch.no_grad():
     for p,b,d in zip(params,before,delta):p.copy_(b+scale*d)
   applied=sum(float((p.detach()-b).double().square().sum()) for p,b in zip(params,before))**.5
   if matched:assert abs(applied-r0[step-1])/r0[step-1]<1e-3,(s,arm,step,applied,r0[step-1])
   row=dict(step=step,loss=lossvalue,zero_width_loss=zerovalue,removed_width_loss=removedvalue if intervention!='original' else 0.,valid_denominator=den,gradient_norm=gn,forward_rng_hash=forward_rng_hash,gradient_assignment_check=gradcheck,initial_forward_max_abs=errors if step==1 else None,actual_adam_delta_l2=rawnorm,applied_delta_l2=applied,R0_target_norm=r0[step-1] if matched else None,scale=scale,**monitor)
   rows.append(row)
   if arm=='R0_original':r0.append(applied)
   del before,delta
   # One replaceable temporary snapshot permits exact continuation without
   # spending any extra Adam steps. It is removed when the branch completes.
   tsave(recovery,dict(model=detach(net.state_dict()),optimizer=opt.state_dict(),rng=rng(),rows=rows,step=step,source_scene=s,arm=arm))
   if step in [1,5]:evaluate(s,arm,step,holdouts)
   print('ADAM_STEP',s,arm,json.dumps(row),flush=True)
  rec=dict(source_scene=s,arm=arm,completed_steps=5,steps=rows,holdout_scenes=opposite,training_frame=0,holdout_frame=128,random_q0_overlap=tr['receipt']['random_overlap'],zero_count=tr['receipt']['zero'],denominator=den,backbone_recomputed=True,shared_native_graph_step1=True,fresh_full_network_steps=[2,3,4,5],all_parameters_Adam=True,training_TF32=False,heldout_TF32=True,train_BN_dropout=True,checkpoint=P['checkpoint'],optimizer_state_restored=True,initial_model_and_buffer_hash=restored_hash,parameters_saved_permanently=False)
  jsonsave(receiptpath,rec);journal.unlink();recovery.unlink(missing_ok=True);del opt;torch.cuda.empty_cache();print('BRANCH_COMPLETE',s,arm,flush=True)
 net.load_state_dict(initial);jsonsave(OUT/'receipts'/f'src_{s:04d}_complete.json',dict(source_scene=s,arms=7,adam_steps=35,parameters_restored=True));print('SOURCE_COMPLETE',s,flush=True)

if args.mode=='trace':
 assert args.scene is not None;trace_train(args.scene)
elif args.mode=='preflight':
 scenes=[args.scene] if args.scene is not None else SCENES
 for s in scenes:prepare_train(s);prepare_eval(s)
 net.load_state_dict(initial);jsonsave(OUT/'preflight_complete.json',dict(scenes=scenes,parameter_updates=0,parameters_restored=True))
else:
 assert args.scene is not None;run_source(args.scene)
