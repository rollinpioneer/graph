"""Zero-training, no new GT-quality calls; exact official selection replay."""
import os,sys,json,time,hashlib,inspect,importlib.util,argparse,resource
from pathlib import Path
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor,as_completed
from scipy.spatial import cKDTree
from scipy.io import loadmat
from PIL import Image
torch.set_num_threads(1)
R=Path(os.environ['TASK_ROOT']);BASE=R.parent
P=BASE/'eg_sub25_b_f05_repl_v1_20261019';L=BASE/'eg_lg1_local_geometry_v1_20261020';RP=BASE/'eg_b_global_rank_pilot_v1_20261020';FAST=BASE/'eg_lg1_c4_confirm_speed_v1_20261020'
DATA=BASE/'eg_sub25_v1_20261015/data/graspnet';CODE=BASE/'eg_sub25_v1_20261015/runtime/EconomicGrasp'
os.environ['R']=str(FAST);sys.path[:0]=[str(FAST/'code'),str(CODE),str(CODE/'libs/pointnet2')]
from geometry_fast import visible_fast,geom_mask,CachedTree
from common import GeoScorer,geom_input
from scoring import fused_score,replace_score
from utils.collision_detector import ModelFreeCollisionDetector
from utils.data_utils import CameraInfo,create_point_cloud_from_depth_image,get_workspace_mask
from graspnetAPI import GraspGroup,GraspNetEval
import graspnetAPI.utils.eval_utils as EU
import graspnetAPI.graspnet_eval as GE
import open3d as o3d

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def util(mu):return ((mu[...,None]>0)&(mu[...,None]<=np.array([.2,.4,.6,.8,1.,1.2])+1e-7)).mean(-1)
W=np.array([sum(1/k for k in range(j,51))/50 for j in range(1,51)])
SOURCE=inspect.getsource(EU.eval_grasp)
SPEC=SOURCE[:SOURCE.index('    ## collision detection')].replace('def eval_grasp(','def select_only(')+'\n    return grasp_list\n'
HEADS=None;D=None

def initialize():
 global HEADS,D
 D=dict(np.load(RP/'prepared.npz'))
 lock=json.loads((L/'method_lock.json').read_text());HEADS=[(h['seed'],GeoScorer(Path(h['path']),h['sha256'])) for h in lock['new_heads'] if h['family']=='G']
 assert len(HEADS)==3

class Scene:
 def __init__(self,s):
  self.s=s;self.ge=GraspNetEval(root=str(DATA),camera='kinect',split='all')
  ids=self.ge.get_model_poses(s,0)[0]
  self.models=[GE.voxel_sample_points(np.asarray(o3d.io.read_point_cloud(str(DATA/f'models/{oid:03d}/nontextured.ply')).points),.008) for oid in ids]
 def frame(self,f,pred):
  ids,self.poses,cam,align=self.ge.get_model_poses(self.s,f)
  self.mt=[GE.transform_points(m,p) for m,p in zip(self.models,self.poses)];self.objpts=np.concatenate(self.mt)
  seg=np.concatenate([np.full(len(m),i) for i,m in enumerate(self.mt)])
  # Exact official nearest-object assignment, once per frame, chunked only in query axis.
  owner_index=np.concatenate([EU.compute_closest_points(pred[k:k+64,13:16].astype(float),self.objpts) for k in range(0,1024,64)])
  lookup={p.tobytes():int(j) for p,j in zip(pred[:,13:16].astype(float),owner_index)}
  self.namespace=EU.__dict__.copy();exec(SPEC,self.namespace)
  self.namespace['compute_closest_points']=lambda a,b:np.array([lookup[np.asarray(v,dtype=float).tobytes()] for v in a])
  transformed={id(m):v for m,v in zip(self.models,self.mt)}
  self.namespace['transform_points']=lambda m,p:transformed[id(m)]
  self.select=self.namespace['select_only']
  folder=DATA/f'scenes/scene_{self.s:04d}/kinect';depth=np.asarray(Image.open(folder/f'depth/{f:04d}.png'));segimg=np.asarray(Image.open(folder/f'label/{f:04d}.png'));meta=loadmat(folder/f'meta/{f:04d}.mat');K=meta['intrinsic_matrix']
  camera=CameraInfo(1280.,720.,K[0,0],K[1,1],K[0,2],K[1,2],meta['factor_depth']);cloud=create_point_cloud_from_depth_image(depth,camera,organized=True)
  work=get_workspace_mask(cloud,segimg,trans=align@cam,organized=True,outlier=.02)
  det=ModelFreeCollisionDetector(cloud[(depth>0)&work],voxel_size=.005)
  self.cd=np.concatenate([det.detect(GraspGroup(pred[k:k+64].copy()),approach_dist=.05,collision_thresh=.01) for k in range(0,1024,64)])
  self.pred=pred.astype(float).copy();self.pred[:,1]=self.pred[:,1].clip(0,.1)
  self.ident={}
  for i,g in enumerate(self.pred):self.ident.setdefault(g[1:].tobytes(),i)
  self.owners=seg[owner_index]
 def selection(self,score,verify=False):
  p=self.pred.copy();p[:,0]=score;g=GraspGroup(p[~self.cd])
  if not len(g):return np.array([],int),np.array([],int)
  # Native NMS is retained; no approximate adjacency used for evaluation.
  nms=g.nms(.03,np.pi/6);nids=np.array([self.ident[v[1:].tobytes()] for v in nms.grasp_group_array])
  selected=np.concatenate(self.select(g,self.models,None,self.poses,None,voxel_size=.008,TOP_K=50));selected=selected[np.argsort(selected[:,0])[::-1]]
  ids=np.array([self.ident[v[1:].tobytes()] for v in selected],int)
  if verify:
   ns=EU.__dict__.copy();exec(SPEC,ns)
   exact=np.concatenate(ns['select_only'](g,self.models,None,self.poses,None,voxel_size=.008,TOP_K=50));exact=exact[np.argsort(exact[:,0])[::-1]]
   assert np.array_equal(exact,selected)
  return ids,nids

def pairs(pred,score,cd,top,seed,scene,frame):
 order=np.lexsort((np.arange(1024),-score));order=order[~cd[order]];rank=np.empty(1024,int);rank[np.lexsort((np.arange(1024),-score))]=np.arange(1024)
 out=[];used=set()
 def add(i,j,kind):
  if i==j:return False
  key=tuple(sorted((int(i),int(j))))
  if key in used:return False
  used.add(key)
  if rank[i]>rank[j]:i,j=j,i
  out.append((int(i),int(j),kind));return True
 if len(order)<2:return out
 leader=int(top[0]) if len(top) else int(order[0])
 for j in order:
  if sum(k[2]=='top1' for k in out)>=16:break
  add(leader,j,'top1')
 adjacent=list(zip(order[:127],order[1:128]));adjacent.sort(key=lambda ij:(abs(score[ij[0]]-score[ij[1]]),rank[ij[0]],int(ij[0])))
 for i,j in adjacent:
  if sum(k[2]=='close_score' for k in out)>=16:break
  add(i,j,'close_score')
 subset=order[:256];edges=list(cKDTree(pred[subset,13:16]).query_pairs(.03));edges.sort(key=lambda ij:(max(rank[subset[ij[0]]],rank[subset[ij[1]]]),min(rank[subset[ij[0]]],rank[subset[ij[1]]])))
 for ii,jj in edges:
  if sum(k[2]=='nms' for k in out)>=16:break
  i,j=subset[ii],subset[jj]
  if tuple(sorted((int(i),int(j)))) in used:continue
  if len(GraspGroup(pred[[i,j]].astype(float)).nms(.03,np.pi/6))==1:add(i,j,'nms')
 rng=np.random.default_rng(np.random.SeedSequence([20261021,seed,scene,frame]));tail=order[64:]
 for k,j in enumerate(rng.permutation(tail)):
  if sum(v[2]=='explore' for v in out)>=16:break
  add(order[k%min(16,len(order))],j,'explore')
 for i,j in zip(order[:-1],order[1:]):
  if len(out)>=64:break
  add(i,j,'fill_adjacent')
 return out[:64]

def coefficients(ids):
 c=np.zeros(1024);np.add.at(c,ids[:50],W[:min(len(ids),50)]);return c
def bounds(c,known,value):
 base=float(np.dot(c[known],value[known]));v=c[~known];return base+float(np.minimum(v,0).sum()),base+float(np.maximum(v,0).sum())

def job(task):
 scene,only=task;os.nice(10);start=time.time();cpu=time.process_time();n=Scene(scene);records=[];fr=[];identities=[]
 indices=np.flatnonzero(D['scene']==scene)
 if only is not None:indices=indices[:only]
 for idx in indices:
  # User removed the time cap on 2026-10-21; record actual cost without truncating scenes.
  f=int(D['frame'][idx]);ep=P/f'B/exports/subset/scene_{scene:04d}/{f:04d}.npz';eh=sha(ep)
  with np.load(ep) as z:pred=z['pred'];feat=z['feature320'];act=z['original_action'];assert np.array_equal(feat[D['ids'][idx]],D['feature'][idx]);assert np.array_equal(act[D['ids'][idx]],D['action'][idx,:,0])
  n.frame(f,pred);gp=L/f'geometry/head/{scene:04d}/{f:04d}.npz'
  with np.load(gp) as z:assert str(z['export_sha256'])==eh;label_geom=z['raw'][:,0]
  cloud,obs=visible_fast(scene,f,pred);tree=CachedTree(cKDTree(cloud),pred);raw=np.asarray([geom_mask(cloud,tree,p) for p in pred],np.float32)
  assert np.array_equal(raw[D['ids'][idx]],label_geom,equal_nan=True)
  known=np.zeros(1024,bool);known[D['ids'][idx]]=True;mu=np.full(1024,np.nan);mu[D['ids'][idx]]=D['mu'][idx,:,0];u=util(mu);valid=((mu>0)&(mu<=.4000001)).astype(float)
  for seed,head in HEADS:
   q=head.quality(feat,act,raw,scene,f);score,info=fused_score(pred[:,0],q);ids,nids=n.selection(score,verify=(idx==indices[0] and seed==20261018));base_c=coefficients(ids)
   gg=geom_input(label_geom,head.ck['geom_norm']);x=np.concatenate([(D['feature'][idx]-head.ck['mean'])/head.ck['std'],D['action'][idx,:,0],gg],-1).astype(np.float32)
   with torch.no_grad():q_label=head.net(torch.from_numpy(x)).squeeze(-1).numpy()
   assert np.allclose(q[D['ids'][idx]],q_label,atol=1e-7,rtol=1e-6)
   ps=pairs(pred,score,n.cd,ids,seed,scene,f);rr=[]
   for i,j,kind in ps:
    qq=q.copy();qq[i],qq[j]=q[j],q[i];ss,_=fused_score(pred[:,0],qq)
    if np.array_equal(ss,score):new,nn=ids,nids
    else:new,nn=n.selection(ss)
    c=coefficients(new)-base_c;lo,hi=bounds(c,known,u)
    tc=np.zeros(1024)
    if len(ids):tc[ids[0]]-=1
    if len(new):tc[new[0]]+=1
    tl,th=bounds(tc,known,valid);changed=not np.array_equal(ids,new);nchange=set(nids)!=set(nn);rev=bool(ss[j]>ss[i])
    certified=bool(lo>=-1e-12 and tl>=-1e-12 and (lo>1e-12 or tl>1e-12))
    pair_known=bool(known[i] and known[j]);ordinary=float(u[j]-u[i]) if pair_known else None
    # A context conflict is not an ordinary utility tie: known strict utility direction vs identifiable AP direction.
    conflict=bool(rev and pair_known and ((ordinary>1e-12 and hi< -1e-12) or (ordinary< -1e-12 and lo>1e-12)))
    rec=dict(scene=scene,frame=f,role=str(D['role'][idx]),seed=seed,i=i,j=j,kind=kind,changed=changed,nms_set_changed=nchange,selected_set_changed=set(ids)!=set(new),pair_reversed=rev,pair_known=pair_known,ordinary_utility_gain=ordinary,ap_lower=100*lo,ap_upper=100*hi,top1_lower=tl,top1_upper=th,certified=certified,supervision_eligible=certified and rev,context_conflict=conflict,unknown_changed=int(((np.abs(c)>1e-12)&~known).sum()),baseline_top1=int(ids[0]) if len(ids) else -1,new_top1=int(new[0]) if len(new) else -1)
    records.append(rec);rr.append(rec)
   supported=[r for r in rr if r['supervision_eligible']]
   fr.append(dict(scene=scene,frame=f,role=str(D['role'][idx]),seed=seed,pairs=len(rr),changed=sum(r['changed'] for r in rr),certified=sum(r['certified'] for r in rr),eligible=len(supported),conflicts=sum(r['context_conflict'] for r in rr),nms_eligible=sum(r['nms_set_changed'] for r in supported),known_top1=bool(len(ids) and known[ids[0]]),known_top50=int(known[ids[:50]].sum()),selected_count=min(50,len(ids)),best_AP_lower=max([0]+[r['ap_lower'] for r in supported]),best_Top1_lower=max([0]+[r['top1_lower'] for r in supported])))
  identities.append(dict(scene=scene,frame=f,export_sha256=eh,geometry_cache_sha256=sha(gp),geometry_matches_G=True,CD_rejected=int(n.cd.sum()),**obs))
 out=R/'results'/('pilot' if only else 'scenes');out.mkdir(exist_ok=True)
 with (out/f'{scene:04d}_swaps.jsonl').open('w') as ff:
  for r in records:ff.write(json.dumps(r,allow_nan=False)+'\n')
 result=dict(scene=scene,status='PASS',pid=os.getpid(),start=start,end=time.time(),seconds=time.time()-start,cpu_seconds=time.process_time()-cpu,frames=fr,identities=identities,swaps=len(records),max_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,new_quality_labels=0)
 save(out/f'{scene:04d}.json',result);return {k:v for k,v in result.items() if k not in ['frames','identities']}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--scenes',default='');p.add_argument('--workers',type=int,default=1);p.add_argument('--node',required=True);a=p.parse_args();initialize()
 scenes=[int(x) for x in a.scenes.split(',')] if a.scenes else [int(D['scene'][0])]
 start=time.time();results=[]
 if a.pilot:results=[job((scenes[0],1))]
 else:
  with ProcessPoolExecutor(max_workers=a.workers) as ex:
   fut=[ex.submit(job,(s,None)) for s in scenes]
   for ff in as_completed(fut):r=ff.result();results.append(r);print('SCENE_DONE',json.dumps(r),flush=True)
 save(R/'results'/f'{a.node}_{"pilot" if a.pilot else "run"}.json',dict(status='PASS',node=a.node,workers=a.workers,wall_seconds=time.time()-start,worker_seconds=sum(r['seconds'] for r in results),receipts=results));print('DONE',a.node,time.time()-start,flush=True)
