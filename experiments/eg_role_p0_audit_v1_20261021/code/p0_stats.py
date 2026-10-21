import os,sys,json,time,hashlib,csv,argparse,importlib.util
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from scipy.stats import spearmanr
torch.set_num_threads(1)
B=Path.home()/'work/grasp';R=B/'eg_role_p0_audit_v1_20261021';P=B/'eg_sub25_b_f05_repl_v1_20261019';L=B/'eg_lg1_local_geometry_v1_20261020';RP=B/'eg_b_global_rank_pilot_v1_20261020'
sys.path.insert(0,str(B/'eg_lqr_build_confirm_v3_20261019/code'))
from scoring import fused_score
SEEDS=[20261018,20261019,20261020];GROUPS=dict(seen=[100,106,115,124],similar=[132,139,150,157],novel=[160,169,176,186])
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,allow_nan=False));q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def util(mu):return ((mu[...,None]>0)&(mu[...,None]<=np.array([.2,.4,.6,.8,1.,1.2])+1e-7)).mean(-1)
def corr(x,y):return float(np.corrcoef(x,y)[0,1]) if len(x)>1 and np.std(x)>0 and np.std(y)>0 else None
def rho(x,y):return float(spearmanr(x,y).statistic) if len(x)>1 and np.std(x)>0 and np.std(y)>0 else None
def rank(score):
 o=np.lexsort((np.arange(len(score)),-score));r=np.empty(len(o),int);r[o]=np.arange(1,len(o)+1);return r
def basic(scene,frame,seed,s,q,cd,mu,ids,role):
 known=np.isfinite(mu);valid=(mu>0)&(mu<=.4000001);score,info=fused_score(s,q);avail=np.flatnonzero(~cd);top=int(ids[0]) if len(ids) else -1;tg=int(avail[np.argmax(q[avail])]) if len(avail) else -1
 failed=top>=0 and known[top] and not valid[top];good=known&valid;rr=bool(failed and (good&~cd).any());dd=bool(failed and (good&cd).any());unknown=bool((~known).any());over=bool(top>=0 and tg>=0 and known[top] and known[tg] and valid[tg] and not valid[top]);rescue=bool(top>=0 and tg>=0 and known[top] and known[tg] and valid[top] and not valid[tg]);uncertain=bool(top!=tg and top>=0 and tg>=0 and not(known[top] and known[tg]))
 other=(good|~known)&~cd
 if top>=0:other[top]=False
 sr,qr=rank(score),rank(q);witness=np.flatnonzero(good&~cd);best=min((int(sr[i]) for i in witness),default=None);bestq=min((int(qr[i]) for i in witness),default=None)
 row=dict(scene=scene,frame=frame,seed=seed,role=role,top=top,gonly_top=tg,known_top=bool(top>=0 and known[top]),failed=bool(failed),R=rr,R_in50=bool(failed and good[ids[:50]].any()),D=dd,A=None,A_cd=None,C_cert=False,U=bool(failed and not(rr or dd) and unknown),known_fraction=float(known.mean()),R_best_S_rank=best,R_best_Q_rank=bestq,ranking_space_upper=bool(failed and other.any()),overturn=over,rescue=rescue,overturn_upper=bool(over or uncertain),fallback=info['fallback'],spearman_all=rho(s,q),spearman_cd=rho(s[~cd],q[~cd]))
 order=np.argsort(s,kind='stable');row['spearman_deciles']=[rho(s[v],q[v]) for v in np.array_split(order,10)]
 if over:
  es=.5*float(s[top]-s[tg])/float(np.std(s,dtype=np.float64));gs=.5*float(q[top]-q[tg])/float(np.std(q,dtype=np.float64));assert gs<=1e-12 and es+gs>=-1e-10
  row['witness']=dict(scene=scene,frame=frame,seed=seed,failure=top,gonly_success=tg,s=[float(s[top]),float(s[tg])],q=[float(q[top]),float(q[tg])],S=[float(score[top]),float(score[tg])],EG_contribution=es,G_contribution=gs,mu=[float(mu[top]),float(mu[tg])])
 return row
def c3_scene(scene):
 start=time.time();rows=[];pair=[];role=next(k for k,v in GROUPS.items() if scene in v)
 for frame in range(256):
  ep=P/f'B/exports/subset/scene_{scene:04d}/{frame:04d}.npz'
  with np.load(ep) as z:s=z['pred'][:,0].astype(float)
  with np.load(R/f'union/{scene:04d}/{frame:04d}.npz') as z:mu=z['quality'];cd=z['cd_mask']
  with np.load(L/f'evaluation/development/{scene:04d}/{frame:04d}.npz') as z:
   for seed in SEEDS:
    q=z[f'Q_G_{seed}'].astype(float);ids=z[f'selected_G_{seed}_CD'];row=basic(scene,frame,seed,s,q,cd,mu,ids,role);rows.append(row);score,_=fused_score(s,q)
    order=np.lexsort((np.arange(1024),-q));pool=np.union1d(ids[:50],order[~cd[order]][:50]);pool=pool[np.isfinite(mu[pool])&~cd[pool]];u=util(mu[pool]);i,j=np.triu_indices(len(pool),1);delta=u[i]-u[j];mask=delta!=0;i,j,delta=i[mask],j[mask],delta[mask];ei=np.sign(s[pool[i]]-s[pool[j]])*np.sign(delta);qi=np.sign(q[pool[i]]-q[pool[j]])*np.sign(delta);fi=np.sign(score[pool[i]]-score[pool[j]])*np.sign(delta)
    pair.append(dict(scene=scene,frame=frame,seed=seed,n=len(delta),both_correct=int(((ei>0)&(qi>0)).sum()),EG_correct_G_wrong=int(((ei>0)&(qi<0)).sum()),EG_wrong_G_correct=int(((ei<0)&(qi>0)).sum()),both_wrong=int(((ei<0)&(qi<0)).sum()),ties=int(((ei==0)|(qi==0)).sum()),overturn=int(((ei<0)&(qi>0)&(fi<0)).sum()),rescue=int(((ei>0)&(qi<0)&(fi>0)).sum())))
 save(R/f'results/c3_{scene:04d}.json',dict(rows=rows,pairs=pair,seconds=time.time()-start,cpu_seconds=time.process_time(),host=os.uname().nodename));return dict(scene=scene,seconds=time.time()-start,rows=len(rows))
def head_init():
 global A,D,GG,HEADS
 src=B/'eg_decision_relation_v1_20261021/code/diagnose.py';spec=importlib.util.spec_from_loader('a_reference',loader=None);A=importlib.util.module_from_spec(spec);sys.modules['a_reference']=A;A.__file__=str(src);os.environ['TASK_ROOT']=str(R)
 # Preserve a reference to the already-created collision detector; no scientific calculation changes.
 exec(compile(src.read_text().replace('self.cd=np.concatenate','self.det=det;self.cd=np.concatenate'),str(src),'exec'),A.__dict__);A.initialize();D=A.D;HEADS=A.HEADS;GG=dict(np.load(L/'prepared_geometry.npz'))
def head_scene(scene,limit=8):
 start=time.time();n=A.Scene(scene);rows=[];disc=[];bias=[];ident=[]
 for idx in np.flatnonzero(D['scene']==scene)[:limit]:
  frame=int(D['frame'][idx]);ep=P/f'B/exports/subset/scene_{scene:04d}/{frame:04d}.npz'
  with np.load(ep) as z:pred=z['pred'];feat=z['feature320'];act=z['original_action'];hyp=z['hypotheses'];assert np.array_equal(hyp[:,0],pred)
  n.frame(frame,pred);cloud,obs=A.visible_fast(scene,frame,pred);tree=A.CachedTree(A.cKDTree(cloud),pred);raw=np.asarray([A.geom_mask(cloud,tree,p) for p in pred],np.float32)
  gp=L/f'geometry/head/{scene:04d}/{frame:04d}.npz'
  with np.load(gp) as z:assert np.array_equal(raw[D['ids'][idx]],z['raw'][:,0],equal_nan=True)
  mu=np.full(1024,np.nan);mu[D['ids'][idx]]=D['mu'][idx,:,0];role=str(D['role'][idx]);u=util(D['mu'][idx]);a0map={int(i):j for j,i in enumerate(D['ids'][idx])};decile=np.minimum((rank(-pred[:,0])-1)//103,9)
  for seed,head in HEADS:
   q=head.quality(feat,act,raw,scene,frame);score,_=fused_score(pred[:,0],q);ids,_=n.selection(score,verify=(frame==0 and seed==SEEDS[0]));row=basic(scene,frame,seed,pred[:,0].astype(float),q.astype(float),n.cd,mu,ids,role)
   top=row['top'];j=a0map.get(top)
   if row['failed'] and j is not None:
    alt=(D['mu'][idx,j,1:]>0)&(D['mu'][idx,j,1:]<=.4000001);row['A']=bool(alt.any());cdalt=n.det.detect(A.GraspGroup(hyp[top,1:].copy()),approach_dist=.05,collision_thresh=.01);row['A_cd']=bool((alt&~cdalt).any());row['U']=bool(not(row['R'] or row['D'] or row['A']))
   if row['overturn'] and j is not None:row['potential_gap_at_failure']=float(u[j].max()-u[j,0]);row['failure_EG_decile']=int(decile[top]);row['potential_above_actual_one_bin']=bool(u[j].max()-u[j,0]>=1/6-1e-12)
   rows.append(row)
   x=np.concatenate([np.broadcast_to((D['feature'][idx]-head.ck['mean'])[:,None,:]/head.ck['std'],(128,6,320)),D['action'][idx],GG['geometry'][idx]],-1).astype(np.float32)
   with torch.no_grad():qq=head.net(torch.from_numpy(x)).squeeze(-1).numpy().astype(float)
   assert np.allclose(q[D['ids'][idx]],qq[:,0],atol=1e-6,rtol=1e-6)
   for pool,sl in [('Top64',slice(0,64)),('random64',slice(64,128))]:
    y=u[sl];p=qq[sl];uc=y-y.mean(1,keepdims=True);qc=p-p.mean(1,keepdims=True);gm=(p.mean(1)-y.mean(1))**2;within=((qc-uc)**2).mean(1);mse=((p-y)**2).mean(1);assert np.allclose(mse,gm+within,atol=1e-14)
    ii,jj=np.triu_indices(6,1);du=y[:,ii]-y[:,jj];dq=p[:,ii]-p[:,jj];different=du!=0;correct=(np.sign(du)==np.sign(dq))&different;a0pairs=different&(ii==0)[None,:];var=float((uc**2).sum());err=float(((qc-uc)**2).sum());regret=y.max(1)-y[np.arange(len(y)),p.argmax(1)]
    disc.append(dict(scene=scene,frame=frame,seed=seed,role=role,pool=pool,mse=float(mse.mean()),group_mean_mse=float(gm.mean()),within_mse=float(within.mean()),within_R2=1-err/var if var>0 else None,within_SSE=err,within_SST=var,constant_groups=int((np.ptp(y,axis=1)==0).sum()),pair_accuracy=float(correct.sum()/different.sum()) if different.any() else None,a0_pair_accuracy=float((correct&a0pairs).sum()/a0pairs.sum()) if a0pairs.any() else None,regret=float(regret.mean())))
   xx=u.mean(1)-u[:,0];yy=qq[:,0]-u[:,0]
   strata=[('all',np.ones(128,bool))]+[(f'u0_bin{b}',np.rint(u[:,0]*6)==b) for b in range(7)]+[(f'EG_decile{b}',decile[D['ids'][idx]]==b) for b in range(10)]
   for name,mask in strata:
    xxm,yym=xx[mask],yy[mask];slope=float(np.mean((xxm-xxm.mean())*(yym-yym.mean()))/np.var(xxm)) if len(xxm)>1 and np.var(xxm)>0 else None
    bias.append(dict(scene=scene,frame=frame,seed=seed,role=role,stratum=name,n=int(mask.sum()),correlation=corr(xxm,yym),slope=slope))
  ident.append(dict(scene=scene,frame=frame,export_sha256=sha(ep),geometry_sha256=sha(gp),geometry_matches_G=True,**obs))
 save(R/f'results/head_{scene:04d}_{limit}.json',dict(rows=rows,discrimination=disc,bias=bias,identities=ident,seconds=time.time()-start,cpu_seconds=time.process_time(),host=os.uname().nodename));return dict(scene=scene,frames=limit,seconds=time.time()-start)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['C3','head'],required=True);p.add_argument('--scenes',required=True);p.add_argument('--workers',type=int,default=1);p.add_argument('--frames',type=int,default=8);a=p.parse_args();facts=json.loads((R/'P0_fact_checks.json').read_text());assert facts['F4'] and facts['F5']
 if a.mode=='head':head_init()
 with ProcessPoolExecutor(a.workers) as pool:
  futures=[pool.submit(head_scene,int(s),a.frames) if a.mode=='head' else pool.submit(c3_scene,int(s)) for s in a.scenes.split(',')]
  for f in as_completed(futures):print(json.dumps(f.result()),flush=True)
