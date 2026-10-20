import os,sys,json,time,hashlib,resource
from pathlib import Path
R=Path(os.environ['R']);sys.path.insert(0,str(R/'code'))
from compact import *
from concurrent.futures import ProcessPoolExecutor,as_completed
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

D=R/'diagnosis_s_vs_g';L=G/'eg_lg1_local_geometry_v1_20261020'
def auc(y,q):return float(roc_auc_score(y,q)) if len(np.unique(y))==2 else None
def metric(mu,q,collision,empty,top64=True):
 if top64:mu=mu[:,:64];q=q[:,:64];collision=collision[:,:64];empty=empty[:,:64]
 valid=(mu>0)&(mu<=.4000001);safe=~collision&~empty;strong=safe&valid;weak=safe&((mu>=.5999999)|(mu<=0));keep=strong|weak;pair=[]
 for v,z in zip(valid,q):
  if v.any() and (~v).any():pair.append(float(((z[v,None]<z[~v][None,:]).astype(float)+.5*(z[v,None]==z[~v][None,:])).mean()))
 return dict(success_AUROC=auc(valid.ravel(),q.ravel()),safe_strong_weak_AUROC=auc(strong[keep],q[keep]) if keep.any() else None,within_frame_success_pair_inversion=float(np.mean(pair)) if pair else None,frames_with_pairs=len(pair),candidates=int(mu.size))

def checkpoint_job(task):
 family,seed,step=task;t=time.time();ct=time.process_time();os.nice(10);torch.set_num_threads(1)
 path=(L if family=='G' else R)/f'heads/{family}/{seed}/step{step}.pt';ck=torch.load(path,map_location='cpu',weights_only=False);net=network(390) if family=='G' else initialize(family,seed);net.load_state_dict(ck['model']);net.eval();q=np.zeros((256,128,6),np.float32)
 with torch.no_grad():
  for i in range(256):
   xx=torch.from_numpy(ORIG[i])
   if family=='G':x=np.concatenate([ORIG[i],GG[i]],-1);out=net(torch.from_numpy(x)).squeeze(-1)
   elif family=='C':out=net(xx,torch.from_numpy(CC[i].copy()))
   else:out=net(xx,torch.from_numpy(TOK[i].copy()),torch.from_numpy(MASK[i].copy()))
   q[i]=out.numpy()
 assert np.isfinite(q).all();scores={}
 for role in ['fit','model_val','gate_cal']:
  ix=BASE['role']==role;mu=BASE['mu'][ix];pred=q[ix];target=utility(mu)
  scores[role]=dict(MSE=float(np.mean((pred-target)**2)),original_pose_MSE=float(np.mean((pred[:,:,0]-target[:,:,0])**2)),Top64_original=metric(mu[:,:,0],pred[:,:,0],BASE['collision'][ix,:,0],BASE['empty'][ix,:,0]),all128_original=metric(mu[:,:,0],pred[:,:,0],BASE['collision'][ix,:,0],BASE['empty'][ix,:,0],False))
 out=D/f'results/Q_{family}_{seed}_{step}.npz';np.savez_compressed(out,Q=q,checkpoint_sha256=np.array(sha(path)))
 result=dict(family=family,seed=seed,step=step,checkpoint=str(path),checkpoint_sha256=sha(path),roles=scores,seconds=time.time()-t,cpu_seconds=time.process_time()-ct,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,Q_path=str(out),Q_sha256=sha(out),optimizer_steps=0)
 write(D/f'results/checkpoint_{family}_{seed}_{step}.json',result);return result

if __name__=='__main__':
 start=time.time();torch.set_num_threads(1);BASE=dict(np.load(RP/'prepared.npz'));nf=(BASE['feature']-BASE['mean'])/BASE['std'];ORIG=np.concatenate([np.broadcast_to(nf[:,:,None,:],(256,128,6,320)),BASE['action']],-1).astype(np.float32)
 with np.load(L/'prepared_geometry.npz') as z:GG=z['geometry'].copy()
 CC=np.load(R/'cache/compact.npy',mmap_mode='r');TOK=np.load(R/'cache/tokens.npy',mmap_mode='r');MASK=np.load(R/'cache/mask.npy',mmap_mode='r')
 tasks=[(f,s,k) for f in ['C','S'] for s in SEEDS for k in [250,500,750,1000]]+[('G',s,1000) for s in SEEDS];records=[];workers=24
 with ProcessPoolExecutor(max_workers=workers) as pool:
  for fu in as_completed([pool.submit(checkpoint_job,x) for x in tasks]):
   v=fu.result();records.append(v);print('CKPT',v['family'],v['seed'],v['step'],round(v['seconds'],2),flush=True)
 write(D/'results/checkpoint_diagnostics.json',dict(status='PASS',tasks=records,workers=workers,wall_seconds=time.time()-start,CPU_worker_seconds_upper=workers*(time.time()-start),GPU_used=False,training_updates=0,no_C3_checkpoint_selection=True))
 print('CHECKPOINT_DIAG_COMPLETE',time.time()-start,flush=True)
