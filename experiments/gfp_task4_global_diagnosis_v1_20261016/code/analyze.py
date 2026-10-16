"""Trajectory-aware descriptive contrasts; no causal action-value claims."""
from pathlib import Path
import json, numpy as np, time
R=Path(__file__).resolve().parents[1]
P=Path('/home/xushijie3/work/gfp_task4_baseline_pilot_v1_20261015')
Z=Path('/home/xushijie3/work/gfp_guidance_distance_z01_20261015')
GROUPS=['detour_progress','aligned_progress','regress','stationary']
def save(n,x):(R/'reports'/n).write_text(json.dumps(x,indent=2,allow_nan=False))
def clustered(y,tid):
 y=np.asarray(y,dtype=float);tid=np.asarray(tid);ok=np.isfinite(y);y=y[ok];tid=tid[ok]
 u,inv=np.unique(tid,return_inverse=True)
 if not len(u):return dict(states=0,episodes=0)
 m=np.bincount(inv,weights=y)/np.bincount(inv);rng=np.random.default_rng(20261016)
 boot=m[rng.integers(0,len(m),size=(400,len(m)))].mean(1)
 return dict(states=len(y),episodes=len(u),trajectory_equal_mean=float(m.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),state_mean=float(y.mean()))
def controlled(y,s,positive=0,negative=2):
 # Same cell, head height, upright and COM velocity; cluster-robust association.
 g=s['group'];cell=s['cell'];common=np.intersect1d(np.unique(cell[g==positive]),np.unique(cell[g==negative]))
 keep=np.isin(g,[positive,negative])&np.isin(cell,common)&np.isfinite(y)
 if keep.sum()<40:return dict(status='INSUFFICIENT_COMMON_SUPPORT',n=int(keep.sum()))
 obs=s['observations'][keep];cat=cell[keep];x=np.c_[obs[:,23],obs[:,38],obs[:,39:42]];x=(x-x.mean(0))/np.maximum(x.std(0),1e-6)
 X=np.column_stack([np.ones(len(x)),(g[keep]==positive).astype(float),x]+[(cat==c).astype(float) for c in np.unique(cat)[1:]]);Y=y[keep];tid=s['trajectory'][keep];unique=np.unique(tid)
 beta=np.linalg.lstsq(X,Y,rcond=None)[0];res=Y-X@beta;bread=np.linalg.pinv(X.T@X);scores=np.array([X[tid==t].T@res[tid==t] for t in unique]);cov=bread@(scores.T@scores)@bread
 if len(unique)>1:cov*=len(unique)/(len(unique)-1)*(len(Y)-1)/max(1,len(Y)-np.linalg.matrix_rank(X))
 se=float(np.sqrt(max(0,cov[1,1])))
 return dict(status='ASSOCIATION_ONLY',n=len(Y),episodes=len(unique),cells=len(common),detour_minus_regress=float(beta[1]),ci95_cluster_normal=[float(beta[1]-1.96*se),float(beta[1]+1.96*se)],rank=int(np.linalg.matrix_rank(X)),columns=X.shape[1])
start=time.time();proposal=[];contrasts=[];uniform=[]
for task in [1,4]:
 src=Z if task==1 else P
 for seed in [0,1]:
  for cp in [200000,600000,1000000]:
   prior=dict(np.load(src/'arrays'/f's{seed}_{cp}.npz'))
   for split in ['train','validation']:
    s=dict(np.load(R/'arrays'/f't{task}_{split}_selected.npz'));v=dict(np.load(R/'arrays'/f't{task}_s{seed}_{cp}_{split}_probe.npz'))
    qd=v['qdata'].mean(-1);qa=v['qactor'].mean(-1);qv=v['qvabc'].mean(-1)
    metric=dict(actor_single_rmse=np.sqrt(v['actor_mse'][:,0]),actor_mean_rmse=np.sqrt(v['actor_mse']).mean(1),actor_best8_rmse=np.sqrt(v['actor_mse'].min(1)),vabc_single_rmse=np.sqrt(v['vabc_mse'][:,0]),vabc_mean_rmse=np.sqrt(v['vabc_mse']).mean(1),vabc_best8_rmse=np.sqrt(v['vabc_mse'].min(1)),actor_vabc_rmse=np.sqrt(v['actor_vabc_mse'].mean(1)),vabc_flow_mse=v['vabc_flow_mse'].mean(1),qdata_minus_actor=(qd-qa).mean(1),qdata_minus_vabc=(qd-qv).mean(1),data_rank_actor=(qd>=qa).mean(1),data_rank_vabc=(qd>=qv).mean(1),critic_agreement_actor=(np.prod(v['qdata']-v['qactor'],axis=-1)>=0).mean(1))
    for group,label in enumerate(GROUPS):
     m=s['group']==group
     proposal.append(dict(task=task,seed=seed,checkpoint=cp,split=split,group=label,metrics={k:clustered(y[m],s['trajectory'][m]) for k,y in metric.items()}))
    for k in ['actor_mean_rmse','vabc_mean_rmse','vabc_flow_mse','actor_vabc_rmse','qdata_minus_actor','qdata_minus_vabc','data_rank_actor','data_rank_vabc']:
     contrasts.append(dict(task=task,seed=seed,checkpoint=cp,split=split,metric=k,**controlled(metric[k],s)))
    # Reuse ORIGINAL uniform batches and original guidance denominator.
    sample=dict(np.load(src/'arrays'/f'{split}_samples.npz'));labels=dict(np.load(R/'arrays'/f't{task}_{split}_labels.npz'));ix=sample['indices'];which=(prior['repeat']==0)&(prior['split']==(split=='validation'));u={k:a[which] for k,a in prior.items()}
    assert np.array_equal(u['sample_index'],np.arange(len(ix)))
    good=labels['valid_window'][ix]&labels['corridor'][ix];pr=labels['progress100'][ix];ed=labels['euclidean_change100'][ix]
    masks=[good&(pr>=1)&(ed>.5),good&(pr>=1)&(ed<-.5),good&(pr<=-1)]
    qd0=u['qdata'].mean(1);qa0=u['qactor'].mean(1);qv0=u['qvabc'].mean(1)
    measures=dict(guidance=u['g'],neutral=((u['g']>=.4)&(u['g']<=.6)).astype(float),delta=u['delta'],data_minus_actor=qd0-qa0,data_minus_vabc=qd0-qv0,critic_disagreement=(np.prod(u['member_delta'],axis=1)<0).astype(float),data_below_actor=(qd0<qa0).astype(float),data_below_vabc=(qd0<qv0).astype(float))
    for label,m in zip(GROUPS[:3],masks):uniform.append(dict(task=task,seed=seed,checkpoint=cp,split=split,group=label,metrics={k:clustered(y[m],sample['trajectory'][m]) for k,y in measures.items()}))
save('proposal_fit.json',proposal);save('controlled_contrasts.json',contrasts);save('original_batch_guidance.json',uniform)
save('analysis_receipt.json',dict(seconds=time.time()-start,environment_steps=0,optimizer_updates=0,inference_reused_only=True,cluster_resamples=400,controlled_intervals='cluster sandwich normal approximation, exploratory',proxy='100-step observed path progress under changing collection goals; not action advantage'))
print('ANALYSIS_DONE',len(proposal),len(contrasts),len(uniform),flush=True)
