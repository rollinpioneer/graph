import numpy as np, glob, os, sys, json
S=sys.argv[1]; out=sys.argv[2]; rng=np.random.default_rng(0)
PP=f'{S}/n0pp/eg_n0_nref_main_20261024/per_pose'
def q_of(mu,col,emp):
    q=np.zeros(mu.shape); ok=~col&~emp&(mu>0); q[ok]=(1.4-mu[ok])/1.2; return q
def feats(n,q,nom,logit=None):
    vp=rng.normal(0,1,(n,256)); vp[:,:12]=q+rng.normal(0,0.15,q.shape)
    if logit is None: logit=rng.normal(0,1,(n,12))
    return dict(vp=vp.astype(np.float32),ang=rng.normal(0,1,(n,64)).astype(np.float32),logit=logit.astype(np.float32))
# probe frames: 80 scenes x 2 frames synthetic
N0=[67,4,50,56,14,65,62,30,45,1,96,71];EX=[2,9,31,33,64,68,76,82]
for s in sorted(set(range(100))-set(N0)-set(EX)):
    for f in [0,32]:
        n=40;nom=rng.integers(0,12,n);mu=rng.choice([0.2,0.4,0.6,0.8,1.0,1.2,-1],(n,12));col=rng.random((n,12))<0.3;emp=np.zeros((n,12),bool)
        col[np.arange(n),nom]=False;mu[col]=-1;I1=rng.random((n,12));cid=np.arange(n)
        np.savez(f'{out}/probe_eval/{s:04d}_{f:04d}.npz',candidate_id=cid,original_angle=nom,mu=mu,collision=col,empty=emp,I1=I1)
        np.savez(f'{out}/features/probe_{s:04d}_{f:04d}.npz',candidate_id=cid,original_angle=nom,**feats(n,q_of(mu,col,emp),nom))
for fp in sorted(glob.glob(f'{PP}/train_*.npz')):
    a=np.load(fp);s,f=map(int,os.path.basename(fp)[:-4].split('_')[1:]);n=len(a['candidate_id'])
    mu=rng.choice([0.2,0.4,0.6,0.8,1.0,1.2,-1],(n,12)).astype(float);mu[a['collision']]=-1;I1=rng.random((n,12))
    np.savez(f'{out}/cal_eval/{s:04d}_{f:04d}.npz',candidate_id=a['candidate_id'],original_angle=a['original_angle'],mu=mu,collision=a['collision'],empty=a['empty'],I1=I1)
    np.savez(f'{out}/features/cal_{s:04d}_{f:04d}.npz',candidate_id=a['candidate_id'],original_angle=a['original_angle'],**feats(n,q_of(mu,a['collision'],a['empty']),a['original_angle']))
for split,ed in [('C3',f'{S}/n0pp/eg_physical_diagnostics_20261024/eg_i1_angle_physics_20261024/evaluation'),('P2',f'{S}/p2/eg_p2_i1_mu_20261024/evaluation')]:
    for fp in sorted(glob.glob(f'{PP}/{split}_*.npz')):
        s,f=map(int,os.path.basename(fp)[:-4].split('_')[1:]);e=np.load(f'{ed}/{s:04d}_{f:04d}.npz');n=len(e['candidate_id'])
        lg=e['angle_logits'][:12].T if 'angle_logits' in e.files else None
        np.savez(f'{out}/features/{split}_{s:04d}_{f:04d}.npz',candidate_id=e['candidate_id'],original_angle=e['original_angle'],**feats(n,q_of(e['mu'],e['collision'],e['empty']),e['original_angle'],lg))
json.dump(dict(features_dir=f'{out}/features',probe_eval_dir=f'{out}/probe_eval',cal_eval_dir=f'{out}/cal_eval',
  c3_eval_dir=f'{S}/n0pp/eg_physical_diagnostics_20261024/eg_i1_angle_physics_20261024/evaluation',p2_eval_dir=f'{S}/p2/eg_p2_i1_mu_20261024/evaluation',
  per_pose_dir=PP,frozen_models=f'{S}/testA/out/frozen_models_P2.json'),open(f'{out}/config.json','w'),indent=1)
