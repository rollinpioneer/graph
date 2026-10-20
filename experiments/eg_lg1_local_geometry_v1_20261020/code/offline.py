from common import *
import torch
from sklearn.metrics import roc_auc_score
torch.set_num_threads(1);t=time.time();d={k:v for k,v in np.load(RP/'prepared.npz').items()};z=np.load(R/'prepared_geometry.npz');gg=z['geometry'];rows=[]
lock=json.load(open(R/'method_lock.json'));heads=[('O',v['seed'],Path(v['path'])) for v in lock['old_O_heads']]+[(v['family'],v['seed'],Path(v['path'])) for v in lock['new_heads']]
for family,seed,path in heads:
 ck=torch.load(path,map_location='cpu',weights_only=False);net=network(324 if family=='O' else 390);net.load_state_dict(ck['model']);net.eval()
 for role in ['model_val','gate_cal']:
  for i in np.flatnonzero(d['role']==role):
   nf=(d['feature'][i]-ck['mean'])/ck['std'];base=np.concatenate([np.broadcast_to(nf[:,None,:],(128,6,320)),d['action'][i]],-1)
   if family!='O':geo=gg[i] if family=='G' else gg[i,permutation(d['scene'][i],d['frame'][i],128)];base=np.concatenate([base,geo],-1)
   with torch.no_grad():q=net(torch.from_numpy(base.copy())).squeeze(-1).numpy()
   loss=(q-utility(d['mu'][i]))**2;mu=d['mu'][i,:,0];positive=(mu>0)&(mu<=.4000001);c=d['collision'][i,:,0];e=d['empty'][i,:,0]
   for zone,ix in [('top64',np.arange(64)),('random64',np.arange(64,128)),('all128',np.arange(128))]:
    for comparison in ['noncollision_mu','all_failures']:
     use=(~c[ix])&(~e[ix])&(mu[ix]>0) if comparison=='noncollision_mu' else np.ones(len(ix),bool);y=positive[ix][use];score=q[ix,0][use];auc=float(roc_auc_score(y,score)) if len(np.unique(y))==2 else None
     rows.append(dict(family=family,seed=seed,role=role,scene=int(d['scene'][i]),frame=int(d['frame'][i]),zone=zone,comparison=comparison,n=len(y),positive=int(y.sum()),auc=auc,inversion=None if auc is None else 1-auc,six_pose_MSE=float(loss[ix].mean()),original_pose_MSE=float(loss[ix,0].mean())))
csv_write(R/'results/offline_per_frame.csv',rows);sr=[]
for family in ['O','G','Gshuffle']:
 for seed in SEEDS:
  for role in ['model_val','gate_cal']:
   for zone in ['top64','random64','all128']:
    for comp in ['noncollision_mu','all_failures']:
     vv=[v for v in rows if (v['family'],v['seed'],v['role'],v['zone'],v['comparison'])==(family,seed,role,zone,comp)];sv=[]
     for scene in sorted({v['scene'] for v in vv}):
      xx=[v for v in vv if v['scene']==scene];aa=[v['auc'] for v in xx if v['auc'] is not None]
      if aa:sv.append(np.mean(aa))
     sr.append(dict(family=family,seed=seed,role=role,zone=zone,comparison=comp,auc=float(np.mean(sv)) if sv else None,inversion=1-float(np.mean(sv)) if sv else None,six_pose_MSE=float(np.mean([v['six_pose_MSE'] for v in vv])),original_pose_MSE=float(np.mean([v['original_pose_MSE'] for v in vv]))))
csv_write(R/'results/offline_summary.csv',sr);write(R/'results/offline_complete.json',dict(status='PASS',seconds=time.time()-t,checkpoint_selection=False,application_decision=False));print('OFFLINE_DONE',time.time()-t,flush=True)
