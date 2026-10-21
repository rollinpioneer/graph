from common import *
import torch
from scipy.stats import rankdata
torch.set_num_threads(1)
L=G/'eg_lg1_local_geometry_v1_20261020';P0=G/'eg_role_p0_audit_v1_20261021';d=dict(np.load(RP/'prepared.npz'));z=dict(np.load(L/'prepared_geometry.npz'));nf=(d['feature']-d['mean'])/d['std'];X=np.concatenate([np.broadcast_to(nf[:,:,None,:],(256,128,6,320)),d['action'],z['geometry']],-1).astype(np.float32);Y=utility(d['mu']);results=[];predictions={};heads=[];training=[]
for seed in SEEDS:
 ck=torch.load(L/f'heads/G/{seed}/step1000.pt',map_location='cpu',weights_only=False)
 for fam,folder in [('G',L),('Grerun',R),('GW',R)]:
  path=folder/f"heads/{'G' if fam=='Grerun' else fam}/{seed}/step1000.pt";v=torch.load(path,map_location='cpu',weights_only=False);net=network(390);net.load_state_dict(v['model']);net.eval();q=[]
  with torch.no_grad():
   for xx in X:q.append(net(torch.from_numpy(xx)).squeeze(-1).numpy())
  predictions[fam,seed]=np.asarray(q)
  if fam!='G':
   assert v['initial_state_sha256']==ck['initial_state_sha256'];assert v['sample_stream_sha256']==ck['sample_stream_sha256']
   rr=json.loads((R/f"results/train_{'G' if fam=='Grerun' else fam}_{seed}.json").read_text());training.append(dict(rr,reported_family=fam))
  if fam=='GW':heads.append(dict(family='GW',seed=seed,path=str(path),sha256=sha(path)))
 new=torch.load(R/f'heads/G/{seed}/step1000.pt',map_location='cpu',weights_only=False);exact=all(torch.equal(v,new['model'][k]) for k,v in ck['model'].items());maxq=float(np.abs(predictions['G',seed]-predictions['Grerun',seed]).max());status='REPRODUCED' if exact else 'NUMERICALLY_EQUIVALENT' if maxq<=1e-5 else 'INCLUDE_C3';results.append(dict(seed=seed,status=status,all_tensors_byte_equal=exact,max_head_Q_delta=maxq,initial_and_stream_equal=True))
include=any(v['status']=='INCLUDE_C3' for v in results)
if include:
 for seed in SEEDS:
  p=R/f'heads/G/{seed}/step1000.pt';heads.append(dict(family='Grerun',seed=seed,path=str(p),sha256=sha(p)))
write(R/'results/reproduction.json',dict(results=results,Grerun_in_C3=include));write(R/'results/training_summary.json',dict(heads=training,GPU_process_seconds=sum(v['seconds'] for v in training),effective_updates=6000))
offline=[]
for (fam,seed),q in predictions.items():
 for i in range(256):
  mu=d['mu'][i,:64,0];eligible=(~d['collision'][i,:64,0])&(((mu>0)&(mu<=.4000001))|(mu>=.5999999));labels=(mu[eligible]<=.4000001);scores=q[i,:64,0][eligible];n1=int(labels.sum());n0=len(labels)-n1;auc=float((rankdata(scores)[labels].sum()-n1*(n1+1)/2)/(n1*n0)) if n1 and n0 else None
  offline.append(dict(family=fam,seed=seed,scene=int(d['scene'][i]),frame=int(d['frame'][i]),role=str(d['role'][i]),MSE=float(((q[i]-Y[i])**2).mean()),a0_MSE=float(((q[i,:,0]-Y[i,:,0])**2).mean()),other_MSE=float(((q[i,:,1:]-Y[i,:,1:])**2).mean()),decision_AUROC=auc))
summary=[]
for fam in ['G','Grerun','GW']:
 for role in ['fit','model_val','gate_cal']:
  rows=[v for v in offline if v['family']==fam and v['role']==role];summary.append(dict(family=fam,role=role,**{k:float(np.mean([v[k] for v in rows if v[k] is not None])) for k in ['MSE','a0_MSE','other_MSE','decision_AUROC']}))
write(R/'results/offline.json',dict(summary=summary,per_frame_seed=offline,diagnostic_only=True))
union=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in (P0/'union').glob('*/*.npz')];write(R/'results/P0_union_manifest.json',union)
write(R/'method_lock.json',dict(config_sha256=sha(R/'frozen_config.json'),heads=heads,P0_union_manifest_sha256=sha(R/'results/P0_union_manifest.json'),evaluation_code_sha256=sha(R/'code/eval_p1.py'),protected_G_sha256=[sha(L/f'heads/G/{s}/step1000.pt') for s in SEEDS]))
print(json.dumps(dict(reproduction=results,Grerun_in_C3=include,offline=summary)))
