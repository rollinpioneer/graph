from pathlib import Path
import numpy as np,json,hashlib,time
from collections import Counter
R=Path(__file__).resolve().parents[1]; P=Path('/home/xushijie3/work/flow_critic_comparative_train_v1_20261014')
protocol=json.loads((R/'configs/protocol.json').read_text()); geom=json.loads((R/'reports/geometry.json').read_text()); distances=np.array(geom['tasks'][0]['bfs_to_goal'])
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def bfs(xy):
 ji=((xy+6)/4).astype(int); j,i=ji.T; valid=(i>=0)&(i<8)&(j>=0)&(j<8)
 out=np.full(len(xy),-1,dtype=np.int16);out[valid]=distances[i[valid],j[valid]];return out
manifest={}; hashes={}; start=time.time()
for split,suffix in [('train',''),('validation','-val')]:
 p=P/'data'/f'humanoidmaze-medium-navigate-v0{suffix}.npz'
 with np.load(p) as z: data={k:z[k] for k in z.files}
 o=data['observations']; a=data['actions']; t=data['terminals'].astype(bool)
 assert o.shape[1]==69 and a.shape[1]==21 and t[-1]
 valid=np.flatnonzero(~t); ends=np.flatnonzero(t); tid=np.searchsorted(ends,np.arange(len(o)),side='left')
 starts=np.r_[0,ends[:-1]+1]; hashes[split]=set(hashlib.sha256(o[i:j+1].tobytes()).hexdigest() for i,j in zip(starts,ends))
 idx=np.random.default_rng(protocol['index_rng_seeds'][split]).choice(valid,100*256,replace=True)
 xyz=o[:,:2]; d=bfs(xyz); eid=tid[idx]; f={'indices':idx,'trajectory':eid,'bfs':d[idx],'euclidean':np.linalg.norm(xyz[idx]-[20,20],axis=1),'head_height':o[idx,23],'upright':o[idx,38],'speed':np.linalg.norm(o[idx,39:42],axis=1)}
 for horizon in [25,100]:
  j=idx+horizon; ok=(j<=ends[eid]) & (d[idx]>=0); j=np.minimum(j,len(o)-1); ok &=d[j]>=0
  f['progress'+str(horizon)]=np.where(ok,d[idx]-d[j],np.nan).astype(np.float32)
 f.update(observations=o[idx].astype(np.float32),actions=np.clip(a[idx],-1+1e-5,1-1e-5).astype(np.float32))
 np.savez_compressed(R/'arrays'/f'{split}_samples.npz',**f)
 reward=(np.linalg.norm(xyz[valid]-[20,20],axis=1)<=.5).astype(float)-1
 manifest[split]={'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'fields':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in data.items()},'valid_transitions':len(valid),'trajectories':len(ends),'selected_trajectories':len(np.unique(eid)),'selected_count':len(idx),'selected_index_sha256':hashlib.sha256(idx.tobytes()).hexdigest(),'sample_sha256':sha(R/'arrays'/f'{split}_samples.npz'),'invalid_geometry_count':int((d[idx]<0).sum()),'bfs_counts':dict(Counter(map(str,d[idx]))),'reward_range':[float(reward.min()),float(reward.max())],'task1_success_fraction':float((reward==0).mean()),'boundary_rule':'raw terminal row excluded as current transition; horizon cannot exceed same episode terminal row','action_clip_eps':1e-5,'qpos_xy_equals_observation':bool(np.array_equal(data['qpos'][:,:2],xyz)) if 'qpos' in data else None}
 print(split,manifest[split]['valid_transitions'],len(ends),manifest[split]['bfs_counts'],flush=True)
 del data,o,a,t,xyz
manifest['train_validation_duplicate_observation_trajectory_hashes']=len(hashes['train'] & hashes['validation']); manifest['environment_steps']=0;manifest['elapsed_seconds']=time.time()-start
(R/'receipts/data_identity.json').write_text(json.dumps(manifest,indent=2))
