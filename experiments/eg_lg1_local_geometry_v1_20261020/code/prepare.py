from common import *
import torch
t=time.time();d={k:v for k,v in np.load(RP/'prepared.npz').items()};raw=[];hashes=[]
for i,(s,f) in enumerate(zip(d['scene'],d['frame'])):
 p=R/f'geometry/head/{s:04d}/{f:04d}.npz'
 with np.load(p) as z:raw.append(z['raw'])
 hashes.append(sha(p))
raw=np.stack(raw);assert raw.shape==(256,128,6,33)
with np.load(DG/'features.npz') as z:assert np.array_equal(raw[:,:,0],z['geometry'],equal_nan=True)
fit=d['role']=='fit';ff=raw[fit].reshape(-1,33);median=np.nanmedian(ff,0);median=np.where(np.isfinite(median),median,0).astype(np.float32);missing=~np.isfinite(ff);imputed=np.concatenate([np.where(missing,median,ff),missing.astype(np.float32)],-1);mean=imputed.mean(0);std=imputed.std(0).clip(1e-6);norm=dict(median=median,mean=mean,std=std);geometry=geom_input(raw,norm)
assert np.isfinite(geometry).all()
np.savez_compressed(R/'prepared_geometry.npz',geometry=geometry,median=median,mean=mean,std=std)
heads=[]
for v in json.load(open(P/'C3_method_lock.json'))['heads']['B']:
 assert sha(v['path'])==v['sha256'];ck=torch.load(v['path'],map_location='cpu',weights_only=False);assert np.array_equal(ck['mean'],d['mean']) and np.array_equal(ck['std'],d['std']);old=next(h for h in json.load(open(P/'results/B_heads.json'))['heads'] if h['seed']==v['seed']);rng=np.random.default_rng(v['seed']);order=rng.permutation(24576);cursor=0;stream=hashlib.sha256()
 for step in range(1000):
  if cursor+128>len(order):order=rng.permutation(24576);cursor=0
  ix=order[cursor:cursor+128];cursor+=128;stream.update(ix.tobytes())
 assert stream.hexdigest()==old['sample_stream_sha256'];heads.append(dict(seed=v['seed'],old_stream=stream.hexdigest(),old_head_sha256=v['sha256']))
changed=float(np.mean(np.any(np.nan_to_num(raw[:,:,1:]-raw[:,:,:1])!=0,axis=(2,3))))
assert changed>.8,'Six-pose geometry suspiciously identical'
write(R/'results/preparation.json',dict(status='PASS',original_pose_equals_D7=True,per_pose_geometry_changed_fraction=changed,geometry_norm_fit_only=True,missing_fraction=float((~np.isfinite(raw)).mean()),label_actions=196608,new_training_quality_labels=0,prepared_source_sha256=sha(RP/'prepared.npz'),geometry_cache_sha256=sha(R/'prepared_geometry.npz'),geometry_norm={k:v.tolist() for k,v in norm.items()},source_geometry_hashes=hashes,old_streams=heads,seconds=time.time()-t))
print('PREPARATION_PASS',time.time()-t,flush=True)
