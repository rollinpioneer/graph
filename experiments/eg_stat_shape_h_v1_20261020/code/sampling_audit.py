from common import *
old=G/'eg_geometry_compact_v1_20261020';d=np.load(RP/'prepared.npz')
new=np.load(R/'cache/mask.npy',mmap_mode='r');prev=np.load(old/'cache/mask.npy',mmap_mode='r')
a=new.sum(-1);b=prev.sum(-1);result={}
assert np.all(new[prev])
for role in ['fit','model_val','gate_cal']:
 ix=d['role']==role;result[role]={}
 for tag,x,y in [('original',a[ix,:,0],b[ix,:,0]),('six_pose',a[ix],b[ix])]:
  result[role][tag]=dict(count=int(x.size),old_mean=float(y.mean()),filled_mean=float(x.mean()),old_padding_fraction=float(1-y.mean()/128),filled_padding_fraction=float(1-x.mean()/128),increased_fraction=float((x>y).mean()),full128_fraction=float((x==128).mean()),empty_fraction=float((x==0).mean()),no_point_loss=bool((x>=y).all()))
write(R/'results/full_sampling_audit.json',result)
