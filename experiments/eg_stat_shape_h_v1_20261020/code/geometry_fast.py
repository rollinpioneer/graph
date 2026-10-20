from geometry_core import *
import ctypes
_lib=ctypes.CDLL(str(R/'code/masks.so'))
_lib.masks.argtypes=[ctypes.c_void_p,ctypes.c_int64,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_void_p,ctypes.c_void_p]
_lib.masks.restype=ctypes.c_int64
def cov_exact(points):
 X=np.array(points.T,ndmin=2,dtype=np.float64)
 avg=np.average(X,axis=1)
 X-=avg[:,None]
 c=np.dot(X,X.T.conj())
 c*=np.true_divide(1,len(points)-1)
 return c.squeeze()

class CachedTree:
 def __init__(self,tree,poses):
  centers={p[13:16].tobytes():p[13:16] for p in poses}
  values=list(centers.values());queries=tree.query_ball_point(np.asarray(values),.16,return_sorted=False,workers=1)
  self.lookup={k:np.asarray(q,dtype=np.int64) for k,q in zip(centers,queries)}
 def query_ball_point(self,c,r):
  assert r==.16
  return self.lookup[c.tobytes()]

def visible_fast(scene,frame,pred):
 root=G/f'eg_sub25_v1_20261015/data/graspnet/scenes/scene_{scene:04d}/kinect'
 dp=root/f'depth/{frame:04d}.png';mp=root/f'meta/{frame:04d}.mat'
 depth=np.asarray(Image.open(dp))[::2,::2];m=loadmat(mp,variable_names=['intrinsic_matrix','factor_depth'])
 K=m['intrinsic_matrix'];scale=float(m['factor_depth'].squeeze());yy,xx=np.indices(depth.shape);z=depth/scale
 cloud=np.stack([(xx*2-K[0,2])*z/K[0,0],(yy*2-K[1,2])*z/K[1,1],z],-1).reshape(-1,3)
 cloud=cloud[(cloud[:,2]>.05)&(cloud[:,2]<2.5)]
 lo=pred[:,13:16].min(0)-.16;hi=pred[:,13:16].max(0)+.16;cloud=cloud[((cloud>lo)&(cloud<hi)).all(1)]
 grid=np.floor(cloud/.005).astype(np.int64);lo_grid=grid.min(0);span=grid.max(0)-lo_grid+1
 assert int(span[0])*int(span[1])*int(span[2])<np.iinfo(np.int64).max
 shifted=grid-lo_grid;key=(shifted[:,0]*span[1]+shifted[:,1])*span[2]+shifted[:,2]
 u,inv=np.unique(key,return_inverse=True)
 n=np.bincount(inv);cloud=np.stack([np.bincount(inv,weights=cloud[:,k])/n for k in range(3)],1)
 return cloud,dict(depth_sha256=sha(dp),meta_sha256=sha(mp),cloud_points=len(cloud))

def geom_mask(cloud,tree,p):
 c=p[13:16];rot=p[4:13].reshape(3,3);w=np.clip(p[1],0,.1);h=p[2];d=p[3]
 pts=(cloud[tree.query_ball_point(c,.16)]-c)@rot
 x,y,z=pts.T if len(pts) else (np.array([]),)*3
 counts=np.zeros(5,np.int64);indices=np.empty(len(pts),np.int64)
 ni=_lib.masks(pts.ctypes.data,len(pts),float(w),float(h),float(d),counts.ctypes.data,indices.ctypes.data)
 counts=counts.astype(float);inner=pts[indices[:ni]]
 out=list(np.log1p(counts))+list(counts/max(1,len(pts)))
 contacts=[];norms=[];plans=[];cx=[]
 for sign in [-1,1]:
  if len(inner):
   edge=inner[:,1].min() if sign<0 else inner[:,1].max()
   band=inner[np.abs(inner[:,1]-edge)<=.005]
  else:band=np.empty((0,3))
  contacts.append(len(band));cx.append(float(band[:,0].mean()) if len(band) else np.nan)
  if len(band)>=6:
   vals,vec=np.linalg.eigh(np.cov(band.T))
   if vals[1]>1e-10:
    norms.append(vec[:,0]);plans.append(vals[0]/max(vals.sum(),1e-12));continue
  norms.append(None);plans.append(np.nan)
 ny=[abs(n[1]) if n is not None else np.nan for n in norms]
 agreement=abs(np.dot(*norms)) if all(n is not None for n in norms) else np.nan
 span=float(np.ptp(inner[:,1])) if len(inner) else np.nan
 center=inner.mean(0) if len(inner) else [np.nan]*3
 vals=np.linalg.eigvalsh(np.cov(inner.T)) if len(inner)>=6 else np.array([np.nan]*3)
 vals=vals/max(float(np.sum(vals)),1e-12) if np.isfinite(vals).all() else vals
 mn=min(ny) if np.isfinite(ny).all() else np.nan
 tangent=np.sqrt(max(0,1-mn*mn))/max(mn,.05) if np.isfinite(mn) else np.nan
 out+=list(np.log1p(contacts))+ny+plans+[agreement]+cx+[float(n is None) for n in norms]+[span,w-span]+list(center)+list(vals)+[abs(contacts[0]-contacts[1])/max(1,sum(contacts)),mn,tangent,np.log1p(len(pts))]
 assert len(out)==len(NAMES)
 return out

def geom_cov(cloud,tree,p):
 c=p[13:16];rot=p[4:13].reshape(3,3);w=np.clip(p[1],0,.1);h=p[2];d=p[3]
 pts=(cloud[tree.query_ball_point(c,.16)]-c)@rot
 x,y,z=pts.T if len(pts) else (np.array([]),)*3
 counts=np.zeros(5,np.int64);indices=np.empty(len(pts),np.int64)
 ni=_lib.masks(pts.ctypes.data,len(pts),float(w),float(h),float(d),counts.ctypes.data,indices.ctypes.data)
 counts=counts.astype(float);inner=pts[indices[:ni]]
 out=list(np.log1p(counts))+list(counts/max(1,len(pts)))
 contacts=[];norms=[];plans=[];cx=[]
 for sign in [-1,1]:
  if len(inner):
   edge=inner[:,1].min() if sign<0 else inner[:,1].max()
   band=inner[np.abs(inner[:,1]-edge)<=.005]
  else:band=np.empty((0,3))
  contacts.append(len(band));cx.append(float(band[:,0].mean()) if len(band) else np.nan)
  if len(band)>=6:
   vals,vec=np.linalg.eigh(cov_exact(band))
   if vals[1]>1e-10:
    norms.append(vec[:,0]);plans.append(vals[0]/max(vals.sum(),1e-12));continue
  norms.append(None);plans.append(np.nan)
 ny=[abs(n[1]) if n is not None else np.nan for n in norms]
 agreement=abs(np.dot(*norms)) if all(n is not None for n in norms) else np.nan
 span=float(np.ptp(inner[:,1])) if len(inner) else np.nan
 center=inner.mean(0) if len(inner) else [np.nan]*3
 vals=np.linalg.eigvalsh(cov_exact(inner)) if len(inner)>=6 else np.array([np.nan]*3)
 vals=vals/max(float(np.sum(vals)),1e-12) if np.isfinite(vals).all() else vals
 mn=min(ny) if np.isfinite(ny).all() else np.nan
 tangent=np.sqrt(max(0,1-mn*mn))/max(mn,.05) if np.isfinite(mn) else np.nan
 out+=list(np.log1p(contacts))+ny+plans+[agreement]+cx+[float(n is None) for n in norms]+[span,w-span]+list(center)+list(vals)+[abs(contacts[0]-contacts[1])/max(1,sum(contacts)),mn,tangent,np.log1p(len(pts))]
 assert len(out)==len(NAMES)
 return out

def extract(scene,frame,pred,variant):
 cloud,identity=visible(scene,frame,pred) if variant=='baseline' else visible_fast(scene,frame,pred)
 tree=cKDTree(cloud)
 if variant!='baseline':tree=CachedTree(tree,pred)
 fn={'baseline':geom,'index_query':geom,'mask':geom_mask,'cov':geom_cov}[variant]
 raw=np.array([fn(cloud,tree,p) for p in pred],np.float32)
 return raw,cloud,identity
