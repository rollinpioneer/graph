from common import *
from concurrent.futures import ProcessPoolExecutor
from scipy.io import loadmat
from scipy.spatial import cKDTree
from PIL import Image
import argparse

NAMES=['log_left','log_right','log_palm','log_approach','log_inner','left_fraction','right_fraction','palm_fraction','approach_fraction','inner_fraction','log_contact_left','log_contact_right','normal_y_left','normal_y_right','planarity_left','planarity_right','normal_agreement','contact_x_left','contact_x_right','missing_left','missing_right','object_span_y','width_margin','inner_mean_x','inner_mean_y','inner_mean_z','pca0','pca1','pca2','support_asymmetry','min_normal_alignment','tan_max_angle','log_local_count']
def geom(cloud,tree,p):
 c=p[13:16];rot=p[4:13].reshape(3,3);w=np.clip(p[1],0,.1);h=p[2];d=p[3]
 pts=(cloud[tree.query_ball_point(c,.16)]-c)@rot
 x,y,z=pts.T if len(pts) else (np.array([]),)*3
 height=np.abs(z)<h/2;f=(x>d-.06)&(x<d);side=np.abs(y)<w/2+.01
 masks=[height&f&(y>-w/2-.01)&(y<-w/2),height&f&(y>w/2)&(y<w/2+.01),height&side&(x<=d-.06)&(x>d-.07),height&side&(x<=d-.07)&(x>d-.12),height&f&(np.abs(y)<=w/2)]
 counts=np.array([m.sum() for m in masks],float);inner=pts[masks[-1]]
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
def visible(scene,frame,pred):
 root=G/f'eg_sub25_v1_20261015/data/graspnet/scenes/scene_{scene:04d}/kinect'
 dp=root/f'depth/{frame:04d}.png';mp=root/f'meta/{frame:04d}.mat'
 depth=np.asarray(Image.open(dp))[::2,::2];m=loadmat(mp,variable_names=['intrinsic_matrix','factor_depth'])
 K=m['intrinsic_matrix'];scale=float(m['factor_depth'].squeeze());yy,xx=np.indices(depth.shape);z=depth/scale
 cloud=np.stack([(xx*2-K[0,2])*z/K[0,0],(yy*2-K[1,2])*z/K[1,1],z],-1).reshape(-1,3)
 cloud=cloud[(cloud[:,2]>.05)&(cloud[:,2]<2.5)]
 lo=pred[:,13:16].min(0)-.16;hi=pred[:,13:16].max(0)+.16;cloud=cloud[((cloud>lo)&(cloud<hi)).all(1)]
 grid=np.floor(cloud/.005).astype(np.int64);u,inv=np.unique(grid,axis=0,return_inverse=True)
 n=np.bincount(inv);cloud=np.stack([np.bincount(inv,weights=cloud[:,k])/n for k in range(3)],1)
 return cloud,dict(depth_sha256=sha(dp),meta_sha256=sha(mp),cloud_points=len(cloud))
