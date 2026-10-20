from common import *
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter

class Observation:
 def __init__(self,scene,frame):
  p=DATA/f'scenes/scene_{scene:04d}/kinect'
  self.dp=p/f'depth/{frame:04d}.png';self.mp=p/f'meta/{frame:04d}.mat'
  meta=loadmat(self.mp,variable_names=['intrinsic_matrix','factor_depth'])
  self.K=meta['intrinsic_matrix'];self.depth=np.asarray(Image.open(self.dp),float)/float(meta['factor_depth'].squeeze())
  yy,xx=np.indices(self.depth.shape);z=self.depth
  self.points=np.stack([(xx-self.K[0,2])*z/self.K[0,0],(yy-self.K[1,2])*z/self.K[1,1],z],-1)
  dx=np.zeros_like(self.points);dy=dx.copy()
  dx[:,1:-1]=self.points[:,2:]-self.points[:,:-2];dy[1:-1]=self.points[2:]-self.points[:-2]
  n=np.cross(dx,dy);nn=np.linalg.norm(n,axis=-1)
  jump=maximum_filter(z,3)-minimum_filter(np.where(z>0,z,1e6),3)
  valid=(z>0)&(nn>1e-12)&(jump<=.01)&(minimum_filter(z,3)>0)
  n/=np.maximum(nn[...,None],1e-12);n[~valid]=0
  mass=uniform_filter(valid.astype(float),5)
  ns=np.stack([uniform_filter(n[:,:,i],5) for i in range(3)],-1)/np.maximum(mass[...,None],1e-12)
  length=np.linalg.norm(ns,axis=-1)
  self.normals=ns/np.maximum(length[...,None],1e-12);self.nvar=1-length
  self.nvalid=valid&(mass>=.6)&(length>.1)
 def query(self,points,eps=.004):
  shape=points.shape[:-1];p=points.reshape(-1,3);z=p[:,2]
  safe=np.where(z>0,z,1)
  uv=np.rint(np.stack([p[:,0]/safe*self.K[0,0]+self.K[0,2],p[:,1]/safe*self.K[1,1]+self.K[1,2]],-1)).astype(int)
  h,w=self.depth.shape;ok=(z>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
  u=uv[:,0].clip(0,w-1);v=uv[:,1].clip(0,h-1);d=self.depth[v,u];ok&=d>0
  r=z-d;state=np.full(len(p),3,np.int8);state[ok & (r< -eps)]=0;state[ok&(np.abs(r)<=eps)]=1;state[ok&(r>eps)]=2
  return state.reshape(shape),np.where(ok,r,np.nan).reshape(shape),u.reshape(shape),v.reshape(shape)
 def inputs(self,rows):
  rows=np.asarray(rows,float);n=len(rows);rot=rows[:,4:13].reshape(-1,3,3);tr=rows[:,13:16];w=rows[:,1].clip(0,.1);d=rows[:,3]
  def world(local):return np.einsum('nki,nji->nkj',local,rot)+tr[:,None,:]
  feats=[];regions=[];region_valid=[]
  # Exact 5mm cell centres, two z layers; padded cells masked per grasp.
  for region in range(4):
   if region<2:
    xx,yy,zz=np.meshgrid(np.arange(-.0175,.041,.005),[.0025,.0075],[-.005,.005],indexing='ij')
    loc=np.broadcast_to(np.stack([xx,yy,zz],-1).reshape(1,-1,3),(n,xx.size,3)).copy()
    loc[:,:,1]=(w[:,None]/2+loc[:,:,1])*(-1 if region==0 else 1)
    mask=loc[:,:,0]<d[:,None]
   else:
    xs=[-.0275,-.0225] if region==2 else np.arange(-.0175,.041,.005)
    xx,ii,zz=np.meshgrid(xs,np.arange(24),[-.005,.005],indexing='ij')
    loc=np.broadcast_to(np.stack([xx,ii,zz],-1).reshape(1,-1,3),(n,xx.size,3)).copy()
    margin=.01 if region==2 else 0
    loc[:,:,1]=-w[:,None]/2-margin+.0025+loc[:,:,1]*.005
    mask=(loc[:,:,1]<w[:,None]/2+margin)&(loc[:,:,0]<d[:,None])
   st,res,_,_=self.query(world(loc));den=np.maximum(mask.sum(1),1)
   f=[((st==i)&mask).sum(1)/den for i in range(4)]
   for lo,hi in [(.004,.01),(.01,.02),(.02,.04),(.04,np.inf)]:
    f.append(((res>lo)&(res<=hi)&mask).sum(1)/den)
   occ=(st==2)&mask;minimum=np.min(np.where(occ,res,np.inf),1);minimum[~occ.any(1)]=0
   f.extend([minimum,((st==0)|~mask).all(1).astype(float)])
   feats.extend(f);regions.append(st);region_valid.append(mask)
  V=np.stack(feats,1).astype(np.float32)
  kout=[];contact_points=[];contact_states=[]
  for dx,dz in [(0,0),(0,.005),(0,-.005),(.005,0)]:
   side_states=[];side_positions=[]
   for sign in [-1,1]:
    t=np.arange(51)*.001
    loc=np.zeros((n,51,3));loc[:,:,0]=d[:,None]-dx;loc[:,:,1]=sign*np.maximum(w[:,None]/2-t,0);loc[:,:,2]=dz
    mask=t[None,:]<=w[:,None]/2+1e-10
    wp=world(loc);st,_,u,v=self.query(wp);hit=(st!=0)&mask;found=hit.any(1);idx=hit.argmax(1)
    selected=st[np.arange(n),idx];selected[~found]=0
    pt=wp[np.arange(n),idx];y=loc[np.arange(n),idx,1]/np.maximum(w/2,1e-6);y[~found]=0
    ui=u[np.arange(n),idx];vi=v[np.arange(n),idx];normal=self.normals[vi,ui]
    reliable=found&(selected==1)&self.nvalid[vi,ui]
    alignment=np.abs((normal*rot[:,:,1]).sum(1)).clip(0,1);angle=np.arccos(alignment)
    ray=pt/np.maximum(np.linalg.norm(pt,axis=1)[:,None],1e-12);grazing=np.abs((normal*ray).sum(1))
    vals=[(selected==j).astype(float) for j in range(4)]+[y]
    vals.extend([np.where(reliable,a,np.nan) for a in [angle,np.tan(np.minimum(angle,1.55)),grazing,self.nvar[vi,ui]]])
    vals.append((~reliable).astype(float));kout.extend(vals);side_states.append(selected);side_positions.append(y)
    if dx==0 and dz==0:contact_points.append(pt);contact_states.append(selected)
   kout.extend([((side_states[0]==1)&(side_states[1]==1)).astype(float),side_positions[1]-side_positions[0]])
  K=np.stack(kout,1).astype(np.float32)
  return V,K,np.stack(contact_points,1),np.stack(contact_states,1)

def official_aligned(obs,rows):
 # Visible depth only; same 5mm voxel-centroid reduction as G. No GT segmentation.
 from scipy.spatial import cKDTree
 cloud=obs.points[::2,::2].reshape(-1,3);cloud=cloud[(cloud[:,2]>.05)&(cloud[:,2]<2.5)]
 grid=np.floor(cloud/.005).astype(np.int64);_,inv=np.unique(grid,axis=0,return_inverse=True);ct=np.bincount(inv)
 cloud=np.stack([np.bincount(inv,weights=cloud[:,i])/ct for i in range(3)],1);tree=cKDTree(cloud);out=[]
 for p in rows:
  w=np.clip(p[1],0,.1);d=p[3];pts=(cloud[tree.query_ball_point(p[13:16],.16)]-p[13:16])@p[4:13].reshape(3,3)
  x,y,z=pts.T;hh=np.abs(z)<.01;ff=(x>-.02)&(x<d);side=np.abs(y)<w/2+.01
  masks=[hh&ff&(y>-w/2-.01)&(y<-w/2),hh&ff&(y>w/2)&(y<w/2+.01),hh&side&(x>-.03)&(x<-.02),hh&side&(x>-.08)&(x<=-.03),hh&ff&(np.abs(y)<=w/2)]
  counts=np.array([m.sum() for m in masks]);inner=pts[masks[-1]];contact=[];cx=[]
  for sign in [-1,1]:
   edge=(inner[:,1].min() if sign<0 else inner[:,1].max()) if len(inner) else 0
   band=inner[np.abs(inner[:,1]-edge)<=.005];contact.append(len(band));cx.append(band[:,0].mean() if len(band) else np.nan)
  span=np.ptp(inner[:,1]) if len(inner) else np.nan
  out.append(list(np.log1p(counts))+list(counts/max(1,len(pts)))+list(np.log1p(contact))+cx+[span,w-span]+list(inner.mean(0) if len(inner) else np.full(3,np.nan))+[abs(contact[0]-contact[1])/max(1,sum(contact)),np.log1p(len(pts))])
 return np.asarray(out,np.float32)
