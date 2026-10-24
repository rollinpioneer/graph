"""Independent NumPy projective TSDF; states 0 free, 1 occupied, 2 unknown."""
import numpy as np
VOXEL=.0025
TRUNC=.010
EPS=.0025

def classify(phi,weight):
    return np.where(weight>0,np.where(phi>EPS,0,1),2).astype(np.uint8)

def query(points,depth,K,voxel_centers=False):
    p=np.asarray(points,dtype=np.float32)
    if voxel_centers:p=(np.floor(p/np.float32(VOXEL))+.5)*np.float32(VOXEL)
    z=p[:,2]
    uv=p[:,:2]/z[:,None]*np.array([K[0,0],K[1,1]],np.float32)+np.array([K[0,2],K[1,2]],np.float32)
    # nvblox image coordinates have pixel centers at half integers; closest uses floor.
    ij=np.floor(uv).astype(np.int64)
    h,w=depth.shape
    valid=(z>0)&(z<=7)&(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)
    d=np.zeros(len(p),np.float32)
    d[valid]=depth[ij[valid,1],ij[valid,0]]
    phi=d-z
    observed=valid&(d>0)&(phi>=-TRUNC)
    return np.clip(phi,-TRUNC,TRUNC),observed.astype(np.float32)

def box_samples(lo,hi,step=VOXEL):
    axes=[]
    for a,b in zip(lo,hi):
        n=max(1,int(np.ceil((b-a)/step)))
        axes.append(a+(np.arange(n)+.5)*(b-a)/n)
    return np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)

def gripper_samples(row):
    w=float(row[1]);d=float(row[3]);R=row[4:13].reshape(3,3)
    parts=[box_samples([-.02,-w/2-.01,-.01],[d,-w/2,.01]),
           box_samples([-.02,w/2,-.01],[d,w/2+.01,.01]),
           box_samples([-.03,-w/2-.01,-.01],[-.02,w/2+.01,.01])]
    q=np.concatenate(parts)
    assert np.isfinite(q).all()
    return (q@R.T+row[13:16]).astype(np.float32)
