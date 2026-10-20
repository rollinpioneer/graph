from visibility import *
from graspnetAPI.utils.dexnet.grasping.grasp import ParallelJawPtGrasp3D
from graspnetAPI.utils.dexnet.grasping.quality import PointGraspMetrics3D
from graspnetAPI.utils.rotation import matrix_to_dexnet_params

def contact(native,p,obs,owner=None):
 p=p.astype(float).copy();p[1]=np.clip(p[1],0,.1)
 owner=int(native.owners(p[None])[0]) if owner is None else int(owner)
 pose=native.poses[owner];rot=p[4:13].reshape(3,3)
 center=(rot@np.array([p[3],0,0])+p[13:16]-pose[:3,3])@pose[:3,:3]
 axis,angle=matrix_to_dexnet_params(pose[:3,:3].T@rot)
 g=ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(center,axis,p[1],angle),p[3])
 found,cs=g.close_fingers(native.dex[owner],check_approach=native.config['metrics']['force_closure']['check_approach'])
 result=np.full(22,np.nan);result[0]=float(found);result[1]=-1
 if not found or any(c is None or c.normal is None for c in cs):return result
 points=np.stack([c.point for c in cs]);normals=np.stack([c.normal for c in cs])
 cam=points@pose[:3,:3].T+pose[:3,3];normals=normals@pose[:3,:3].T
 # Sort by gripper y so correspondence is independent of Dex-Net axis sign.
 order=np.argsort(((cam-p[13:16])@rot)[:,1]);cam=cam[order];normals=normals[order]
 normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-12)
 line=cam[1]-cam[0];line/=max(np.linalg.norm(line),1e-12)
 alpha=np.arccos(np.abs(normals@line).clip(0,1));need=np.tan(alpha.max())
 bins=np.array([.2,.4,.6,.8,1,1.2]);eligible=bins[bins>=need]
 result[1]=eligible[0] if len(eligible) else -1
 # Cross-check analytical closure against official function, no re-search.
 official=[f for f in bins if PointGraspMetrics3D.force_closure(cs[0],cs[1],float(f))]
 result[2]=official[0] if official else -1
 for j,eps in enumerate([.003,.004,.006]):result[3+j*2:5+j*2]=obs.query(cam,eps)[0]
 state,_,u,v=obs.query(cam);dn=obs.normals[v,u];nv=obs.nvalid[v,u]
 rays=cam/np.linalg.norm(cam,axis=1)[:,None];grazing=np.abs((normals*rays).sum(1))
 result[9:11]=grazing
 result[11:13]=np.where(nv,np.degrees(np.arccos(np.abs((normals*dn).sum(1)).clip(0,1))),np.nan)
 _,_,pc,ps=obs.inputs(p[None]);result[13:15]=np.linalg.norm(pc[0]-cam,axis=1)
 # Probe normals at estimated contacts, only genuine surface/reliable normals.
 _,_,pu,pv=obs.query(pc[0]);pn=obs.normals[pv,pu]
 result[15:17]=np.where((ps[0]==1)&obs.nvalid[pv,pu],np.degrees(np.arccos(np.abs((pn*normals).sum(1)).clip(0,1))),np.nan)
 result[17:19]=ps[0];result[19:21]=nv.astype(float);result[21]=need
 return result

def collision_evidence(native,p,obs,owner=None):
 p=p.astype(float).copy();p[1]=np.clip(p[1],0,.1)
 owner=int(native.owners(p[None])[0]) if owner is None else int(owner)
 model=native.mt[owner];lo=model.min(0)-.05;hi=model.max(0)+.05
 points=native.allpts[((native.allpts>lo)&(native.allpts<hi)).all(1)]
 xyz=(points-p[13:16])@p[4:13].reshape(3,3);x,y,z=xyz.T;w=p[1];d=p[3]
 hh=(z>-.01)&(z<.01);ff=(x>-.02)&(x<d)
 left=hh&ff&(y>-(w/2+.01))&(y<-w/2);right=hh&ff&(y<w/2+.01)&(y>w/2)
 base=hh&(y>-(w/2+.01))&(y<w/2+.01)&(x>-.03)&(x<-.02)
 inside=hh&ff&(y>=-w/2)&(y<=w/2);hit=left|right|base
 state=obs.query(points[hit])[0]
 return np.array([len(state),inside.sum()]+[(state==i).mean() if len(state) else np.nan for i in range(4)],float)
