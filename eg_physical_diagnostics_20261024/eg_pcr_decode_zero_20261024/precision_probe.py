import json,sys
import check as c
import numpy as np
import torch
from scipy.io import loadmat
sys.path.insert(0,str(c.R0))
import pcr as previous

rows=[]
for s,k in [(100,170),(106,551),(176,471),(100,439)]:
 a=np.load(c.OLD/f'cache/C3_{s:04d}_0000.npz');e=np.load(c.OLD/f'evaluation/C3_{s:04d}_0000.npz');obj=int(e['object_id'][k]);g=np.load(c.R0/f'sdf/{obj:03d}/grid_sampled_sdf.npz');grid=c.Grid(c.R0/f'sdf/{obj:03d}/grid_sampled_sdf.npz')
 m=loadmat(c.DATA/f'scenes/scene_{s:04d}/kinect/meta/0000.mat');pose=torch.tensor(m['poses'][:,:,np.flatnonzero(m['cls_indexes'].reshape(-1)==obj+1)[0]],dtype=torch.float32)
 p=torch.tensor(a['pred'][[k]],dtype=torch.float32);v=torch.tensor(a['view_xyz'][[k]],dtype=torch.float32);ang=torch.tensor(a['angle_index'][[k]])*np.pi/12
 rotations={'cached':p[:,4:13].reshape(-1,3,3),'cpu':c.rotation(-v,ang),'cuda':c.rotation(-v.cuda(),ang.cuda()).cpu()}
 al=torch.tensor(a['angle_logits'][:,[k]],dtype=torch.float32,requires_grad=True);dl=torch.tensor(a['depth_logits'][:,[k]],dtype=torch.float32,requires_grad=True)
 raw=torch.tensor(a['raw_width'].reshape(-1)[[k]],dtype=torch.float32,requires_grad=True)
 ah=al[:12].argmax(0)*np.pi/12;dh=(dl[:4].argmax(0)+1)*.01;aa=c.angle_st(al,ah);dd=c.depth_st(dl,dh);ww=(1.2*raw/10).clamp(0,.1)
 terms,_,_=c.physics(p[:,13:16],v,aa,dd,ww,pose,grid)
 rows.append(dict(scene=s,id=k,type='compare_decode',angle_cache=ang.tolist(),angle_decode=aa.tolist(),width_cache=p[:,1].tolist(),width_decode=ww.tolist(),depth_cache=p[:,3].tolist(),depth_decode=dd.tolist(),PCR=float(terms['PCR'].detach()[0])))
 for name,rot in rotations.items():
  rot=rot.detach().requires_grad_();w=p[:,1];d=p[:,3]
  lp=(p[:,13:16]+rot[:,:,0]*(d-.002)[:,None]-rot[:,:,1]*w[:,None]/2-pose[:,3])@pose[:,:3]
  rp=(p[:,13:16]+rot[:,:,0]*(d-.002)[:,None]+rot[:,:,1]*w[:,None]/2-pose[:,3])@pose[:,:3]
  ls,ln,_=grid.sample(lp);rs,rn,_=grid.sample(rp);osl,oln,_=previous.sample(lp,g);osr,orn,_=previous.sample(rp,g)
  ln=ln@pose[:,:3].T;rn=rn@pose[:,:3].T
  ra=.5*(1-c.F.cosine_similarity(rot[:,:,1],rn,dim=1))+.5*(1-c.F.cosine_similarity(-rot[:,:,1],ln,dim=1))
  rows.append(dict(scene=s,id=k,rotation=name,rot_error=float((rot-rotations['cached']).abs().max()),RA=ra.detach().tolist(),sdf=[ls.detach().tolist(),rs.detach().tolist()],normals=[ln.tolist(),rn.tolist()],sample_diff=float((oln@pose[:,:3].T-ln).abs().max()),left_object=lp.detach().tolist(),right_object=rp.detach().tolist()))
(c.ROOT/'precision_probe.json').write_text(json.dumps(rows));print(json.dumps(rows))
