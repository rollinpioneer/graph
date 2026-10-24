import sys,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
ROOT=Path.home()/'work/grasp/eg_pcr_r0_20261024';OLD=ROOT.parent/'eg_pose_quality_diagnosis_v1_20261022';DATA=ROOT.parent/'eg_sub25_v1_20261015/data/graspnet'

def sample(pos,grid):
 pts=grid['points'];lo=torch.tensor(pts[0,0,0],dtype=torch.float32);hi=torch.tensor(pts[-1,-1,-1],dtype=torch.float32)
 p=pos.flip(-1);p.retain_grad();coords=((p-lo.flip(-1))/(hi-lo).flip(-1)-.5)*2
 values=torch.tensor(grid['sdf'],dtype=torch.float32)[None,None]
 val=F.grid_sample(values,coords.reshape(1,-1,1,1,3),mode='bilinear',padding_mode='border',align_corners=False).reshape(-1)
 norm=torch.autograd.grad(val.sum(),p,retain_graph=True)[0].flip(-1)
 outside=((pos<lo)|(pos>hi)).any(-1)
 return val,norm,outside

def run(scene,frame):
 a=np.load(OLD/f'cache/C3_{scene:04d}_{frame:04d}.npz');e=np.load(OLD/f'evaluation/C3_{scene:04d}_{frame:04d}.npz');ids=e['final_ids']
 meta=loadmat(DATA/f'scenes/scene_{scene:04d}/kinect/meta/{frame:04d}.mat');objects=meta['cls_indexes'].reshape(-1);poses=meta['poses'];rows=[];grads=[]
 contacts={r['candidate_id']:r for r in json.loads((ROOT/f'contacts/{scene:04d}_{frame:04d}.json').read_text())['rows']}
 for obj in np.unique(e['object_id'][ids]):
  ii=ids[e['object_id'][ids]==obj];gpath=ROOT/f'sdf/{obj:03d}/grid_sampled_sdf.npz'
  if not gpath.exists():raise FileNotFoundError(gpath)
  grid=np.load(gpath);pose=torch.tensor(poses[:,:,np.flatnonzero(objects==obj+1)[0]],dtype=torch.float32)
  pred=torch.tensor(a['pred'][ii],dtype=torch.float32);base=pred[:,4:13].reshape(-1,3,3)
  theta=torch.zeros(len(ii),requires_grad=True);cc=theta.cos();ss=theta.sin();zz=theta*0;oo=zz+1
  rx=torch.stack([oo,zz,zz,zz,cc,-ss,zz,ss,cc],dim=-1).reshape(-1,3,3);rot=base@rx
  # Actual EG hard-angle/depth decode: no gradient to either classification head.
  al=torch.tensor(a['angle_logits'][:,ii],dtype=torch.float32,requires_grad=True);dl=torch.tensor(a['depth_logits'][:,ii],dtype=torch.float32,requires_grad=True)
  angle=al[:12].argmax(0).float()*np.pi/12;depth=(dl[:4].argmax(0).float()+1)*.01
  raw=torch.tensor(np.asarray(a['raw_width']).reshape(-1)[ii],dtype=torch.float32,requires_grad=True);width=(1.2*raw/10).clamp(0,.1)
  assert torch.allclose(width,pred[:,1],atol=1e-6) and torch.allclose(depth,pred[:,3],atol=1e-6)
  left=pred[:,13:16]+rot[:,:,0]*(depth-.002)[:,None]-rot[:,:,1]*width[:,None]/2
  right=pred[:,13:16]+rot[:,:,0]*(depth-.002)[:,None]+rot[:,:,1]*width[:,None]/2
  lp=(left-pose[:,3])@pose[:,:3];rp=(right-pose[:,3])@pose[:,:3]
  ls,ln,lo=sample(lp,grid);rs,rn,ro=sample(rp,grid);ln=ln@pose[:,:3].T;rn=rn@pose[:,:3].T
  ra=.5*(1-F.cosine_similarity(right-left,rn,dim=1))+.5*(1-F.cosine_similarity(left-right,ln,dim=1))
  rc=F.relu(.005-ls)+F.relu(.005-rs);ss=F.relu(ls-.02)+F.relu(rs-.02);risk=ra+rc+ss
  terms={}
  for name,term in [('RA',ra),('RC',rc),('RS',ss)]:
   gg=torch.autograd.grad(term.sum(),[raw,al,dl,theta],allow_unused=True,retain_graph=True)
   terms[name]={key:None if value is None else dict(norm=float(value.norm()),finite=bool(torch.isfinite(value).all()),nonzero=int((value.abs()>1e-7).sum())) for key,value in zip(['raw_width','angle_logits','depth_logits','continuous_roll'],gg)}
  grad=torch.autograd.grad(risk.sum(),[raw,al,dl,theta],allow_unused=True)
  grads.append(dict(scene=scene,frame=frame,object_id=int(obj),n=len(ii),width_finite=bool(torch.isfinite(grad[0]).all()),width_nonzero=int((grad[0].abs()>1e-9).sum()),angle_logits_connected=grad[1] is not None,depth_logits_connected=grad[2] is not None,continuous_roll_finite=bool(torch.isfinite(grad[3]).all()),continuous_roll_nonzero=int((grad[3].abs()>1e-9).sum()),term_gradients=terms))
  points=torch.tensor(np.array([contacts.get(int(k),{}).get('contact_points_object',[[float('nan')]*3]*2) for k in ii]),dtype=torch.float32).reshape(-1,3).requires_grad_();cv,cn,co=sample(points,grid)
  contact_sdf=cv.detach().numpy().reshape(-1,2)
  for j,k in enumerate(ii):rows.append(dict(scene=scene,frame=frame,candidate_id=int(k),object_id=int(obj),RA=float(ra[j].detach()),RC=float(rc[j].detach()),RS=float(ss[j].detach()),PCR=float(risk[j].detach()),sdf_left=float(ls[j].detach()),sdf_right=float(rs[j].detach()),normal_left_norm=float(ln[j].norm()),normal_right_norm=float(rn[j].norm()),outside_grid=bool(lo[j]|ro[j]),gt_contact_sdf_abs_max=float(abs(contact_sdf[j]).max())))
 out=ROOT/'pcr_all';out.mkdir(exist_ok=True);(out/f'{scene:04d}_{frame:04d}.json').write_text(json.dumps(dict(rows=rows,gradients=grads)));print(scene,frame,len(rows),flush=True)

if __name__=='__main__':
 p=json.loads((OLD/'study_plan.json').read_text());tasks=[(s,f) for s in p['C3_scenes'] for f in p['C3_frames']]
 if sys.argv[1]=='pilot':run(*tasks[0])
 else:
  pending=tasks[int(sys.argv[1])::3];deadline=time.time()+7200
  while pending and time.time()<deadline:
   progressed=False
   for s,f in pending[:]:
    e=np.load(OLD/f'evaluation/C3_{s:04d}_{f:04d}.npz');ii=e['final_ids'];objs=np.unique(e['object_id'][ii])
    if all((ROOT/f'sdf/{o:03d}/grid_sampled_sdf.npz').exists() for o in objs):
     run(s,f);pending.remove((s,f));progressed=True
   if not progressed:time.sleep(10)
  if pending:raise RuntimeError(('SDF not ready',pending))
