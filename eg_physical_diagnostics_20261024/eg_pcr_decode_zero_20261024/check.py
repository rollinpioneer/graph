import ast, hashlib, json, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from adapter import angle_st, depth_st

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
R0=BASE/'eg_pcr_r0_20261024'
OLD=BASE/'eg_pose_quality_diagnosis_v1_20261022'
DATA=BASE/'eg_sub25_v1_20261015/data/graspnet'
EG=BASE/'eg_full_repro_v1_20261015/runtime/EconomicGrasp'
SCENES=[100,106,115,124,132,139,150,157,160,169,176,186]
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def extract(path,name):
    text=path.read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.get_source_segment(text,node)

ns=dict(torch=torch,np=np,cfgs=SimpleNamespace(num_angle=12,num_depth=4,grasp_max_width=.1,m_point=1024),angle_st=angle_st,depth_st=depth_st)
rot_source=extract(EG/'utils/loss_utils.py','batch_viewpoint_params_to_matrix')
decode_source=extract(EG/'models/economicgrasp.py','pred_decode')
exec(rot_source,ns)
exec(decode_source,ns)
official=ns['pred_decode']
adapted_source=decode_source.replace('def pred_decode(', 'def adapted_decode(').replace(
    'grasp_angle = grasp_angle_indxs * np.pi / 12',
    'grasp_angle = grasp_angle_indxs * np.pi / 12\n        grasp_angle = angle_st(grasp_angle_pred.squeeze(0), grasp_angle)').replace(
    'grasp_depth = (grasp_depth_indxs + 1) * 0.01',
    'grasp_depth = (grasp_depth_indxs + 1) * 0.01\n        grasp_depth = depth_st(grasp_depth_pred.squeeze(0), grasp_depth)')
assert adapted_source.count('= angle_st(')==1 and adapted_source.count('= depth_st(')==1
exec(adapted_source,ns)
adapted=ns['adapted_decode']
# Official decoder always has 1024 vectors. Grouped geometry can have exactly
# three: torch.cross's implicit dimension would then cross across candidates.
# Use an explicit vector axis ONLY in this grouped diagnostic helper.
group_rot=rot_source.replace('def batch_viewpoint_params_to_matrix(', 'def grouped_rotation(').replace('torch.cross(axis_x, axis_y)', 'torch.cross(axis_x, axis_y, dim=-1)')
exec(group_rot,ns)
rotation=ns['grouped_rotation']

def endpoints(a,device):
    tt=lambda v:torch.tensor(v,device=device,dtype=torch.float32)
    return dict(point_clouds=[None],xyz_graspable=tt(a['pred'][:,13:16])[None],
        grasp_score_pred=tt(a['logits'].T)[None],grasp_angle_pred=tt(a['angle_logits'])[None],
        grasp_depth_pred=tt(a['depth_logits'])[None],grasp_width_pred=tt(a['raw_width'].reshape(-1))[None],
        grasp_top_view_xyz=tt(a['view_xyz'])[None])

def forward():
    rows=[]
    (ROOT/'official_decoder.py.txt').write_text(rot_source+'\n\n'+decode_source)
    (ROOT/'adapted_decoder.py.txt').write_text(adapted_source)
    for s in SCENES:
        for f in [0,64,128,192]:
            p=OLD/f'cache/C3_{s:04d}_{f:04d}.npz';a=np.load(p)
            for dev in ['cpu','cuda']:
                ep=endpoints(a,dev)
                with torch.no_grad(): o=official(ep)[0];v=adapted(ep)[0]
                ep['grasp_angle_pred'].requires_grad_();ep['grasp_depth_pred'].requires_grad_()
                vg=adapted(ep)[0]
                raw=torch.tensor(a['pred'],device=dev,dtype=torch.float32)
                rows.append(dict(scene=s,frame=f,device=dev,n=len(o),adapted_equal=bool(torch.equal(o,v)),
                  max_adaptation_error=float((o-v).abs().max()),cached_equal=bool(torch.equal(o,raw)),
                  grad_enabled_equal=bool(torch.equal(o,vg)),
                  cached_max_error=float((o-raw).abs().max()),score_equal=bool(torch.equal(o[:,0],raw[:,0])),
                  width_equal=bool(torch.equal(o[:,1],raw[:,1])),depth_equal=bool(torch.equal(o[:,3],raw[:,3])),
                  cache_sha256=sha(p)))
    # Synthetic periodic seam, tie and ignored-class checks, not efficacy data.
    al=torch.zeros(13,3,requires_grad=True);dl=torch.zeros(5,3,requires_grad=True)
    with torch.no_grad(): al[0,0]=1;al[11,1]=1;al[12,:]=100;dl[4,:]=100
    ah=al[:12].argmax(0)*np.pi/12;dh=(dl[:4].argmax(0)+1)*.01
    aa=angle_st(al,ah);dd=depth_st(dl,dh);ga,gd=torch.autograd.grad(aa.sum()+dd.sum(),[al,dl])
    synthetic=dict(forward_equal=bool(torch.equal(aa,ah) and torch.equal(dd,dh)),
      angle_seam_11_from_0_gradient=float(ga[11,0]),angle_seam_0_from_11_gradient=float(ga[0,1]),
      invalid_grad_zero=bool((ga[12]==0).all() and (gd[4]==0).all()),finite=bool(torch.isfinite(ga).all() and torch.isfinite(gd).all()))
    (ROOT/'forward.json').write_text(json.dumps(dict(rows=rows,synthetic=synthetic,
      decoder_sha256=sha(EG/'models/economicgrasp.py'),rotation_sha256=sha(EG/'utils/loss_utils.py'),
      parameter_updates=0,network_forward=False,torch=torch.__version__,tf32=True)))
    print('forward complete',len(rows),flush=True)

class Grid:
    def __init__(self,path):
        a=np.load(path);self.lo=torch.tensor(a['points'][0,0,0],dtype=torch.float32)
        self.hi=torch.tensor(a['points'][-1,-1,-1],dtype=torch.float32)
        self.v=torch.tensor(a['sdf'],dtype=torch.float32)[None,None]
    def sample(self,pos):
        p=pos.flip(-1);coords=((p-self.lo.flip(-1))/(self.hi-self.lo).flip(-1)-.5)*2
        val=F.grid_sample(self.v,coords.reshape(1,-1,1,1,3),mode='bilinear',padding_mode='border',align_corners=False).reshape(-1)
        norm=torch.autograd.grad(val.sum(),p,retain_graph=True)[0].flip(-1)
        return val,norm,((pos<self.lo)|(pos>self.hi)).any(-1)

def physics(center,view,angle,depth,width,pose,grid,frozen=None):
    rot=rotation(-view,angle)
    left=center+rot[:,:,0]*(depth-.002)[:,None]-rot[:,:,1]*width[:,None]/2
    right=center+rot[:,:,0]*(depth-.002)[:,None]+rot[:,:,1]*width[:,None]/2
    ls,ln,lo=grid.sample((left-pose[:,3])@pose[:,:3]);rs,rn,ro=grid.sample((right-pose[:,3])@pose[:,:3])
    ln=ln@pose[:,:3].T;rn=rn@pose[:,:3].T
    if frozen is not None:ln,rn=frozen
    ra=.5*(1-F.cosine_similarity(right-left,rn,dim=1))+.5*(1-F.cosine_similarity(left-right,ln,dim=1))
    rc=F.relu(.005-ls)+F.relu(.005-rs);ss=F.relu(ls-.02)+F.relu(rs-.02)
    return dict(RA=ra,RC=rc,RS=ss,PCR=ra+rc+ss), (ln,rn),lo|ro

def gradients(s):
    started=time.time();f=0;cp=OLD/f'cache/C3_{s:04d}_{f:04d}.npz';ev=OLD/f'evaluation/C3_{s:04d}_{f:04d}.npz'
    a=np.load(cp);e=np.load(ev);ids=e['final_ids'];ids=ids[e['P0_mu'][ids]>0]
    meta=loadmat(DATA/f'scenes/scene_{s:04d}/kinect/meta/{f:04d}.mat');objects=meta['cls_indexes'].reshape(-1)
    old={int(r['candidate_id']):r for r in json.loads((R0/f'pcr/{s:04d}_{f:04d}.json').read_text())['rows']}
    rows=[];directions=[]
    for obj in np.unique(e['object_id'][ids]):
        ii=ids[e['object_id'][ids]==obj];n=len(ii);tt=lambda x:torch.tensor(x,dtype=torch.float32)
        grid=Grid(R0/f'sdf/{obj:03d}/grid_sampled_sdf.npz');pose=tt(meta['poses'][:,:,np.flatnonzero(objects==obj+1)[0]])
        center=tt(a['pred'][ii,13:16]);view=tt(a['view_xyz'][ii]);al=tt(a['angle_logits'][:,ii]).requires_grad_();dl=tt(a['depth_logits'][:,ii]).requires_grad_()
        raw=tt(a['raw_width'].reshape(-1)[ii]).requires_grad_();width=(1.2*raw/10).clamp(0,.1)
        ah=al[:12].argmax(0)*np.pi/12;dh=(dl[:4].argmax(0)+1)*.01
        angle=angle_st(al,ah);depth=depth_st(dl,dh)
        terms,normals,outside=physics(center,view,angle,depth,width,pose,grid)
        assert torch.allclose(rotation(-view,angle),tt(a['pred'][ii,4:13]).reshape(-1,3,3),atol=5e-7,rtol=0)
        assert max(abs(float(terms['PCR'][j].detach())-old[int(k)]['PCR']) for j,k in enumerate(ii))<1e-4,(s,obj,'PCR replay')
        per=[dict(scene=s,frame=f,object_id=int(obj),candidate_id=int(k),mu=float(e['P0_mu'][k]),
              outside=bool(outside[j]),PCR=float(terms['PCR'][j].detach()),
              previous_PCR=old[int(k)]['PCR'],RC=float(terms['RC'][j].detach()),RS=float(terms['RS'][j].detach())) for j,k in enumerate(ii)]
        for name,term in terms.items():
            gs=torch.autograd.grad(term.sum(),[al,dl,raw,angle,depth],retain_graph=True,allow_unused=True)
            for key,g in zip(['angle_logits','depth_logits','raw_width','angle_rad','depth_m'],gs):
                assert g is not None and torch.isfinite(g).all(),(s,obj,name,key)
                v=g.abs().amax(0) if g.ndim==2 else g
                for j in range(n):per[j][name+'_'+key]=float(v[j])
            assert (gs[0][12]==0).all() and (gs[1][4]==0).all()
        # Temperature changes only Jacobians, never hard outputs.
        for temp in [.5,2.]:
            aa=angle_st(al,ah,temp);dd=depth_st(dl,dh,temp)
            assert torch.equal(aa,angle) and torch.equal(dd,depth)
            term=physics(center,view,aa,dd,width,pose,grid)[0]['PCR']
            ga,gd=torch.autograd.grad(term.sum(),[al,dl],retain_graph=True)
            for j in range(n):
                per[j][f'PCR_angle_logits_T{temp}']=float(ga[:,j].abs().max())
                per[j][f'PCR_depth_logits_T{temp}']=float(gd[:,j].abs().max())
        # A faithful check of the officially detached-normal surrogate, and the
        # distinct physical value with normals recomputed at each probe pose.
        for coord,steps in [('angle',[np.pi/180*.1,np.pi/180*.5]),('depth',[.00005,.0002])]:
            for step in steps:
                vals={}
                for sign in [-1,1]:
                    aa=(ah.detach()+sign*step if coord=='angle' else ah.detach()).requires_grad_()
                    dd=(dh.detach()+sign*step if coord=='depth' else dh.detach()).requires_grad_()
                    for mode in ['frozen','recomputed']:
                        terms2,_,_=physics(center,view,aa,dd,width.detach(),pose,grid,normals if mode=='frozen' else None)
                        vals[(sign,mode)]={k:v.detach().numpy() for k,v in terms2.items()}
                for name in terms:
                    g=np.array([r[name+('_angle_rad' if coord=='angle' else '_depth_m')] for r in per])
                    for j,k in enumerate(ii):
                        r=dict(scene=s,candidate_id=int(k),object_id=int(obj),mu=float(e['P0_mu'][k]),coordinate=coord,step=step,term=name,gradient=float(g[j]),baseline=float(terms[name][j].detach()))
                        for mode in ['frozen','recomputed']:
                            minus=vals[(-1,mode)][name][j];plus=vals[(1,mode)][name][j]
                            r[mode+'_fd']=float((plus-minus)/(2*step))
                            r[mode+'_descent_change']=float((minus if g[j]>0 else plus if g[j]<0 else r['baseline'])-r['baseline'])
                        directions.append(r)
        rows.extend(per)
    (ROOT/f'grad_{s}.json').write_text(json.dumps(dict(rows=rows,directions=directions,seconds=time.time()-started,
        cache_sha256=sha(cp),evaluation_sha256=sha(ev),parameter_updates=0)))
    print(s,len(rows),round(time.time()-started,2),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='forward':forward()
    elif sys.argv[1]=='pilot':gradients(SCENES[0])
    else:
        for s in SCENES[int(sys.argv[1])::3]:
            if not (ROOT/f'grad_{s}.json').exists():gradients(s)
