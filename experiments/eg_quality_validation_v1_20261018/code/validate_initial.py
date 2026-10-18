"""Frozen-candidate diagnosis; native geometry/SDF quality, no network changes."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k] = '1'
import sys, json, time, argparse, hashlib, resource
from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.spatial import cKDTree
R = Path('/home/xushijie3/work/eg_sub25_v1_20261015')
N = Path('/home/xushijie3/work/eg_quality_validation_v1_20261018')
E = R/'runs/EG_SUB25_V1/evaluation'
sys.path.insert(0, str(R/'d2d4_20261017'))
from d2_lookup import SceneLabels, V
import graspnetAPI.graspnet_eval as GE
import graspnetAPI.utils.eval_utils as EU
from graspnetAPI import GraspNetEval
from graspnetAPI.utils.dexnet.grasping.grasp_quality_config import GraspQualityConfigFactory
FR = np.array([.2,.4,.6,.8,1.,1.2])

def utility(q):
    q = np.asarray(q)
    return ((q[...,None] > 0) & (q[...,None] <= FR + 1e-7)).sum(-1)

def ap(q):
    z = np.zeros((50,6))
    for j,f in enumerate(FR):
        c = np.cumsum((q > 0) & (q <= f + 1e-7))
        if len(c): z[:,j] = c[np.minimum(np.arange(50),len(c)-1)] / np.arange(1,51)
    return z

def basis(x):
    y = np.array([-x[1],x[0],0.])
    if np.linalg.norm(y) < 1e-10: y = np.array([0.,1.,0.])
    y /= np.linalg.norm(y)
    return np.column_stack([x,y,np.cross(x,y)])

def rx(t):
    c,s = np.cos(t),np.sin(t)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

class Geometry:
    def __init__(self, scene):
        self.scene = scene
        self.ge = GraspNetEval(root=str(R/'data/graspnet'),camera='kinect',split='all')
        ml,self.dex,_ = self.ge.get_scene_models(scene,ann_id=0)
        self.models = [GE.voxel_sample_points(m,.008) for m in ml]
        self.table = GE.create_table_points(1.,1.,.05,dx=-.5,dy=-.5,dz=-.05,grid_size=.008)
        self.labels = SceneLabels(scene)
        cfg = GE.get_config(); self.fcq = {}
        for f in FR:
            cfg['metrics']['force_closure']['friction_coef'] = round(float(f),2)
            self.fcq[round(float(f),2)] = GraspQualityConfigFactory.create_config(cfg['metrics']['force_closure'])
        self.calls = 0

    def frame(self, f):
        ids,self.poses,cam,align = self.ge.get_model_poses(self.scene,f)
        self.ids = list(ids)
        self.mt = [GE.transform_points(m,p) for m,p in zip(self.models,self.poses)]
        self.objpts = np.concatenate(self.mt)
        self.seg = np.concatenate([np.full(len(m),i) for i,m in enumerate(self.mt)])
        self.tree = cKDTree(self.objpts)
        self.allpts = np.concatenate([self.objpts,GE.transform_points(self.table,np.linalg.inv(align@cam))])
        self.meta = sio.loadmat(R/f'data/graspnet/scenes/scene_{self.scene:04d}/kinect/meta/{f:04d}.mat')
        self.cls = self.meta['cls_indexes'].ravel().astype(int)
        assert self.ids == list(self.cls - 1), (self.ids, self.cls)
        self.lp = [self.labels.obj[c]['points']@self.poses[i][:3,:3].T+self.poses[i][:3,3] for i,c in enumerate(self.cls)]
        self.lt = [cKDTree(p) for p in self.lp]
        self.lglobal = cKDTree(np.concatenate(self.lp))
        self.lseg = np.concatenate([np.full(len(p),i) for i,p in enumerate(self.lp)])

    def owners(self,g):
        return self.seg[self.tree.query(g[:,13:16])[1]]

    def score(self, rows, owner):
        rows = np.asarray(rows).copy()
        rows[:,1] = rows[:,1].clip(0,.1)
        gl = [np.zeros((0,17)) for _ in self.mt]; gl[owner] = rows
        cm,em,dg = EU.collision_detection(gl,self.mt,self.dex,self.poses,self.allpts,outlier=.05,return_dexgrasps=True)
        q = np.full(len(rows),-1.)
        for j,g in enumerate(dg[owner]):
            if not cm[owner][j] and g is not None:
                q[j] = EU.get_grasp_score(g,self.dex[owner],FR[::-1],self.fcq)
        self.calls += len(rows)
        return q,np.asarray(cm[owner]),np.asarray(em[owner])

    def variants(self,g,owner):
        dist,pt = self.lt[owner].query(g[13:16])
        globdist,globidx = self.lglobal.query(g[13:16])
        cid = self.cls[owner]; lab = self.labels.obj[cid]
        Rp = self.poses[owner][:3,:3]; rot = g[4:13].reshape(3,3)
        ro = Rp.T @ rot
        j = int(((-V) @ ro[:,0]).argmax())
        rb = basis(-V[j].astype(float)); raw_base = Rp@rb
        offsets = np.asarray(lab['offsets'][pt,j]).reshape(48,3).astype(float)
        mu = np.asarray(lab['scores'][pt,j]).reshape(48).astype(float)
        col = np.asarray(lab['collision'][pt,j]).reshape(48)
        label_q = np.where((mu>0)&~col&(offsets[:,2]<=.1),mu,-1.)
        raw_rots = np.stack([raw_base@rx(t) for t in offsets[:,0]])
        # Nearest label rotation under parallel-jaw half-turn symmetry.
        rel = np.einsum('nji,jk->nik',raw_rots,rot)
        tr = np.maximum(np.trace(rel,axis1=1,axis2=2),rel[:,0,0]-rel[:,1,1]-rel[:,2,2])
        ai = int(tr.reshape(12,4)[:,0].argmax())
        di = int(np.abs(offsets.reshape(12,4,3)[ai,:,1]-g[3]).argmin()); cell=ai*4+di
        y=raw_base[:,1]-rot[:,0]*np.dot(raw_base[:,1],rot[:,0]); y/=np.linalg.norm(y)
        fixedbase=np.column_stack([rot[:,0],y,np.cross(rot[:,0],y)])
        grid=np.tile(g,(48,1)); grid[:,3]=offsets[:,1]; grid[:,1]=offsets[:,2].clip(0,.1)
        grid[:,4:13]=np.stack([fixedbase@rx(t) for t in offsets[:,0]]).reshape(48,9)
        width=g.copy(); width[1]=offsets[cell,2].clip(0,.1)
        exact=g.copy(); exact[13:16]=self.lp[owner][pt]; exact[4:13]=raw_rots[cell].ravel()
        exact[1]=offsets[cell,2].clip(0,.1); exact[3]=offsets[cell,1]
        meta=dict(distance=float(dist),global_distance=float(globdist),owner_diff=bool(self.lseg[globidx]!=owner),object_id=int(cid-1),label_point=int(pt),view=int(j),cell=int(cell),label_actual=float(label_q[cell]),label_best=float(label_q[utility(label_q).argmax()]),raw_label_actual=float(mu[cell]),label_collision=bool(col[cell]),width_delta=float(width[1]-g[1]),angle_error_deg=float(np.degrees(np.arccos(np.clip((tr[cell]-1)/2,-1,1)))),view_error_deg=float(np.degrees(np.arccos(np.clip(np.dot(raw_base[:,0],rot[:,0]),-1,1)))))
        return grid,width,exact,meta

def run(scene,role,part,pilot=False):
    os.nice(10); t0=time.perf_counter(); ctx=Geometry(scene)
    out=N/'results'/f'{role}_{part}_{scene:04d}'; out.mkdir(exist_ok=False)
    summaries=[]; records=[]; source_hashes={}; verify_n=0
    frames=[0] if pilot else list(range(0,256,4))
    for f in frames:
        ctx.frame(f)
        path=E/f'results/{role}/{part}/scene_{scene:04d}/evaluated/{f:04d}.npz'
        source_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        z=np.load(path); gs=z['grasps']; qs=z['quality']; cs=z['collision']
        pool=f in [0,64,128,192]; inds=range(len(gs)) if pool else range(min(1,len(gs)))
        owners=ctx.owners(gs); poolrows=[]
        for k in inds:
            g=gs[k]; grid,w,exact,m=ctx.variants(g,int(owners[k]))
            rows=np.vstack([g,w,exact,grid])
            if k==0:
                pose=grid.copy(); pose[:,1]=g[1]; rows=np.vstack([rows,pose])
            q,c,empty=ctx.score(rows,int(owners[k])); verify_n+=1
            if q[0]!=qs[k] or c[0]!=cs[k]:
                raise RuntimeError(f'native reproduction failed {scene}/{f}/{k}: {q[0]}/{qs[k]} {c[0]}/{cs[k]}')
            bestq=q[np.r_[0,np.arange(3,51)][utility(q[np.r_[0,np.arange(3,51)]]).argmax()]]
            rec=dict(score=float(g[0]),later_valid=bool(((qs[k+1:]>0)&(qs[k+1:]<=.4000001)).any()),scene=scene,role=role,partition=part,frame=f,rank=k,actual=float(q[0]),collision=bool(c[0]),empty=bool(empty[0]),width_only=float(q[1]),exact_cell=float(q[2]),exact_collision=bool(c[2]),best=float(bestq),best_grid=float(q[3+utility(q[3:51]).argmax()]),**m)
            if k==0:
                rec['pose_only']=float(q[51+utility(q[51:]).argmax()])
                rec['grid_only_fixed_width_best_index']=int(utility(q[51:]).argmax())
            records.append(rec); poolrows.append(rec)
            if pool:
                np.savez_compressed(out/f'grid_{f:04d}_{k:02d}.npz',q=q,collision=c,empty=empty)
        if pool:
            keys={name:np.array([r[name] for r in poolrows]) for name in ('actual','best','label_actual','label_best')}
            rank=np.arange(len(gs)); orders={'original':rank}
            for name,key in keys.items(): orders[name]=np.lexsort((rank,-utility(key)))
            orders['best_ties_actual']=np.lexsort((rank,-utility(keys['actual']),-utility(keys['best'])))
            for name,order in orders.items():
                ac=ap(qs[order]); summaries.append(dict(scene=scene,role=role,partition=part,frame=f,variant=name,n=len(gs),ap=float(ac.mean()*100),ap04=float(ac[:,1].mean()*100),ap08=float(ac[:,3].mean()*100)))
            saved=np.load(E/f'results/{role}/{part}/scene_{scene:04d}/ap.npy')[f]
            assert np.allclose(ap(qs),saved,rtol=0,atol=1e-12)
        print('FRAME',scene,role,part,f,round(time.perf_counter()-t0,2),ctx.calls,flush=True)
    (out/'records.json').write_text(json.dumps(records))
    (out/'ap.json').write_text(json.dumps(summaries))
    (out/'sources.json').write_text(json.dumps(source_hashes,indent=2))
    (out/'COMPLETE.json').write_text(json.dumps(dict(scene=scene,role=role,partition=part,frames=len(frames),verified=verify_n,geometry_calls=ctx.calls,elapsed=time.perf_counter()-t0,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,pilot=pilot),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('scene',type=int); p.add_argument('role'); p.add_argument('partition'); p.add_argument('--pilot',action='store_true'); a=p.parse_args()
    run(a.scene,a.role,a.partition,a.pilot)
