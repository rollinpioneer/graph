import os, json, hashlib, time
from pathlib import Path
import numpy as np

G = Path(os.environ['GRASP_ROOT'])
R = G/'eg_nqs_pilot_v2_20261018'
OLD = G/'eg_sub25_v1_20261015'
DATA = OLD/'data/graspnet'
CODE = OLD/'runtime/EconomicGrasp'
FR = np.array([.2,.4,.6,.8,1.,1.2])
SPLIT = json.loads((R/'planned_split.json').read_text())

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seed(*parts): return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:4],'little')
def utility(q):
    q=np.asarray(q)
    return ((q[...,None]>0)&(q[...,None]<=FR+1e-7)).mean(-1).astype(np.float32)
def ap(q):
    v=(np.asarray(q)[:,None]>0)&(np.asarray(q)[:,None]<=FR[None,:])
    c=np.cumsum(v,axis=0); out=np.zeros((50,6))
    if len(c): out[:min(50,len(c))]=c[:50]; out[min(50,len(c)):]=c[-1]
    return out/np.arange(1,51)[:,None]
def write_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.partial.json');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def save_npz(p,**arrays):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.partial.npz');np.savez(tmp,**arrays);tmp.replace(p)
def frames(scene):
    if scene in SPLIT['probe_fit']+SPLIT['probe_calibration']:return SPLIT['fit_and_calibration_frames']
    if scene in SPLIT['probe_development']:return SPLIT['development_frames']
    return SPLIT['confirmation_frames']

class Native:
    def __init__(self,scene):
        import graspnetAPI.graspnet_eval as GE
        import graspnetAPI.utils.eval_utils as EU
        from graspnetAPI import GraspNetEval
        from graspnetAPI.utils.dexnet.grasping.grasp_quality_config import GraspQualityConfigFactory
        self.GE,self.EU=GE,EU;self.scene=scene
        self.ge=GraspNetEval(root=str(DATA),camera='kinect',split='all')
        models,self.dex,_=self.ge.get_scene_models(scene,ann_id=0)
        self.models=[GE.voxel_sample_points(m,.008) for m in models]
        self.table=GE.create_table_points(1.,1.,.05,dx=-.5,dy=-.5,dz=-.05,grid_size=.008)
        self.config=GE.get_config();self.fc={}
        for f in FR:
            self.config['metrics']['force_closure']['friction_coef']=round(float(f),2)
            self.fc[round(float(f),2)]=GraspQualityConfigFactory.create_config(self.config['metrics']['force_closure'])
    def frame(self,f):
        from scipy.spatial import cKDTree
        self.ids,self.poses,cam,align=self.ge.get_model_poses(self.scene,f)
        self.mt=[self.GE.transform_points(m,p) for m,p in zip(self.models,self.poses)]
        self.objpts=np.concatenate(self.mt)
        self.seg=np.concatenate([np.full(len(m),i) for i,m in enumerate(self.mt)])
        self.tree=cKDTree(self.objpts)
        self.table_trans=self.GE.transform_points(self.table,np.linalg.inv(align@cam))
        self.allpts=np.concatenate([self.objpts,self.table_trans])
    def owners(self,rows):return self.seg[self.tree.query(rows[:,13:16])[1]]
    def score(self,rows):
        rows=np.asarray(rows).astype(np.float64).copy();rows[:,1]=rows[:,1].clip(0,.1)
        owner=self.owners(rows);q=np.full(len(rows),-1.);coll=np.zeros(len(rows),bool);empty=coll.copy()
        for o in np.unique(owner):
            idx=np.flatnonzero(owner==o);gl=[np.zeros((0,17)) for _ in self.models];gl[o]=rows[idx]
            cs,es,ds=self.EU.collision_detection(gl,self.mt,self.dex,self.poses,self.allpts,outlier=.05,return_dexgrasps=True)
            coll[idx]=cs[o];empty[idx]=es[o]
            for j,g in enumerate(ds[o]):
                if not cs[o][j] and g is not None:q[idx[j]]=self.EU.get_grasp_score(g,self.dex[o],FR[::-1],self.fc)
        assert np.isfinite(q).all()
        return q,coll,empty,owner
    def official(self,rows):
        from graspnetAPI import GraspGroup
        rows=rows.astype(np.float64).copy();rows[:,1]=rows[:,1].clip(0,.1)
        if not len(rows):return np.zeros((0,17)),np.zeros(0),np.zeros(0,bool)
        gs,qs,cs=self.EU.eval_grasp(GraspGroup(rows),self.models,self.dex,self.poses,self.config,table=self.table_trans,voxel_size=.008,TOP_K=50)
        gs=np.concatenate(gs);qs=np.concatenate(qs);cs=np.concatenate(cs)
        order=np.argsort(gs[:,0])[::-1]
        return gs[order],qs[order],cs[order]
