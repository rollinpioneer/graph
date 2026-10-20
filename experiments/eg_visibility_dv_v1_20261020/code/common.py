import os,sys,json,time,hashlib,importlib.util
from pathlib import Path
import numpy as np
R=Path(os.environ['R']); B=R.parent
G=B;D=R
P=B/'eg_sub25_b_f05_repl_v1_20261019'
LG=B/'eg_lg1_local_geometry_v1_20261020'
CP=B/'eg_geometry_compact_v1_20261020'
H=B/'eg_stat_shape_h_v1_20261020'
RP=B/'eg_b_global_rank_pilot_v1_20261020'
DATA=B/'eg_sub25_v1_20261015/data/graspnet'
SEEDS=[20261018,20261019,20261020]
C3=[100,106,115,124,132,139,150,157,160,169,176,186]
CIND=[0,1,2,3,4,5,6,7,8,9,10,11,17,18,21,22,23,24,25,29,32]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def native(scene):
 return module('native_reference',B/'eg_nqs_pilot_v2_20261018/code/common.py').Native(scene)
def export(scene,frame):
 return P/f'B/exports/subset/scene_{scene:04d}/{frame:04d}.npz'
def utility(mu):return ((mu[...,None]>0)&(mu[...,None]<=np.array([.2,.4,.6,.8,1,1.2])+1e-7)).mean(-1)
def seed(s,f):return int.from_bytes(hashlib.sha256(f'DV|20261020|{s}|{f}'.encode()).digest()[:4],'little')
def frame_sources(scene,frame):
 path=f'evaluation/development/{scene:04d}/{frame:04d}.npz'
 a=np.load(LG/path);b=np.load(CP/path);c=np.load(H/path)
 ids=np.unique(np.concatenate([z[k] for z,fs in [(a,('O_','G_')),(b,('C_','S_')),(c,('SF_','H_'))] for k in z.files if k.startswith(tuple('selected_'+f for f in fs)) and k.endswith('_CD')]))
 # Latest cache is cumulative; never interpret NaN as a negative label.
 mu=c['quality'];co=c['collision'];em=c['empty'];assert np.isfinite(mu[ids]).all()
 top=np.stack([a[f'selected_G_{s}_CD'][:10] for s in SEEDS])
 q=np.stack([a[f'Q_G_{s}'] for s in SEEDS])
 return ids,mu,co,em,c['cd_mask'],top,q
