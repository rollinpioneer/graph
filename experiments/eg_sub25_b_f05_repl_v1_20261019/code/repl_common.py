import os,sys,json,time,hashlib,importlib.util
from pathlib import Path
import numpy as np
R=Path(os.environ['REPL_ROOT']);G=R.parent;V3=G/'eg_lqr_build_confirm_v3_20261019';DATA=G/'eg_sub25_v1_20261015/data/graspnet';CODE=G/'eg_sub25_v1_20261015/runtime/EconomicGrasp';P=json.loads((R/'protocol.json').read_text());SEEDS=[20261018,20261019,20261020]
sys.path.insert(0,str(V3/'code'))
from scoring import sha256,FrozenScorer,fused_score,replace_score

def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.partial.json');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def backbone(bone):
 if bone=='A':return G/'eg_sub25_v1_20261015/runs/EG_SUB25_V1/train/economicgrasp_epoch10.tar',P['A_backbone_sha256']
 q=json.loads((R/'results/backbone_status.json').read_text());assert q['status']=='PASS';return Path(q['checkpoint']),q['sha256']
def native_class():
 spec=importlib.util.spec_from_file_location('native_reference',G/'eg_nqs_pilot_v2_20261018/code/common.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.Native,m.ap

def utility(mu):return ((mu[...,None]>0)&(mu[...,None]<=np.array([.2,.4,.6,.8,1.,1.2])+1e-7)).mean(-1).astype(np.float32)

def sample_ids(pred,s,f):
 ids=np.lexsort((np.arange(len(pred)),-pred[:,0]));seed=int.from_bytes(hashlib.sha256(f'EG-POSE-SELECT-PILOT-V1|sample|20261018|{s:04d}|{f:04d}'.encode()).digest()[:8],'little');return np.r_[ids[:64],np.random.default_rng(seed).choice(ids[64:],64,replace=False)]
