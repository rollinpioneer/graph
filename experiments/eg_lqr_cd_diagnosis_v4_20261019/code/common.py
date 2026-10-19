import os,sys,json,csv,hashlib
from pathlib import Path
import numpy as np

G=Path(os.environ['GRASP_ROOT'])
R=Path(os.environ['DIAG_ROOT'])
V=G/'eg_lqr_build_confirm_v3_20261019'
DATA=G/'eg_sub25_v1_20261015/data/graspnet'
CODE=G/'eg_sub25_v1_20261015/runtime/EconomicGrasp'
CFG=json.loads((R/'frozen_config.json').read_text())
SEEDS=[h['seed'] for h in CFG['heads']]
SPLITS=['seen','similar','novel']
sys.path.insert(0,str(V/'code'))
from scoring import sha256, fused_score,replace_score

def write(name,value):
 p=R/name;p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix('.partial'+p.suffix);tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def csv_write(name,rows):
 p=R/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def utility(mu):
 return ((mu[:,None]>0)&(mu[:,None]<=np.array([.2,.4,.6,.8,1.,1.2]))).mean(1)

def split_for(scene,groups=None):
 return next(k for k,v in (groups or CFG['exploration']).items() if scene in v)
