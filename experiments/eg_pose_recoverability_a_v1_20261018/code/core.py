import os,sys,json,time,hashlib
from pathlib import Path
import numpy as np
G=Path(os.environ['GRASP_ROOT']);R=Path(os.environ['POSE_ROOT']);OLD=G/'eg_sub25_v1_20261015';PRIOR=G/'eg_quality_validation_v1_20261018';E=OLD/'runs/EG_SUB25_V1/evaluation';CODE=OLD/'runtime/EconomicGrasp';DATA=OLD/'data/graspnet'
SCENES=[8,36,41,44,58,65,73,75,80,86];FRAMES=list(range(0,256,4))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def rx(t):
 c,s=np.cos(t),np.sin(t);return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def grid(g):
 v=np.tile(g,(48,1));v[:,3]=np.tile(np.array([.01,.02,.03,.04],np.float32),12)
 for k in range(12):v[k*4:(k+1)*4,4:13]=(g[4:13].reshape(3,3)@rx(k*np.pi/12)).ravel()
 return v
