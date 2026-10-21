import os,sys
from pathlib import Path
import numpy as np
import torch
from adapters import Adapter,attributes
R=Path(os.environ['TASK_ROOT']);os.environ['R']=str(R);sys.path.insert(0,str(R.parent/'eg_lg1_local_geometry_v1_20261020/code'))
from common import *
class ConditionedScorer:
 def __init__(self,path,expected):
  assert sha(path)==expected;self.ck=torch.load(path,map_location='cpu',weights_only=False);self.net=Adapter(network(390),self.ck['family']);self.net.load_state_dict(self.ck['model']);self.net.eval()
 def quality(self,feat,act,raw,scene,frame,pred):
  ck=self.ck;x=np.concatenate([(feat-ck['mean'])/ck['std'],act,geom_input(raw,ck['geom_norm'])],-1).astype(np.float32);att=attributes(pred[:,None,:],pred[:,0])[:,0];att=(att-ck['attribute_mean'])/ck['attribute_std']
  with torch.no_grad():return self.net(torch.from_numpy(x),torch.from_numpy(att)).squeeze(-1).numpy()
