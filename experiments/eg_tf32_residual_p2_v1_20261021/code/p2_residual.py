from common import *
import torch
torch.set_num_threads(1)
L=G/'eg_lg1_local_geometry_v1_20261020'
FAMILIES=['Eact','Epot','Gcap65']

class FrozenG(GeoScorer):
 @torch.no_grad()
 def logits(self,feat,act,raw):
  x=np.concatenate([(feat-self.ck['mean'])/self.ck['std'],act,geom_input(raw,self.ck['geom_norm'])],-1).astype(np.float32)
  return self.net[:-1](torch.from_numpy(x)).squeeze(-1).numpy()

def token(feat,ck):
 return (feat[:,256:].astype(np.float64)-ck['mean'][256:].astype(np.float64))/ck['std'][256:].astype(np.float64)

def predict(family,theta,t,score,logit,calibration):
 delta=t@theta[:64]+theta[64]
 if family in ['Eact','Epot']:
  cal=calibration[family]
  return cal['alpha']*(score.astype(np.float64)+delta)+cal['beta']
 return torch.sigmoid(torch.from_numpy(logit.astype(np.float64)+delta)).numpy()

def gheads(cfg):
 return {h['seed']:FrozenG(Path(h['path']),h['sha256']) for h in cfg['protected_G']}
