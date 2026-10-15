import json
from pathlib import Path
import numpy as np
import torch
from .model import load_model

def ensemble_predictions(predictions):
    n=np.asarray(predictions); mean=n.mean(0); std=n.std(0); return {'mean':mean,'std':std}

class _Stream:
    def __init__(self, parent, task_id): self.parent=parent; self.task_id=task_id; self.history=[]
    def step(self, feature_dict): return self.parent.step(self, feature_dict)

class PathGraphEnsemble:
    """Three-seed inference bundle consumed by Stage 5."""
    def __init__(self, checkpoints, device='cpu', history_steps=32):
        self.device=device if device.startswith('cuda') and torch.cuda.is_available() else 'cpu'; self.history_steps=history_steps
        self.models=[load_model(p,self.device) for p in checkpoints]; self.checkpoints=checkpoints
    @classmethod
    def from_bundle(cls,bundle_path,device='cpu'):
        b=json.loads(Path(bundle_path).read_text()); return cls([x['path'] if isinstance(x,dict) else x for x in b['checkpoints']],device,b.get('history_steps',32))
    def new_stream(self,task_id): return _Stream(self,task_id)
    def _feature(self, f):
        if isinstance(f,dict):
            vals=[]
            for k in ('eef_pos','object_pos','target_pos','gripper_state','action'):
                v=f.get(k,0.0); vals.extend(np.asarray(v,dtype=np.float32).reshape(-1).tolist()[:3] or [0.0])
            a=np.zeros(14,np.float32); a[:min(14,len(vals))]=vals[:14]; return a
        a=np.asarray(f,dtype=np.float32).reshape(-1); out=np.zeros(14,np.float32); out[:min(14,len(a))]=a[:14]; return out
    def step(self,state,feature_dict):
        state.history.append(self._feature(feature_dict)); state.history=state.history[-self.history_steps:]; x=np.asarray(state.history,np.float32); xt=torch.from_numpy(x).unsqueeze(0).to(self.device)
        outs=[m(xt) for m in self.models];
        def stack(k): return np.stack([o[k].detach().cpu().numpy()[0] for o in outs])
        npv=stack('node_probs'); epv=stack('edge_type_probs'); eiv=stack('edge_id_probs'); ph=stack('phi'); co=stack('remaining_cost')
        nmean=npv.mean(0); emean=epv.mean(0);
        entropy=lambda p: float(-(p*np.log(np.clip(p,1e-8,1))).sum())
        return {'node_probs_mean':nmean,'node_predictive_entropy':entropy(nmean),'node_mutual_information':entropy(nmean)-float(np.mean([entropy(p) for p in npv])),'edge_type_probs_mean':emean,'edge_predictive_entropy':entropy(emean),'edge_mutual_information':entropy(emean)-float(np.mean([entropy(p) for p in epv])),'edge_id_probs_mean':eiv.mean(0),'phi_mean':float(np.clip(ph.mean(),0,1)),'phi_std':float(ph.std()),'remaining_cost_mean':float(max(0,co.mean())),'remaining_cost_std':float(co.std()),'per_model_phi':ph.tolist(),'per_model_remaining_cost':co.tolist()}
