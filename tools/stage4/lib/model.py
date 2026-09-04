import torch
from torch import nn
class GraphStateModel(nn.Module):
    def __init__(self,input_dim=14,hidden_dim=64,node_classes=16,edge_id_classes=32,edge_type_classes=6):
        super().__init__(); self.proj=nn.Linear(input_dim,hidden_dim); self.gru=nn.GRU(hidden_dim,hidden_dim,batch_first=True); self.node_head=nn.Linear(hidden_dim,node_classes); self.edge_type_head=nn.Linear(hidden_dim,edge_type_classes); self.edge_id_head=nn.Linear(hidden_dim,edge_id_classes); self.phi_head=nn.Sequential(nn.Linear(hidden_dim,1),nn.Sigmoid()); self.cost_head=nn.Sequential(nn.Linear(hidden_dim,1),nn.Softplus()); self.event_cost_head=nn.Sequential(nn.Linear(input_dim,64),nn.ReLU(),nn.Linear(64,1));
        nn.init.zeros_(self.event_cost_head[-1].weight)
        nn.init.zeros_(self.event_cost_head[-1].bias)
    def forward(self,x,task_mask=None):
        h,_=self.gru(torch.relu(self.proj(x))); z=h[:,-1]; nl=self.node_head(z); et=self.edge_type_head(z); ei=self.edge_id_head(z)
        node_probs=nl.softmax(-1); edge_id_probs=ei.softmax(-1)
        if task_mask is not None:
            nm=task_mask.get('node') if isinstance(task_mask,dict) else None; em=task_mask.get('edge_id') if isinstance(task_mask,dict) else None
            if nm is not None: node_probs=(node_probs*nm.to(node_probs.device)); node_probs=node_probs/node_probs.sum(-1,keepdim=True).clamp_min(1e-8)
            if em is not None: edge_id_probs=(edge_id_probs*em.to(edge_id_probs.device)); edge_id_probs=edge_id_probs/edge_id_probs.sum(-1,keepdim=True).clamp_min(1e-8)
        phi=self.phi_head(z).squeeze(-1).clamp(0,1)
        # Structural boundary condition: completed node progress implies zero residual cost.
        learned_cost=self.cost_head(z).squeeze(-1)*((1.0-phi).clamp_min(0.0))
        # Graph-order prior keeps remaining cost monotone across recognised nodes;
        # the residual head still models within-node variation.
        rank=torch.tensor([6.,1.,2.,3.,4.,0.,5.,7.,8.,9.,10.,11.,12.,13.,14.,15.],device=z.device)
        expected_rank=(node_probs*rank[:node_probs.shape[-1]]).sum(-1)
        graph_cost=((5.0-expected_rank).clamp_min(0.0)/5.0 + 0.20*(1.0-phi)).clamp_min(0.0)
        event_correction=self.event_cost_head(x[:,-1]).squeeze(-1)
        remaining=(0.75*graph_cost+0.25*learned_cost+event_correction).clamp_min(0)
        return {'node_logits':nl,'node_probs':node_probs,'edge_type_logits':et,'edge_type_probs':et.softmax(-1),'edge_id_logits':ei,'edge_id_probs':edge_id_probs,'phi':phi,'remaining_cost':remaining,'event_cost_correction':event_correction,'history_embedding':z}
def load_model(path,device='cpu'):
    m=GraphStateModel(); ck=torch.load(path,map_location=device); m.load_state_dict(ck.get('model',ck),strict=False); return m.to(device).eval()
