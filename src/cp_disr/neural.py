"""Production shared RGCN, goal-aligned differences and seven policy variants.

Torch/PyG are deliberately optional at import of the package/logic CLI.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch_geometric.nn import RGCNConv
from .graph import RELATIONS,four_views,view
from .facts import Truth
from .common import DataIntegrityError

METHODS=('B0','B1','B2','Full','A_DD','A_Q','A_B')
def mlp(inp,out=128):return nn.Sequential(nn.Linear(inp,128),nn.ReLU(),nn.Linear(128,out),nn.ReLU())

class NodeFeatures(nn.Module):
    def __init__(self,action_schemas,predicate_schemas,argument_types):
        super().__init__();self.actions={s:i for i,s in enumerate(sorted(action_schemas))};self.predicates={s:i for i,s in enumerate(sorted(predicate_schemas))};self.types={s:i for i,s in enumerate(sorted(argument_types))}
        self.kind=nn.Embedding(2,128);self.action=nn.Embedding(max(1,len(self.actions)),128);self.predicate=nn.Embedding(max(1,len(self.predicates)),128)
        self.arg=nn.Embedding(max(1,len(self.types)),128);self.fact=nn.Linear(4,128)
    def forward(self,graph):
        device=self.kind.weight.device; facts=dict(graph.values);signs={g.fact_id:g.sign for g in graph.template.goals};rows=[]
        for n in graph.template.nodes:
            action=n.kind=='ACTION';row=self.kind.weight[0 if action else 1]
            row=row+(self.action.weight[self.actions[n.schema]] if action else self.predicate.weight[self.predicates[n.schema]])
            # Ordered parameter-role weights, never arbitrary object/instance IDs.
            for role,t in enumerate(n.argument_types):row=row+self.arg.weight[self.types[t]]/(role+1)
            if not action:
                code=[float(facts[n.id]==t) for t in (Truth.TRUE,Truth.FALSE,Truth.UNKNOWN)]+[float(signs.get(n.id,0))]
                row=row+self.fact(torch.tensor(code,device=device,dtype=row.dtype))
            rows.append(row)
        return torch.stack(rows)

class GoalReadout(nn.Module):
    def __init__(self):
        super().__init__();self.goal=mlp(129);self.global_readout=mlp(384)
    def forward(self,h,template):
        ids=template.node_ids
        goals=torch.stack([self.goal(torch.cat((h[ids.index(g.fact_id)],h.new_tensor([g.sign])))) for g in template.goals]) if template.goals else h.new_zeros((0,128))
        def pool(kind):
            idx=[i for i,n in enumerate(template.nodes) if n.kind==kind];return h[idx].mean(0) if idx else h.new_zeros(128)
        global_row=self.global_readout(torch.cat((pool('ACTION'),pool('PROPOSITION'),goals.mean(0) if len(goals) else h.new_zeros(128))))
        return torch.cat((global_row.unsqueeze(0),goals),0)

class GraphEncoder(nn.Module):
    def __init__(self,actions,predicates,types):
        super().__init__();self.features=NodeFeatures(actions,predicates,types)
        self.layers=nn.ModuleList([RGCNConv(128,128,len(RELATIONS),num_bases=4,aggr='mean',root_weight=True) for _ in range(4)])
        self.norms=nn.ModuleList([nn.LayerNorm(128) for _ in range(4)]);self.readout=GoalReadout();self.forward_calls=0
    def forward(self,graph):
        self.forward_calls+=1;h=self.features(graph);ids={v:i for i,v in enumerate(graph.template.node_ids)};edges=graph.edges
        ei=torch.tensor([[ids[a] for a,b,r in edges],[ids[b] for a,b,r in edges]],device=h.device,dtype=torch.long).reshape(2,-1)
        et=torch.tensor([RELATIONS.index(r) for a,b,r in edges],device=h.device,dtype=torch.long)
        for layer,norm in zip(self.layers,self.norms):h=norm(torch.relu(layer(h,ei,et)))
        return self.readout(h,graph.template)

class TupleEncoder(nn.Module):
    """B0: typed fact/contract tuples + shared MLP; no graph convolution/nominal view."""
    def __init__(self,actions,predicates,types):
        super().__init__();self.features=NodeFeatures(actions,predicates,types);self.rel=nn.Embedding(4,128);self.tuple_net=mlp(384);self.node_net=mlp(128);self.readout=GoalReadout()
    def forward(self,graph):
        h=self.features(graph);ids={v:i for i,v in enumerate(graph.template.node_ids)}
        tuples=[self.tuple_net(torch.cat((h[ids[a]],h[ids[b]],self.rel.weight[i]))) for a,b,r in graph.template.edges for i in range(4) if r==RELATIONS[i]]
        context=torch.stack(tuples).mean(0) if tuples else h.new_zeros(128)
        return self.readout(self.node_net(h)+context,graph.template)

@dataclass
class Differences:
    zk:torch.Tensor
    zh:torch.Tensor
    dk:torch.Tensor
    dh:torch.Tensor
    dp:torch.Tensor
    encodings:tuple

def differences(encoder,graphs):
    k,h,ki,hi=graphs;zk=encoder(k);zki=zk if ki.values==k.values else encoder(ki)
    if h is k:
        dk=zki.float()-zk.float();zero=torch.zeros_like(dk)
        return Differences(zk,zk,dk,dk,zero,(zk,zk,zki,zki))
    zh=encoder(h);zhi=zh if hi.values==h.values else encoder(hi)
    dk=zki.float()-zk.float();dh=zhi.float()-zh.float()
    return Differences(zk,zh,dk,dh,dh-dk,(zk,zh,zki,zhi))

class CandidateReadout(nn.Module):
    def __init__(self,context_dim):
        super().__init__();self.query=nn.Linear(context_dim,128);self.key=nn.Linear(128,128);self.value=nn.Linear(128,128);self.net=mlp(context_dim+128)
    def forward(self,context,rows,mask=None):
        mask=torch.ones(len(rows),dtype=torch.bool,device=rows.device) if mask is None else mask
        if not mask.any():raise DataIntegrityError('Goal/global readout entirely masked')
        score=(self.key(rows)*self.query(context)).sum(-1)/math.sqrt(128)
        attention=torch.softmax(score.masked_fill(~mask,-torch.inf),dim=-1)
        return self.net(torch.cat((context,(attention[:,None]*self.value(rows)).sum(0))))

class AnchoredPrior(nn.Module):
    def __init__(self,context_dim=512):
        super().__init__();self.readout=CandidateReadout(context_dim);self.final=nn.Linear(128,1,bias=False)
    def forward(self,context,dp,B=.5,unbounded=False,mask=None):
        # Both paths share context, goal mask and parameters. Exact zero shortcut.
        if torch.count_nonzero(dp).item()==0:
            zero=dp.sum()*0+context.sum()*0
            return zero.expand(128),zero
        up=self.readout(context,dp,mask)-self.readout(context,torch.zeros_like(dp),mask)
        s=self.final(up).squeeze(-1);return up,B*s if unbounded else B*torch.tanh(s)

@dataclass
class PolicyOutput:
    candidate_ids:tuple
    logits:torch.Tensor
    mask:torch.Tensor
    distribution:object
    value:torch.Tensor
    q:torch.Tensor
    hidden:torch.Tensor
    diagnostics:dict
    ended_reason:str|None=None
    def select(self,deterministic=False):
        if self.distribution is None:raise DataIntegrityError('NO_SAFE_CANDIDATES')
        if deterministic:
            maximum=self.logits[self.mask].max();choices=[i for i,k in enumerate(self.candidate_ids) if self.mask[i] and self.logits[i]==maximum]
            index=min(choices,key=lambda i:self.candidate_ids[i])
        else:index=int(self.distribution.sample())
        return self.candidate_ids[index],index

class Policy(nn.Module):
    """One parameter owner. B1 alone intentionally reads current enhanced structure."""
    def __init__(self,actions,predicates,types,observation_dim,candidate_dim,method='Full',B=.5):
        super().__init__()
        if method not in METHODS:raise ValueError(method)
        self.method=method;self.B=B;self.encoder=GraphEncoder(actions,predicates,types);self.tuples=TupleEncoder(actions,predicates,types)
        self.observation=nn.Linear(observation_dim,128);self.gru=nn.GRUCell(128,128);self.candidate=nn.Linear(candidate_dim,128)
        self.contract=CandidateReadout(384);self.b0_readout=mlp(512);self.prior=AnchoredPrior(512)
        self.base=nn.Linear(128,1);self.v_head=nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Linear(128,1));self.q_head=nn.Sequential(nn.Linear(640,128),nn.ReLU(),nn.Linear(128,1))
    @property
    def q_coefficient(self):return 0. if self.method=='A_Q' else .1
    def initial_hidden(self):return next(self.parameters()).new_zeros(128)
    def advance_hidden(self,base_input,hidden):
        x=torch.as_tensor(base_input,device=hidden.device,dtype=hidden.dtype)
        return self.gru(torch.relu(self.observation(x)),hidden)
    def forward(self,snapshot,hidden=None):
        hidden=self.initial_hidden() if hidden is None else hidden
        zo=self.advance_hidden(snapshot.base_input,hidden);mask=torch.tensor(snapshot.mask,device=zo.device,dtype=torch.bool)
        if not mask.any():
            empty=zo.new_zeros(len(mask));return PolicyOutput(snapshot.candidate_ids,empty,mask,None,zo.sum()*0,empty,zo,{},'NO_SAFE_CANDIDATES')
        facts=snapshot.facts.values;contracts={c.id:c for c in snapshot.template.contracts}
        edges=() if self.method in ('B0','B2') else snapshot.prior_edges
        k=view(snapshot.template,facts)
        zk=self.tuples(k) if self.method=='B0' else self.encoder(k)
        zh=(self.encoder(view(snapshot.template,facts,edges)) if edges else zk) if self.method=='B1' else None
        uk_all=[];up_all=[];logits=[];diagnostics={'differences':{},'prior_inputs':{},'up':{},'delta':{}}
        for i,cid in enumerate(snapshot.candidate_ids):
            if not snapshot.mask[i]:
                uk_all.append(zo.new_zeros(128));up_all.append(zo.new_zeros(128));logits.append(zo.sum()*0);continue
            ca=self.candidate(torch.as_tensor(snapshot.candidate_features[i],device=zo.device,dtype=zo.dtype))
            context=torch.cat((zo,ca,zk.mean(0)))
            if self.method=='B0':
                uk=self.b0_readout(torch.cat((context,zk.mean(0))));prior_input=torch.zeros_like(zk)
            elif self.method=='B1':uk=self.contract(context,zk);prior_input=zh.float()-zk.float() if edges else torch.zeros_like(zk)
            else:
                delta=differences(self.encoder,four_views(snapshot.template,facts,edges,contracts[cid]))
                uk=self.contract(context,delta.dk);prior_input=(delta.dh if self.method=='A_DD' else delta.dp) if edges else torch.zeros_like(delta.dk)
                diagnostics['differences'][cid]=delta
            up,residual=self.prior(torch.cat((context,uk)),prior_input,self.B,self.method=='A_B')
            uk_all.append(uk);up_all.append(up);logits.append(self.base(uk).squeeze(-1)+residual)
            diagnostics['prior_inputs'][cid]=prior_input;diagnostics['up'][cid]=up;diagnostics['delta'][cid]=residual
        uk=torch.stack(uk_all);up=torch.stack(up_all);mean_k=uk[mask].mean(0);mean_p=up[mask].mean(0)
        value=self.v_head(torch.cat((zo,zk.mean(0),mean_k,mean_p))).squeeze(-1)
        q=self.q_head(torch.cat((uk,up,zo.expand(len(uk),-1),mean_k.expand(len(uk),-1),mean_p.expand(len(uk),-1)),dim=-1)).squeeze(-1)
        logits=torch.stack(logits).masked_fill(~mask,-torch.inf)
        return PolicyOutput(snapshot.candidate_ids,logits,mask,torch.distributions.Categorical(logits=logits),value,q,zo,diagnostics)
