"""Generic fixed-graph incremental mean adapter. I and S are the same arm.

This is an independently written adapter of established incremental algebra,
not the official InkStream implementation or a new planning algorithm.
"""
import torch
from cp_disr.pddl.fast_alt.fast_value import FastValue

class IncrementalValue:
    def __init__(self, model, device, task):
        self.fv = FastValue(model, device)
        self.fv.prepare(task, task.template())
        self.dtype = self.fv.static.dtype
        self.degrees=[]
        for edges in self.fv.edges_for(1):
            self.degrees.append(torch.bincount(edges[1],minlength=self.fv.N).to(self.dtype).clamp_min(1))
        # A float64 verification model needs its unregistered static pair matrix
        # cast as well. This affects only the bounded high-precision path.
        gs=model.attn._goal_static(self.fv.st,self.fv.gprop,self.fv.template)
        gs.M=gs.M.to(self.dtype)

    def codes(self, states):
        return self.fv.pack(states).to(self.dtype)

    def initial(self, codes):
        f=self.fv;st=f.st;G=codes.shape[0]
        dyn=f.enc.features.fact(torch.cat((codes,st.prop_goal_sign.unsqueeze(0).expand(G,-1,-1)),dim=-1))
        h=f.static.unsqueeze(0).repeat(G,1,1)
        h[:,st.prop_pos]=h[:,st.prop_pos]+dyn
        return h

    def read(self, H, codes):
        f=self.fv;m=f.model
        x=m._features(f.st,f.gprop,codes,H)
        z0=m.mg.heads.phi(x)
        gs=m.attn._goal_static(f.st,f.gprop,f.template)
        pf,allowed=m.attn.pair_features(gs,codes,f.gprop)
        z=z0+m.attn(z0,pf,allowed)
        value=m.mg.heads.rho(z.sum(1)).squeeze(-1)
        return dict(features=x,phi=z0,pair_features=pf,goal_encoding=z,values=value)

    @torch.no_grad()
    def full(self, codes, read=True):
        f=self.fv;G=codes.shape[0];N=f.N
        h=self.initial(codes); hs=[h];pres=[]
        hflat=h.reshape(G*N,128);edges=f.edges_for(G)
        for li,(layer,norm) in enumerate(zip(f.enc.layers,f.enc.norms)):
            out=hflat.new_zeros((G*N,128))
            for r in range(f.R):
                hh=layer.propagate(edges[r],x=hflat,edge_type_ptr=None,size=(G*N,G*N))
                out=out+(hh@f.weights[li][r])
            out=out+hflat@layer.root
            out=out+layer.bias
            pres.append(out.reshape(G,N,128))
            hflat=norm(torch.relu(out));hs.append(hflat.reshape(G,N,128))
        return dict(layers=hs,pres=pres,read=self.read(hs[-1],codes) if read else None,codes=codes)

    @torch.no_grad()
    def parent_cache(self, parent):
        return self.full(self.codes([parent]),read=False)

    @torch.no_grad()
    def children(self, cache, states, trace=False):
        f=self.fv;N=f.N;G=len(states)
        assert G<=min(48,max(1,f.edge_budget//max(f.E,1)))
        codes=self.codes(states);h=self.initial(codes)
        delta=(h-cache['layers'][0]).reshape(G*N,128)
        affected_bool=torch.zeros((G,N),device=h.device,dtype=torch.bool)
        affected_bool[:,f.st.prop_pos]=(codes!=cache['codes']).any(-1)
        affected_bool=affected_bool.reshape(-1)
        hs=[h] if trace else None
        for li,(layer,norm) in enumerate(zip(f.enc.layers,f.enc.norms)):
            active=affected_bool.nonzero().reshape(-1)
            events=[];parts=[active]
            for r,edges in enumerate(f.edges_for(G)):
                selected=affected_bool[edges[0]]
                src,dst=edges[:,selected]
                events.append((r,src,dst))
                if dst.numel():parts.append(dst)
            next_active=torch.unique(torch.cat(parts),sorted=True)
            pre=cache['pres'][li][0,next_active%N].clone()
            for r,src,dst in events:
                if not dst.numel():continue
                targets,inverse=torch.unique(dst,sorted=True,return_inverse=True)
                agg=delta.new_zeros((targets.numel(),128)).index_add_(0,inverse,delta[src])
                agg=agg/self.degrees[r][targets%N].unsqueeze(-1)
                inc=agg@f.weights[li][r]
                pre.index_add_(0,torch.searchsorted(next_active,targets),inc)
            if active.numel():
                pre.index_add_(0,torch.searchsorted(next_active,active),delta[active]@layer.root)
            new=cache['layers'][li+1].repeat(G,1,1).reshape(G*N,128)
            if next_active.numel():new[next_active]=norm(torch.relu(pre))
            delta=new-cache['layers'][li+1].expand(G,-1,-1).reshape(G*N,128)
            h=new.reshape(G,N,128)
            affected_bool.zero_();affected_bool[next_active]=True
            if trace:hs.append(h)
        return dict(layers=hs,read=self.read(h,codes),codes=codes)
