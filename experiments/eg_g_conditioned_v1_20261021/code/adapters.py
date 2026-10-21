"""P0 definitions only. No optimizer or training entry point."""
import torch
from torch import nn

class Adapter(nn.Module):
    def __init__(self, base, family):
        super().__init__()
        self.base=base
        self.family=family
        for p in base.parameters(): p.requires_grad_(False)
        if family=='FiLM':
            self.encoder=nn.Sequential(nn.Linear(14,64),nn.ReLU(),nn.Linear(64,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
            self.mod=nn.Linear(128,256)
            nn.init.zeros_(self.mod.weight);nn.init.zeros_(self.mod.bias)
        else:
            # Equal-depth, similar parameter-count residual controls.
            dim=128 if family=='cap' else 142
            width=186 if family=='cap' else 181
            self.residual=nn.Sequential(nn.Linear(dim,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,1))
            nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
    def forward(self,x,a):
        h=self.base[1](self.base[0](x))
        if self.family=='FiLM':
            gamma,beta=self.mod(self.encoder(a)).chunk(2,-1)
            logit=self.base[2](h*(1+gamma)+beta)
        else:
            inp=h if self.family=='cap' else torch.cat([h,a],-1)
            logit=self.base[2](h)+self.residual(inp)
        return torch.sigmoid(logit)

def attributes(hyp,parent_score):
    import numpy as np
    return np.concatenate([hyp[...,4:13],hyp[...,13:16],hyp[...,1:2],np.broadcast_to(parent_score[...,None,None],hyp.shape[:-1]+(1,))],-1).astype(np.float32)
