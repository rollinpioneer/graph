from __future__ import annotations
import torch
from torch import nn

class ChunkPolicy(nn.Module):
    def __init__(self, obs_dim=14, action_dim=4, horizon=16, hidden_dim=128):
        super().__init__(); self.horizon=horizon; self.action_dim=action_dim
        self.net=nn.Sequential(nn.Linear(obs_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,horizon*action_dim))
    def forward(self,x): return self.net(x).reshape(len(x),self.horizon,self.action_dim)
