"""Causal student and offline teacher architectures for U2 boundaries."""
from __future__ import annotations

import torch
from torch import nn


class CausalBoundaryGRU(nn.Module):
    input_dim = 17; history_steps = 32; projection_dim = 64; hidden_dim = 96
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(self.input_dim, self.projection_dim), nn.ReLU())
        self.gru = nn.GRU(self.projection_dim, self.hidden_dim, batch_first=True)
        self.boundary_head = nn.Linear(self.hidden_dim, 1); self.event_head = nn.Linear(self.hidden_dim, 11)
        self.unknown_head = nn.Linear(self.hidden_dim, 1)
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h, _ = self.gru(self.projection(x))
        return {"boundary_logit": self.boundary_head(h).squeeze(-1), "event_logits": self.event_head(h), "unknown_logit": self.unknown_head(h).squeeze(-1), "embedding": h}


class OfflineBoundaryTeacher(nn.Module):
    """Bidirectional teacher; it is never emitted as a causal deployment model."""
    input_dim = 17; past_context = 16; future_context = 8; hidden_dim = 96
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(self.input_dim, 64), nn.ReLU())
        self.gru = nn.GRU(64, self.hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.embedding_projection = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.boundary_head = nn.Linear(self.hidden_dim, 1); self.event_head = nn.Linear(self.hidden_dim, 11); self.unknown_head = nn.Linear(self.hidden_dim, 1)
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h, _ = self.gru(self.projection(x)); h = torch.tanh(self.embedding_projection(h))
        return {"boundary_logit": self.boundary_head(h).squeeze(-1), "event_logits": self.event_head(h), "unknown_logit": self.unknown_head(h).squeeze(-1), "embedding": h}
