"""Rank-free causal value models for U1."""
from __future__ import annotations

import torch
from torch import nn


class ValueModel(nn.Module):
    def __init__(self, input_dim: int = 11, hidden_dim: int = 64, variant: str = "dual_value_nograph"):
        super().__init__()
        if variant not in {"cost_only_norank", "success_only_norank", "dual_value_nograph"}:
            raise ValueError(variant)
        self.variant = variant
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.q_head = nn.Linear(hidden_dim, 1) if variant != "cost_only_norank" else None
        self.d_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid()) if variant != "success_only_norank" else None

    def forward(self, observation_history: torch.Tensor, masks: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
        hidden, _ = self.gru(torch.relu(self.proj(observation_history)))
        z = hidden[:, -1]
        return {"q_logit": None if self.q_head is None else self.q_head(z).squeeze(-1),
                "d_normalized": None if self.d_head is None else self.d_head(z).squeeze(-1), "embedding": z}
