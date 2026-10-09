"""C0: the non-attention control of card CP-DISR-C1-COMPACT-DIAGNOSIS-V2 (plan 5.3).

D0 (= the earlier DENSE-G recipe) adds one residual 4-head softmax attention over ALL goal pairs on top of the goal embeddings z: z' = z + O(softmax(QK^T/sqrt(d) + b(r)) V z).
C0 keeps everything else (graph encoder, phi / rho, residual position, global read-out, loss, data) and replaces ONLY that module by a non-softmax pairwise MLP message mean over ALL goal pairs, self included, with the SAME raw pair information r_ij
(the nine features of ``GeneralGoalAttention.pair_features``) and a zero-initialised output projection:

    m_i = mean_j MLP([z_i, z_j, r_ij]);   z'_i = z_i + W_out m_i;   W_out = 0 at initialisation  (so C0 and D0 both start from the common base action distribution).

The hidden width is the integer closest to equal NEW-parameter count with D0 (66,088): MLP = Linear(2*128 + 9, h) -> ReLU -> Linear(h, 128), W_out = Linear(128, 128); 394 h + 16,640 parameters -> h = 126 (66,284, +0.30 %).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...blocksworld.method_serial.model import NEW_SEED, D
from ..model import GeneralGoalAttention, PddlSerialModel
from .. import model as PM

C0_HIDDEN = 126
D0_NEW_PARAMS = 66088


class GeneralGoalPairMLP(GeneralGoalAttention):
    """Same goal graph statics / pair features as D0's attention (inherited ``_goal_static`` and ``pair_features``), different interaction computation."""

    def __init__(self, hidden=C0_HIDDEN):
        super().__init__("dense")
        for name in ("q", "k", "v", "bias", "o"):                      # the softmax attention parameters do not exist in C0 (no idle parameters)
            delattr(self, name)
        self.mlp1 = nn.Linear(2 * D + self.R_DIM, hidden)
        self.mlp2 = nn.Linear(hidden, D)
        self.out = nn.Linear(D, D)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z, feats, allowed):                               # ``allowed`` is ignored: every pair, self included
        G, ng, _ = z.shape
        zi = z.unsqueeze(2).expand(G, ng, ng, D)
        zj = z.unsqueeze(1).expand(G, ng, ng, D)
        h = F.relu(self.mlp1(torch.cat((zi, zj, feats), dim=-1)))
        m = self.mlp2(h).mean(dim=2)
        return self.out(m)


class PddlSerialModelC0(PddlSerialModel):
    def __init__(self, mg, seed=NEW_SEED, hidden=C0_HIDDEN):
        super().__init__(mg, "mg", seed)
        self.mode = "c0"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.attn = GeneralGoalPairMLP(hidden)
        self.kind = "GOAL_C0"
        self.to(next(mg.parameters()).device)


def make_model_v2(mode, device, seed=20261008):
    """mg / dense / rel: the unchanged original factory (D0 is the original DENSE-G construction). c0: same common base, MLP interaction."""
    if mode in ("mg", "dense", "rel"):
        return PM.make_model(mode, device, seed=seed)
    assert mode == "c0", mode
    from ...blocksworld import gp_attribution as A
    base = PM.make_base(device, seed)
    mg = A.GoalProgressModelGoal(base, "global", "production", seed=seed).to(device)
    return PddlSerialModelC0(mg)


def count_new(model):
    return sum(p.numel() for p in model.new_params())


def count_trainable(model):
    ps = model.encoder_params() + model.existing_head_params() + model.new_params()
    return sum(p.numel() for p in ps)
