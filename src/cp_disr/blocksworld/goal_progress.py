"""CP-DISR C1 goal-progress runbook, phase 2 (C1-BW-GOAL-PROGRESS-V1): matched GLOBAL vs ADDITIVE goal-progress scoring, plus the B2-CONTINUE control.

M1 / M2 share ONE goal-free state encoder (the A02 B2 graph encoder with the goal-mark channel zeroed; no goal pooling), identical per-goal inputs x(F,g) (386-d), identical two-layer ``phi`` / ``rho`` heads
(82,688 parameters each, identical initial tensors), identical data and losses. The only difference is the aggregation order:
    V1(F,G) = rho( sum_g phi(x(F,g)) )        global: aggregate, then judge
    V2(F,G) = sum_g rho( phi(x(F,g)) )        additive: judge each goal, then aggregate
    score_j(F,a,G) = Vj(F,G) - Vj(T_nom(F,a),G)      logits = score over the legal set (no time input, no GRU, no residual B2 logits)
Training labels (planner A* and per-candidate distances) are used only in the losses; they never enter a forward pass.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import imitation as I
from . import state as S
from .planner import Solver

MODES = ("global", "additive")
X_DIM = 386
HEAD_SEED = 20261006
EXISTING_LR = 1e-4
NEW_HEAD_LR = 3e-4
EPOCHS = 100
BATCH = 32
GRAD_CLIP = 0.5
SHUFFLE_SEED = 0
RANK_WEIGHT = 1.0
CHUNK = 192
RUN_IDS = {"M0": "R-C1-GP-M0-0", "M1": "R-C1-GP-M1-0", "M2": "R-C1-GP-M2-0"}


class ProgressHeads(nn.Module):
    """Identical module for M1 and M2 (the mode only changes where the sum is taken)."""

    def __init__(self, seed=HEAD_SEED):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.phi = nn.Sequential(nn.Linear(X_DIM, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
            self.rho = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1, bias=False))


class GoalProgressModel(nn.Module):
    """A02 B2 graph encoder (fine-tuned) + new heads. ``base`` is the loaded B2CachedPolicy; only ``base.encoder`` is used and trained."""

    def __init__(self, base, mode, seed=HEAD_SEED):
        super().__init__()
        assert mode in MODES
        self.mode = mode
        self.base = base
        self.heads = ProgressHeads(seed)
        self._st0 = {}
        for name, mod in base.named_children():
            for p in mod.parameters():
                p.requires_grad_(name == "encoder")

    def trainable_encoder_params(self):
        return [p for p in self.base.encoder.parameters() if p.requires_grad]

    def head_params(self):
        return list(self.heads.parameters())

    def goal_free_static(self, template):
        hit = self._st0.get(id(template))
        if hit is not None and hit[0] is template:
            return hit[1]
        st = self.base._static(template)
        st0 = dataclasses.replace(st, prop_goal_sign=torch.zeros_like(st.prop_goal_sign))
        gprop = torch.tensor([st.prop_index[g.fact_id] for g in template.goals], device=st.prop_pos.device)
        self._st0[id(template)] = (template, (st0, gprop))
        return st0, gprop

    def features(self, st0, gprop, codes):
        """x(F,g) for G graphs: (G, ng, 386). Nothing here depends on the other goals or on the goal set."""
        _ro, H = self.base._encode(st0, codes)                       # goal marks are zero; the (unused) goal readout is ignored
        act = H[:, st0.action_pos].mean(1)
        prop = H[:, st0.prop_pos].mean(1)
        hg = H[:, st0.goal_pos]                                      # (G, ng, 128) hidden state of each goal proposition node
        truth = codes[:, gprop, 0].unsqueeze(-1)                     # current truth of the goal atom
        ng = hg.shape[1]
        sign = st0.goal_sign.unsqueeze(0).expand(codes.shape[0], -1, -1)
        return torch.cat((hg, act.unsqueeze(1).expand(-1, ng, -1), prop.unsqueeze(1).expand(-1, ng, -1), truth, sign), dim=-1)

    def values(self, x, goal_mask=None):
        """(G,) value of each graph; ``goal_mask`` (ng,) bool selects a subset of the goals (used by the additivity tests)."""
        if goal_mask is not None:
            x = x[:, goal_mask]
        phi = self.heads.phi(x)
        if self.mode == "global":
            return self.heads.rho(phi.sum(1)).squeeze(-1)
        return self.heads.rho(phi).squeeze(-1).sum(1)

    def group_scores(self, snaps, goal_mask=None):
        """(S, n) masked scores of snapshots sharing one template: V(F) - V(T_nom(F, a)) on the legal candidates."""
        base = self.base
        st0, gprop = self.goal_free_static(snaps[0].template)
        st = base._static(snaps[0].template)
        n = len(snaps[0].candidate_ids)
        dev = st.prop_pos.device
        codes_list, offsets, legal_per, off = [], [], [], 0
        for sn in snaps:
            legal = [i for i, m in enumerate(sn.mask) if m]
            legal_per.append(legal)
            codes0 = base._codes(st, sn.facts.values)
            rows = [st.action_row[sn.candidate_ids[i]] for i in legal]
            codes_list.append(torch.cat((codes0.unsqueeze(0), base._successor_codes(st, codes0, rows)), 0))
            offsets.append(off)
            off += codes_list[-1].shape[0]
        x = self.features(st0, gprop, torch.cat(codes_list, 0))
        v = self.values(x, goal_mask)
        out = v.new_full((len(snaps), n), -torch.inf)
        for si, sn in enumerate(snaps):
            o, legal = offsets[si], legal_per[si]
            out[si, legal] = v[o] - v[o + 1: o + 1 + len(legal)]
        return out


class ScorePolicy:
    """Adapter so the frozen episode runner can evaluate M1 / M2 (no hidden state, no time input)."""

    def __init__(self, model):
        self.model = model

    def initial_hidden(self):
        return torch.zeros(1)

    @torch.no_grad()
    def __call__(self, snap, hidden):
        logits = self.model.group_scores([snap])[0]
        return SimpleNamespace(logits=logits, distribution=torch.distributions.Categorical(logits=logits), hidden=hidden)


def load_base(root, device):
    from .eval_a03 import load_model
    return load_model(root, "B2", device)


# ------------------------------------------------------------------------------------------------ data
def candidate_distances(solver, case, state):
    """{legal action id: exact d*(T(state,a), goal)} (labels only)."""
    out = {}
    for a, nxt in S.successors(tuple(state)):
        d = solver.cost_to_go(nxt, case.problem.goal)
        if d is None:
            raise RuntimeError("unsolvable successor")
        out[S.action_id(case.problem.names, a)] = d
    return out


def build_rank_labels(trajs, cases, solver=None):
    solver = solver or Solver()
    by_id = {c.case_id: c for c in cases}
    labels = {}
    for t in trajs:
        c = by_id[t["case_id"]]
        for s in t["states"][:-1]:
            key = "%s|%s" % (t["case_id"], ",".join(map(str, s)))
            if key not in labels:
                labels[key] = candidate_distances(solver, c, s)
    return labels


def rank_key(case_id, state):
    return "%s|%s" % (case_id, ",".join(map(str, state)))


def nll_and_rank(scores, astar_mask, dist, legal_mask):
    """Per-decision L_IL (-log sum_{a in A*} softmax) and mean pairwise rank loss over strictly ordered legal pairs (0 if the state has none)."""
    nll = I.imitation_nll(scores, astar_mask)
    z = scores.masked_fill(~legal_mask, 0.0)
    diff = z.unsqueeze(2) - z.unsqueeze(1)
    pair = (dist.unsqueeze(2) < dist.unsqueeze(1)) & legal_mask.unsqueeze(2) & legal_mask.unsqueeze(1)
    cnt = pair.sum((1, 2))
    rank = (F.softplus(-diff) * pair).sum((1, 2)) / cnt.clamp(min=1)
    return nll, rank, cnt


class GPTrainer:
    """Matched trainer for M1 / M2. Trajectories are equal-weight; decisions are independent (no recurrence)."""

    def __init__(self, model, cases, rank_labels, device):
        self.model, self.device, self.rank_labels = model, device, rank_labels
        self.cases = {c.case_id: c for c in cases}
        enc = model.trainable_encoder_params()
        self.params = enc + model.head_params()
        self.optimizer = torch.optim.Adam([{"params": enc, "lr": EXISTING_LR}, {"params": model.head_params(), "lr": NEW_HEAD_LR}], betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        self.steps, self.nan_events = 0, 0
        self._snaps = {}

    def snapshots(self, traj):
        got = self._snaps.get(traj["tid"])
        if got is None:
            c = self.cases[traj["case_id"]]
            got = [I.snapshot_at(c, s, t) for t, s in enumerate(traj["states"][:-1])]
            self._snaps[traj["tid"]] = got
        return got

    def _group_loss(self, snaps, astar, keys):
        groups = {}
        for i, sn in enumerate(snaps):
            groups.setdefault(id(sn.template), []).append(i)
        nll, rank, hit = [None] * len(snaps), [None] * len(snaps), [None] * len(snaps)
        for idx in groups.values():
            sub = [snaps[i] for i in idx]
            scores = self.model.group_scores(sub)
            ids = sub[0].candidate_ids
            am = torch.zeros_like(scores, dtype=torch.bool)
            dist = torch.zeros_like(scores)
            lm = torch.tensor([s.mask for s in sub], device=scores.device, dtype=torch.bool)
            for j, i in enumerate(idx):
                for a in astar[i]:
                    am[j, ids.index(a)] = True
                for a, d in self.rank_labels[keys[i]].items():
                    dist[j, ids.index(a)] = d
            nl, rk, _cnt = nll_and_rank(scores, am, dist, lm)
            flags = am.gather(1, scores.argmax(-1).unsqueeze(1)).squeeze(1)
            for j, i in enumerate(idx):
                nll[i], rank[i], hit[i] = nl[j], rk[j], flags[j]
        return torch.stack(nll), torch.stack(rank), torch.stack(hit)

    def step(self, trajs):
        self.model.train()
        per = [self.snapshots(t) for t in trajs]
        snaps = [s for p in per for s in p]
        astar = [a for t in trajs for a in t["astar"]]
        keys = [rank_key(t["case_id"], s) for t in trajs for s in t["states"][:-1]]
        w = torch.tensor([x for ws in I.decision_weights(trajs) for x in ws], device=self.device)
        self.optimizer.zero_grad()
        tot = {"loss": 0.0, "nll": 0.0, "rank": 0.0, "mass": 0.0, "hit": 0.0}
        for a in range(0, len(snaps), CHUNK):
            b = min(a + CHUNK, len(snaps))
            nll, rank, hit = self._group_loss(snaps[a:b], astar[a:b], keys[a:b])
            part = (w[a:b] * (nll + RANK_WEIGHT * rank)).sum()
            if not torch.isfinite(part):
                self.nan_events += 1
                raise I.ImitationError("non-finite loss")
            part.backward()
            tot["loss"] += float(part)
            tot["nll"] += float((w[a:b] * nll).sum())
            tot["rank"] += float((w[a:b] * rank).sum())
            tot["mass"] += float((w[a:b] * torch.exp(-nll.detach())).sum())
            tot["hit"] += float((w[a:b] * hit.float()).sum())
        norm = float(torch.nn.utils.clip_grad_norm_(self.params, GRAD_CLIP, error_if_nonfinite=True))
        self.optimizer.step()
        self.steps += 1
        return {**tot, "grad_norm": norm, "decisions": len(snaps)}

    def epoch(self, trajs, seed):
        order = list(range(len(trajs)))
        random.Random(seed).shuffle(order)
        rows = [self.step([trajs[i] for i in order[k:k + BATCH]]) for k in range(0, len(order), BATCH)]
        n = len(rows)
        return {"batches": n, "decisions": sum(r["decisions"] for r in rows), **{k: sum(r[k] for r in rows) / n for k in ("loss", "nll", "rank", "mass", "hit", "grad_norm")}}


# ------------------------------------------------------------------------------------------------ labels of section 13 (applied only after every result exists)
CORE = ("C5-2", "C5-23", "S6-2", "S7-2", "S8-2", "K6-3")
CLOSE = 0.10
ADVANTAGE = 0.10


def core_rates(table, cond, field):
    vals = [table[(cond, s)][field] / table[(cond, s)]["n"] for s in CORE if (cond, s) in table and table[(cond, s)]["n"]]
    return sum(vals) / len(vals) if vals else None


def direction_labels(table):
    """table[(condition, slice)] = {'n','success','perfect'}; returns the applicable section-13 labels (all computed from the complete result set)."""
    D = {c: core_rates(table, c, "perfect") for c in ("M0", "M1", "M2", "C0", "C3", "G1")}
    Sc = {c: core_rates(table, c, "success") for c in ("M0", "M1", "M2", "C0", "C3", "G1")}
    nonneg = sum(1 for s in CORE if (("M2", s) in table and ("M1", s) in table and table[("M2", s)]["perfect"] / table[("M2", s)]["n"] >= table[("M1", s)]["perfect"] / table[("M1", s)]["n"]))
    a1 = lambda c: sum(table[(c, s)]["success"] for s in ("A1-4", "A1-5") if (c, s) in table)
    labels = []
    if D["M2"] - D["M1"] >= ADVANTAGE and Sc["M2"] >= Sc["M1"] and nonneg >= 4 and a1("C0") - a1("M2") <= 1:
        labels.append("DECOMPOSITION_PROMISING")
    if D["M1"] > D["C0"] and D["M2"] > D["C0"] and abs(D["M2"] - D["M1"]) < ADVANTAGE:
        labels.append("NO_CLEAR_ADDITIVE_ADVANTAGE")
    if D["M2"] > D["M0"] and abs(D["M2"] - D["M0"]) < CLOSE:
        labels.append("M0_CLOSE_TO_M2_TRAINING_EXPLANATION_REMAINS")
    for c in ("C3", "G1"):
        if D[c] >= D["M2"] - CLOSE:
            labels.append("%s_CLOSE_OR_STRONGER_THAN_M2" % c)
    if D["M2"] - D["M0"] >= ADVANTAGE and a1("C0") - a1("M2") > 1:
        labels.append("COMPOSITION_REVERSAL_TRADEOFF")
    if D["M2"] <= D["C0"] and D["M1"] <= D["C0"] and D["M0"] <= D["C0"]:
        labels.append("NEW_MODELS_NOT_SUPPORTED")
    return {"D_core": D, "S_core": Sc, "core_slices_M2_ge_M1": nonneg, "A1_success_C0": a1("C0"), "A1_success_M2": a1("M2"), "labels": labels}
