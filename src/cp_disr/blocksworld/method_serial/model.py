"""C1-BW-METHOD-SERIAL-SUITE-V3 models. One wrapper around the loaded M1-GOAL final model (encoder + phi/rho) with the five method variants:

  kind  MG          frozen reference (no new module)
        EVENT       next-goal / action joint head (BASE, FACT, JOINT differ only in the loss)
        CAL         value calibration (ABS, REL differ only in the loss): V -> W = a0 V + c0, logits = beta (W(s) - W(sa))
        REC_REL     goal-anchored recurrent relational processor (true neighbours)
        REC_SELF    same processor, every existing neighbour slot reads the node's own state
        GOAL_DENSE  residual 4-head goal attention over all goal pairs
        GOAL_REL    same module and pair features, mask restricted by state / binding / contract relations

Every new output layer starts at zero, so each variant reproduces the M1-GOAL action distribution at initialisation. Labels never enter a forward pass.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import state as S
from ...graph import RELATIONS
from .heads_math import NextGoalActionHead, calibrated_logits

KINDS = ("MG", "EVENT", "CAL", "REC_REL", "REC_SELF", "GOAL_DENSE", "GOAL_REL")
NEW_SEED = 20261007
D = 128


def codes_for_state(st, problem, state):
    """(P,3) TRUE/FALSE one-hot per template proposition for a state tuple (same arithmetic as ``BwMixin._codes`` on a snapshot's facts)."""
    facts = S.fact_values(problem, tuple(state))
    return torch.tensor([[1.0, 0.0, 0.0] if facts[p] else [0.0, 1.0, 0.0] for p in st.prop_ids], device=st.prop_pos.device)


class RecurrentProcessor(nn.Module):
    """h_{t+1} = GRU([H0, m_t, goalmark], h_t); output H0 + Wo (h_T - H0) with Wo = 0 at start. REL: per-relation mean of true in-neighbours; SELF: slot present -> own state."""

    def __init__(self, self_mode):
        super().__init__()
        self.self_mode = self_mode
        self.wr = nn.ModuleList([nn.Linear(D, D, bias=False) for _ in RELATIONS])
        self.gru = nn.GRUCell(2 * D + 1, D)
        self.wo = nn.Linear(D, D, bias=False)
        nn.init.zeros_(self.wo.weight)
        self._rel = {}

    def _rel_static(self, st):
        hit = self._rel.get(id(st))
        if hit is not None and hit[0] is st:
            return hit[1]
        out = []
        for r in range(len(RELATIONS)):
            idx = (st.et == r).nonzero().reshape(-1)
            if len(idx) == 0:
                out.append(None)
                continue
            src, dst = st.ei[0, idx], st.ei[1, idx]
            deg = torch.zeros(st.N, device=src.device).index_add_(0, dst, torch.ones(len(dst), device=src.device))
            out.append((src, dst, deg.clamp(min=1.0), deg > 0))
        mark = torch.zeros((st.N, 1), device=st.prop_pos.device)
        mark[st.prop_pos] = st.prop_goal_sign
        self._rel[id(st)] = (st, (out, mark))
        return out, mark

    def forward(self, st, H0, steps):
        """Returns {t: H_t} for every requested t (t counted in iterations)."""
        rel, mark = self._rel_static(st)
        G, N, _ = H0.shape
        h = H0
        res = {}
        for t in range(1, max(steps) + 1):
            m = torch.zeros_like(H0)
            for r, ent in enumerate(rel):
                if ent is None:
                    continue
                src, dst, deg, has = ent
                hw = self.wr[r](h)
                if self.self_mode:
                    m = m + has.view(1, N, 1).to(h.dtype) * hw
                else:
                    m = m + torch.zeros_like(H0).index_add(1, dst, hw[:, src]) / deg.view(1, N, 1)
            inp = torch.cat((H0, m, mark.unsqueeze(0).expand(G, -1, -1)), dim=-1)
            h = self.gru(inp.reshape(G * N, -1), h.reshape(G * N, -1)).reshape(G, N, D)
            if t in steps:
                res[t] = H0 + self.wo(h - H0)
        return res


class GoalAttention(nn.Module):
    R_DIM = 9

    def __init__(self, relational):
        super().__init__()
        self.relational = relational
        self.q, self.k, self.v = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D)
        self.bias = nn.Linear(self.R_DIM, 4)
        self.o = nn.Linear(D, D)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)
        self._gs = {}

    def _goal_static(self, st, gprop, template):
        hit = self._gs.get(id(st))
        if hit is not None and hit[0] is st:
            return hit[1]
        dev = st.prop_pos.device
        goals = [g.fact_id for g in template.goals]
        ng = len(goals)
        parts = [g.split(":") for g in goals]
        objs = [set(p[2:]) for p in parts]
        is_on = torch.tensor([1.0 if p[1] == "On" else 0.0 for p in parts], device=dev)
        share = torch.tensor([[1.0 if (i != j and objs[i] & objs[j]) else 0.0 for j in range(ng)] for i in range(ng)], device=dev)
        add_sets = [{a.id for a in c.effects.add} for c in template.contracts]
        del_sets = [{a.id for a in c.effects.delete} for c in template.contracts]
        threat = torch.zeros((ng, ng), device=dev)
        for ad, de in zip(add_sets, del_sets):
            ai = [i for i, g in enumerate(goals) if g in ad]
            dj = [j for j, g in enumerate(goals) if g in de]
            for i in ai:
                for j in dj:
                    if i != j:
                        threat[i, j] = threat[j, i] = 1.0
        on_idx, mats = [], []
        for pi, pid in enumerate(st.prop_ids):
            pp = pid.split(":")
            if pp[1] == "On":
                a, b = pp[2], pp[3]
                m = torch.zeros((ng, ng), device=dev)
                for i in range(ng):
                    for j in range(ng):
                        if i != j and ((a in objs[i] and b in objs[j]) or (a in objs[j] and b in objs[i])):
                            m[i, j] = 1.0
                on_idx.append(pi)
                mats.append(m.reshape(-1))
        M = torch.stack(mats) if mats else torch.zeros((0, ng * ng), device=dev)
        out = SimpleNamespace(ng=ng, is_on=is_on, share=share, threat=threat, on_idx=torch.tensor(on_idx, device=dev, dtype=torch.long), M=M, eye=torch.eye(ng, device=dev))
        self._gs[id(st)] = (st, out)
        return out

    def pair_features(self, gs, codes, gprop):
        G = codes.shape[0]
        ng = gs.ng
        truth = codes[:, gprop, 0]                                                     # (G, ng)
        if len(gs.on_idx):
            conn = ((codes[:, gs.on_idx, 0] @ gs.M) > 0).reshape(G, ng, ng).to(codes.dtype)
        else:
            conn = codes.new_zeros((G, ng, ng))
        oi, oj = gs.is_on[:, None], gs.is_on[None, :]
        both_on, both_tab = oi * oj, (1 - oi) * (1 - oj)
        one_on = 1.0 - both_on - both_tab
        e = lambda t: t.unsqueeze(0).expand(G, -1, -1)
        feats = torch.stack((e(both_on), e(one_on), e(both_tab), truth.unsqueeze(2).expand(G, ng, ng), truth.unsqueeze(1).expand(G, ng, ng), e(gs.share), conn, e(gs.threat), e(gs.eye)), dim=-1)
        allowed = (gs.eye.unsqueeze(0) + gs.share.unsqueeze(0) + conn + gs.threat.unsqueeze(0)) > 0
        return feats, allowed

    def forward(self, z, feats, allowed):
        G, ng, _ = z.shape
        q, k, v = (f(z).reshape(G, ng, 4, 32) for f in (self.q, self.k, self.v))
        e = torch.einsum("gihd,gjhd->ghij", q, k) / math.sqrt(32) + self.bias(feats).permute(0, 3, 1, 2)
        if self.relational:
            e = e.masked_fill(~allowed.unsqueeze(1), -torch.inf)
        a = torch.softmax(e, dim=-1)
        return self.o(torch.einsum("ghij,gjhd->gihd", a, v).reshape(G, ng, D))


class SerialModel(nn.Module):
    def __init__(self, mg, kind, seed=NEW_SEED):
        super().__init__()
        assert kind in KINDS
        self.kind = kind
        self.mg = mg
        self.rec = self.event = self.attn = None
        self.log_beta = None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if kind == "EVENT":
                self.event = NextGoalActionHead(D, seed)
            elif kind in ("REC_REL", "REC_SELF"):
                self.rec = RecurrentProcessor(kind == "REC_SELF")
            elif kind in ("GOAL_DENSE", "GOAL_REL"):
                self.attn = GoalAttention(kind == "GOAL_REL")
        if kind == "CAL":
            self.log_beta = nn.Parameter(torch.zeros(()))
            self.register_buffer("a0", torch.ones(()))
            self.register_buffer("c0", torch.zeros(()))
        dev = next(mg.parameters()).device
        self.to(dev)

    # ---- parameter groups
    def encoder_params(self):
        return self.mg.trainable_encoder_params()

    def existing_head_params(self):
        return list(self.mg.heads.parameters())

    def new_params(self):
        mods = [m for m in (self.rec, self.event, self.attn) if m is not None]
        return [p for m in mods for p in m.parameters()]

    def eta_params(self):
        return [self.log_beta] if self.log_beta is not None else []

    def set_calibration(self, a0, c0):
        self.a0.fill_(float(a0))
        self.c0.fill_(float(c0))
        with torch.no_grad():
            self.log_beta.fill_(-math.log(float(a0)))

    # ---- core
    def layout(self, snaps, extra=None):
        mg = self.mg
        base = mg.base
        st, gprop = mg.goal_free_static(snaps[0].template)
        n = len(snaps[0].candidate_ids)
        codes_list, info, off = [], [], 0
        for i, sn in enumerate(snaps):
            legal = [j for j, m in enumerate(sn.mask) if m]
            c0 = base._codes(st, sn.facts.values)
            rows = [st.action_row[sn.candidate_ids[j]] for j in legal]
            parts = [c0.unsqueeze(0), base._successor_codes(st, c0, rows)]
            m_extra = 0
            if extra is not None and extra[i] is not None and len(extra[i]):
                parts.append(extra[i])
                m_extra = len(extra[i])
            codes_list.append(torch.cat(parts, 0))
            info.append((off, legal, m_extra))
            off += codes_list[-1].shape[0]
        return st, gprop, torch.cat(codes_list, 0), info, n

    def zv(self, st, gprop, codes, steps=(None,), template=None):
        """{step: (z (G,ng,128), v (G,), truth (G,ng))}; step is None for non-recurrent kinds."""
        mg = self.mg
        _ro, H = mg.base._encode(st, codes)
        if self.rec is not None:
            Hs = self.rec(st, H, tuple(steps))
        else:
            Hs = {None: H}
        out = {}
        feats = attn_feats = None
        for t in steps:
            Ht = Hs[t]
            x = self._features(st, gprop, codes, Ht)
            z = mg.heads.phi(x)
            if self.attn is not None:
                gs = self.attn._goal_static(st, gprop, template)
                if attn_feats is None:
                    attn_feats = self.attn.pair_features(gs, codes, gprop)
                z = z + self.attn(z, attn_feats[0], attn_feats[1])
            v = mg.heads.rho(z.sum(1)).squeeze(-1)
            out[t] = (z, v, codes[:, gprop, 0])
        return out

    def _features(self, st, gprop, codes, H):
        """x(F,g) (G, ng, 386) from a node-state tensor H (identical arithmetic to ``GoalProgressModel.features``)."""
        act = H[:, st.action_pos].mean(1)
        prop = H[:, st.prop_pos].mean(1)
        hg = H[:, st.goal_pos]
        truth = codes[:, gprop, 0].unsqueeze(-1)
        ng = hg.shape[1]
        sign = st.goal_sign.unsqueeze(0).expand(codes.shape[0], -1, -1)
        return torch.cat((hg, act.unsqueeze(1).expand(-1, ng, -1), prop.unsqueeze(1).expand(-1, ng, -1), truth, sign), dim=-1)

    def forward_group(self, snaps, extra=None, steps=(None,)):
        st, gprop, codes, info, n = self.layout(snaps, extra)
        zv = self.zv(st, gprop, codes, steps, snaps[0].template)
        dev = codes.device
        Sn = len(snaps)
        o_idx = torch.tensor([o for o, _, _ in info], device=dev)
        pos, src, ex_src, ex_cnt = [], [], [], []
        for si, (o, lg, mx) in enumerate(info):
            pos += [si * n + j for j in lg]
            src += [o + 1 + k for k in range(len(lg))]
            ex_src.append((o + 1 + len(lg), mx))
        pos_t, src_t = torch.tensor(pos, device=dev, dtype=torch.long), torch.tensor(src, device=dev, dtype=torch.long)
        legal = torch.zeros(Sn * n, dtype=torch.bool, device=dev)
        legal[pos_t] = True
        legal = legal.reshape(Sn, n)
        res = {}
        for t, (z, v, truth) in zv.items():
            ng = z.shape[1]
            v0 = v[o_idx]
            vn = v.new_zeros(Sn * n).index_copy(0, pos_t, v[src_t]).reshape(Sn, n)
            vx = [v[a:a + m] if m else v.new_zeros(0) for a, m in ex_src]
            zn = tn = z0 = t0 = None
            if self.kind == "EVENT":
                zn = z.new_zeros((Sn * n, ng, D)).index_copy(0, pos_t, z[src_t]).reshape(Sn, n, ng, D)
                tn = truth.new_zeros((Sn * n, ng)).index_copy(0, pos_t, truth[src_t]).reshape(Sn, n, ng)
                z0, t0 = z[o_idx], truth[o_idx]
            res[t] = {"legal": legal, "v0": v0, "vn": vn, "vx": vx, "logits": (v0[:, None] - vn).masked_fill(~legal, -torch.inf), "z0": z0, "zn": zn, "t0": t0, "tn": tn, "info": info}
        return res

    # ---- evaluation interface: logits of one snapshot
    @torch.no_grad()
    def eval_logits(self, snap, steps=None):
        key = steps
        out = self.forward_group([snap], steps=(key,))[key]
        legal = out["legal"]
        base = out["logits"]
        if self.kind == "EVENT":
            t0 = out["t0"]
            pending = t0 < 0.5
            gv = torch.ones_like(pending)
            jo = self.event(out["z0"], out["zn"], t0, out["tn"], base.masked_fill(~legal, 0.0), gv, pending, legal)
            return jo.log_action[0]
        if self.kind == "CAL":
            return calibrated_logits(out["v0"], out["vn"], legal, float(self.a0), float(self.c0), self.log_beta)[0]
        return base[0]

    @torch.no_grad()
    def eval_event(self, snap):
        """(JointOutput, pending, legal) of one snapshot for the event diagnostics."""
        out = self.forward_group([snap], steps=(None,))[None]
        legal, t0 = out["legal"], out["t0"]
        pending = t0 < 0.5
        jo = self.event(out["z0"], out["zn"], t0, out["tn"], out["logits"].masked_fill(~legal, 0.0), torch.ones_like(pending), pending, legal)
        return jo, pending, legal

    @torch.no_grad()
    def state_values(self, template, problem, states, steps=None, chunk=192):
        """V (or W for CAL) of arbitrary state tuples of one template (lookahead leaves)."""
        st, gprop = self.mg.goal_free_static(template)
        out = []
        for a in range(0, len(states), chunk):
            codes = torch.stack([codes_for_state(st, problem, s) for s in states[a:a + chunk]])
            v = self.zv(st, gprop, codes, (steps,), template)[steps][1]
            out.append(v * self.a0 + self.c0 if self.kind == "CAL" else v)
        return torch.cat(out) if out else torch.zeros(0)

    def mask_density(self, snap):
        """Fraction of non-self goal pairs the attention may read in this state (DENSE reads all pairs)."""
        if not self.attn.relational:
            return 1.0
        st, gprop = self.mg.goal_free_static(snap.template)
        gs = self.attn._goal_static(st, gprop, snap.template)
        codes = self.mg.base._codes(st, snap.facts.values).unsqueeze(0)
        _f, allowed = self.attn.pair_features(gs, codes, gprop)
        ng = gs.ng
        off = ~torch.eye(ng, dtype=torch.bool, device=allowed.device)
        return float(allowed[0][off].float().mean()) if ng > 1 else 1.0


class SerialPolicy:
    """Adapter for the frozen episode runner (no hidden state, no time input)."""

    def __init__(self, model, steps=None):
        self.model, self.steps = model, steps

    def initial_hidden(self):
        return torch.zeros(1)

    @torch.no_grad()
    def __call__(self, snap, hidden):
        logits = self.model.eval_logits(snap, self.steps)
        return SimpleNamespace(logits=logits, distribution=torch.distributions.Categorical(logits=logits), hidden=hidden)
