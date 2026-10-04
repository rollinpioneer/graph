"""Batched graph encoding for the shared Policy (method-independent engineering speed-up).

All candidate nominal graphs of one decision (and their prior-weighted twins) are encoded as ONE
disjoint-union RGCN forward instead of one forward per graph. The maths is identical to
GraphEncoder.forward / Policy.forward (checked numerically in tests); parameters are shared.
Applies to B2, A_STAT, Full, A_CAT and PRIOR_BIAS; every other method falls back to Policy.forward.
"""
import math

import torch

from .common import DataIntegrityError
from .facts import Truth
from .graph import RELATIONS, GraphView, successor, view
from .neural import Differences, Policy, PolicyOutput, canonical_method

FAST = ("B2", "A_STAT", "Full", "A_CAT", "PRIOR_BIAS")
_CODE = {Truth.TRUE: 0, Truth.FALSE: 1, Truth.UNKNOWN: 2}


class _Static:
    """Parameter-independent tensors of one GraphTemplate."""

    def __init__(self, template, feats):
        nodes = template.nodes
        self.N = len(nodes)
        ids = {n.id: i for i, n in enumerate(nodes)}
        self.is_action = torch.tensor([n.kind == "ACTION" for n in nodes])
        self.kind_idx = torch.tensor([0 if n.kind == "ACTION" else 1 for n in nodes])
        self.schema_idx = torch.tensor([feats.actions[n.schema] if n.kind == "ACTION" else feats.predicates[n.schema] for n in nodes])
        width = max([len(n.argument_types) for n in nodes] + [1])
        arg = torch.full((self.N, width), -1, dtype=torch.long)
        for i, n in enumerate(nodes):
            for role, t in enumerate(n.argument_types):
                arg[i, role] = feats.types[t]
        self.arg_idx = arg
        self.arg_scale = torch.tensor([1.0 / (r + 1) for r in range(width)])
        self.prop_idx = torch.tensor([i for i, n in enumerate(nodes) if n.kind == "PROPOSITION"], dtype=torch.long)
        self.prop_ids = [nodes[i].id for i in self.prop_idx.tolist()]
        self.action_idx = torch.tensor([i for i, n in enumerate(nodes) if n.kind == "ACTION"], dtype=torch.long)
        signs = {g.fact_id: g.sign for g in template.goals}
        self.sign = torch.tensor([float(signs.get(i, 0)) for i in self.prop_ids])
        self.goal_idx = torch.tensor([ids[g.fact_id] for g in template.goals], dtype=torch.long)
        self.goal_sign = torch.tensor([float(g.sign) for g in template.goals])
        base = sorted(template.edges)
        self.base_set = {tuple(e) for e in base}
        self.base_ei = torch.tensor([[ids[a] for a, b, r in base], [ids[b] for a, b, r in base]], dtype=torch.long).reshape(2, -1)
        self.base_et = torch.tensor([RELATIONS.index(r) for a, b, r in base], dtype=torch.long)
        self.ids = ids


_CACHE = {}


def _static(template, feats):
    hit = _CACHE.get(id(template))
    if hit is None or hit[0] is not template:
        hit = (template, _Static(template, feats))
        _CACHE[id(template)] = hit
        if len(_CACHE) > 20000:
            _CACHE.clear()
            _CACHE[id(template)] = hit
    return hit[1]


def encode_views(encoder, template, views):
    """views: list of GraphView over `template`. Returns (G, 1 + n_goals, 128) readout rows."""
    feats, dev = encoder.features, encoder.features.kind.weight.device
    s = _static(template, feats)
    G, N = len(views), s.N
    S = feats.kind.weight[s.kind_idx.to(dev)]
    act = s.is_action.to(dev).unsqueeze(-1)
    S = S + torch.where(act, feats.action.weight[s.schema_idx.to(dev) % feats.action.num_embeddings], feats.predicate.weight[s.schema_idx.to(dev) % feats.predicate.num_embeddings])
    ai = s.arg_idx.to(dev)
    arg_rows = feats.arg.weight[ai.clamp(min=0)] * (ai >= 0).unsqueeze(-1).to(S.dtype) * s.arg_scale.to(dev).view(1, -1, 1)
    S = S + arg_rows.sum(1)
    codes = torch.zeros(G, len(s.prop_ids), 4)
    for g, v in enumerate(views):
        vals = dict(v.values)
        for j, pid in enumerate(s.prop_ids):
            codes[g, j, _CODE[vals[pid]]] = 1.0
    codes[:, :, 3] = s.sign.unsqueeze(0)
    fact = feats.fact(codes.to(dev, S.dtype))
    H0 = S.unsqueeze(0).repeat(G, 1, 1)
    H0[:, s.prop_idx.to(dev)] = H0[:, s.prop_idx.to(dev)] + fact
    # edges: template topology for every graph, plus per-graph soft edges (with optional weights)
    any_w = False
    ei = [s.base_ei + g * N for g in range(G)]
    et = [s.base_et for _ in range(G)]
    ew = [torch.ones(s.base_et.numel()) for _ in range(G)]
    for g, v in enumerate(views):
        extra = [e for e in v.soft_edges if tuple(e[:3]) not in s.base_set]
        if extra:
            ei.append(torch.tensor([[s.ids[e[0]] + g * N for e in extra], [s.ids[e[1]] + g * N for e in extra]], dtype=torch.long))
            et.append(torch.tensor([RELATIONS.index(e[2]) for e in extra], dtype=torch.long))
            w = [float(e[3]) if len(e) > 3 else 1.0 for e in extra]
            any_w = any_w or any(len(e) > 3 for e in extra)
            ew.append(torch.tensor(w))
    EI = torch.cat(ei, 1).to(dev)
    ET = torch.cat(et).to(dev)
    EW = torch.cat(ew).to(dev, S.dtype) if any_w else None
    h = H0.reshape(G * N, -1)
    for layer, norm in zip(encoder.layers, encoder.norms):
        h = norm(torch.relu(layer.forward_sparse(h, EI, ET, EW)))
    h = h.reshape(G, N, -1)
    ro = encoder.readout
    gi = s.goal_idx.to(dev)
    if len(gi):
        goals = ro.goal(torch.cat((h[:, gi], s.goal_sign.to(dev, h.dtype).view(1, -1, 1).expand(G, -1, 1)), -1))
        goal_mean = goals.mean(1)
    else:
        goals = h.new_zeros((G, 0, 128))
        goal_mean = h.new_zeros((G, 128))
    pool_a = h[:, s.action_idx.to(dev)].mean(1) if len(s.action_idx) else h.new_zeros((G, 128))
    pool_p = h[:, s.prop_idx.to(dev)].mean(1) if len(s.prop_idx) else h.new_zeros((G, 128))
    glob = ro.global_readout(torch.cat((pool_a, pool_p, goal_mean), -1))
    return torch.cat((glob.unsqueeze(1), goals), 1)




def _readout(module, context, rows):
    """CandidateReadout.forward for K candidates at once: context (K, C), rows (K, R, 128) -> (K, 128)."""
    score = (module.key(rows) * module.query(context).unsqueeze(1)).sum(-1) / math.sqrt(128)
    attention = torch.softmax(score, dim=-1)
    pooled = (attention.unsqueeze(-1) * module.value(rows)).sum(1)
    return module.net(torch.cat((context, pooled), -1))


def _anchored(prior, context, dp, B):
    """AnchoredPrior.forward vectorised over candidates; a candidate whose dp is exactly zero gets exactly zero."""
    nonzero = (dp != 0).flatten(1).any(1)
    up = _readout(prior.readout, context, dp) - _readout(prior.readout, context, torch.zeros_like(dp))
    s = prior.final(up).squeeze(-1)
    keep = nonzero.to(up.dtype)
    return up * keep.unsqueeze(-1), B * torch.tanh(s) * keep


class BatchedPolicy(Policy):
    """Policy with batched graph encoding and vectorised candidate heads for FAST methods.

    Mathematically identical to Policy.forward (checked in tests); every other method falls back to it.
    diagnostics keep 'delta' and 'up' per candidate; the heavy per-candidate 'differences' are not built.
    """

    def forward(self, snapshot, hidden=None):
        method = canonical_method(self.method)
        if method not in FAST:
            return super().forward(snapshot, hidden)
        hidden = self.initial_hidden() if hidden is None else hidden
        zo = self.advance_hidden(snapshot.base_input, hidden)
        mask = torch.tensor(snapshot.mask, device=zo.device, dtype=torch.bool)
        if not mask.any():
            empty = zo.new_zeros(len(mask))
            return PolicyOutput(snapshot.candidate_ids, empty, mask, None, zo.sum() * 0, empty, zo, {"successor_used": False, "method": method}, "NO_SAFE_CANDIDATES")
        template = snapshot.template
        facts = snapshot.facts.values
        contracts = {c.id: c for c in template.contracts}
        edges = () if method in ("B2", "PRIOR_BIAS") else snapshot.prior_edges
        k = view(template, facts)
        views = [k]
        valid = [i for i in range(len(mask)) if snapshot.mask[i]]
        if edges:
            h = view(template, facts, edges)
            views.append(h)
        kis = [successor(k, contracts[snapshot.candidate_ids[i]]) for i in valid]
        views.extend(kis)
        n_valid = len(valid)
        if edges:
            views.extend(GraphView(template, ki.values, h.soft_edges) for ki in kis)
        Z = encode_views(self.encoder, template, views)
        K = n_valid
        zk = Z[0]
        first_ki = 2 if edges else 1
        Zki = Z[first_ki:first_ki + K]
        dk = Zki.float() - zk.float()
        feats = torch.as_tensor([snapshot.candidate_features[i] for i in valid], device=zo.device, dtype=zo.dtype)
        ca = self.candidate(feats)
        zbar = zk.mean(0)
        context = torch.cat((zo.expand(K, -1), ca, zbar.expand(K, -1)), -1)
        uk = _readout(self.contract, context, dk)
        cat_ctx = torch.cat((context, uk), -1)
        zero_res = zo.new_zeros(K)
        if method in ("B2", "PRIOR_BIAS"):
            up = zo.new_zeros(K, 128)
            residual = zero_res
            if method == "PRIOR_BIAS":
                scores = torch.as_tensor(self.prior_bias_scores(snapshot, snapshot.mask), device=zo.device, dtype=zo.dtype)[valid]
                residual = self.B * torch.tanh(self.prior_beta) * scores
        elif not edges:
            up, residual = zo.new_zeros(K, 128), zero_res
        else:
            zh = Z[1]
            Zhi = Z[first_ki + K:first_ki + 2 * K]
            if method == "A_STAT":
                dp = (zh.float() - zk.float()).unsqueeze(0).expand(K, -1, -1)
                up, residual = _anchored(self.prior, cat_ctx, dp, self.B)
            elif method == "Full":
                dp = (Zhi.float() - zh.float()) - dk
                up, residual = _anchored(self.prior, cat_ctx, dp, self.B)
            else:  # A_CAT
                zkk, zhh = zk.unsqueeze(0).expand(K, -1, -1), zh.unsqueeze(0).expand(K, -1, -1)
                C = torch.cat((zkk, Zki, zhh, Zhi), -1)
                C0 = torch.cat((zkk, Zki, zkk, Zki), -1)
                pi, an = self.cat_proj(C), self.cat_proj(C0)
                up = _readout(self.prior.readout, cat_ctx, pi) - _readout(self.prior.readout, cat_ctx, an)
                residual = self.B * torch.tanh(self.prior.final(up).squeeze(-1))
        if self.shadow_zero_prior:
            residual = residual * 0
        base = self.base(uk).squeeze(-1)
        vidx = torch.as_tensor(valid, device=zo.device)
        logits = zo.new_zeros(len(mask)).index_put((vidx,), base + residual)
        uk_all = zo.new_zeros(len(mask), 128).index_put((vidx,), uk)
        up_all = zo.new_zeros(len(mask), 128).index_put((vidx,), up)
        diagnostics = {"differences": {}, "prior_inputs": {}, "up": {}, "delta": {}, "effect_tokens": {}, "stat_relation": {}, "successor_used": True, "method": method, "actor_episode_discount_weight": False}
        for j, i in enumerate(valid):
            cid = snapshot.candidate_ids[i]
            diagnostics["delta"][cid] = residual[j]
            diagnostics["up"][cid] = up[j]
        mean_k = uk_all[mask].mean(0)
        mean_p = up_all[mask].mean(0)
        value = self.v_head(torch.cat((zo, zbar, mean_k, mean_p))).squeeze(-1)
        q = self.q_head(torch.cat((uk_all, up_all, zo.expand(len(mask), -1), mean_k.expand(len(mask), -1), mean_p.expand(len(mask), -1)), dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~mask, -torch.inf)
        return PolicyOutput(snapshot.candidate_ids, logits, mask, torch.distributions.Categorical(logits=logits), value, q, zo, diagnostics)
