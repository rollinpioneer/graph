"""Batched graph encoding for the shared Policy (method-independent engineering speed-up).

All candidate nominal graphs of one decision (and their prior-weighted twins) are encoded as ONE
disjoint-union RGCN forward instead of one forward per graph. The maths is identical to
GraphEncoder.forward / Policy.forward (checked numerically in tests); parameters are shared.
Applies to B2, A_STAT, Full, A_CAT and PRIOR_BIAS; every other method falls back to Policy.forward.
"""
import torch

from .common import DataIntegrityError
from .facts import Truth
from .graph import RELATIONS, successor, view
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
        h = norm(torch.relu(layer(h, EI, ET) if EW is None else layer(h, EI, ET, EW)))
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


class BatchedPolicy(Policy):
    """Policy with the batched encoder for FAST methods; otherwise identical to Policy."""

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
        views, index = [k], {}
        valid = [i for i in range(len(mask)) if snapshot.mask[i]]
        if edges:
            h = view(template, facts, edges)
            views.append(h)
        for i in valid:
            ki = successor(k, contracts[snapshot.candidate_ids[i]])
            index[i] = len(views)
            views.append(ki)
            if edges:
                from .graph import GraphView
                index[(i, "h")] = len(views)
                views.append(GraphView(template, ki.values, h.soft_edges))
        Z = encode_views(self.encoder, template, views)
        zk = Z[0]
        uk_all, up_all, logits = [], [], []
        diagnostics = {"differences": {}, "prior_inputs": {}, "up": {}, "delta": {}, "effect_tokens": {}, "stat_relation": {}, "successor_used": True, "method": method, "actor_episode_discount_weight": False}
        for i, cid in enumerate(snapshot.candidate_ids):
            if not snapshot.mask[i]:
                uk_all.append(zo.new_zeros(128))
                up_all.append(zo.new_zeros(128))
                logits.append(zo.sum() * 0)
                continue
            ca = self.candidate(torch.as_tensor(snapshot.candidate_features[i], device=zo.device, dtype=zo.dtype))
            zki = Z[index[i]]
            if edges:
                zh, zhi = Z[1], Z[index[(i, "h")]]
                dk = zki.float() - zk.float()
                dh = zhi.float() - zh.float()
                delta = Differences(zk, zh, dk, dh, dh - dk, (zk, zh, zki, zhi))
            else:
                dk = zki.float() - zk.float()
                delta = Differences(zk, zk, dk, dk, torch.zeros_like(dk), (zk, zk, zki, zki))
            context = torch.cat((zo, ca, zk.mean(0)))
            uk = self._phi_k(context, delta.dk)
            if method == "A_CAT":
                zk_e, zh_e, zki_e, zhi_e = delta.encodings
                C = torch.cat((zk_e, zki_e, zh_e, zhi_e), dim=-1)
                C0 = torch.cat((zk_e, zki_e, zk_e, zki_e), dim=-1)
                prior_input = self.cat_proj(C)
                anchor = self.cat_proj(C0)
                cat_ctx = torch.cat((context, uk))
                if edges:
                    up = self.prior.readout(cat_ctx, prior_input) - self.prior.readout(cat_ctx, anchor)
                    s_ = self.prior.final(up).squeeze(-1)
                    residual = self.B * torch.tanh(s_)
                else:
                    zero = prior_input.sum() * 0 + cat_ctx.sum() * 0
                    up, residual = zero.expand(128), zero
            else:
                if method == "A_STAT":
                    prior_input = delta.zh.float() - delta.zk.float() if edges else torch.zeros_like(delta.dk)
                    diagnostics["stat_relation"][cid] = prior_input
                elif method in ("B2", "PRIOR_BIAS"):
                    prior_input = torch.zeros_like(delta.dk)
                else:
                    prior_input = delta.dp if edges else torch.zeros_like(delta.dk)
                up, residual = self.prior(torch.cat((context, uk)), prior_input, self.B, False)
                if method == "PRIOR_BIAS":
                    if "bias_scores" not in diagnostics:
                        diagnostics["bias_scores"] = self.prior_bias_scores(snapshot, snapshot.mask)
                    residual = self.B * torch.tanh(self.prior_beta) * diagnostics["bias_scores"][i]
            diagnostics["differences"][cid] = delta
            if self.shadow_zero_prior:
                residual = residual * 0
            uk_all.append(uk)
            up_all.append(up)
            logits.append(self.base(uk).squeeze(-1) + residual)
            diagnostics["prior_inputs"][cid] = prior_input
            diagnostics["up"][cid] = up
            diagnostics["delta"][cid] = residual
        uk = torch.stack(uk_all)
        up = torch.stack(up_all)
        mean_k = uk[mask].mean(0)
        mean_p = up[mask].mean(0)
        value = self.v_head(torch.cat((zo, zk.mean(0), mean_k, mean_p))).squeeze(-1)
        q = self.q_head(torch.cat((uk, up, zo.expand(len(uk), -1), mean_k.expand(len(uk), -1), mean_p.expand(len(uk), -1)), dim=-1)).squeeze(-1)
        logits = torch.stack(logits).masked_fill(~mask, -torch.inf)
        return PolicyOutput(snapshot.candidate_ids, logits, mask, torch.distributions.Categorical(logits=logits), value, q, zo, diagnostics)
