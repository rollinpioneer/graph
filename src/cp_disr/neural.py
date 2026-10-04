"""Production shared RGCN, Set Transformer B0, B1-K, A_CAT and remaining variants.

Torch/PyG are deliberately optional at import of the package/logic CLI.
Method profile: CP-DISR v2.1.1.
"""
from dataclasses import dataclass
import hashlib
import math
import torch
from torch import nn
from torch_geometric.nn import RGCNConv
from .graph import RELATIONS, four_views, view
from .facts import Truth
from .common import DataIntegrityError

METHODS = ("B0", "B1", "B1-K", "B2", "Full", "A_DD", "A_Q", "A_B", "A_CAT", "B1-H", "B1-K+E", "A_STAT", "PRIOR_BIAS")
CANONICAL = {"B1-K": "B1", "B1": "B1"}


class WeightedRGCNConv(RGCNConv):
    """RGCNConv whose messages may carry a per-edge multiplier (mean aggregation unchanged).

    With edge_weight None, or all-ones weights, the output equals RGCNConv exactly. The weight only
    scales a message inside its own relation type; it adds no relation, node, truth value or mask.
    """

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        if edge_weight is None:
            return super().forward(x, edge_index, edge_type)
        n = x.size(0)
        weight = (self.comp @ self.weight.view(self.num_bases, -1)).view(self.num_relations, self.in_channels_l, self.out_channels) if self.num_bases is not None else self.weight
        out = x.new_zeros(n, self.out_channels)
        for i in range(self.num_relations):
            m = edge_type == i
            if not bool(m.any()):
                continue
            src, dst = edge_index[0][m], edge_index[1][m]
            msg = x[src] * edge_weight[m].unsqueeze(-1)
            agg = x.new_zeros(n, x.size(1)).index_add(0, dst, msg)
            cnt = x.new_zeros(n).index_add(0, dst, torch.ones_like(dst, dtype=x.dtype)).clamp(min=1).unsqueeze(-1)
            out = out + (agg / cnt) @ weight[i]
        if self.root is not None:
            out = out + x @ self.root
        if self.bias is not None:
            out = out + self.bias
        return out


def canonical_method(method):
    if method not in METHODS:
        raise ValueError(method)
    return CANONICAL.get(method, method)


def mlp(inp, out=128):
    return nn.Sequential(nn.Linear(inp, 128), nn.ReLU(), nn.Linear(128, out), nn.ReLU())


class NodeFeatures(nn.Module):
    def __init__(self, action_schemas, predicate_schemas, argument_types):
        super().__init__()
        self.actions = {s: i for i, s in enumerate(sorted(action_schemas))}
        self.predicates = {s: i for i, s in enumerate(sorted(predicate_schemas))}
        self.types = {s: i for i, s in enumerate(sorted(argument_types))}
        self.kind = nn.Embedding(2, 128)
        self.action = nn.Embedding(max(1, len(self.actions)), 128)
        self.predicate = nn.Embedding(max(1, len(self.predicates)), 128)
        self.arg = nn.Embedding(max(1, len(self.types)), 128)
        self.fact = nn.Linear(4, 128)

    def forward(self, graph):
        device = self.kind.weight.device
        facts = dict(graph.values)
        signs = {g.fact_id: g.sign for g in graph.template.goals}
        rows = []
        for n in graph.template.nodes:
            action = n.kind == "ACTION"
            row = self.kind.weight[0 if action else 1]
            row = row + (self.action.weight[self.actions[n.schema]] if action else self.predicate.weight[self.predicates[n.schema]])
            for role, t in enumerate(n.argument_types):
                row = row + self.arg.weight[self.types[t]] / (role + 1)
            if not action:
                code = [float(facts[n.id] == t) for t in (Truth.TRUE, Truth.FALSE, Truth.UNKNOWN)] + [float(signs.get(n.id, 0))]
                row = row + self.fact(torch.tensor(code, device=device, dtype=row.dtype))
            rows.append(row)
        return torch.stack(rows)


class GoalReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.goal = mlp(129)
        self.global_readout = mlp(384)

    def forward(self, h, template):
        ids = template.node_ids
        goals = torch.stack([self.goal(torch.cat((h[ids.index(g.fact_id)], h.new_tensor([g.sign])))) for g in template.goals]) if template.goals else h.new_zeros((0, 128))

        def pool(kind):
            idx = [i for i, n in enumerate(template.nodes) if n.kind == kind]
            return h[idx].mean(0) if idx else h.new_zeros(128)

        global_row = self.global_readout(torch.cat((pool("ACTION"), pool("PROPOSITION"), goals.mean(0) if len(goals) else h.new_zeros(128))))
        return torch.cat((global_row.unsqueeze(0), goals), 0)


class GraphEncoder(nn.Module):
    def __init__(self, actions, predicates, types):
        super().__init__()
        self.features = NodeFeatures(actions, predicates, types)
        self.layers = nn.ModuleList([WeightedRGCNConv(128, 128, len(RELATIONS), num_bases=4, aggr="mean", root_weight=True) for _ in range(4)])
        self.norms = nn.ModuleList([nn.LayerNorm(128) for _ in range(4)])
        self.readout = GoalReadout()
        self.forward_calls = 0

    def forward(self, graph):
        self.forward_calls += 1
        h = self.features(graph)
        ids = {v: i for i, v in enumerate(graph.template.node_ids)}
        edges = graph.edges
        ei = torch.tensor([[ids[a] for a, b, r in edges], [ids[b] for a, b, r in edges]], device=h.device, dtype=torch.long).reshape(2, -1)
        et = torch.tensor([RELATIONS.index(r) for a, b, r in edges], device=h.device, dtype=torch.long)
        weights = getattr(graph, "weights", None)
        ew = torch.tensor([weights.get((a, b, r), 1.0) for a, b, r in edges], device=h.device, dtype=h.dtype) if weights else None
        for layer, norm in zip(self.layers, self.norms):
            h = norm(torch.relu(layer(h, ei, et) if ew is None else layer(h, ei, et, ew)))
        return self.readout(h, graph.template)


class MAB(nn.Module):
    def __init__(self, dim=128, heads=4, ffn=256):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ffn), nn.ReLU(), nn.Linear(ffn, dim))
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, q, kv, key_padding_mask=None):
        h, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(q + h)
        return self.ln2(x + self.ff(x))


class SAB(nn.Module):
    def __init__(self, dim=128, heads=4, ffn=256):
        super().__init__()
        self.mab = MAB(dim, heads, ffn)

    def forward(self, x, key_padding_mask=None):
        return self.mab(x, x, key_padding_mask)


class SetTransformerEncoder(nn.Module):
    """B0: two SAB layers, padding mask only, no graph adjacency or nominal patch."""

    FIELD_TYPES = ("PRE_POS", "PRE_NEG", "ADD", "DEL", "UNKNOWN")

    def __init__(self, actions, predicates, types):
        super().__init__()
        self.features = NodeFeatures(actions, predicates, types)
        self.field = nn.Embedding(len(self.FIELD_TYPES), 128)
        self.field_proj = nn.Linear(384, 128)
        self.sab1 = SAB()
        self.sab2 = SAB()
        self.cross = MAB()
        self.forward_calls = 0

    def tokens(self, graph):
        h = self.features(graph)
        ids = {v: i for i, v in enumerate(graph.template.node_ids)}
        tokens = [h]
        rows = []
        for c in graph.template.contracts:
            groups = (
                (c.pre_pos, "PRE_POS"),
                (c.pre_neg, "PRE_NEG"),
                (c.effects.add, "ADD"),
                (c.effects.delete, "DEL"),
                (c.effects.unknown, "UNKNOWN"),
            )
            a_h = h[ids[c.id]]
            for atoms, name in groups:
                rel = self.field.weight[self.FIELD_TYPES.index(name)]
                for atom in atoms:
                    p_h = h[ids[atom.id]] if atom.id in ids else a_h.new_zeros(128)
                    rows.append(self.field_proj(torch.cat((a_h, p_h, rel))))
        if rows:
            tokens.append(torch.stack(rows))
        return torch.cat(tokens, 0)

    def encode(self, tokens):
        self.forward_calls += 1
        x = tokens.unsqueeze(0)
        x = self.sab1(x)
        x = self.sab2(x)
        return x.squeeze(0)

    def candidate_read(self, query, tokens):
        q = query.unsqueeze(0).unsqueeze(0)
        kv = tokens.unsqueeze(0)
        return self.cross(q, kv).squeeze(0).squeeze(0)


@dataclass
class Differences:
    zk: torch.Tensor
    zh: torch.Tensor
    dk: torch.Tensor
    dh: torch.Tensor
    dp: torch.Tensor
    encodings: tuple


def differences(encoder, graphs):
    k, h, ki, hi = graphs
    zk = encoder(k)
    zki = zk if ki.values == k.values else encoder(ki)
    if h is k:
        dk = zki.float() - zk.float()
        zero = torch.zeros_like(dk)
        return Differences(zk, zk, dk, dk, zero, (zk, zk, zki, zki))
    zh = encoder(h)
    zhi = zh if hi.values == h.values else encoder(hi)
    dk = zki.float() - zk.float()
    dh = zhi.float() - zh.float()
    return Differences(zk, zh, dk, dh, dh - dk, (zk, zh, zki, zhi))


class CandidateReadout(nn.Module):
    def __init__(self, context_dim):
        super().__init__()
        self.query = nn.Linear(context_dim, 128)
        self.key = nn.Linear(128, 128)
        self.value = nn.Linear(128, 128)
        self.net = mlp(context_dim + 128)

    def forward(self, context, rows, mask=None):
        mask = torch.ones(len(rows), dtype=torch.bool, device=rows.device) if mask is None else mask
        if not mask.any():
            raise DataIntegrityError("Goal/global readout entirely masked")
        score = (self.key(rows) * self.query(context)).sum(-1) / math.sqrt(128)
        attention = torch.softmax(score.masked_fill(~mask, -torch.inf), dim=-1)
        return self.net(torch.cat((context, (attention[:, None] * self.value(rows)).sum(0))))


class AnchoredPrior(nn.Module):
    def __init__(self, context_dim=512):
        super().__init__()
        self.readout = CandidateReadout(context_dim)
        self.final = nn.Linear(128, 1, bias=False)

    def forward(self, context, dp, B=0.5, unbounded=False, mask=None):
        if torch.count_nonzero(dp).item() == 0:
            zero = dp.sum() * 0 + context.sum() * 0
            return zero.expand(128), zero
        up = self.readout(context, dp, mask) - self.readout(context, torch.zeros_like(dp), mask)
        s = self.final(up).squeeze(-1)
        return up, B * s if unbounded else B * torch.tanh(s)


def _hash_bucket(value, buckets=4096):
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


class EffectTokenReadout(nn.Module):
    """Candidate-local, permutation-invariant contract/effect encoder.

    This path reads only the grounded contract: preconditions, nominal effects,
    conditional guards, ordered typed arguments, and grounded bindings. It
    deliberately does not call ``nominal_overlay`` or construct a successor.
    Stable hash buckets keep the vocabulary independent of candidate order.
    """

    ROLES = (
        "ACTION", "ARG", "BIND", "PRE_POS", "PRE_NEG", "ADD", "DEL",
        "UNKNOWN", "GUARD_POS", "GUARD_NEG", "COND_ADD", "COND_DEL",
        "COND_UNKNOWN",
    )

    def __init__(self, context_dim=384, dim=128, buckets=4096):
        super().__init__()
        self.buckets = buckets
        self.role = nn.Embedding(len(self.ROLES), dim)
        self.symbol = nn.Embedding(buckets, dim)
        self.arguments = nn.Embedding(buckets, dim)
        self.types = nn.Embedding(buckets, dim)
        self.position = nn.Embedding(64, dim)
        self.proj = nn.Sequential(nn.Linear(dim * 5, dim), nn.ReLU())
        self.query = nn.Linear(context_dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)

    def _token(self, role, contract, symbol, arguments, types, position):
        role_row = self.role.weight[self.ROLES.index(role)]
        symbol_row = self.symbol.weight[_hash_bucket(f"{contract.id}|{symbol}", self.buckets)]
        argument_row = self.arguments.weight[_hash_bucket("|".join(map(str, arguments)) or "<none>", self.buckets)]
        type_row = self.types.weight[_hash_bucket("|".join(map(str, types)) or "<none>", self.buckets)]
        position_row = self.position.weight[position % self.position.num_embeddings]
        return self.proj(torch.cat((role_row, symbol_row, argument_row, type_row, position_row)))

    def tokens(self, contract):
        tokens = []
        argument_types = {a.name: a.type for a in contract.arguments}

        def add(role, symbol, arguments=(), types=(), position=0):
            tokens.append(self._token(role, contract, symbol, arguments, types, position))

        add("ACTION", contract.name, contract.bound_arguments,
            tuple(a.type for a in contract.arguments))
        for index, argument in enumerate(contract.arguments):
            add("ARG", argument.name, (argument.name,), (argument.type,), index)
        for index, (argument, bound) in enumerate(zip(contract.arguments, contract.bound_arguments)):
            add("BIND", argument.name, (bound,), (argument.type,), 16 + index)

        def add_atoms(atoms, role, offset):
            for index, atom in enumerate(atoms):
                add(role, atom.predicate, atom.arguments,
                    tuple(argument_types.get(value, "<ground>") for value in atom.arguments),
                    offset + index)

        add_atoms(contract.pre_pos, "PRE_POS", 24)
        add_atoms(contract.pre_neg, "PRE_NEG", 32)
        add_atoms(contract.effects.add, "ADD", 40)
        add_atoms(contract.effects.delete, "DEL", 48)
        add_atoms(contract.effects.unknown, "UNKNOWN", 56)
        for conditional_index, conditional in enumerate(contract.conditional):
            offset = 64 + conditional_index * 4
            add_atoms(conditional.positive, "GUARD_POS", offset)
            add_atoms(conditional.negative, "GUARD_NEG", offset + 1)
            add_atoms(conditional.effects.add, "COND_ADD", offset + 2)
            add_atoms(conditional.effects.delete, "COND_DEL", offset + 3)
            add_atoms(conditional.effects.unknown, "COND_UNKNOWN", offset + 4)
        return torch.stack(tokens)

    def forward(self, contract, context):
        tokens = self.tokens(contract).to(device=context.device, dtype=context.dtype)
        query = self.query(context)
        score = (self.key(tokens) * query).sum(-1) / math.sqrt(tokens.shape[-1])
        weights = torch.softmax(score, dim=-1)
        return (weights[:, None] * self.value(tokens)).sum(0), int(tokens.shape[0])


@dataclass
class PolicyOutput:
    candidate_ids: tuple
    logits: torch.Tensor
    mask: torch.Tensor
    distribution: object
    value: torch.Tensor
    q: torch.Tensor
    hidden: torch.Tensor
    diagnostics: dict
    ended_reason: str | None = None

    def select(self, deterministic=False):
        if self.distribution is None:
            raise DataIntegrityError("NO_SAFE_CANDIDATES")
        if deterministic:
            maximum = self.logits[self.mask].max()
            choices = [i for i, k in enumerate(self.candidate_ids) if self.mask[i] and self.logits[i] == maximum]
            index = min(choices, key=lambda i: self.candidate_ids[i])
        else:
            index = int(self.distribution.sample())
        return self.candidate_ids[index], index


class Policy(nn.Module):
    """One parameter owner. B1 is B1-K: phi_K(c, 0), no prior, no nominal successor."""

    def __init__(self, actions, predicates, types, observation_dim, candidate_dim, method="Full", B=0.5):
        super().__init__()
        self.method = canonical_method(method)
        self.B = B
        self.encoder = GraphEncoder(actions, predicates, types)
        self.set_encoder = SetTransformerEncoder(actions, predicates, types)
        self.observation = nn.Linear(observation_dim, 128)
        self.gru = nn.GRUCell(128, 128)
        self.candidate = nn.Linear(candidate_dim, 128)
        self.contract = CandidateReadout(384)
        self.b0_fuse = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.effect_readout = EffectTokenReadout(384)
        self.effect_fuse = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.prior = AnchoredPrior(512)
        self.cat_proj = nn.Linear(512, 128, bias=False)
        self.base = nn.Linear(128, 1)
        self.v_head = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1))
        self.q_head = nn.Sequential(nn.Linear(640, 128), nn.ReLU(), nn.Linear(128, 1))
        # Only PRIOR_BIAS owns the shared scalar; other methods keep their original parameter set.
        if self.method == "PRIOR_BIAS":
            self.prior_beta = nn.Parameter(torch.zeros(()))
        self.shadow_zero_prior = False  # evaluation switch: zero ONLY the final prior residual

    def prior_bias_scores(self, snapshot, mask):
        """PRIOR_BIAS: centred prior scores scaled to [-1, 1] over the valid candidate set (same bound B)."""
        s = {}
        for e in snapshot.prior_edges:
            if len(e) > 3 and e[2] == "SOFT_RELEVANT_TO_GOAL":
                s[e[0]] = float(e[3])
        raw = [s.get(cid, 0.0) if bool(mask[i]) else 0.0 for i, cid in enumerate(snapshot.candidate_ids)]
        valid = [i for i in range(len(raw)) if bool(mask[i])]
        if not s or not valid:
            return [0.0] * len(raw)
        mean = sum(raw[i] for i in valid) / len(valid)
        centred = [raw[i] - mean if bool(mask[i]) else 0.0 for i in range(len(raw))]
        scale = max(abs(c) for c in centred)
        return [c / scale if scale > 1e-12 else 0.0 for c in centred]

    @property
    def q_coefficient(self):
        return 0.0 if self.method == "A_Q" else 0.1

    def initial_hidden(self):
        return next(self.parameters()).new_zeros(128)

    def advance_hidden(self, base_input, hidden):
        x = torch.as_tensor(base_input, device=hidden.device, dtype=hidden.dtype)
        return self.gru(torch.relu(self.observation(x)), hidden)

    def _phi_k(self, context, dk_or_zero):
        return self.contract(context, dk_or_zero)

    def forward(self, snapshot, hidden=None):
        method = canonical_method(self.method)
        hidden = self.initial_hidden() if hidden is None else hidden
        zo = self.advance_hidden(snapshot.base_input, hidden)
        mask = torch.tensor(snapshot.mask, device=zo.device, dtype=torch.bool)
        if not mask.any():
            empty = zo.new_zeros(len(mask))
            return PolicyOutput(snapshot.candidate_ids, empty, mask, None, zo.sum() * 0, empty, zo, {"successor_used": False, "method": method}, "NO_SAFE_CANDIDATES")
        facts = snapshot.facts.values
        contracts = {c.id: c for c in snapshot.template.contracts}
        k = view(snapshot.template, facts)
        successor_used = False
        zk = None
        encoded_set = None
        if method == "B0":
            tokens = self.set_encoder.tokens(k)
            tokens = torch.cat((tokens, zo.unsqueeze(0)), 0)
            encoded_set = self.set_encoder.encode(tokens)
            zk = encoded_set.mean(0, keepdim=True)
        else:
            zk = self.encoder(k)
        zh = None
        if method == "B1-H":
            edges = snapshot.prior_edges
            zh = self.encoder(view(snapshot.template, facts, edges)) if edges else zk
        uk_all = []
        up_all = []
        logits = []
        diagnostics = {"differences": {}, "prior_inputs": {}, "up": {}, "delta": {}, "effect_tokens": {}, "stat_relation": {}, "successor_used": False, "method": method, "actor_episode_discount_weight": False}
        for i, cid in enumerate(snapshot.candidate_ids):
            if not snapshot.mask[i]:
                uk_all.append(zo.new_zeros(128))
                up_all.append(zo.new_zeros(128))
                logits.append(zo.sum() * 0)
                continue
            ca = self.candidate(torch.as_tensor(snapshot.candidate_features[i], device=zo.device, dtype=zo.dtype))
            if method == "B0":
                uk = self.b0_fuse(torch.cat((zo, ca, self.set_encoder.candidate_read(ca, encoded_set))))
                prior_input = zo.new_zeros(zk.shape)
                up, residual = self.prior(torch.cat((zo, ca, uk, uk)), prior_input, self.B, False)
            elif method == "B1":
                context = torch.cat((zo, ca, zk.mean(0)))
                zero_dk = torch.zeros_like(zk)
                uk = self._phi_k(context, zero_dk)
                prior_input = torch.zeros_like(zk)
                up, residual = self.prior(torch.cat((context, uk)), prior_input, self.B, False)
            elif method == "B1-H":
                context = torch.cat((zo, ca, zk.mean(0)))
                uk = self._phi_k(context, zk)
                prior_input = zh.float() - zk.float() if snapshot.prior_edges else torch.zeros_like(zk)
                up, residual = self.prior(torch.cat((context, uk)), prior_input, self.B, False)
            elif method == "B1-K+E":
                context = torch.cat((zo, ca, zk.mean(0)))
                base_uk = self._phi_k(context, torch.zeros_like(zk))
                effect, token_count = self.effect_readout(contracts[cid], context)
                uk = self.effect_fuse(torch.cat((base_uk, effect)))
                prior_input = torch.zeros_like(zk)
                up, residual = self.prior(torch.cat((context, uk)), prior_input, self.B, False)
                diagnostics["effect_tokens"][cid] = token_count
            else:
                edges = () if method in ("B2", "PRIOR_BIAS") else snapshot.prior_edges
                delta = differences(self.encoder, four_views(snapshot.template, facts, edges, contracts[cid]))
                successor_used = True
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
                        s = self.prior.final(up).squeeze(-1)
                        residual = self.B * torch.tanh(s)
                    else:
                        zero = prior_input.sum() * 0 + cat_ctx.sum() * 0
                        up, residual = zero.expand(128), zero
                else:
                    if method == "A_STAT":
                        prior_input = delta.zh.float() - delta.zk.float() if edges else torch.zeros_like(delta.dk)
                        diagnostics["stat_relation"][cid] = prior_input
                    elif method == "A_DD":
                        prior_input = delta.dh if edges else torch.zeros_like(delta.dk)
                    elif method in ("B2", "PRIOR_BIAS"):
                        prior_input = torch.zeros_like(delta.dk)
                    else:
                        prior_input = delta.dp if edges else torch.zeros_like(delta.dk)
                    up, residual = self.prior(torch.cat((context, uk)), prior_input, self.B, method == "A_B")
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
        diagnostics["successor_used"] = successor_used
        uk = torch.stack(uk_all)
        up = torch.stack(up_all)
        mean_k = uk[mask].mean(0)
        mean_p = up[mask].mean(0)
        zpool = encoded_set.mean(0) if method == "B0" else zk.mean(0)
        value = self.v_head(torch.cat((zo, zpool, mean_k, mean_p))).squeeze(-1)
        q = self.q_head(torch.cat((uk, up, zo.expand(len(uk), -1), mean_k.expand(len(uk), -1), mean_p.expand(len(uk), -1)), dim=-1)).squeeze(-1)
        logits = torch.stack(logits).masked_fill(~mask, -torch.inf)
        return PolicyOutput(snapshot.candidate_ids, logits, mask, torch.distributions.Categorical(logits=logits), value, q, zo, diagnostics)
