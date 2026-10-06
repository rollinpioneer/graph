"""CP-DISR-C1-CROSS-TASK-BLOCKSWORLD-V1 policies (runbook 7): B2-CACHED, QMARK (task adapter) and ASNET-READOUT.

All three are ``neural.Policy`` subclasses on the production parameter owner: the same node-feature embeddings, 4-layer RGCN, LayerNorms, goal readout, CandidateReadout,
GRU, Actor / Value / Q heads and the same PPO. They differ only in the rows handed to the shared candidate readout:

* B2-CACHED      rows_i = E(F_i^after) - E(F)                    1 + K graph encodings (E(F) once)      nominal successor of every legal candidate
* QMARK          rows_i = E(G(F, q=a_i)) - E(G(F, q=none))        1 + K graph encodings                  no successor, one shared query projection (``c1_qmark_policy``)
* ASNET-READOUT  rows_i = H[ACTION(a_i)]                          1 graph encoding                       no query, no successor, final node state of the candidate's ACTION node

The K + 1 graphs of a decision are encoded as ONE disjoint-union batch through the production modules (``features`` embeddings, ``layers``, ``norms``, ``readout``), which is
mathematically the per-graph computation of ``GraphEncoder`` (mean aggregation never crosses graphs); equivalence to the unbatched production code is a pre-training test.
No block name, grounded action id, case id, split or goal hash enters any computation: candidate features are the fixed 8-vector of runbook 6.2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import c1_qmark_policy as Q
from . import neural
from .facts import Truth
from .neural import Policy, PolicyOutput

B2_CACHED = "B2-CACHED"
QMARK_BW = "B1-K+QMARK-BW"
ASNET = "ASNET-READOUT"
METHODS = (B2_CACHED, QMARK_BW, ASNET)
SCHEMA_ORDER = ("PICK_UP", "PUT_DOWN", "UNSTACK", "STACK")
OBS_DIM = 48
CAND_DIM = 8
HIDDEN = 128


def candidate_feature(schema):
    """[one_hot(PICK_UP, PUT_DOWN, UNSTACK, STACK), arity / 2, 0, 0, 0] (runbook 6.2): no identity of any block or grounded action."""
    arity = {"PICK_UP": 1, "PUT_DOWN": 1, "UNSTACK": 2, "STACK": 2}[schema]
    row = [0.0] * CAND_DIM
    row[SCHEMA_ORDER.index(schema)] = 1.0
    row[4] = arity / 2.0
    return tuple(row)


@dataclass
class TemplateStatic:
    """Per-template constant tensors (node-feature indices, edges, goal / action / proposition positions, effect masks)."""
    template: object
    N: int
    kind_idx: torch.Tensor
    action_idx: torch.Tensor
    pred_idx: torch.Tensor
    is_action: torch.Tensor
    arg_idx: torch.Tensor
    arg_w: torch.Tensor
    prop_pos: torch.Tensor
    action_pos: torch.Tensor
    goal_pos: torch.Tensor
    goal_sign: torch.Tensor          # (ng, 1)
    prop_goal_sign: torch.Tensor     # (P, 1)
    ei: torch.Tensor
    et: torch.Tensor
    action_ids: tuple
    action_row: dict                 # action id -> row in the action-position list
    add_mask: torch.Tensor           # (A, P) bool
    del_mask: torch.Tensor           # (A, P) bool
    prop_ids: tuple
    prop_index: dict


class BwMixin:
    """Batched encoder application shared by the three policies (reads the policy's own production modules)."""

    def _static(self, template):
        cache = self.__dict__.setdefault("_static_cache", {})
        hit = cache.get(id(template))
        if hit is not None and hit.template is template:
            return hit
        dev = next(self.parameters()).device
        feats = self.encoder.features
        nodes = template.nodes
        ids = template.node_ids
        N = len(nodes)
        kind_idx, action_idx, pred_idx, is_action = [], [], [], []
        arg_idx = torch.zeros((N, 2), dtype=torch.long)
        arg_w = torch.zeros((N, 2))
        for i, n in enumerate(nodes):
            act = n.kind == "ACTION"
            kind_idx.append(0 if act else 1)
            action_idx.append(feats.actions[n.schema] if act else 0)
            pred_idx.append(0 if act else feats.predicates[n.schema])
            is_action.append(act)
            for role, t in enumerate(n.argument_types):
                arg_idx[i, role] = feats.types[t]
                arg_w[i, role] = 1.0 / (role + 1)
        prop_pos = [i for i, n in enumerate(nodes) if n.kind == "PROPOSITION"]
        action_pos = [i for i, n in enumerate(nodes) if n.kind == "ACTION"]
        signs = {g.fact_id: float(g.sign) for g in template.goals}
        prop_ids = tuple(ids[i] for i in prop_pos)
        prop_index = {p: j for j, p in enumerate(prop_ids)}
        edges = template.edges
        pos = {v: i for i, v in enumerate(ids)}
        ei = torch.tensor([[pos[a] for a, b, r in edges], [pos[b] for a, b, r in edges]], dtype=torch.long).reshape(2, -1)
        et = torch.tensor([neural.RELATIONS.index(r) for a, b, r in edges], dtype=torch.long)
        action_ids = tuple(ids[i] for i in action_pos)
        add_mask = torch.zeros((len(action_ids), len(prop_ids)), dtype=torch.bool)
        del_mask = torch.zeros((len(action_ids), len(prop_ids)), dtype=torch.bool)
        for r, c in enumerate(sorted(template.contracts, key=lambda c: c.id)):
            assert c.id == action_ids[r]
            for a in c.effects.add:
                add_mask[r, prop_index[a.id]] = True
            for a in c.effects.delete:
                del_mask[r, prop_index[a.id]] = True
        st = TemplateStatic(
            template=template, N=N, kind_idx=torch.tensor(kind_idx, device=dev), action_idx=torch.tensor(action_idx, device=dev), pred_idx=torch.tensor(pred_idx, device=dev),
            is_action=torch.tensor(is_action, device=dev), arg_idx=arg_idx.to(dev), arg_w=arg_w.to(dev), prop_pos=torch.tensor(prop_pos, device=dev), action_pos=torch.tensor(action_pos, device=dev),
            goal_pos=torch.tensor([ids.index(g.fact_id) for g in template.goals], device=dev), goal_sign=torch.tensor([[float(g.sign)] for g in template.goals], device=dev),
            prop_goal_sign=torch.tensor([[signs.get(p, 0.0)] for p in prop_ids], device=dev), ei=ei.to(dev), et=et.to(dev), action_ids=action_ids,
            action_row={a: r for r, a in enumerate(action_ids)}, add_mask=add_mask.to(dev), del_mask=del_mask.to(dev), prop_ids=prop_ids, prop_index=prop_index)
        cache[id(template)] = st
        return st

    def _codes(self, st, values):
        """(P, 3) float [TRUE, FALSE, UNKNOWN] one-hot per proposition, in the template's proposition order."""
        dev = st.prop_pos.device
        rows = [[1.0, 0.0, 0.0] if values[p] == Truth.TRUE else [0.0, 1.0, 0.0] if values[p] == Truth.FALSE else [0.0, 0.0, 1.0] for p in st.prop_ids]
        return torch.tensor(rows, device=dev)

    def _successor_codes(self, st, codes0, rows):
        """Nominal successor facts of the legal candidates ``rows`` (the same assignment nominal_overlay applies: ADD -> TRUE, DEL -> FALSE)."""
        K = len(rows)
        out = codes0.unsqueeze(0).repeat(K, 1, 1)
        r = torch.tensor(rows, device=codes0.device)
        true_code = torch.tensor([1.0, 0.0, 0.0], device=codes0.device)
        false_code = torch.tensor([0.0, 1.0, 0.0], device=codes0.device)
        out = torch.where(st.add_mask[r].unsqueeze(-1), true_code, out)
        out = torch.where(st.del_mask[r].unsqueeze(-1), false_code, out)
        return out

    def _encode(self, st, codes, query_nodes=None):
        """Disjoint-union encoding of G graphs. codes (G, P, 3); query_nodes (G,) long node index of the queried ACTION node or -1. Returns (readout rows (G, 1+ng, 128), H (G, N, 128))."""
        enc = self.encoder
        feats = enc.features
        G = codes.shape[0]
        N = st.N
        static = feats.kind.weight[st.kind_idx] + torch.where(st.is_action.unsqueeze(-1), feats.action.weight[st.action_idx], feats.predicate.weight[st.pred_idx])
        static = static + (feats.arg.weight[st.arg_idx] * st.arg_w.unsqueeze(-1)).sum(1)
        dyn = feats.fact(torch.cat((codes, st.prop_goal_sign.unsqueeze(0).expand(G, -1, -1)), dim=-1))
        dyn_full = torch.zeros((G, N, dyn.shape[-1]), device=dyn.device, dtype=dyn.dtype)
        dyn_full[:, st.prop_pos] = dyn
        h = static.unsqueeze(0) + dyn_full
        if query_nodes is not None:
            qmask = torch.zeros((G, N), device=h.device, dtype=h.dtype)
            has = query_nodes >= 0
            qmask[torch.arange(G, device=h.device)[has], query_nodes[has]] = 1.0
            qvec = self.query_projection(torch.ones((1, 1), device=h.device, dtype=h.dtype)).reshape(-1)
            h = h + qmask.unsqueeze(-1) * qvec
        E = st.ei.shape[1]
        offsets = (torch.arange(G, device=h.device) * N).repeat_interleave(E)
        ei = st.ei.repeat(1, G) + offsets.unsqueeze(0)
        et = st.et.repeat(G)
        hflat = h.reshape(G * N, -1)
        for layer, norm in zip(enc.layers, enc.norms):
            hflat = norm(torch.relu(layer(hflat, ei, et)))
        H = hflat.reshape(G, N, -1)
        ro = enc.readout
        goals = ro.goal(torch.cat((H[:, st.goal_pos], st.goal_sign.unsqueeze(0).expand(G, -1, -1)), dim=-1))
        glob = ro.global_readout(torch.cat((H[:, st.action_pos].mean(1), H[:, st.prop_pos].mean(1), goals.mean(1)), dim=-1))
        return torch.cat((glob.unsqueeze(1), goals), dim=1), H

    def _phi_batch(self, contexts, rows):
        """CandidateReadout applied to K (context, rows) pairs at once (identical arithmetic to ``CandidateReadout.forward``)."""
        cr = self.contract
        score = (cr.key(rows) * cr.query(contexts).unsqueeze(1)).sum(-1) / math.sqrt(128)
        attn = torch.softmax(score, dim=-1)
        attended = (attn.unsqueeze(-1) * cr.value(rows)).sum(1)
        return cr.net(torch.cat((contexts, attended), dim=-1))

    def _assemble(self, snapshot, zo, mask, legal, uk_legal, zk_mean_extra, diagnostics):
        """Shared tail: logits, value and Q heads exactly as ``Policy.forward`` (prior terms are zero for these methods)."""
        n = len(snapshot.candidate_ids)
        uk = zo.new_zeros((n, 128))
        uk[torch.tensor(legal, device=zo.device)] = uk_legal
        up = zo.new_zeros((n, 128))
        base = self.base(uk_legal).squeeze(-1)
        logits = zo.new_zeros(n)
        logits = logits.index_put((torch.tensor(legal, device=zo.device),), base)
        mean_k = uk_legal.mean(0)
        mean_p = zo.new_zeros(128)
        value = self.v_head(torch.cat((zo, zk_mean_extra, mean_k, mean_p))).squeeze(-1)
        q = self.q_head(torch.cat((uk, up, zo.expand(n, -1), mean_k.expand(n, -1), mean_p.expand(n, -1)), dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~mask, -torch.inf)
        return PolicyOutput(snapshot.candidate_ids, logits, mask, torch.distributions.Categorical(logits=logits), value, q, zo, diagnostics)

    def _prep(self, snapshot, hidden):
        hidden = self.initial_hidden() if hidden is None else hidden
        zo = self.advance_hidden(snapshot.base_input, hidden)
        mask = torch.tensor(snapshot.mask, device=zo.device, dtype=torch.bool)
        legal = [i for i, m in enumerate(snapshot.mask) if m]
        return zo, mask, legal

    def _contexts(self, snapshot, zo, legal, zk_mean):
        feats = torch.tensor([snapshot.candidate_features[i] for i in legal], device=zo.device, dtype=zo.dtype)
        ca = self.candidate(feats)
        return torch.cat((zo.unsqueeze(0).expand(len(legal), -1), ca, zk_mean.unsqueeze(0).expand(len(legal), -1)), dim=-1)


def _segment_mean(values, seg, n_seg):
    """Mean of rows of ``values`` per segment id (``seg`` sorted not required)."""
    out = values.new_zeros((n_seg,) + values.shape[1:])
    out = out.index_add(0, seg, values)
    cnt = torch.zeros(n_seg, device=values.device, dtype=values.dtype).index_add(0, seg, torch.ones(len(seg), device=values.device, dtype=values.dtype))
    return out / cnt.view(-1, *([1] * (values.dim() - 1)))


def _batched_group(policy, kind, snaps, zos, selected):
    """Batched forward of S snapshots that share ONE template: returns (logp_selected (S,), value (S,), q_selected (S,), entropy (S,)).

    The graphs of all snapshots (1 + K each for B2 / QMARK, 1 for ASNET) go through the production encoder modules as one disjoint-union batch; the heads are applied exactly as in the
    per-snapshot ``forward`` (so this is the same computation, grouped). Equivalence with the per-snapshot forward is a pre-training test."""
    st = policy._static(snaps[0].template)
    dev = zos.device
    S_g = len(snaps)
    n = len(snaps[0].candidate_ids)
    codes_list, qnodes, legal_per, offsets, graph_off = [], [], [], [], 0
    for sn in snaps:
        legal = [i for i, m in enumerate(sn.mask) if m]
        legal_per.append(legal)
        codes0 = policy._codes(st, sn.facts.values)
        rows_idx = [st.action_row[sn.candidate_ids[i]] for i in legal]
        offsets.append(graph_off)
        if kind == "b2":
            codes_list.append(torch.cat((codes0.unsqueeze(0), policy._successor_codes(st, codes0, rows_idx)), 0))
        elif kind == "qmark":
            codes_list.append(codes0.unsqueeze(0).repeat(len(legal) + 1, 1, 1))
            qnodes += [-1] + [int(st.action_pos[r]) for r in rows_idx]
        else:
            codes_list.append(codes0.unsqueeze(0))
        graph_off += codes_list[-1].shape[0]
    codes = torch.cat(codes_list, 0)
    z, H = policy._encode(st, codes, torch.tensor(qnodes, device=dev) if kind == "qmark" else None)
    rows_all, zk_means, seg, flat_pos, feats = [], [], [], [], []
    for si, sn in enumerate(snaps):
        legal, off = legal_per[si], offsets[si]
        K = len(legal)
        zk = z[off]
        zk_means.append(zk.mean(0))
        if kind == "asnet":
            ridx = torch.tensor([int(st.action_pos[st.action_row[sn.candidate_ids[i]]]) for i in legal], device=dev)
            rows_all.append(H[off, ridx].unsqueeze(1))
        else:
            rows_all.append(z[off + 1: off + 1 + K].float() - zk.float().unsqueeze(0))
        seg += [si] * K
        flat_pos += [si * n + i for i in legal]
        feats += [sn.candidate_features[i] for i in legal]
    rows_all = torch.cat(rows_all, 0)
    seg_t = torch.tensor(seg, device=dev)
    zk_mean = torch.stack(zk_means)
    ca = policy.candidate(torch.tensor(feats, device=dev, dtype=zos.dtype))
    ctx = torch.cat((zos[seg_t], ca, zk_mean[seg_t]), dim=-1)
    uk_legal = policy._phi_batch(ctx, rows_all)
    flat = torch.tensor(flat_pos, device=dev)
    uk_full = zos.new_zeros((S_g * n, 128)).index_copy(0, flat, uk_legal).view(S_g, n, 128)
    mask = torch.tensor([sn.mask for sn in snaps], device=dev, dtype=torch.bool)
    logits = policy.base(uk_full).squeeze(-1).masked_fill(~mask, -torch.inf)
    dist = torch.distributions.Categorical(logits=logits)
    sel = torch.tensor([sn.candidate_ids.index(s) for sn, s in zip(snaps, selected)], device=dev)
    mean_k = _segment_mean(uk_legal, seg_t, S_g)
    mean_p = zos.new_zeros((S_g, 128))
    value = policy.v_head(torch.cat((zos, zk_mean, mean_k, mean_p), dim=-1)).squeeze(-1)
    up = zos.new_zeros((S_g, n, 128))
    q = policy.q_head(torch.cat((uk_full, up, zos.unsqueeze(1).expand(-1, n, -1), mean_k.unsqueeze(1).expand(-1, n, -1), mean_p.unsqueeze(1).expand(-1, n, -1)), dim=-1)).squeeze(-1)
    return dist.log_prob(sel), value, q.gather(1, sel.unsqueeze(1)).squeeze(1), dist.entropy()


def batched_outputs(policy, kind, snaps, zos, selected):
    """Grouped forward over snapshots of possibly different templates; results returned in the input order."""
    groups = {}
    for i, sn in enumerate(snaps):
        groups.setdefault(id(sn.template), []).append(i)
    lp = [None] * len(snaps)
    vs, qs, ent = list(lp), list(lp), list(lp)
    for _tid, idx in groups.items():
        a, b, c, d = _batched_group(policy, kind, [snaps[i] for i in idx], zos[idx], [selected[i] for i in idx])
        for j, i in enumerate(idx):
            lp[i], vs[i], qs[i], ent[i] = a[j], b[j], c[j], d[j]
    return torch.stack(lp), torch.stack(vs), torch.stack(qs), torch.stack(ent)


class B2CachedPolicy(BwMixin, Policy):
    """B2 with E(F) computed once per decision: 1 + K encodings, K nominal successors (same parameters and heads as the original B2)."""

    row_kind = "b2"

    def __init__(self, actions, predicates, types, observation_dim, candidate_dim, method=B2_CACHED, B=0.5):
        super().__init__(actions, predicates, types, observation_dim, candidate_dim, method="B2", B=B)
        self.method = method
        self.nominal_apply_calls = 0
        self.encodings = 0

    def forward(self, snapshot, hidden=None):
        zo, mask, legal = self._prep(snapshot, hidden)
        if not legal:
            empty = zo.new_zeros(len(mask))
            return PolicyOutput(snapshot.candidate_ids, empty, mask, None, zo.sum() * 0, empty, zo, {"successor_used": False, "method": B2_CACHED}, "NO_SAFE_CANDIDATES")
        st = self._static(snapshot.template)
        codes0 = self._codes(st, snapshot.facts.values)
        rows_idx = [st.action_row[snapshot.candidate_ids[i]] for i in legal]
        codes = torch.cat((codes0.unsqueeze(0), self._successor_codes(st, codes0, rows_idx)), dim=0)
        z, _H = self._encode(st, codes)
        zk, zki = z[0], z[1:]
        deltas = zki.float() - zk.float().unsqueeze(0)
        ctx = self._contexts(snapshot, zo, legal, zk.mean(0))
        uk = self._phi_batch(ctx, deltas)
        self.nominal_apply_calls += len(legal)
        self.encodings += codes.shape[0]
        diag = {"successor_used": True, "method": B2_CACHED, "encoder_graph_encodings": int(codes.shape[0]), "nominal_apply_calls": len(legal), "actor_episode_discount_weight": False}
        return self._assemble(snapshot, zo, mask, legal, uk, zk.mean(0), diag)


class QmarkBwPolicy(BwMixin, Q.QmarkPolicy):
    """QMARK on Blocksworld: the frozen c1_qmark_policy.QmarkPolicy parameters / mathematics; only the encoder application is batched."""

    row_kind = "qmark"

    def __init__(self, actions, predicates, types, observation_dim, candidate_dim, method=QMARK_BW, B=0.5):
        super().__init__(actions, predicates, types, observation_dim, candidate_dim, method=Q.QMARK_METHOD, B=B)
        self.method = method
        self.encodings = 0

    def reference_forward(self, snapshot, hidden=None):
        """The frozen unbatched QMARK forward (per-candidate encoder calls); used by the equivalence tests."""
        saved = self.method
        self.method = Q.QMARK_METHOD
        try:
            return Q.QmarkPolicy.forward(self, snapshot, hidden)
        finally:
            self.method = saved

    def forward(self, snapshot, hidden=None):
        zo, mask, legal = self._prep(snapshot, hidden)
        if not legal:
            empty = zo.new_zeros(len(mask))
            return PolicyOutput(snapshot.candidate_ids, empty, mask, None, zo.sum() * 0, empty, zo, {"successor_used": False, "method": QMARK_BW}, "NO_SAFE_CANDIDATES")
        st = self._static(snapshot.template)
        codes0 = self._codes(st, snapshot.facts.values)
        K = len(legal)
        codes = codes0.unsqueeze(0).repeat(K + 1, 1, 1)                 # facts are never modified: every graph carries the same fact values
        qnodes = torch.tensor([-1] + [int(st.action_pos[st.action_row[snapshot.candidate_ids[i]]]) for i in legal], device=codes0.device)
        z, _H = self._encode(st, codes, qnodes)
        zk, zq = z[0], z[1:]
        deltas = zq.float() - zk.float().unsqueeze(0)
        ctx = self._contexts(snapshot, zo, legal, zk.mean(0))
        uk = self._phi_batch(ctx, deltas)
        self.encodings += codes.shape[0]
        diag = {"successor_used": False, "method": QMARK_BW, "encoder_graph_encodings": int(codes.shape[0]), "nominal_apply_calls": 0, "actor_episode_discount_weight": False}
        return self._assemble(snapshot, zo, mask, legal, uk, zk.mean(0), diag)


class AsnetReadoutPolicy(BwMixin, Policy):
    """ASNet-style adapted readout: ONE graph encoding per decision; the candidate row is the final hidden state of the candidate's ACTION node (no query, no successor, no extra module)."""

    row_kind = "asnet"

    def __init__(self, actions, predicates, types, observation_dim, candidate_dim, method=ASNET, B=0.5):
        super().__init__(actions, predicates, types, observation_dim, candidate_dim, method="B1-K", B=B)
        self.method = method
        self.encodings = 0

    def forward(self, snapshot, hidden=None):
        zo, mask, legal = self._prep(snapshot, hidden)
        if not legal:
            empty = zo.new_zeros(len(mask))
            return PolicyOutput(snapshot.candidate_ids, empty, mask, None, zo.sum() * 0, empty, zo, {"successor_used": False, "method": ASNET}, "NO_SAFE_CANDIDATES")
        st = self._static(snapshot.template)
        codes0 = self._codes(st, snapshot.facts.values).unsqueeze(0)
        z, H = self._encode(st, codes0)
        zk = z[0]
        rows_idx = torch.tensor([int(st.action_pos[st.action_row[snapshot.candidate_ids[i]]]) for i in legal], device=H.device)
        rows = H[0, rows_idx].unsqueeze(1)                           # (K, 1, 128): H[ACTION(a_i)]
        ctx = self._contexts(snapshot, zo, legal, zk.mean(0))
        uk = self._phi_batch(ctx, rows)
        self.encodings += 1
        diag = {"successor_used": False, "method": ASNET, "encoder_graph_encodings": 1, "nominal_apply_calls": 0, "actor_episode_discount_weight": False}
        return self._assemble(snapshot, zo, mask, legal, uk, zk.mean(0), diag)


POLICY_CLASS = {B2_CACHED: B2CachedPolicy, QMARK_BW: QmarkBwPolicy, ASNET: AsnetReadoutPolicy}


def make_policy(method, device="cpu", seed=0, actions=("PICK_UP", "PUT_DOWN", "STACK", "UNSTACK")):
    from .blocksworld import contracts as C
    torch.manual_seed(seed)
    model = POLICY_CLASS[method](sorted(actions), list(C.PREDICATES), list(C.TYPES), OBS_DIM, CAND_DIM, method=method, B=0.5)
    return model.to(device)
