"""CP-DISR-C1-MECH-CONFIRM-V1 QMARK: a strong "no successor state" control for the C1 mechanism question.

QMARK (``B1-K+QMARK``) is a fairness control, not a new method. It receives exactly what B2 receives (current public facts, the full contract graph,
the goal, the candidate identity, the same 4-layer relational graph encoder, candidate readout, GRU, Actor, Value and Q heads, the same training
protocol) but it never builds a successor state:

* B2     : change fact values (nominal apply), encode the after-state,       rows_i = E(F_after_i) - E(F)
* QMARK  : change NO fact value; mark the queried action node,               rows_i = E(G(F, q=a_i)) - E(G(F, q=none))

The two graphs share facts, goals, contract edges and candidate set; the only difference is one binary query channel on the queried ACTION node,
passed through one small shared projection and added to that node's feature. The PRE / ADD / DEL / UNKNOWN relation edges of the encoder then let the
message passing carry the query to the propositions and goals the action touches. No contract-id / object-string / case / seed / cell dependent
parameter exists: the single projection is shared by every candidate.

``neural.py`` is untouched; the old methods are unaffected (QMARK lives in a ``Policy`` subclass registered in-process, exactly like the
representation controls). Import-light: no simulator.
"""
from __future__ import annotations

import torch
from torch import nn

from . import neural
from .graph import view
from .neural import Policy, PolicyOutput

QMARK_METHOD = "B1-K+QMARK"
METHODS = (QMARK_METHOD,)
QUERY_DIM = 128


def query_channel(graph, queried_action_id, device=None, dtype=torch.float32):
    """(N, 1) binary channel: 1 on the queried ACTION node, 0 elsewhere. Pure function of the node ids; never reads facts."""
    ids = graph.template.node_ids
    if queried_action_id not in ids:
        raise KeyError("unknown action node %s" % queried_action_id)
    index = ids.index(queried_action_id)
    node = graph.template.nodes[index]
    if node.kind != "ACTION":
        raise ValueError("%s is not an ACTION node" % queried_action_id)
    channel = torch.zeros((len(ids), 1), device=device, dtype=dtype)
    channel[index, 0] = 1.0
    return channel


class QmarkPolicy(Policy):
    """B1-K+QMARK on the production Policy parameter owner (only ``query_projection`` is new)."""

    def __init__(self, actions, predicates, types, observation_dim, candidate_dim, method=QMARK_METHOD, B=0.5):
        if method != QMARK_METHOD:
            raise ValueError(method)
        # Every production module is built by the production constructor first, so the shared modules start from the same weights as B2 at a given seed.
        super().__init__(actions, predicates, types, observation_dim, candidate_dim, method="B1-K", B=B)
        self.method = method
        self.query_projection = nn.Linear(1, QUERY_DIM, bias=False)

    def encode_with_action_query(self, graph, queried_action_id=None):
        """Same computation as ``GraphEncoder.forward`` with one extra additive query channel on the node features (identical output when no query)."""
        enc = self.encoder
        enc.forward_calls += 1
        h = enc.features(graph)
        if queried_action_id is not None:
            h = h + self.query_projection(query_channel(graph, queried_action_id, device=h.device, dtype=h.dtype))
        ids = {v: i for i, v in enumerate(graph.template.node_ids)}
        edges = graph.edges
        ei = torch.tensor([[ids[a] for a, b, r in edges], [ids[b] for a, b, r in edges]], device=h.device, dtype=torch.long).reshape(2, -1)
        et = torch.tensor([neural.RELATIONS.index(r) for a, b, r in edges], device=h.device, dtype=torch.long)
        for layer, norm in zip(enc.layers, enc.norms):
            h = norm(torch.relu(layer(h, ei, et)))
        return enc.readout(h, graph.template)

    def forward(self, snapshot, hidden=None):
        hidden = self.initial_hidden() if hidden is None else hidden
        zo = self.advance_hidden(snapshot.base_input, hidden)
        mask = torch.tensor(snapshot.mask, device=zo.device, dtype=torch.bool)
        if not mask.any():
            empty = zo.new_zeros(len(mask))
            return PolicyOutput(snapshot.candidate_ids, empty, mask, None, zo.sum() * 0, empty, zo, {"successor_used": False, "method": QMARK_METHOD}, "NO_SAFE_CANDIDATES")
        facts = snapshot.facts.values
        k = view(snapshot.template, facts)          # the one graph: its fact values are never modified
        zk = self.encoder(k)                        # E(G(F, q=none))
        uk_all, up_all, logits = [], [], []
        diagnostics = {"differences": {}, "prior_inputs": {}, "up": {}, "delta": {}, "effect_tokens": {}, "stat_relation": {}, "successor_used": False, "method": QMARK_METHOD,
                       "actor_episode_discount_weight": False, "query_rows": {}}
        for i, cid in enumerate(snapshot.candidate_ids):
            if not snapshot.mask[i]:
                uk_all.append(zo.new_zeros(128))
                up_all.append(zo.new_zeros(128))
                logits.append(zo.sum() * 0)
                continue
            ca = self.candidate(torch.as_tensor(snapshot.candidate_features[i], device=zo.device, dtype=zo.dtype))
            context = torch.cat((zo, ca, zk.mean(0)))
            zq = self.encode_with_action_query(k, cid)             # E(G(F, q=a_i)): same facts, one marked action node
            rows = zq.float() - zk.float()
            uk = self._phi_k(context, rows)
            prior_input = torch.zeros_like(rows)
            up, residual = self.prior(torch.cat((context, uk)), prior_input, self.B, False)
            uk_all.append(uk)
            up_all.append(up)
            logits.append(self.base(uk).squeeze(-1) + residual)
            diagnostics["prior_inputs"][cid] = prior_input
            diagnostics["up"][cid] = up
            diagnostics["delta"][cid] = residual
            diagnostics["query_rows"][cid] = rows
        uk = torch.stack(uk_all)
        up = torch.stack(up_all)
        mean_k = uk[mask].mean(0)
        mean_p = up[mask].mean(0)
        zpool = zk.mean(0)
        value = self.v_head(torch.cat((zo, zpool, mean_k, mean_p))).squeeze(-1)
        q = self.q_head(torch.cat((uk, up, zo.expand(len(uk), -1), mean_k.expand(len(uk), -1), mean_p.expand(len(uk), -1)), dim=-1)).squeeze(-1)
        logits = torch.stack(logits).masked_fill(~mask, -torch.inf)
        return PolicyOutput(snapshot.candidate_ids, logits, mask, torch.distributions.Categorical(logits=logits), value, q, zo, diagnostics)


def make_qmark_policy_factory(v11, original_make_policy):
    """A `make_policy` that builds QmarkPolicy for QMARK and defers to the already-registered factory otherwise."""
    def make_policy(template, method, device):
        if method != QMARK_METHOD:
            return original_make_policy(template, method, device)
        from .platforms.libero.runtime_factory import PREDICATES, TASK_OBJECTS
        actions = sorted({c.name for c in template.contracts})
        type_set = set()
        for mapping in TASK_OBJECTS.values():
            type_set.update(mapping.values())
        model = QmarkPolicy(actions=actions, predicates=sorted(PREDICATES), types=sorted(type_set), observation_dim=v11.OBS_DIM, candidate_dim=v11.CAND_DIM, method=method, B=v11.B_PRIOR)
        return model.to(device)
    make_policy.__wrapped_production__ = getattr(original_make_policy, "__wrapped_production__", original_make_policy)
    make_policy.__c1_qmark__ = True
    return make_policy


def identity_report(template, snap, repeats=5):
    """Machine-readable parameter / computation identity of QMARK vs B2 on one snapshot (offline, CPU)."""
    import time

    from . import graph as graph_mod
    from . import tb_repctl_checks as C

    def timed(policy):
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            C.forward(policy, snap)
            times.append(time.perf_counter() - t0)
        return sorted(times)[len(times) // 2]

    b2 = C.make_policy(template, "B2", 0)
    from .platforms.libero import runtime_factory as rf
    type_set = set()
    for mapping in rf.TASK_OBJECTS.values():
        type_set.update(mapping.values())
    torch.manual_seed(0)
    qm = QmarkPolicy(sorted({c.name for c in template.contracts}), sorted(rf.PREDICATES), sorted(type_set), C.OBS_DIM, C.CAND_DIM, method=QMARK_METHOD, B=0.5)
    qm.eval()
    legal = int(sum(snap.mask))
    b2_eff, b2_mod = C.effective_parameters(b2, snap)
    qm_eff, qm_mod = C.effective_parameters(qm, snap)
    b2_comp = C.computation_counts(template, "B2", snap)
    before = qm.encoder.forward_calls
    with C.count_nominal_apply() as counter:
        C.forward(qm, snap)
    qm_comp = {"legal_candidates": legal, "encoder_forward_calls_total": qm.encoder.forward_calls - before, "encoder_forward_calls_current_state": 1,
               "encoder_forward_calls_per_candidate": (qm.encoder.forward_calls - before - 1) / legal if legal else None, "nominal_apply_calls_total": counter["calls"],
               "interaction_module_calls_total": {"query_projection": legal}}
    b2_comp = dict(b2_comp, graph_propagation_layers=len(b2.encoder.layers), forward_seconds_cpu_median=timed(b2))
    qm_comp["graph_propagation_layers"] = len(qm.encoder.layers)
    qm_comp["forward_seconds_cpu_median"] = timed(qm)
    return {"card": "CP-DISR-C1-MECH-CONFIRM-V1", "method": QMARK_METHOD,
            "b2": {"effective_trainable_parameters": b2_eff, "by_module": b2_mod, "computation": b2_comp},
            "qmark": {"effective_trainable_parameters": qm_eff, "by_module": qm_mod, "computation": qm_comp,
                      "state_dict_trainable_parameters": sum(p.numel() for p in qm.parameters()), "new_modules": {"query_projection": sum(p.numel() for p in qm.query_projection.parameters())}},
            "extra_parameters": qm_eff - b2_eff, "extra_parameters_pct_of_b2": 100.0 * (qm_eff - b2_eff) / b2_eff,
            "disclosure": "QMARK computes E(G(F,q=none)) once and E(G(F,q=a_i)) once per legal candidate (1 + K encoder forwards); B2 re-encodes the current state per candidate and the "
                          "successor per candidate (1 + 2K). Same 4 relational-graph layers, candidate readout, GRU and heads. QMARK is not strictly equal-compute and is not claimed to be."}


def register(v11):
    """Idempotent in-process registration (QMARK joins neural.METHODS and EMPTY_PRIOR_METHODS like B2 and B1-K+E; no tracked file changes)."""
    neural.METHODS = tuple(dict.fromkeys(tuple(neural.METHODS) + METHODS))
    v11.EMPTY_PRIOR_METHODS = set(v11.EMPTY_PRIOR_METHODS) | set(METHODS)
    if not getattr(v11.make_policy, "__c1_qmark__", False):
        v11.make_policy = make_qmark_policy_factory(v11, v11.make_policy)
    return v11
