"""CP-DISR-TB-REP-CONTROLS-01 representation-control policies (additive; ``neural.py`` is byte-identical to the base commit).

``RepctlPolicy`` subclasses the production ``neural.Policy``. Both controls share the production parameter owner (graph
encoder, observation/GRU, candidate embedding, ``contract`` candidate readout, base/value/Q heads) and differ from B2 only
in what is handed to the shared candidate readout:

* ``B2-ABS``  (Absolute Successor)        rows_i = E(F_after_i)   with F_after_i from the unchanged four_views / nominal_apply
* ``B1-K+NC`` (Matched Neural Composition) rows_i = StateEffectInteraction(F, K(a_i)); no nominal_apply, no F_after_i

Legacy methods are never routed here: ``make_control_policy`` is bound only for the two control names.
"""
from __future__ import annotations

import torch
from torch import nn

from . import neural
from .graph import view
from .neural import Policy, PolicyOutput

CONTROL_METHODS = ("B2-ABS", "B1-K+NC")


class StateEffectInteraction(nn.Module):
    """Matched Neural Composition: a learned (F, K(a_i)) -> candidate rows interaction.

    Reads only the current state F (through the shared NodeFeatures embedding of each grounded atom, which carries
    that atom's current TRUE/FALSE/UNKNOWN code) and the grounded contract K(a_i): action, PRE_POS, PRE_NEG, ADD,
    DEL, UNKNOWN and conditional guards / conditional effects with their conditional index. It never applies the
    contract: there is no nominal overlay, no successor view and no file or cache read.
    """

    ROLES = (
        "ACTION", "PRE_POS", "PRE_NEG", "ADD", "DEL", "UNKNOWN",
        "GUARD_POS", "GUARD_NEG", "COND_ADD", "COND_DEL", "COND_UNKNOWN",
    )
    MAX_CONDITIONALS = 8

    def __init__(self, dim=128):
        super().__init__()
        self.role = nn.Embedding(len(self.ROLES), dim)
        self.conditional = nn.Embedding(self.MAX_CONDITIONALS, dim)
        self.interact = nn.Linear(2 * dim, dim)
        self.forward_calls = 0

    def tokens(self, node_h, node_index, contract):
        rows = []

        def add(role, node_id, conditional_index=None):
            row = node_h[node_index[node_id]] + self.role.weight[self.ROLES.index(role)]
            if conditional_index is not None:
                row = row + self.conditional.weight[min(conditional_index, self.MAX_CONDITIONALS - 1)]
            rows.append(row)

        add("ACTION", contract.id)
        for atoms, role in (
            (contract.pre_pos, "PRE_POS"), (contract.pre_neg, "PRE_NEG"),
            (contract.effects.add, "ADD"), (contract.effects.delete, "DEL"), (contract.effects.unknown, "UNKNOWN"),
        ):
            for atom in atoms:
                add(role, atom.id)
        for index, conditional in enumerate(contract.conditional):
            for atoms, role in (
                (conditional.positive, "GUARD_POS"), (conditional.negative, "GUARD_NEG"),
                (conditional.effects.add, "COND_ADD"), (conditional.effects.delete, "COND_DEL"),
                (conditional.effects.unknown, "COND_UNKNOWN"),
            ):
                for atom in atoms:
                    add(role, atom.id, index)
        return torch.stack(rows)

    def forward(self, node_h, node_index, contract):
        self.forward_calls += 1
        tokens = self.tokens(node_h, node_index, contract)
        pooled = tokens.mean(0, keepdim=True).expand_as(tokens)
        return torch.relu(self.interact(torch.cat((tokens, pooled), dim=-1))), int(tokens.shape[0])


class RepctlPolicy(Policy):
    """B2-ABS or B1-K+NC on the production Policy parameter owner."""

    def __init__(self, actions, predicates, types, observation_dim, candidate_dim, method="B2-ABS", B=0.5):
        if method not in CONTROL_METHODS:
            raise ValueError(method)
        # Every legacy module is built by the production constructor first, so the RNG stream (and therefore the
        # initial weights of all shared modules at a given seed) is identical to B2 / B1-K / B1-K+E.
        super().__init__(actions, predicates, types, observation_dim, candidate_dim, method="B1-K", B=B)
        self.method = method
        if method == "B1-K+NC":
            self.state_effect = StateEffectInteraction()

    def forward(self, snapshot, hidden=None):
        method = self.method
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
        zk = self.encoder(k)
        node_h = node_index = None
        if method == "B1-K+NC":
            node_h = self.encoder.features(k)
            node_index = {v: i for i, v in enumerate(snapshot.template.node_ids)}
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
            context = torch.cat((zo, ca, zk.mean(0)))
            if method == "B2-ABS":
                # Absolute Successor: the B2 call pattern (same four_views / nominal_apply / encoder calls) but the rows
                # are E(F_after_i); the latent difference E(F_after_i) - E(F) is never formed as the consequence.
                delta = neural.differences(self.encoder, neural.four_views(snapshot.template, facts, (), contracts[cid]))
                successor_used = True
                uk = self._phi_k(context, delta.encodings[2].float())
                prior_input = torch.zeros_like(delta.dk)
                diagnostics["differences"][cid] = delta
            else:
                # Matched Neural Composition: no nominal_apply, no F_after, no successor view.
                rows, token_count = self.state_effect(node_h, node_index, contracts[cid])
                uk = self._phi_k(context, rows)
                prior_input = torch.zeros_like(zk)
                diagnostics["effect_tokens"][cid] = token_count
            up, residual = self.prior(torch.cat((context, uk)), prior_input, self.B, False)
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
        zpool = zk.mean(0)
        value = self.v_head(torch.cat((zo, zpool, mean_k, mean_p))).squeeze(-1)
        q = self.q_head(torch.cat((uk, up, zo.expand(len(uk), -1), mean_k.expand(len(uk), -1), mean_p.expand(len(uk), -1)), dim=-1)).squeeze(-1)
        logits = torch.stack(logits).masked_fill(~mask, -torch.inf)
        return PolicyOutput(snapshot.candidate_ids, logits, mask, torch.distributions.Categorical(logits=logits), value, q, zo, diagnostics)


def policy_class(method):
    return RepctlPolicy if method in CONTROL_METHODS else Policy


def make_control_policy(v11, original_make_policy):
    """A `make_policy` that builds RepctlPolicy for the two controls and defers to the production factory otherwise."""
    def make_policy(template, method, device):
        if method not in CONTROL_METHODS:
            return original_make_policy(template, method, device)
        from .platforms.libero.runtime_factory import PREDICATES, TASK_OBJECTS
        actions = sorted({c.name for c in template.contracts})
        type_set = set()
        for mapping in TASK_OBJECTS.values():
            type_set.update(mapping.values())
        model = RepctlPolicy(actions=actions, predicates=sorted(PREDICATES), types=sorted(type_set), observation_dim=v11.OBS_DIM,
                             candidate_dim=v11.CAND_DIM, method=method, B=v11.B_PRIOR)
        return model.to(device)
    make_policy.__wrapped_production__ = original_make_policy
    return make_policy
