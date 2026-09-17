"""PATHGRAPH_REMAINING_WORK_PBRS_V2 planner.

Public-contract remaining-work model. Exact integer progress, unit skill
cost, public one-shot disturbance after each modeled skill edge. No
environment/oracle/MDP imports. Cache keys use task_hash, never id().
"""
from __future__ import annotations

import heapq

from p2cq_research.task_contract import N, TaskContract, dnf_satisfied

class UnreachableWorkError(RuntimeError):
    pass

class SearchTruncatedError(RuntimeError):
    pass


def pack_bits(flags):
    x = 0
    for i, v in enumerate(flags):
        if v:
            x |= 1 << i
    return x


def bit_on(x, i):
    return bool(x >> i & 1)


def set_bit(x, i, val):
    if val:
        return x | (1 << i)
    return x & ~(1 << i)


def scale_s(contract: TaskContract) -> float:
    return float(sum(contract.transport_steps[i] + 4 for i in contract.present_nodes()))


def goal_ok(contract, valid_bits):
    flags = [bit_on(valid_bits, i) for i in range(N)]
    return dnf_satisfied(contract.goal_dnf, contract.goal_clause_mask, flags)


def abstract_from_dynamic(contract: TaskContract, dyn):
    nodes = dyn["nodes"]
    progress = []
    for i in range(N):
        if not contract.node_mask[i]:
            progress.append(0)
        else:
            p = int(nodes[i]["progress"])
            k = contract.transport_steps[i]
            if p < 0 or p > k:
                raise ValueError("progress bounds")
            progress.append(p)
    return (
        pack_bits(nodes[i]["valid"] for i in range(N)),
        int(dyn["held_id"]),
        pack_bits(nodes[i]["open_loss"] for i in range(N)),
        pack_bits(nodes[i]["recovering"] for i in range(N)),
        pack_bits(nodes[i]["invalidated"] for i in range(N)),
        1 if dyn["disturbance_consumed"] else 0,
        tuple(progress),
    )


def _invalidate(contract, valid, invalidated, seeds):
    stack = list(seeds)
    seen = set()
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        flags = [bit_on(valid, k) for k in range(N)]
        for j in contract.present_nodes():
            if not contract.invalidation_dependency[i][j]:
                continue
            if not bit_on(valid, j):
                continue
            if dnf_satisfied(contract.preconditions_dnf[j], contract.precondition_clause_mask[j], flags):
                continue
            valid = set_bit(valid, j, False)
            invalidated = set_bit(invalidated, j, True)
            stack.append(j)
            flags = [bit_on(valid, k) for k in range(N)]
    return valid, invalidated


def _apply_disturbance(contract, state):
    valid, held, lost, rec, inv, dist, prog = state
    rule = contract.disturbance_rule
    if rule.get("kind") != "ONE_SHOT" or dist:
        return state
    trig = rule["trigger_node"]
    if prog[trig] < rule["trigger_progress"]:
        return state
    tgt = rule["target_node"]
    seeds = []
    if bit_on(valid, tgt):
        seeds.append(tgt)
    if held == tgt:
        held = -1
    valid = set_bit(valid, tgt, False)
    lost = set_bit(lost, tgt, True)
    rec = set_bit(rec, tgt, False)
    prog = list(prog)
    prog[tgt] = prog[tgt] // 2
    dist = 1
    if seeds:
        valid, inv = _invalidate(contract, valid, inv, seeds)
    return (valid, held, lost, rec, inv, dist, tuple(prog))


def _neighbors(contract: TaskContract, state):
    valid, held, lost, rec, inv, dist, prog = state
    prog = list(prog)
    present = contract.present_nodes()
    any_held = held >= 0
    out = []

    def emit(nstate):
        out.append(_apply_disturbance(contract, nstate))

    for i in present:
        k = contract.transport_steps[i]
        if (not any_held) and (not bit_on(lost, i)):
            nv, ninv = valid, inv
            seeds = []
            if bit_on(valid, i):
                nv = set_bit(valid, i, False)
                seeds.append(i)
            if seeds:
                nv, ninv = _invalidate(contract, nv, ninv, seeds)
            emit((nv, i, lost, rec, ninv, dist, tuple(prog)))
        if held == i and prog[i] < k:
            nb = list(prog)
            nb[i] = prog[i] + 1
            emit((valid, held, lost, rec, inv, dist, tuple(nb)))
        if held == i and prog[i] == k:
            flags = [bit_on(valid, k2) for k2 in range(N)]
            nv, ninv = valid, inv
            if dnf_satisfied(contract.preconditions_dnf[i], contract.precondition_clause_mask[i], flags):
                nv = set_bit(valid, i, True)
            else:
                ninv = set_bit(inv, i, True)
            emit((nv, -1, lost, rec, ninv, dist, tuple(prog)))
        if bit_on(lost, i) and not any_held:
            emit((valid, held, lost, set_bit(rec, i, True), inv, dist, tuple(prog)))
        if bit_on(lost, i) and bit_on(rec, i) and not any_held:
            emit((valid, i, set_bit(lost, i, False), set_bit(rec, i, False), inv, dist, tuple(prog)))
        if held == i:
            nv, ninv = valid, inv
            if bit_on(valid, i):
                nv = set_bit(valid, i, False)
                ninv = set_bit(inv, i, True)
                nv, ninv = _invalidate(contract, nv, ninv, [i])
            emit((nv, -1, lost, rec, ninv, dist, tuple(prog)))
    return out


class RemainingWorkPlanner:
    def __init__(self, contract: TaskContract, max_expands=200000):
        if not isinstance(contract, TaskContract):
            contract = TaskContract(contract)
        self.contract = contract
        self.task_hash = contract.task_hash()
        self.max_expands = int(max_expands)
        self._cost_cache = {}
        self.contract_summary = {
            "task_hash": self.task_hash,
            "family_id": contract.family_id,
            "motif": contract.motif,
            "split": contract.split,
        }

    def abstract_state(self, dynamic_public):
        return abstract_from_dynamic(self.contract, dynamic_public)

    def neighbors(self, state):
        return _neighbors(self.contract, state)

    def remaining_cost(self, state, *, use_cache=True):
        key = (self.task_hash, state)
        if use_cache and key in self._cost_cache:
            return self._cost_cache[key]
        c = self.contract
        if goal_ok(c, state[0]):
            rec = (0.0, False, 1)
            if use_cache:
                self._cost_cache[key] = rec
            return rec
        inf = float("inf")
        distm = {state: 0.0}
        heap = [(0.0, 0, state)]
        seq = 0
        expands = 0
        while heap:
            d, _, u = heapq.heappop(heap)
            if d != distm.get(u):
                continue
            if goal_ok(c, u[0]):
                rec = (d, False, expands)
                if use_cache:
                    self._cost_cache[key] = rec
                return rec
            expands += 1
            if expands > self.max_expands:
                rec = (inf, True, expands)
                if use_cache:
                    self._cost_cache[key] = rec
                return rec
            for v in _neighbors(c, u):
                nd = d + 1.0
                if nd < distm.get(v, inf):
                    distm[v] = nd
                    seq += 1
                    heapq.heappush(heap, (nd, seq, v))
        rec = (inf, False, expands)
        if use_cache:
            self._cost_cache[key] = rec
        return rec

    def potential(self, dynamic_public, *, use_cache=True):
        state = self.abstract_state(dynamic_public)
        cost, truncated, expands = self.remaining_cost(state, use_cache=use_cache)
        if truncated:
            raise SearchTruncatedError(f"search truncated expands={expands}")
        if cost == float("inf"):
            raise UnreachableWorkError("finite remaining work unavailable")
        s = scale_s(self.contract)
        return -cost / s, cost, expands


def phi_remaining_work(contract, dyn, planner=None, use_cache=True):
    if planner is None:
        planner = RemainingWorkPlanner(contract)
    phi, cost, expands = planner.potential(dyn, use_cache=use_cache)
    return phi, cost, expands