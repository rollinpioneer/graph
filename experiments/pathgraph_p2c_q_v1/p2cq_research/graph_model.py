"""Independent nominal PathGraph model. Does not call environment.step or oracle."""
from __future__ import annotations

import heapq

from .task_contract import N, TaskContract, dnf_satisfied

BIN_ZERO, BIN_MID, BIN_DONE = 0, 1, 2


def progress_bin(progress, k):
    if progress <= 0:
        return BIN_ZERO
    if k > 0 and progress >= k:
        return BIN_DONE
    return BIN_MID


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


def abstract_from_dynamic(contract: TaskContract, dyn):
    nodes = dyn["nodes"]
    valid = pack_bits(nodes[i]["valid"] for i in range(N))
    lost = pack_bits(nodes[i]["open_loss"] for i in range(N))
    rec = pack_bits(nodes[i]["recovering"] for i in range(N))
    inv = pack_bits(nodes[i]["invalidated"] for i in range(N))
    bins = []
    for i in range(N):
        if not contract.node_mask[i]:
            bins.append(BIN_ZERO)
        else:
            bins.append(progress_bin(nodes[i]["progress"], contract.transport_steps[i]))
    return (
        valid, int(dyn["held_id"]), lost, rec, inv,
        1 if dyn["disturbance_consumed"] else 0, tuple(bins),
    )


def _invalidate(contract, valid, seeds):
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
            stack.append(j)
            flags = [bit_on(valid, k) for k in range(N)]
    return valid


def _neighbors(contract: TaskContract, state):
    valid, held, lost, rec, inv, dist, bins = state
    bins = list(bins)
    present = contract.present_nodes()
    any_held = held >= 0
    out = []

    def emit(cost, nstate):
        out.append((cost, nstate))

    # WAIT omitted from model (zero task progress); model uses skill edges only.
    for i in present:
        k = contract.transport_steps[i]
        # ACQUIRE
        if (not any_held) and (not bit_on(lost, i)):
            nv = valid
            seeds = []
            if bit_on(valid, i):
                nv = set_bit(valid, i, False)
                seeds.append(i)
            if seeds:
                nv = _invalidate(contract, nv, seeds)
            emit(1, (nv, i, lost, rec, inv, dist, tuple(bins)))
        # nominal transport
        if held == i and bins[i] != BIN_DONE:
            nb = list(bins)
            nb[i] = BIN_DONE
            emit(k, (valid, held, lost, rec, inv, dist, tuple(nb)))
        # PLACE
        if held == i and bins[i] == BIN_DONE:
            flags = [bit_on(valid, k2) for k2 in range(N)]
            nv = valid
            ninv = inv
            if dnf_satisfied(contract.preconditions_dnf[i], contract.precondition_clause_mask[i], flags):
                nv = set_bit(valid, i, True)
            else:
                ninv = set_bit(inv, i, True)
            emit(1, (nv, -1, lost, rec, ninv, dist, tuple(bins)))
        # START_RECOVERY
        if bit_on(lost, i) and not any_held:
            emit(1, (valid, held, lost, set_bit(rec, i, True), inv, dist, tuple(bins)))
        # REGRASP
        if bit_on(lost, i) and bit_on(rec, i) and not any_held:
            emit(1, (valid, i, set_bit(lost, i, False), set_bit(rec, i, False), inv, dist, tuple(bins)))
        # RELEASE
        if held == i:
            nv = valid
            ninv = inv
            if bit_on(valid, i):
                nv = set_bit(valid, i, False)
                ninv = set_bit(inv, i, True)
                nv = _invalidate(contract, nv, [i])
            emit(1, (nv, -1, lost, rec, ninv, dist, tuple(bins)))
    return out


def goal_ok(contract, valid):
    flags = [bit_on(valid, i) for i in range(N)]
    return dnf_satisfied(contract.goal_dnf, contract.goal_clause_mask, flags)


_COST_CACHE = {}

def model_cost(contract: TaskContract, dyn, max_expands=200000):
    start = abstract_from_dynamic(contract, dyn)
    cache_key = (id(contract), start)
    cached = _COST_CACHE
    if cache_key in cached:
        return cached[cache_key]
    if goal_ok(contract, start[0]):
        cached[cache_key] = (0.0, False, 1)
        return 0.0, False, 1
    inf = float("inf")
    distm = {start: 0.0}
    heap = [(0.0, 0, start)]
    seq = 0
    expands = 0
    while heap:
        d, _, u = heapq.heappop(heap)
        if d != distm.get(u):
            continue
        if goal_ok(contract, u[0]):
            cached[cache_key] = (d, False, expands)
            return d, False, expands
        expands += 1
        if expands > max_expands:
            cached[cache_key] = (inf, True, expands)
            return inf, True, expands
        for cost, v in _neighbors(contract, u):
            nd = d + cost
            if nd < distm.get(v, inf):
                distm[v] = nd
                seq += 1
                heapq.heappush(heap, (nd, seq, v))
    cached[cache_key] = (inf, False, expands)
    return inf, False, expands


def scale_s(contract: TaskContract):
    return float(sum(contract.transport_steps[i] + 4 for i in contract.present_nodes()))


def phi_cost(contract, dyn, clip_hits=None):
    s = scale_s(contract)
    c, truncated, _ = model_cost(contract, dyn)
    if c == float("inf"):
        if clip_hits is not None:
            clip_hits["unreachable"] = clip_hits.get("unreachable", 0) + 1
        return -2.0, True
    clipped = min(c, 2.0 * s)
    hit = clipped != c
    if hit and clip_hits is not None:
        clip_hits["clip"] = clip_hits.get("clip", 0) + 1
    return -clipped / s, hit
