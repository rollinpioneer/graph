"""Independent task-cost oracle. Does not read V1/V2 reward values."""
from __future__ import annotations
from collections import deque
from p2cq_research.environment import SkillEnv
from p2cq_research.task_contract import canonical

ATOL = 1e-10
RTOL = 1e-9


def _close(a, b):
    return abs(a - b) <= max(ATOL, RTOL * max(abs(a), abs(b)))


def local_facts(env):
    return canonical(env.nongraph_public())


def remaining_cost(env, max_states=200000, cache=None):
    if env.contract.goal_satisfied(env.state.valid):
        return 0.0, 1
    key = (env.contract.task_hash(), env.state.fingerprint())
    if cache is not None and key in cache:
        return cache[key], 0
    start = env.snapshot()
    q = deque([(start, 0)])
    seen = {env.state.fingerprint()}
    expanded = 0
    while q:
        snap, dist = q.popleft()
        env.restore(snap)
        env.state.t = 0
        env._terminated = False
        expanded += 1
        if expanded > max_states:
            env.restore(start)
            return float("inf"), expanded
        parent = env.snapshot()
        mask = env.legal_mask()
        for a in range(37):
            if not mask[a]:
                continue
            env.restore(parent)
            env.state.t = 0
            env._terminated = False
            env.step(int(a))
            if env.success():
                env.restore(start)
                rec = float(dist + 1)
                if cache is not None:
                    cache[key] = rec
                return rec, expanded
            fp = env.state.fingerprint()
            if fp in seen:
                continue
            seen.add(fp)
            q.append((env.snapshot(), dist + 1))
        env.restore(parent)
    env.restore(start)
    if cache is not None:
        cache[key] = float("inf")
    return float("inf"), expanded


def action_q(env, max_states=200000, cache=None):
    snap = env.snapshot()
    mask = env.legal_mask()
    qs = {}
    for a in range(37):
        if not mask[a]:
            continue
        env.restore(snap)
        env.state.t = 0
        env._terminated = False
        env.step(int(a))
        if env.success():
            qs[a] = 1.0
        else:
            cost, _ = remaining_cost(env, max_states=max_states, cache=cache)
            qs[a] = float("inf") if cost == float("inf") else 1.0 + cost
    env.restore(snap)
    return qs, mask


def conservative_optimal(qs):
    finite = {a: q for a, q in qs.items() if q != float("inf")}
    if not finite:
        return []
    qmin = min(finite.values())
    return sorted(a for a, q in finite.items() if _close(q, qmin))


def shortest_from_initial(contract, max_states=200000, cache=None):
    env = SkillEnv(contract)
    env.reset()
    if contract.goal_satisfied(env.state.valid):
        return 0.0, 0
    cost, expanded = remaining_cost(env, max_states=max_states, cache=cache)
    return cost, expanded
