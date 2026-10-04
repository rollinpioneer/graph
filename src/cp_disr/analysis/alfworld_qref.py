"""Same-observation Q_ref for ALFWorld pick_and_place_simple (spec 5.2).

Belief: train-split class frequencies (Laplace alpha, frozen) split equally over remaining
same-class instances of feasible, unchecked receptacles. Continuation mu_ref: greedy by the
exchange-optimal index  w_r * g^{c_r} / (1 - g^{c_r})  with the belief recomputed after every
miss. Costs are raw-command counts: CHECK = 1 + p_open(type) (train mean), TAKE = 1,
PUT = 2 + p_open(goal type). No hidden state, no learned network is read.
"""
from ..platforms.alfworld.data import instance_belief
from ..platforms.alfworld.observation import type_of

H = 50.0


def gamma(steps):
    return 2.0 ** (-steps / H)


def check_cost(tables, r):
    return 1.0 + tables.p_open(type_of(r))


def finish_cost(tables, pub, holding=False):
    """Raw steps from 'target seen where the agent stands' (or holding) to success."""
    put = 2.0 + (tables.p_open(type_of(pub.goal_instance)) if pub.goal_instance else 0.0)
    return put if holding else 1.0 + put


def _index(w, c):
    g = gamma(c)
    return w * g / (1.0 - g)


def greedy_order(tables, pub, first=None):
    """Visit order of unchecked receptacles: `first` (if given) then belief-index greedy."""
    remaining = list(pub.unchecked)
    order = []
    if first is not None:
        order.append(first)
        remaining.remove(first)
    while remaining:
        b = instance_belief(tables, pub.goal_otype, remaining)
        best = max(remaining, key=lambda r: (_index(b[r], check_cost(tables, r)), r))
        order.append(best)
        remaining.remove(best)
    return order


def belief(tables, pub):
    return instance_belief(tables, pub.goal_otype, pub.unchecked)


def q_ref(tables, pub):
    """Q_ref(h, a) for every legal macro; deterministic given the public state."""
    out = {}
    if pub.done:
        return out
    t0 = pub.raw_steps
    fin = finish_cost(tables, pub)
    for kind, target in pub.legal:
        if kind == "PUT":
            T = t0 + finish_cost(tables, pub, holding=True)
            out[(kind, target)] = gamma(T) if T <= pub.max_steps else 0.0
        elif kind == "TAKE":
            T = t0 + fin
            out[(kind, target)] = gamma(T) if T <= pub.max_steps else 0.0
    checks = [t for k, t in pub.legal if k == "CHECK"]
    if checks:
        b = belief(tables, pub)
        for a in checks:
            order = greedy_order(tables, pub, first=a)
            t = t0
            value = 0.0
            for r in order:
                t += check_cost(tables, r)
                T = t + fin
                value += b[r] * (gamma(T) if T <= pub.max_steps else 0.0)
            out[("CHECK", a)] = value
    return out


def best_action(tables, pub):
    q = q_ref(tables, pub)
    return max(q, key=lambda a: (q[a], tuple(reversed(a)))) if q else None
