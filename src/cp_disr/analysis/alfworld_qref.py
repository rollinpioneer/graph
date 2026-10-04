"""Same-observation Q_ref for ALFWorld pick_and_place_simple (multi-holder revision).

Belief: every unchecked, canContain-feasible receptacle instance r independently holds a target with
probability h_r = holder_rate(target class, class of r) (train-split counts, shrunk to the class marginal,
frozen tau). Hypotheses are the non-empty holder sets (the target exists somewhere), so this replaces the
earlier mutually-exclusive single-location assumption, which is wrong for the roughly half of all games that
have several holder receptacles.

Because success is reached at the FIRST holder visited, the value of a visiting order r_1..r_m is
  sum_k  prod_{j<k}(1 - h_j) * h_k / Z  *  2^{-(t0 + cumulative check cost_k + finish)/50},  Z = 1 - prod_j (1 - h_j),
and the exchange-optimal order sorts by  h_r g^{c_r} / (1 - g^{c_r})  (g = 2^{-1/50}); beliefs about other
receptacles do not change after a miss, so the order is static. mu_ref = that order; Q_ref(h, CHECK a) is the
value of visiting a first and then mu_ref. Costs are raw-command counts: CHECK = 1 + p_open(class) (train
mean), TAKE = 1, PUT = 2 + p_open(goal class). No hidden state and no learned network is read.
"""
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


def holder_probs(tables, pub):
    return {r: tables.holder_rate(pub.goal_otype, type_of(r)) for r in pub.unchecked}


def belief(tables, pub):
    """P(r holds a target | at least one unchecked receptacle does): the marginal used for calibration."""
    h = holder_probs(tables, pub)
    miss = 1.0
    for v in h.values():
        miss *= 1.0 - v
    z = 1.0 - miss
    return {r: min(1.0, v / z) for r, v in h.items()} if z > 0 else {}


def _index(h, c):
    g = gamma(c)
    return h * g / (1.0 - g)


def greedy_order(tables, pub, first=None):
    """mu_ref visiting order of the unchecked receptacles (static, exchange-optimal)."""
    h = holder_probs(tables, pub)
    rest = sorted((r for r in h if r != first), key=lambda r: (-_index(h[r], check_cost(tables, r)), r))
    return ([first] if first is not None else []) + rest


def order_value(tables, pub, order, h=None):
    """Expected start-discounted return of visiting `order` then finishing; hypotheses are non-empty holder sets."""
    h = holder_probs(tables, pub) if h is None else h
    miss_all = 1.0
    for r in order:
        miss_all *= 1.0 - h[r]
    z = 1.0 - miss_all
    if z <= 0:
        return 0.0
    fin = finish_cost(tables, pub)
    t, prefix, value = pub.raw_steps, 1.0, 0.0
    for r in order:
        t += check_cost(tables, r)
        T = t + fin
        value += prefix * h[r] / z * (gamma(T) if T <= pub.max_steps else 0.0)
        prefix *= 1.0 - h[r]
    return value


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
        h = holder_probs(tables, pub)
        for a in checks:
            out[("CHECK", a)] = order_value(tables, pub, greedy_order(tables, pub, first=a), h)
    return out


def best_action(tables, pub):
    q = q_ref(tables, pub)
    return max(q, key=lambda a: (q[a], tuple(reversed(a)))) if q else None
