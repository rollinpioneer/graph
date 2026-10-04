"""Zero-training macro scripts for ALFWorld pick_and_place_simple (spec 4.2).

All scripts see only PublicState (+ frozen prior for Fixed Prior). Belief-Optimal uses the
train-split belief of the same-observation Q_ref; it is NOT a clairvoyant oracle.
"""
import random

from ..analysis.alfworld_qref import best_action, q_ref
from ..platforms.alfworld.adapter import AlfEpisode
from ..platforms.alfworld.observation import type_of


def natural_key(name):
    t, i = name.rsplit(" ", 1)
    return (t, int(i))


def _finish(pub):
    """Non-search macro when one is legal (TAKE/PUT), else None."""
    rest = sorted((a for a in pub.legal if a[0] != "CHECK"), key=lambda a: (a[0], natural_key(a[1])))
    return rest[0] if rest else None


def random_legal(pub, ctx):
    return ctx["rng"].choice(sorted(pub.legal, key=lambda a: (a[0], natural_key(a[1]))))


def contract_search(pub, ctx):
    f = _finish(pub)
    if f:
        return f
    return min((a for a in pub.legal if a[0] == "CHECK"), key=lambda a: natural_key(a[1]))


def fixed_prior_search(pub, ctx):
    f = _finish(pub)
    if f:
        return f
    s = ctx["prior"].scores(pub, ctx["tables"])
    return min((a for a in pub.legal if a[0] == "CHECK"), key=lambda a: (-s[a[1]], natural_key(a[1])))


def belief_optimal(pub, ctx):
    return best_action(ctx["tables"], pub)


SCRIPTS = {
    "random_legal": random_legal,
    "contract_search": contract_search,
    "fixed_prior": fixed_prior_search,
    "belief_optimal": belief_optimal,
}


def run_episode(gamefile, tables, policy, prior=None, seed=0, record_states=False):
    ctx = {"tables": tables, "prior": prior, "rng": random.Random(seed)}
    ep = AlfEpisode(gamefile, tables)
    trace = []
    try:
        while not ep.done:
            pub = ep.public()
            action = policy(pub, ctx)
            n_check = sum(1 for a in pub.legal if a[0] == "CHECK")
            res = ep.execute(*action)
            row = {"raw_steps": pub.raw_steps, "n_legal": len(pub.legal), "n_check_legal": n_check,
                   "action": list(action), "duration": res.duration, "status": res.status}
            if record_states:
                row["pub"] = pub
            trace.append(row)
        T = ep.success_step
        return {"gamefile": gamefile, "won": ep.won, "raw_steps": ep.raw_steps, "J": 2.0 ** (-T / 50.0) if ep.won else 0.0,
                "n_checks": sum(1 for r in trace if r["action"][0] == "CHECK"), "trace": trace,
                "macros": [tuple(r["action"]) for r in trace]}
    finally:
        ep.close()
