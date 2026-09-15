from __future__ import annotations
from collections import defaultdict
from typing import Any

def ess(weights):
    s = sum(weights); q = sum(w*w for w in weights)
    return 0.0 if q == 0 else (s*s)/q

def signed_vs_weight(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    g = defaultdict(list)
    for r in details:
        g[(r["episode_id"], r["method"])].append(r)
    out = []
    for (eid, method), rows in sorted(g.items()):
        rewards = [float(r["reward_mu"]) for r in rows]
        w = [max(0.0, x) for x in rewards]
        rec = [float(r["reward_mu"]) for r in rows if r.get("edge_type") == "recovery" and float(r["reward_mu"]) > 0]
        out.append(dict(episode_id=eid, method=method, signed_return=sum(rewards),
                        positive_weight_sum=sum(w), ess=ess(w),
                        recovery_positive_share=(sum(rec)/sum(w) if sum(w) else 0.0),
                        n=len(rows), policy_harm_claimed=False))
    return out