"""Policy snapshot for ALFWorld: built from PublicState + frozen prior only."""
import math

from ...common import digest
from ...rl import Snapshot
from ...contracts import precondition_value
from ...facts import Truth
from .pddl_contracts import GOAL_FACT, KINDS, candidate_key, episode_template, fact_store, facts_from_public

BASE_DIM = 8
CAND_DIM = len(KINDS)
SOFT = "SOFT_RELEVANT_TO_GOAL"


def prior_edges_for(template, scores):
    """Weighted soft edges CHECK(r) -> goal; only strictly positive instance scores become edges."""
    edges = []
    for c in template.contracts:
        kind, r = candidate_key(c)
        if kind == "CHECK" and scores.get(r, 0.0) > 0.0:
            edges.append((c.id, GOAL_FACT, SOFT, round(float(scores[r]), 6)))
    return tuple(sorted(edges))


def base_input(pub):
    n = max(1, len(pub.feasible))
    u = len(pub.unchecked)
    return (
        pub.raw_steps / pub.max_steps,
        (pub.max_steps - pub.raw_steps) / pub.max_steps,
        u / n,
        (n - u) / n,
        float(pub.target_found),
        float(pub.holding is not None),
        float(pub.placed),
        math.log1p(u) / 4.0,
    )


def build_snapshot(pub, prior_edges, env_id, episode_id, decision_id):
    template = episode_template(pub.feasible, pub.goal_instance)
    values = facts_from_public(template, pub)
    facts = fact_store(values, float(pub.raw_steps))
    cids = tuple(c.id for c in template.contracts)
    legal = set(pub.legal)
    mask = tuple(candidate_key(c) in legal for c in template.contracts)
    feats = tuple(tuple(float(c.name == k) for k in KINDS) for c in template.contracts)
    return Snapshot(
        env_id=env_id, episode_id=episode_id, decision_id=decision_id, template=template, facts=facts,
        candidate_ids=cids, mask=mask, prior_edges=prior_edges, prior_hash=digest(prior_edges),
        base_input=base_input(pub), candidate_features=feats,
        observation_ref=digest([pub.raw_steps, pub.checked, pub.holding, pub.placed]),
        clock_seconds=float(pub.raw_steps),
    )


def mask_matches_preconditions(snapshot):
    """Structural check: the legal mask equals 'contract precondition is TRUE' on the public facts."""
    contracts = {c.id: c for c in snapshot.template.contracts}
    values = snapshot.facts.values
    expected = tuple(precondition_value(contracts[i], values) == Truth.TRUE for i in snapshot.candidate_ids)
    return expected == snapshot.mask
