"""CP-DISR-C1-CROSS-TASK-BLOCKSWORLD-V1 frozen outcome rules (runbook 13.4, 14, 15) with amendment A01 (hop > 4 is structurally empty).

Amendment A01 (user decision, recorded before any training): HOP_GT4_STATUS = STRUCTURALLY_EMPTY, RECEPTIVE_FIELD_STATUS = NOT_TESTABLE_STRUCTURALLY_EMPTY. The hop quota, the H5 experiment and the
LIMIT_RECEPTIVE_FIELD label are removed; every decision is a hop <= 4 decision, so D_core is the plain A1 / A2 decision-perfect rate. All thresholds are pre-registered decision rules, not
significance tests. Pure python: no torch.
"""
from __future__ import annotations

EPS = 1e-9
METHODS = ("B2", "QMARK", "ASNET")

RULES = {
    "id_gate": {"id_dev_success_min": 33, "id_dev_decision_perfect_min": 33, "id_dev_n": 36, "nan": 0, "hard_failure": 0, "else": "PPO_ID_GATE_FAIL"},
    "a0_gate": {"decision_perfect_min": 0.90, "else": "EQUIVARIANCE_OR_BINDING_GATE_FAIL"},
    "D_core": "mean of the A1 and A2 episode decision-perfect rates (all decisions are hop <= 4 by amendment A01)",
    "REP_SUCCESSOR_ADVANTAGE": {"D_core_B2_minus_max_other_min": 0.15, "each_slice_B2_lead_min": 0.10},
    "REP_QUERY_ADVANTAGE": {"D_core_QMARK_minus_max_other_min": 0.15, "each_slice_QMARK_lead_min": 0.10},
    "REP_CANDIDATE_CONDITIONING_NEEDED": {"abs_D_core_B2_minus_QMARK_max": 0.10, "min_D_core_B2_QMARK_minus_ASNET_min": 0.15},
    "REP_DIRECT_READOUT_SUFFICIENT": {"D_core_ASNET_minus_max_other_min": -0.10, "slice_behind_best_max": 0.15},
    "REP_MIXED": "none of the above",
    "LIMIT_COORDINATION": {"slice": "A2 coordination_slice cases", "D_each_method_below": 0.50, "coordination_type_first_divergence_share_min": 0.60},
    "LIMIT_TRAINING": "ID gate failed (no OOD interpretation)",
    "LIMIT_NONE": {"min_D_core_over_methods_min": 0.70},
    "LIMIT_MIXED": "otherwise",
    "LIMIT_RECEPTIVE_FIELD": "REMOVED by amendment A01 (hop > 4 structurally empty): status NOT_TESTABLE_STRUCTURALLY_EMPTY",
}
INTERPRETATIONS = {
    "coordination_type_first_divergence": "operational definition fixed before any result: a first divergence is of coordination type when (a) the diverging action itself destroys a satisfied goal atom, or (b) the episode contains a repeated "
                                          "(state, action) pair, or (c) the episode destroys a satisfied goal atom at any step (covers wrong goal order that undoes completed work)",
    "LIMIT_NONE": "'no obvious common failure' is fixed as every method having D_core >= 0.70",
    "pooling": "one seed per method; the unit of evidence is the case, with the caveat that a deterministic policy on one seed is not an independent-seed estimate",
}


def _ge(a, b):
    return a >= b - EPS


def id_gate(success_n, perfect_n, nan, hard_fail):
    g = RULES["id_gate"]
    return success_n >= g["id_dev_success_min"] and perfect_n >= g["id_dev_decision_perfect_min"] and nan == 0 and not hard_fail


def a0_gate(decision_perfect_rate):
    return decision_perfect_rate >= RULES["a0_gate"]["decision_perfect_min"] - EPS


def d_core(a1, a2):
    return (a1 + a2) / 2.0


def representation_label(D):
    """D: {method: {'A1': rate, 'A2': rate}} for B2, QMARK, ASNET (episode decision-perfect rates)."""
    core = {m: d_core(D[m]["A1"], D[m]["A2"]) for m in METHODS}

    def lead(m, others, sl):
        return D[m][sl] - max(D[o][sl] for o in others)
    others_b2 = ("QMARK", "ASNET")
    others_q = ("B2", "ASNET")
    facts = {"D_core": core, "slice_rates": D}
    if _ge(core["B2"] - max(core[o] for o in others_b2), 0.15) and all(_ge(lead("B2", others_b2, s), 0.10) for s in ("A1", "A2")):
        return "REP_SUCCESSOR_ADVANTAGE", facts
    if _ge(core["QMARK"] - max(core[o] for o in others_q), 0.15) and all(_ge(lead("QMARK", others_q, s), 0.10) for s in ("A1", "A2")):
        return "REP_QUERY_ADVANTAGE", facts
    if abs(core["B2"] - core["QMARK"]) <= 0.10 + EPS and _ge(min(core["B2"], core["QMARK"]) - core["ASNET"], 0.15):
        return "REP_CANDIDATE_CONDITIONING_NEEDED", facts
    best = {s: max(D[m][s] for m in METHODS) for s in ("A1", "A2")}
    if _ge(core["ASNET"], max(core["B2"], core["QMARK"]) - 0.10) and all(best[s] - D["ASNET"][s] <= 0.15 + EPS for s in ("A1", "A2")):
        return "REP_DIRECT_READOUT_SUFFICIENT", facts
    return "REP_MIXED", facts


def limitation_label(id_ok, D, coordination_D, coordination_div_share, a2_all=None):
    """coordination_D: {method: D on A2 coordination-slice cases}; coordination_div_share: share of first divergences (pooled over the three methods) of coordination type."""
    if not id_ok:
        return "LIMIT_TRAINING", {}
    core = {m: d_core(D[m]["A1"], D[m]["A2"]) for m in METHODS}
    facts = {"coordination_D": coordination_D, "coordination_type_share": coordination_div_share, "D_core": core}
    if all(coordination_D[m] < 0.50 - EPS for m in METHODS) and coordination_div_share is not None and _ge(coordination_div_share, 0.60):
        return "LIMIT_COORDINATION", facts
    if all(core[m] >= 0.70 - EPS for m in METHODS):
        return "LIMIT_NONE", facts
    return "LIMIT_MIXED", facts


MAINLINE = {
    "REP_SUCCESSOR_ADVANTAGE": "explicit successor as the algorithm mainline; study its extra role in multi-step composition",
    "REP_QUERY_ADVANTAGE": "query propagation as the main implementation candidate; tighten the difference to the ASNet-style readout",
    "REP_CANDIDATE_CONDITIONING_NEEDED": "develop a unified candidate-conditioned relational computation; do not rush to choose between B2 and QMARK",
    "REP_DIRECT_READOUT_SUFFICIENT": "do not package QMARK as a new algorithm; paper follows unified explanation, diagnostic protocol and conditional comparison",
    "REP_MIXED": "do not widen the matrix; locate one minimal decidable question from the existing logs",
}
LIMIT_MAINLINE = {
    "LIMIT_COORDINATION": "a minimal new algorithm for goal preservation / ordering is allowed",
    "LIMIT_TRAINING": "decide first whether to switch all methods to planner imitation; do not interpret OOD",
    "LIMIT_NONE": "no common failure; no new algorithm indicated by this card",
    "LIMIT_MIXED": "failure sources differ or evidence is insufficient; do not widen the matrix",
}
