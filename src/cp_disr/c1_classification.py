"""CP-DISR-C1-MECH-CONFIRM-V1 frozen result-classification rules (runbook section 10 and 11). Pure python: no torch, no simulator.

All differences are absolute success-rate differences. T(m) is the total Fresh Confirm success rate and C(m, c) the success rate in cell c, each averaged over
the QMARK seeds that have been released (matched seeds for B2 and ABS). The three clauses the runbook leaves open are fixed here, before any Fresh Confirm
result exists, and are listed in INTERPRETATIONS.
"""
from __future__ import annotations

EPS = 1e-9
CELLS = ("IN_S", "BUF_T", "BUF_T+BUF_S", "IN_S+BUF_T")
DUAL_CELLS = ("BUF_T+BUF_S", "IN_S+BUF_T")
SINGLE_CELLS = ("IN_S", "BUF_T")

RULES = {
    "id_qualification": {"qmark_final_dev12_success": 12, "nan": 0, "hard_failure": 0, "else": "CONTROL_NOT_ID_QUALIFIED"},
    "M1_EXPLICIT_APPLICATION_SHARED_ADVANTAGE": {"T_B2_minus_T_Q_min": 0.20, "T_ABS_minus_T_Q_min": 0.15, "cells_b2_leads_q_by": 0.10, "cells_b2_min_count": 3,
                                                 "cells_abs_leads_q_by": 0.10, "cells_abs_min_count": 3},
    "M2_B2_SPECIFIC_ADVANTAGE": {"T_B2_minus_T_Q_min": 0.20, "not": "M1", "any_of": {"T_B2_minus_T_ABS_min": 0.15, "ABS_close_to_Q_abs_diff_max": 0.10,
                                                                                    "b2_advantage_concentrated_in_dual_goal_cells": {"each_dual_cell_advantage_min": 0.20, "mean_single_cell_advantage_max": 0.10}}},
    "M3_RELATIONAL_QUERY_IS_SUFFICIENT": {"T_Q_minus_T_B2_min": -0.10, "T_Q_minus_T_ABS_min": -0.10, "cells_q_trails_b2_by_more_than": 0.20, "cells_q_trails_b2_max_count": 1},
    "M4_INCONCLUSIVE": {"definition": "none of M1, M2, M3"},
    "DIFFERENCE_STABILITY_SIGNAL": {"T_B2_minus_T_ABS_min": 0.15, "cells_b2_leads_abs_by": 0.20, "cells_min_count": 2},
    "ABSOLUTE_SUCCESSOR_SUFFICIENT": {"T_ABS_minus_T_B2_min": -0.10, "cells_abs_trails_b2_by_more_than": 0.20, "cells_max_count": 1},
    "decision": {"strong_total_success_min": 0.70, "failed_total_success_max": 0.50},
    "seed_release": "QMARK seed1 only if seed0 is ID-qualified and the seed0 classification is M4; seed2 only if the seed0/1 pooled classification is M4; stop at the first non-M4 class; "
                    "after seed2 a pooled M4 is MECHANISM_INCONCLUSIVE",
}
INTERPRETATIONS = {
    "ABS_close_to_QMARK": "the runbook says 'ABS 与 QMARK 接近' without a number: fixed as |T(ABS) - T(QMARK)| <= 0.10 (the M3 tolerance)",
    "advantage_concentrated_in_multiple_dual_goal_cells": "the runbook says 'B2 的优势集中在多个双目标 cell' without a number: fixed as B2 leads QMARK by >= 0.20 in BOTH dual-goal cells "
                                                         "(BUF_T+BUF_S, IN_S+BUF_T) while the mean B2 lead in the two single-goal cells (IN_S, BUF_T) is <= 0.10",
    "strong_and_failed": "the runbook says 'B2/ABS 保持强结果' / '明显失效' without a number: strong = T >= 0.70, failed = T < 0.50 (both B2 and ABS)",
    "pooling": "with several seeds T and C are means over seeds of the per-seed rates over the 32 final cases; B2 and ABS use the matched seeds of QMARK",
}


def _ge(a, b):
    return a >= b - EPS


def _gt(a, b):
    return a > b + EPS


def _count(cell_rates, fn):
    return sum(1 for c in CELLS if fn(cell_rates[c]))


def id_qualified(final_dev12_success, nan_count, hard_failure):
    return final_dev12_success == 12 and nan_count == 0 and not hard_failure


def classify_main(T, C):
    """T: {method: total rate}, C: {method: {cell: rate}} for methods B2, ABS, QMARK. Returns {'class': M1|M2|M3|M4, 'facts': {...}}."""
    b2, ab, q = "B2", "ABS", "QMARK"
    d_b2_q, d_ab_q, d_b2_ab = T[b2] - T[q], T[ab] - T[q], T[b2] - T[ab]
    lead_b2 = {c: C[b2][c] - C[q][c] for c in CELLS}
    lead_ab = {c: C[ab][c] - C[q][c] for c in CELLS}
    trail_q_b2 = {c: C[b2][c] - C[q][c] for c in CELLS}
    r = RULES
    m1 = (_ge(d_b2_q, r["M1_EXPLICIT_APPLICATION_SHARED_ADVANTAGE"]["T_B2_minus_T_Q_min"]) and _ge(d_ab_q, r["M1_EXPLICIT_APPLICATION_SHARED_ADVANTAGE"]["T_ABS_minus_T_Q_min"])
          and sum(_ge(v, 0.10) for v in lead_b2.values()) >= 3 and sum(_ge(v, 0.10) for v in lead_ab.values()) >= 3)
    dual_each = all(_ge(lead_b2[c], 0.20) for c in DUAL_CELLS)
    single_mean = sum(lead_b2[c] for c in SINGLE_CELLS) / len(SINGLE_CELLS)
    concentrated = dual_each and single_mean <= 0.10 + EPS
    m2_extra = {"b2_exceeds_abs_by_0.15": _ge(d_b2_ab, 0.15), "abs_close_to_qmark": abs(d_ab_q) <= 0.10 + EPS, "b2_advantage_concentrated_in_dual_goal_cells": bool(concentrated)}
    m2 = _ge(d_b2_q, 0.20) and not m1 and any(m2_extra.values())
    m3 = (_ge(T[q], T[b2] - 0.10) and _ge(T[q], T[ab] - 0.10) and sum(_gt(v, 0.20) for v in trail_q_b2.values()) <= 1)
    cls = "M1" if m1 else "M2" if m2 else "M3" if m3 else "M4"
    return {"class": cls, "facts": {"T_B2_minus_T_Q": d_b2_q, "T_ABS_minus_T_Q": d_ab_q, "T_B2_minus_T_ABS": d_b2_ab, "b2_lead_over_q_by_cell": lead_b2, "abs_lead_over_q_by_cell": lead_ab,
                                    "m2_clauses": m2_extra, "single_cell_mean_b2_lead": single_mean}}


def classify_b2_vs_abs(T, C):
    d = T["B2"] - T["ABS"]
    lead = {c: C["B2"][c] - C["ABS"][c] for c in CELLS}
    if _ge(d, 0.15) and sum(_ge(v, 0.20) for v in lead.values()) >= 2:
        label = "DIFFERENCE_STABILITY_SIGNAL"
    elif _ge(T["ABS"], T["B2"] - 0.10) and sum(_gt(-v, 0.20) for v in lead.values()) <= 1:
        label = "ABSOLUTE_SUCCESSOR_SUFFICIENT"
    else:
        label = "UNRESOLVED"
    return {"label": label, "T_B2_minus_T_ABS": d, "b2_lead_over_abs_by_cell": lead}


def pool(per_seed):
    """per_seed: {seed: {'T': {method: rate}, 'C': {method: {cell: rate}}}} -> pooled T and C (mean over seeds)."""
    seeds = sorted(per_seed)
    methods = per_seed[seeds[0]]["T"]
    T = {m: sum(per_seed[s]["T"][m] for s in seeds) / len(seeds) for m in methods}
    C = {m: {c: sum(per_seed[s]["C"][m][c] for s in seeds) / len(seeds) for c in CELLS} for m in methods}
    return T, C


def route_recommendation(main_class, b2_total, abs_total, qmark_qualified, inconclusive_after_three, physical_ok=True):
    """Runbook section 11 (recommendation only)."""
    strong = RULES["decision"]["strong_total_success_min"]
    failed = RULES["decision"]["failed_total_success_max"]
    if not physical_ok or not qmark_qualified or inconclusive_after_three or (b2_total < failed - EPS and abs_total < failed - EPS):
        return {"decision": "C", "recommendation": "PAUSE"}
    if main_class in ("M1", "M2") and (b2_total >= strong - EPS or abs_total >= strong - EPS):
        return {"decision": "A", "recommendation": "ENTER_MAINLINE"}
    if main_class == "M3":
        return {"decision": "B", "recommendation": "NARROW_AND_CONTINUE"}
    return {"decision": "B*", "recommendation": "NARROW_AND_CONTINUE", "note": "M1/M2 but neither B2 nor ABS reaches the 'strong' level, or an unclassified residual: treated as the narrower option"}
