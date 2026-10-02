"""Offline tests for the T_P-EF discovery card pure logic (no simulator, no provider)."""
import math

from cp_disr.analysis import tp_ef_discovery as m

A, B = m.PAIR


def trace(*ends, success_at=None, extra=()):
    rows = []
    for i, e in enumerate(ends):
        rows.append({"action": f"s{i}", "elapsed_end": e, "task_success": success_at == i})
    return rows


def test_branch_outcome_success_discounted():
    out = m.branch_outcome(trace(4, 8, 12, 16, 20, success_at=4), nominal_depth=5)
    assert out["success"] and out["skill_count"] == 5 and out["rework_count"] == 0
    assert math.isclose(out["g_start_discounted"], 2 ** (-20 / m.H_SECONDS))


def test_branch_outcome_failure_is_zero_and_counted():
    out = m.branch_outcome(trace(4, 8), nominal_depth=5)
    assert not out["success"] and out["g_start_discounted"] == 0.0 and out["rework_count"] is None


def test_branch_outcome_late_success_not_credited():
    out = m.branch_outcome(trace(30, 61, success_at=1), nominal_depth=5)
    assert out["g_start_discounted"] == 0.0


def row(case, cand, g, t, success=True, valid=True, skills=5, term="TASK_SUCCESS"):
    return {"case_id": case, "candidate_id": cand, "valid": valid, "g_start_discounted": g, "completion_seconds": t,
            "success": success, "skill_count": skills, "termination": term}


def test_epsilon_uses_floor_and_repeat_range_only():
    rows = [row("c", A, 0.5, 20.0), row("c", A, 0.5, 20.0), row("c", B, 0.4, 25.0), row("c", B, 0.4, 25.0)]
    eps = m.epsilon_from_repeats(rows)
    assert eps["epsilon_q"] == m.EPS_G_FLOOR and eps["epsilon_t"] == m.EPS_T_FLOOR and eps["repeat_groups_complete"]
    noisy = rows[:1] + [row("c", A, 0.6, 22.0)] + rows[2:]
    assert math.isclose(m.epsilon_from_repeats(noisy)["epsilon_q"], 0.1)


def test_compare_pair_reliable_direction():
    eps = {"epsilon_q": 0.02, "epsilon_t": 2.1}
    rows = [row("c", A, 0.55, 20.0), row("c", B, 0.40, 30.0, skills=7)]
    out = m.compare_pair("c", rows, eps)
    assert out["reliable"] and out["direction"] == A and "return" in out["reasons"]


def test_compare_pair_within_tolerance_is_low_value():
    eps = {"epsilon_q": 0.02, "epsilon_t": 2.1}
    out = m.compare_pair("c", [row("c", A, 0.50, 20.0), row("c", B, 0.51, 20.5)], eps)
    assert not out["reliable"] and out["status"] == "LOW_DIAGNOSTIC_VALUE"


def test_compare_pair_planner_artefact_excluded():
    eps = {"epsilon_q": 0.02, "epsilon_t": 2.1}
    out = m.compare_pair("c", [row("c", A, 0.5, 20.0), row("c", B, 0.0, 60.0, success=False, term="NO_PLAN")], eps)
    assert not out["reliable"] and out["status"] == "ARTEFACT_EXCLUDED"


def test_compare_pair_invalid_branch_not_established():
    eps = {"epsilon_q": 0.02, "epsilon_t": 2.1}
    out = m.compare_pair("c", [row("c", A, 0.5, 20.0), row("c", B, 0.0, 0.0, valid=False)], eps)
    assert out["status"] == "NOT_ESTABLISHED_INVALID_BRANCH"


def test_state_rank_is_deterministic_and_outcome_free():
    keys = [m.state_rank_key(f"T_B_dev_{i:02d}") for i in range(20)]
    assert keys == [m.state_rank_key(f"T_B_dev_{i:02d}") for i in range(20)] and len(set(keys)) == 20


def test_branch_ids_unique_and_budget_fits():
    ids = {m.branch_id_of(c, cand, r) for c in ("a", "b", "c") for cand in m.PAIR for r in (0, 1)}
    assert len(ids) == 12
    assert 4 + 2 + 2 <= m.PHYSICAL_EPISODE_CAP


def test_classify_e6_mapping():
    assert m.classify_e6(0, True, True, True) == "PROVIDER_SCHEMA_LIMITATION"
    assert m.classify_e6(0, False, False, False) == "CONTRACT_SUFFICIENT"
    assert m.classify_e6(0, False, False, True) == "HEURISTIC_INCONCLUSIVE"
    assert m.classify_e6(3, True, True, True) == "BENEFICIAL_IMPERFECT"
    assert m.classify_e6(3, False, True, True) == "HARMFUL_BUT_INFORMATIVE"
