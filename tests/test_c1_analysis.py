"""CP-DISR-C1-MECH-CONFIRM-V1: analysis helpers on small synthetic episodes (pure python)."""
from cp_disr import c1_analysis as A


def decision(sel, ok=True, binding=None, div=None, planner=None, ref="AVAILABLE"):
    d = {"decision_index": 0, "reference_status": ref, "selected": sel, "diverged_from_public_optimal": div is not None, "divergence_type": div, "placement_binding": binding, "repeated_action": False,
         "destroyed_completed_goals": [], "encoder_forward_calls": 5, "forward_seconds": 0.1, "probs": [0.9, 0.1], "mask": [True, True], "selected_index": 0}
    if planner:
        d["planner"] = planner
    return d


def episode(case, cell, success, first_div=None, nocand=False, destroyed=0, repeated=0, decisions=None):
    return {"case_id": case, "cell": cell, "success": success, "reason": "TASK_SUCCESS" if success else ("NO_CANDIDATE_SAFE_TERMINATION:x" if nocand else "TIMEOUT"), "steps": 2, "elapsed_seconds": 10.0, "n_decisions": 1,
            "first_divergence": first_div, "repeated_action_count": repeated, "destroyed_completed_goal_count": destroyed, "reference_unavailable_decisions": 0,
            "ended_with_no_candidate_safe_termination": nocand, "no_candidate_after_earlier_divergence": bool(nocand and first_div), "decisions": decisions or [decision("a")]}


def test_rates_completeness_and_cell_rows():
    eps = [episode("c%d" % i, A.CELLS[i % 4], i % 2 == 0) for i in range(8)]
    runs = {("B2", 0): eps}
    assert A.completeness(runs, ["c%d" % i for i in range(8)]) == []
    assert A.completeness(runs, ["c%d" % i for i in range(7)] + ["zz"])
    assert A.total_rate(eps) == 0.5
    rates, counts = A.cell_rates(eps)
    assert all(counts[c][1] == 2 for c in A.CELLS) and rates["IN_S"] == 1.0
    rows = A.cell_rows(runs)
    assert len(rows) == 5 and rows[-1]["cell"] == "ALL"
    assert A.case_rows(runs)[0]["method"] == "B2"


def test_failure_categories_and_mechanism_counts():
    div_before = {"index": 0, "selected": "x", "type": "PLACES_TRAIN_DEFAULT_BINDING", "completed_goals_before": {"g": False}, "placement_binding": "target->container"}
    div_after = {"index": 2, "selected": "x", "type": "PICK_NOT_OPTIMAL", "completed_goals_before": {"g": True}, "placement_binding": None}
    assert A.failure_category(episode("a", "IN_S", True)) == "SUCCESS"
    assert A.failure_category(episode("a", "IN_S", False, first_div=div_before)) == "FIRST_DIVERGENCE_BEFORE_ANY_GOAL"
    assert A.failure_category(episode("a", "IN_S", False, first_div=div_after)) == "FIRST_DIVERGENCE_AFTER_A_COMPLETED_GOAL"
    assert A.failure_category(episode("a", "IN_S", False, destroyed=1, first_div=div_after)) == "DESTROYED_A_COMPLETED_GOAL"
    assert A.failure_category(episode("a", "IN_S", False, nocand=True)) == "OBSERVATION_LOSS_LIMITED_NO_CANDIDATE_WITHOUT_DIVERGENCE"
    assert A.failure_category(episode("a", "IN_S", False)) == "NO_DIVERGENCE_FROM_PUBLIC_OPTIMAL"
    eps = [episode("a", "IN_S", False, first_div=div_before, decisions=[decision("p", binding="target->container", div="PLACES_TRAIN_DEFAULT_BINDING")]), episode("b", "IN_S", True, repeated=1)]
    m = A.mechanism(eps)
    assert m["failures"] == 1 and m["first_divergence_rate"] == 0.5 and m["training_default_binding_error_episodes"] == 1 and m["repeated_action_episode_rate"] == 0.5
    assert m["failure_categories"] == {"FIRST_DIVERGENCE_BEFORE_ANY_GOAL": 1}


def test_planner_summary_and_interpretation():
    plan = {"status": "PLAN_FOUND", "plan": ["a", "b"], "cpu_seconds": 0.001, "expanded_nodes": 3, "generated_nodes": 8, "cost": 8.4, "depth": 2}
    nop = {"status": "NO_PLAN", "plan": [], "cpu_seconds": 0.002, "expanded_nodes": 5, "generated_nodes": 9, "cost": 0.0, "depth": 0}
    ok = [episode("c%d" % i, A.CELLS[i % 4], True, decisions=[decision("a", planner=plan)]) for i in range(7)]
    bad = [episode("z", "IN_S", False, nocand=True, decisions=[decision("a", planner=nop)])]
    s = A.planner_summary(ok + bad)
    assert s["episodes"] == 8 and s["no_plan_or_timeout_episodes"] == 1 and s["search_status_counts"] == {"PLAN_FOUND": 7, "NO_PLAN": 1}
    assert A.planner_interpretation(dict(s, success_rate=0.95, cpu_seconds_p95=0.01), {"B2": 0.9})["class"] == "HIGH_SUCCESS_LOW_COST"
    assert A.planner_interpretation(dict(s, success_rate=0.6, no_plan_or_timeout_episodes=4, episodes=8), {"B2": 0.9})["class"] == "PLANNER_FRAGILE_TO_PUBLIC_UNKNOWN"
    assert A.planner_interpretation(dict(s, success_rate=0.99, cpu_seconds_p95=1.0), {"B2": 0.5})["class"] == "PLANNER_OUTPERFORMS_ALL_LEARNED"


def test_efficiency_and_single_vs_dual():
    eps = [episode("a", "BUF_T", True), episode("b", "BUF_T+BUF_S", False), episode("c", "IN_S", True), episode("d", "IN_S+BUF_T", True)]
    e = A.efficiency_neural(eps)
    assert e["encoder_forward_calls_per_decision_mean"] == 5 and e["forward_seconds_per_decision_mean"] == 0.1
    sd = A.single_vs_dual(eps)
    assert sd["BUF_T_minus_BUF_T+BUF_S"] == 1.0 and sd["IN_S_minus_IN_S+BUF_T"] == 0.0
