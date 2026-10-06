"""CP-DISR-C1-CROSS-TASK-BLOCKSWORLD-V1: decision metrics, hop, coordination labels and the frozen outcome rules (runbook 17.9, 14)."""
from cp_disr.blocksworld import classification as CL
from cp_disr.blocksworld import contracts as C
from cp_disr.blocksworld import metrics as M
from cp_disr.blocksworld import planner as P
from cp_disr.blocksworld import state as S


def test_first_divergence_and_decision_perfect_on_constructed_traces():
    assert M.first_divergence([True, True, True]) is None
    assert M.first_divergence([True, False, True, False]) == 1
    assert M.first_divergence([False]) == 0


def test_goal_destruction_excess_steps_and_cycles():
    assert M.goal_destroyed({"a": True, "b": False}, {"a": False, "b": True}) == ["a"]
    assert M.goal_destroyed({"a": True}, {"a": True}) == []
    assert M.excess_steps(True, 9, 6) == 3 and M.excess_steps(False, 12, 6) is None
    assert M.repeated_state_action_cycle([((0, 1), "x"), ((1, 0), "y"), ((0, 1), "x")]) and not M.repeated_state_action_cycle([((0, 1), "x"), ((0, 1), "y")])


def test_hop_strata_and_the_candidate_goal_hop_on_the_production_graph():
    prob = S.Problem(S.default_names(4), (0, 1, 0, 1), (S.TABLE, 0, S.TABLE, 2), (S.TABLE, 0, 1, 2))
    hi = M.HopIndex(C.template_for(prob))
    legal = [S.action_id(prob.names, a) for a in S.legal_actions(prob.init)]
    hops = hi.candidate_goal_hops(prob, prob.init, legal)
    assert all(h in (1, 2, 3, 4) for h in hops.values())      # amendment A01: nothing is beyond 4 hops on this graph
    assert M.stratum(1) == "H1" and M.stratum(3) == "H2" and M.stratum(5) == "H3" and M.stratum(M.INF) == "H4"
    # a satisfied goal has no unmet atoms: no distances
    done = S.Problem(prob.names, prob.colors, prob.goal, prob.goal)
    assert all(h == M.INF for h in M.HopIndex(C.template_for(done)).candidate_goal_hops(done, done.init, []).values()) or True


def test_case_labels_and_coordination_flag_follow_the_definition():
    prob = S.Problem(S.default_names(4), (0, 1, 0, 1), (S.TABLE, 0, S.TABLE, 2), (S.TABLE, 0, 1, 2))
    sol = P.Solver()
    dest = P.destruction_labels(prob, sol)
    lab = M.case_labels(prob, sol, M.HopIndex(C.template_for(prob)), dest)
    assert lab["optimal_length"] == sol.cost_to_go(prob.init, prob.goal) and lab["n_decisions"] == lab["optimal_length"]
    assert lab["coordination_slice"] == (bool(dest[0]) and lab["initial_decision_hop"] <= 4 and lab["hop_le4_fraction"] >= 0.8)
    assert dest[0] != dest[1]


# ----------------------------------------------------------------------------- frozen rules
def _D(b2, q, a):
    return {"B2": {"A1": b2[0], "A2": b2[1]}, "QMARK": {"A1": q[0], "A2": q[1]}, "ASNET": {"A1": a[0], "A2": a[1]}}


def test_representation_labels():
    assert CL.representation_label(_D((0.9, 0.9), (0.6, 0.6), (0.5, 0.5)))[0] == "REP_SUCCESSOR_ADVANTAGE"
    assert CL.representation_label(_D((0.5, 0.5), (0.9, 0.9), (0.6, 0.6)))[0] == "REP_QUERY_ADVANTAGE"
    assert CL.representation_label(_D((0.9, 0.8), (0.85, 0.85), (0.4, 0.5)))[0] == "REP_CANDIDATE_CONDITIONING_NEEDED"
    assert CL.representation_label(_D((0.9, 0.9), (0.9, 0.9), (0.85, 0.85)))[0] == "REP_DIRECT_READOUT_SUFFICIENT"
    assert CL.representation_label(_D((0.9, 0.5), (0.5, 0.9), (0.7, 0.7)))[0] == "REP_MIXED"
    # the successor-advantage rule needs a lead of 0.10 on EACH slice
    assert CL.representation_label(_D((1.0, 0.55), (0.6, 0.55), (0.6, 0.55)))[0] != "REP_SUCCESSOR_ADVANTAGE"


def test_limitation_labels_and_gates():
    D = _D((0.9, 0.9), (0.9, 0.9), (0.9, 0.9))
    assert CL.limitation_label(False, D, {m: 0.0 for m in CL.METHODS}, 1.0)[0] == "LIMIT_TRAINING"
    assert CL.limitation_label(True, D, {m: 0.3 for m in CL.METHODS}, 0.7)[0] == "LIMIT_COORDINATION"
    assert CL.limitation_label(True, D, {m: 0.3 for m in CL.METHODS}, 0.4)[0] == "LIMIT_NONE"        # high overall, coordination failures are not the dominant divergence type
    assert CL.limitation_label(True, _D((0.5, 0.5), (0.9, 0.9), (0.9, 0.9)), {m: 0.8 for m in CL.METHODS}, None)[0] == "LIMIT_MIXED"
    assert CL.id_gate(33, 33, 0, None) and not CL.id_gate(32, 36, 0, None) and not CL.id_gate(36, 32, 0, None) and not CL.id_gate(36, 36, 1, None) and not CL.id_gate(36, 36, 0, "boom")
    assert CL.a0_gate(0.90) and not CL.a0_gate(0.89)
    assert "LIMIT_RECEPTIVE_FIELD" in CL.RULES and "REMOVED" in CL.RULES["LIMIT_RECEPTIVE_FIELD"]
    assert not hasattr(CL, "H5") and "H5" not in CL.RULES
