"""M01-M09: the mechanism gate only accepts an all-success, symmetric-cost, reversed-advantage result set."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cp_disr.analysis import s4_family_a_soft_ordering_mvp as m

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/final_master/s4_family_a_soft_ordering_mvp.yaml"
CFG = m.load_config(CONFIG)
SRC = Path(m.__file__).read_text(encoding="utf-8")

# A passing result set: target-first wins in SO_TARGET_FIRST, second-first wins in SO_SECOND_FIRST,
# and SO_NEUTRAL is nearly tied. The same numbers are used for every structural variant.
PASS_TIMES = {
    ("SO_TARGET_FIRST", "target_first"): [10.0, 10.2],
    ("SO_TARGET_FIRST", "second_first"): [12.0, 12.2],
    ("SO_SECOND_FIRST", "target_first"): [12.4, 12.6],
    ("SO_SECOND_FIRST", "second_first"): [10.4, 10.6],
    ("SO_NEUTRAL", "target_first"): [11.00, 11.02],
    ("SO_NEUTRAL", "second_first"): [11.03, 11.05],
}


def _reach_doc():
    return {"both_routes_reachable": True, "route_lengths_equal": True, "skill_multisets_equal": True,
            "b_plan_nominal_costs_equal": True, "canonical_first_action": m.INITIAL_CANDIDATES[1],
            "canonical_tie_break_reads_geometry": False}


def _build_out(tmp_path, times=None, **overrides):
    """Materialize a complete synthetic evidence tree for classify(): registration, results, ledger, attempts, reach."""
    times = times or PASS_TIMES
    out = tmp_path
    phys = out / "physical"
    (phys / "branch_results").mkdir(parents=True, exist_ok=True)
    (out / "geometry").mkdir(parents=True, exist_ok=True)
    (out / "decision").mkdir(parents=True, exist_ok=True)
    (out / "stage_manifest.json").write_text('{"phases_done": []}', encoding="utf-8")
    (out / "geometry/contract_reachability.json").write_text(json.dumps(_reach_doc()), encoding="utf-8")
    branches, results, attempts = [], {}, {}
    for cid in m.CONFIG_ORDER:
        for route in (m.ROUTE_TARGET_FIRST, m.ROUTE_SECOND_FIRST):
            for repeat in (0, 1):
                b = m.make_branch(cid, route, repeat, str(tmp_path / "m.yaml"), f"{m.TASK_ID}_{cid}",
                                  {m.RUNTIME_REL: "0" * 64, "src/cp_disr/platforms/libero/tp_sr_instrumentation.py": "1" * 64},
                                  "technical" if (cid == "SO_TARGET_FIRST" and repeat == 0) else "remaining")
                bid = b["branch_id"]
                branches.append(b)
                t = times[(cid, route)][repeat]
                over = dict(overrides.get(bid, {}))
                trace = [{"candidate_id": c, "controller_exit": "NORMAL_TERMINATION", "elapsed_seconds": (t / 3.0) * i,
                          "sim_duration": 1.0, "task_success": (i == 3)} for i, c in enumerate(m.ROUTES[route])]
                results[bid] = {"branch_id": bid, "task_success": True, "execution_status": "TERMINATED",
                                "termination_reason": "TASK_SUCCESS", "protocol_complete": True,
                                "witness_validity": "VALID", "controller_exit": "NORMAL_TERMINATION",
                                "recorder_errors": 0, "global_perception_mask_unchanged": True,
                                "provider_module_loaded": False, "trace": trace, **over}
                attempts[bid] = "COMPLETED"
    (phys / "witnesses").mkdir(parents=True, exist_ok=True)
    (phys / "witnesses/e4_branch_registration.json").write_text(json.dumps({"branches": branches}), encoding="utf-8")
    for bid, r in results.items():
        (phys / "branch_results" / f"{bid}.json").write_text(json.dumps(r), encoding="utf-8")
    (phys / "attempt_registry.json").write_text(json.dumps(attempts), encoding="utf-8")
    (phys / "budget_ledger.json").write_text(json.dumps({
        "physical_witness_episodes": {"cap": 12, "used": 12}, "environment_resets": {"cap": 12, "used": 12},
        "total_branch_attempts": {"cap": 12, "used": 12}}), encoding="utf-8")
    return out, branches


def _classify(tmp_path, **kw):
    out, branches = _build_out(tmp_path, **kw)
    return m.classify(ROOT, CONFIG, out), out, branches


def test_M01_all_routes_must_succeed(tmp_path):
    gate, _out, _b = _classify(tmp_path)
    assert gate["common_integrity"]["all_12_task_success"] is True
    assert gate["common_integrity_ok"] is True
    assert gate["mechanism_feasibility"] == "ESTABLISHED_FOR_CONTROLLED_VALIDATION"
    # break one branch's success and the whole common-integrity block fails
    out, branches = _build_out(tmp_path / "b")
    bad = branches[0]["branch_id"]
    p = out / "physical/branch_results" / f"{bad}.json"
    r = json.loads(p.read_text(encoding="utf-8"))
    r["task_success"] = False
    p.write_text(json.dumps(r), encoding="utf-8")
    gate2 = m.classify(ROOT, CONFIG, out)
    assert gate2["common_integrity"]["all_12_task_success"] is False
    assert gate2["mechanism_feasibility"] == "NOT_ESTABLISHED"


def test_M02_skill_count_difference_cannot_prove_mechanism(tmp_path):
    out, branches = _build_out(tmp_path)
    bad = branches[0]["branch_id"]
    p = out / "physical/branch_results" / f"{bad}.json"
    r = json.loads(p.read_text(encoding="utf-8"))
    r["trace"] = r["trace"][:3]
    p.write_text(json.dumps(r), encoding="utf-8")
    gate = m.classify(ROOT, CONFIG, out)
    assert gate["common_integrity"]["all_skill_count_4"] is False
    assert gate["common_integrity"]["all_sequences_match_a_frozen_route"] is False
    assert gate["mechanism_feasibility"] == "NOT_ESTABLISHED"


def test_M03_no_plan_cannot_prove_mechanism(tmp_path):
    out, branches = _build_out(tmp_path)
    bad = branches[0]["branch_id"]
    p = out / "physical/branch_results" / f"{bad}.json"
    r = json.loads(p.read_text(encoding="utf-8"))
    r["execution_status"] = "NO_PLAN"
    p.write_text(json.dumps(r), encoding="utf-8")
    gate = m.classify(ROOT, CONFIG, out)
    assert gate["common_integrity"]["no_no_plan"] is False
    assert gate["mechanism_feasibility"] == "NOT_ESTABLISHED"


def test_M04_target_first_reversal_required(tmp_path):
    # neutralise the target-first advantage: SO_TARGET_FIRST no longer reverses
    times = dict(PASS_TIMES)
    times[("SO_TARGET_FIRST", "target_first")] = [12.0, 12.2]
    times[("SO_TARGET_FIRST", "second_first")] = [10.0, 10.2]
    gate, _out, _b = _classify(tmp_path, times=times)
    assert gate["per_config_criterion_met"]["SO_TARGET_FIRST"] is False
    assert gate["mechanism_feasibility"] == "NOT_ESTABLISHED"


def test_M05_second_first_reversal_required(tmp_path):
    times = dict(PASS_TIMES)
    times[("SO_SECOND_FIRST", "second_first")] = [12.4, 12.6]
    times[("SO_SECOND_FIRST", "target_first")] = [10.4, 10.6]
    gate, _out, _b = _classify(tmp_path, times=times)
    assert gate["per_config_criterion_met"]["SO_SECOND_FIRST"] is False
    assert gate["mechanism_feasibility"] == "NOT_ESTABLISHED"


def test_M06_neutral_control_required(tmp_path):
    times = dict(PASS_TIMES)
    times[("SO_NEUTRAL", "target_first")] = [8.0, 8.0]
    times[("SO_NEUTRAL", "second_first")] = [12.0, 12.0]
    gate, _out, _b = _classify(tmp_path, times=times)
    assert gate["per_config_criterion_met"]["SO_NEUTRAL"] is False
    assert gate["mechanism_feasibility"] == "NOT_ESTABLISHED"


def test_M07_same_nominal_cost_required(tmp_path):
    out, _branches = _build_out(tmp_path)
    reach = _reach_doc()
    reach["b_plan_nominal_costs_equal"] = False
    (out / "geometry/contract_reachability.json").write_text(json.dumps(reach), encoding="utf-8")
    gate = m.classify(ROOT, CONFIG, out)
    assert gate["common_integrity"]["b_plan_nominal_costs_equal"] is False
    assert gate["contract_cannot_rank"]["b_plan_nominal_costs_equal"] is False
    assert gate["mechanism_feasibility"] == "NOT_ESTABLISHED"


def test_M08_wall_time_cannot_replace_sim_time(tmp_path):
    gate, out, _b = _classify(tmp_path)
    assert gate["sim_time_only"] is True
    assert gate["wall_time_used_for_mechanism"] is False
    # the primary metric is defined from sim-time trace fields only, never wall-clock fields
    metric = m._sim_completion({"trace": [{"elapsed_seconds": 3.0, "task_success": False},
                                          {"elapsed_seconds": 7.5, "task_success": True}]})
    assert metric == pytest.approx(4.5)
    assert "wall" not in json.dumps(m._sim_completion.__doc__ or "").lower()
    assert "wall_seconds" not in SRC.split("def _sim_completion", 1)[1].split("def _sum_skill_sim", 1)[0]
    assert "sim time" in (m._sim_completion.__doc__ or "").lower()


def test_M09_no_automatic_provider_or_s2_authorization(tmp_path):
    gate, out, _b = _classify(tmp_path)
    assert gate["mechanism_feasibility"] == "ESTABLISHED_FOR_CONTROLLED_VALIDATION"
    na = json.loads((out / "decision/next_action.json").read_text(encoding="utf-8"))
    assert na["provider_authorized"] is False
    assert na["representation_authorized"] is False
    assert na["s2_authorized"] is False
    assert na["s3_authorized"] is False
    assert na["next_action"] == "REQUEST_PROVIDER_BASELINE_QUALIFICATION"
    assert na["no_automatic_third_attempt"] is True
    assert na["no_automatic_fourth_config"] is True
    assert CFG["authorization"]["provider_authorized"] is False
    assert CFG["authorization"]["representation_authorized"] is False
    assert CFG["authorization"]["s2_authorized"] is False
    assert CFG["authorization"]["s3_authorized"] is False
    for key in ("provider_first_calls", "provider_retries", "representation_forwards", "rl_transitions",
                "optimizer_steps", "training_attempts", "elastic_attempts", "formal_test_episodes",
                "standalone_capture_resets"):
        assert int(CFG["budgets"][key]) == 0, key