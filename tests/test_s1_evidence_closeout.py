"""Regression tests for the offline S1-REV1 r3 evidence closeout (engineering tests; fixtures are labelled as such).

Tests that need saved evidence read the real r3 directory read-only and write only into tmp_path.
"""
import json
import os
from pathlib import Path

import pytest

from cp_disr.analysis import s1_evidence_closeout as m

ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / m.S1_REL / "e4_recovery_23fab0c_r3"
needs_data = pytest.mark.skipif(not R3.is_dir(), reason="saved r3 evidence not present")


def ctx(tmp_path):
    return m.Ctx(ROOT, R3, tmp_path / "out")


@pytest.fixture
def guards():
    """Install the zero-call guards and restore the originals afterwards (the guards patch module attributes globally)."""
    import importlib
    saved = []
    for mod in ("cp_disr.runtime", "cp_disr.baselines.b_plan", "cp_disr.analysis.s1_integration", "cp_disr.analysis.s1_revision_resume",
                "cp_disr.analysis.s1_e4_recovery", "cp_disr.platforms.libero.d0_env", "cp_disr.platforms.libero.runtime_factory", "cp_disr.vlm_provider"):
        try:
            mod_obj = importlib.import_module(mod)
        except Exception:
            continue
        saved.append((mod_obj, dict(vars(mod_obj))))
    prov = None
    try:
        prov = importlib.import_module("cp_disr.vlm_provider").DashScopeProvider
        init = prov.__init__
    except Exception:
        init = None
    env = dict(os.environ)
    yield m.install_guards()
    for mod_obj, d in saved:
        for k, v in d.items():
            setattr(mod_obj, k, v)
    if prov is not None and init is not None:
        prov.__init__ = init
    os.environ.clear(); os.environ.update(env)


# 1 ---------------------------------------------------------------- budget arithmetic
@needs_data
def test_budget_is_recomputed_8_2_2_8_and_unknown_not_refunded(tmp_path):
    chk = m.cmd_budget(ctx(tmp_path))
    assert chk["reserved_by_round"] == {"original": 8, "r1": 2, "r2": 2, "r3": 8}
    assert chk["reserved_total"] == 20 and chk["remaining"] == 0 and chk["status"] == "CONSISTENT"
    assert chk["unknown_not_refunded"] is True and chk["failed_not_refunded"] is True
    assert chk["attempt_states_by_round"]["r2"] == {"UNKNOWN": 2}
    assert chk["attempt_states_by_round"]["r3"] == {"COMPLETED": 8}


# 2 ---------------------------------------------------------------- repeats are not independent configurations
@needs_data
def test_repeated_case_not_counted_as_independent_configuration(tmp_path):
    m.cmd_branches(ctx(tmp_path))
    unit = json.loads((tmp_path / "out/audit/branches/branch_outcomes.json").read_text())["unit_of_analysis"]
    assert unit["independent_case_configurations"] == 2 and unit["paired_comparisons"] == 4 and unit["branch_attempts"] == 8
    assert "no significance" in unit["statistical_note"]


# 3 ---------------------------------------------------------------- NORMAL_TERMINATION != task_success
def test_normal_termination_is_not_task_success():
    actions = [{"candidate_id": "x", "evaluator": {"success": False, "terminated": True, "reason": "NORMAL_TERMINATION"}}]
    o = m.branch_outcome({"actions": actions, "planner": []})
    assert o["last_evaluator_success"] is False and o["termination_by_evaluator"] is True


# 4 ---------------------------------------------------------------- CONTINUE + NO_PLAN != Evaluator terminal failure
def test_continue_then_no_plan_is_not_evaluator_failure():
    actions = [{"candidate_id": "p", "evaluator": {"success": False, "terminated": False, "truncated": False, "reason": "CONTINUE"}}]
    o = m.branch_outcome({"actions": actions, "planner": [{"status": "NO_PLAN"}]})
    assert o["termination_by_evaluator"] is False and o["termination_by_planner"] is True and o["planner_stop_status"] == "NO_PLAN"
    assert o["last_evaluator_reason"] == "CONTINUE"


# 5 ---------------------------------------------------------------- action sequence difference alone is not E4
def _o(ids):
    return {"last_evaluator_success": True, "last_evaluator_reason": "TASK_SUCCESS", "termination_by_evaluator": True, "termination_by_planner": False,
            "action_ids": ids, "measured_cost": None, "measured_rework": None}


def _e4_check(diff):
    assert not (diff["action_sequence_difference_descriptive_only"] and not diff["evaluator_confirmed_success_difference"]
                and not diff["continuation_termination_difference"] and diff["measured_execution_cost_difference"] in (False, "NOT_MEASURED")
                and diff["measured_rework_difference"] in (False, "NOT_MEASURED") and diff["e4_outcome_difference_established"])


def test_sequence_difference_without_outcome_difference_is_not_e4():
    d = m.pair_difference(_o(["a", "b"]), _o(["a", "c", "b"]))
    assert d["action_sequence_difference_descriptive_only"] is True and d["e4_outcome_difference_established"] is False
    _e4_check(d)


def test_reverse_regression_sequence_implies_consequence_is_rejected():
    """Re-introduce the old 'sequence differs => consequence differs' rule; the same check must reject it."""
    def buggy(open_o, pick_o):
        d = m.pair_difference(open_o, pick_o)
        d["e4_outcome_difference_established"] = bool(d["action_sequence_difference_descriptive_only"])
        return d
    with pytest.raises(AssertionError):
        _e4_check(buggy(_o(["a", "b"]), _o(["a", "c", "b"])))


# 6 ---------------------------------------------------------------- admission and multi-hop explanation are compatible
@needs_data
def test_admitted_relation_and_multihop_explanation_both_true(tmp_path):
    A = m.load_actual_contracts(ctx(tmp_path))
    rel = {"source_ref": m.OPEN_ID, "target_ref": "p:Inside:second_object:container", "type": "SOFT_RELEVANT_TO_GOAL", "effect_fact_ref": "p:Open:container"}
    goals = [g.fact_id for g in A["template"].goals]
    assert m.local_redundancy(A["contracts"], rel, goals) is False          # frozen local rule admits it
    paths = m.multihop_paths(A["contracts"], rel["source_ref"], rel["target_ref"])
    assert paths and any(step.get("required_by_precondition_of", "").startswith("a:PLACE") for p in paths for step in p)


# 7 ---------------------------------------------------------------- production nominal functions distinguish OPEN / PICK
@needs_data
def test_reachability_uses_production_functions_and_distinguishes_open_and_pick(tmp_path):
    c = ctx(tmp_path)
    A = m.load_actual_contracts(c)
    from cp_disr.contracts import nominal_overlay
    rrs = [m.jread(p) for p in sorted((R3 / "witnesses/restore_receipts").glob("*.json")) if m.jread(p)["case_id"] == "T_A_dev_14"]
    bind = m.bind_case_facts(c, "T_A_dev_14", rrs)
    facts = m.to_truth_map(bind["facts"])
    by_id = {ct.id: ct for ct in A["contracts"]}
    tpl = A["template"]
    res = {}
    for cid in (m.OPEN_ID, m.PICK_ID):
        ov = dict(nominal_overlay(by_id[cid], facts, derived=tpl.derived_rules, exclusive_groups=tpl.exclusive_groups))
        res[cid] = m.symbolic_closure(A["contracts"], ov, list(m.GOALS_T_A), tpl.derived_rules, tpl.exclusive_groups)
    assert res[m.OPEN_ID]["status"] == "COMPLETE" and res[m.OPEN_ID]["goal_reachable"] is True
    assert res[m.PICK_ID]["status"] == "COMPLETE" and res[m.PICK_ID]["goal_reachable"] == "PROVEN_UNREACHABLE"


# 8 ---------------------------------------------------------------- truncated search never proves unreachability
@needs_data
def test_truncated_search_is_never_proved_unreachable(tmp_path):
    c = ctx(tmp_path)
    A = m.load_actual_contracts(c)
    rrs = [m.jread(p) for p in sorted((R3 / "witnesses/restore_receipts").glob("*.json")) if m.jread(p)["case_id"] == "T_A_dev_14"]
    facts = m.to_truth_map(m.bind_case_facts(c, "T_A_dev_14", rrs)["facts"])
    tpl = A["template"]
    r = m.symbolic_closure(A["contracts"], facts, list(m.GOALS_T_A), tpl.derived_rules, tpl.exclusive_groups, limit=2)
    assert r["status"] == "ENUMERATION_LIMIT_REACHED" and r["goal_reachable"] != "PROVEN_UNREACHABLE"


# 9 ---------------------------------------------------------------- missing post-action facts
def test_missing_post_action_facts_are_reported_not_invented(tmp_path):
    (tmp_path / "witnesses").mkdir(); (tmp_path / "branch_results").mkdir()
    (tmp_path / "witnesses/branch_x.jsonl").write_text('{"phase": "action_complete", "candidate_id": "a"}\n')
    c = m.Ctx(ROOT, tmp_path, tmp_path / "out")
    scan = m.scan_post_action_facts(c)
    assert scan["status"] == "NOT_RECORDED_OR_NOT_FOUND"


@needs_data
def test_real_r3_has_no_saved_post_action_facts_and_successors_are_nominal(tmp_path):
    scan = m.scan_post_action_facts(ctx(tmp_path))
    assert scan["status"] == "NOT_RECORDED_OR_NOT_FOUND"
    m.cmd_contracts(ctx(tmp_path))
    nom = json.loads((tmp_path / "out/contract/nominal_successors.json").read_text())
    for case in nom.values():
        for succ in case["successors"].values():
            assert succ["provenance"].startswith("NOMINAL_DERIVED")


# 10 --------------------------------------------------------------- protected-file hash change fails verification
def test_protected_file_hash_change_is_detected(tmp_path):
    f = tmp_path / "p.txt"; f.write_text("a")
    c = m.Ctx(tmp_path, tmp_path, tmp_path / "out")
    before = {"entries": {"p.txt": {"sha256": m.sha256_file(f)}}}
    assert m.compare_protected(c, before)[1:] == ([], [])
    f.write_text("b")
    after, changed, missing = m.compare_protected(c, before)
    assert changed == ["p.txt"]
    f.unlink()
    assert m.compare_protected(c, before)[2] == ["p.txt"]


# 11 --------------------------------------------------------------- zero-call guard
def test_zero_call_guard_blocks_runtime_env_provider_and_budget(guards):
    assert guards
    import cp_disr.runtime as rt
    import cp_disr.analysis.s1_integration as integ
    with pytest.raises(m.ZeroCallViolation):
        rt.load_runtime("x")
    with pytest.raises(m.ZeroCallViolation):
        integ.reserve_branch_attempt()
    try:
        import cp_disr.platforms.libero.d0_env as env
        with pytest.raises(m.ZeroCallViolation):
            env.make_env()
    except ImportError:
        pass
    try:
        from cp_disr.vlm_provider import DashScopeProvider
        with pytest.raises(m.ZeroCallViolation):
            DashScopeProvider()
    except ImportError:
        pass
    assert os.environ.get("DASHSCOPE_API_KEY") is None


def test_zero_call_violation_maps_to_exit_code_5(guards, tmp_path, monkeypatch, capsys):
    import cp_disr.analysis.s1_evidence_closeout as mod
    monkeypatch.setitem(mod.COMMANDS, "inventory", lambda c: mod.install_guards() and __import__("cp_disr.runtime").runtime.load_runtime("x"))
    assert mod.main(["inventory", "--root", str(ROOT), "--source", str(R3), "--output", str(tmp_path / "o")]) == 5


# 12 --------------------------------------------------------------- unmet gates cannot authorize anything
@pytest.mark.parametrize("n_rel,explainable", [(2, True), (0, False), (2, False), (4, True)])
def test_eligibility_never_authorizes_training_s2_or_new_attempts(n_rel, explainable):
    e = m.build_eligibility("FOUND_AND_RECONCILED", n_rel, explainable, {"E4": "X"})
    assert e["tp_training_authorized"] is False and e["additional_physical_attempts_authorized"] == 0 and e["method_upgrade_authorized"] is False
    assert e["next_action"] == "EXPLICIT_RESEARCH_DECISION_REQUIRED" and e["route_if_key_gate_unmet"] == "S4_RESEARCH_DECISION"
    assert e["eligibility_decision"] == "DERIVED_FROM_AVAILABLE_EVIDENCE"
    assert "not located" not in e["plan_condition_citation"]


# ================================================================ authority reconciliation (A01-A18)
import shutil
import subprocess

OUT = ROOT / m.S1_REL / "evidence_closeout_cc1ad64"
needs_out = pytest.mark.skipif(not (OUT / "evidence/reviewed_eligibility.json").is_file(), reason="closeout evidence not present")


def _auth_root(tmp_path, mutate=None):
    d = tmp_path / "repo" / m.AUTH_DIR
    d.mkdir(parents=True)
    for f in ("CP_DISR_Final_Experimental_Plan_v3.md", "CP_DISR_Final_Research_Content_v5.md", "authority_manifest.json"):
        shutil.copyfile(ROOT / m.AUTH_DIR / f, d / f)
    if mutate:
        man = json.loads((d / "authority_manifest.json").read_text(encoding="utf-8"))
        mutate(man)
        (d / "authority_manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return tmp_path / "repo"


def _ev(name):
    return json.loads((OUT / "evidence" / name).read_text(encoding="utf-8"))


def test_A01_manifest_matching_gives_found_and_reconciled(tmp_path):
    r = m.verify_authority(_auth_root(tmp_path))
    assert r["authority_status"] == "FOUND_AND_RECONCILED" and r["master_id"] == "CP-DISR-FINAL-EXEC-3.0"
    assert m.verify_authority(ROOT)["authority_status"] == "FOUND_AND_RECONCILED"


def test_A02_wrong_v3_hash_fails_closed(tmp_path):
    root = _auth_root(tmp_path, lambda man: man["experimental_plan"].__setitem__("sha256", "0" * 64))
    with pytest.raises(m.AuthorityMismatch):
        m.verify_authority(root)


def test_A03_wrong_v5_hash_fails_closed(tmp_path):
    root = _auth_root(tmp_path, lambda man: man["research_content"].__setitem__("sha256", "f" * 64))
    with pytest.raises(m.AuthorityMismatch):
        m.verify_authority(root)
    root2 = _auth_root(tmp_path / "b", lambda man: man.__setitem__("master_id", "OTHER"))
    with pytest.raises(m.AuthorityMismatch):
        m.verify_authority(root2)
    root3 = _auth_root(tmp_path / "c", lambda man: man["research_content"].__setitem__("document_version", "4.0"))
    with pytest.raises(m.AuthorityMismatch):
        m.verify_authority(root3)
    with pytest.raises(m.AuthorityMismatch):
        m.build_eligibility("NOT_FOUND", 2, True, {})


@needs_out
def test_A04_historical_command_config_not_found_unchanged():
    p = OUT / "inventory/command_config.json"
    cfg = json.loads(p.read_text(encoding="utf-8"))
    assert cfg["authoritative_definitions"]["Final_Experimental_Plan_v3"]["status"] == "NOT_FOUND"
    r = subprocess.run(["git", "-C", str(ROOT), "show", "7dc6965cc909cb0a79087e433006d56d6415096c:" + str(p.relative_to(ROOT))], capture_output=True)
    if r.returncode == 0:
        assert r.stdout == p.read_bytes()
    ar = _ev("authority_reconciliation.json")
    assert ar["execution_time_authority_availability"] == "NOT_FOUND_ON_SERVER" and ar["current_authority_status"] == "FOUND_AND_RECONCILED"


@needs_out
def test_A05_eligibility_has_no_not_located_or_candidate_pending():
    txt = (OUT / "evidence/reviewed_eligibility.json").read_text(encoding="utf-8")
    assert "not located" not in txt and "CANDIDATE_SATISFIED_PENDING_AUTHORITATIVE_DEFINITION" not in txt
    for f in ("reviewed_summary.md", "e4_review.md"):
        t = (OUT / "evidence" / f).read_text(encoding="utf-8")
        assert "not located" not in t and "CANDIDATE_SATISFIED_PENDING_AUTHORITATIVE_DEFINITION" not in t


@needs_out
def test_A06_summary_does_not_write_current_authority_as_not_found():
    t = (OUT / "evidence/reviewed_summary.md").read_text(encoding="utf-8")
    assert "权威文档 v3 查找结果：NOT_FOUND" not in t
    assert "已从原始权威附件恢复并完成对照" in t and "历史记录保留不变" in t


@needs_out
def test_A07_to_A11_key_gate_statuses():
    g = _ev("reviewed_eligibility.json")["key_gate_status"]
    assert g["E1_relation_existence"] == "NOT_ESTABLISHED"
    assert g["E2_E3_representation"] == "PRIOR_VALUES_RETAINED_NOT_REVALIDATED"
    assert g["E4_real_consequence"] == "LIMITED_SUPPORT_FEASIBILITY_ONLY"
    assert g["E5_contract_insufficiency"] == "NOT_SATISFIED"
    assert g["E6_prior_classification"] == "UNKNOWN"
    assert set(_ev("reviewed_eligibility.json")["key_gate_notes"]) == set(g)


@needs_out
def test_A12_to_A15_no_authorization_and_s4_route():
    e = _ev("reviewed_eligibility.json")
    assert e["tp_training_authorized"] is False
    assert e["additional_physical_attempts_authorized"] == 0
    assert e["method_upgrade_authorized"] is False
    assert e["next_action"] == "EXPLICIT_RESEARCH_DECISION_REQUIRED" and e["route_if_key_gate_unmet"] == "S4_RESEARCH_DECISION"
    ar = _ev("authority_reconciliation.json")
    assert ar["next_action"] == "S4_RESEARCH_DECISION" and ar["tp_training_authorized"] is False and ar["additional_physical_attempts_authorized"] == 0
    assert ar["new_samples"] == 0 and ar["scientific_result_changed"] is False
    assert ar["experimental_plan_sections_applied"] == ["5.2", "5.3", "5.4", "5.6"] and ar["research_content_additional_e1_e6_gate_definition"] is False


def test_A16_pick_no_plan_not_written_as_evaluator_failure():
    actions = [{"candidate_id": "p", "evaluator": {"success": False, "terminated": False, "truncated": False, "reason": "CONTINUE"}}]
    o = m.branch_outcome({"actions": actions, "planner": [{"status": "NO_PLAN"}]})
    assert o["termination_by_evaluator"] is False and o["planner_stop_status"] == "NO_PLAN"
    sec = m.E4_V3_SECTION
    assert "Evaluator为CONTINUE，随后B_PLAN返回NO_PLAN" in sec and "不表示PICK被Evaluator判失败" in sec and "NOT_ESTABLISHED" in sec


def test_A17_action_sequence_difference_alone_not_e4_evidence():
    d = m.pair_difference(_o(["a", "b"]), _o(["a", "c", "b"]))
    assert d["action_sequence_difference_descriptive_only"] is True and d["e4_outcome_difference_established"] is False
    review = m.review_e4([], {}, {})
    assert review["cases_with_reliable_protocol_level_difference"] == 0
    assert review["frozen_E4_gate_status"] == "NOT_SATISFIED_ON_AVAILABLE_EVIDENCE"


@needs_out
def test_A17b_e4_review_ends_with_v3_section_and_keeps_e4_e5_separate():
    t = (OUT / "evidence/e4_review.md").read_text(encoding="utf-8")
    assert t.rstrip().endswith("E4与E5必须保持分离。")
    assert "## 对照 Final Experimental Plan v3 §5.2" in t and "LIMITED_SUPPORT_FEASIBILITY_ONLY" in t


def test_A18_no_automatic_official_e6_classification():
    e = m.build_eligibility("FOUND_AND_RECONCILED", 2, True, {"E6_prior_classification": "UNKNOWN"})
    assert "CONTRACT_SUFFICIENT" not in json.dumps(e)
    assert e["diagnostic_reason"] == "CONTRACT_EXPLAINABLE_BRANCH_DIFFERENCE"
    assert "no official E6 class is assigned" in e["key_gate_notes"]["E6_prior_classification"]
    assert e["authoritative_plan_v3_status"] == e["authoritative_research_v5_status"] == "FOUND_AND_RECONCILED"
