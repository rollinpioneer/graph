"""R01-R08: the Wave P mechanism verdict (runbook section 15) and the Wave P root-cause rules."""
from __future__ import annotations

import copy

import pytest

from cp_disr.analysis import s4_tp_sr_pilot_v2_diag as D
from cp_disr.analysis import s4_tp_sr_pilot_v2_proto as P


def row(cid, kind, route, rep, success, label=None, eng=False, contacts=0):
    return {"config_id": cid, "kind": kind, "route": route, "repeat": rep, "success": success, "label": label or ("NOT_APPLICABLE_SUCCESS" if success else "UNRESOLVED"),
            "engineering_failure": eng, "contact_events": contacts}


def established_rows():
    rows = []
    for cid in ("V2_A_INTERFERENCE", "V2_B_INTERFERENCE"):
        rows += [row(cid, "INTERFERENCE", "direct", 0, False, "PHYSICAL_GRASP_INTERFERENCE", contacts=12), row(cid, "INTERFERENCE", "direct", 1, False, "PHYSICAL_GRASP_INTERFERENCE", contacts=9)]
        rows += [row(cid, "INTERFERENCE", "relocation", r, True) for r in (0, 1)]
    for cid in ("V2_A_OFFSET_CONTROL", "V2_B_OFFSET_CONTROL"):
        rows += [row(cid, "OFFSET_CONTROL", "direct", r, True) for r in (0, 1)] + [row(cid, "OFFSET_CONTROL", "relocation", r, True) for r in (0, 1)]
    return rows


def ev(rows, pair=True, reach=True):
    return P.evaluate_mechanism(rows, pair, reach)


def test_R01_all_conditions_met_is_established():
    r = ev(established_rows())
    assert r["verdict"] == "ESTABLISHED" and r["reasons"] == [] and r["branches"] == 16


def test_R02_direct_success_in_both_repeats_of_an_interference_config_is_not_established():
    rows = established_rows()
    for r in rows[:2]:
        r.update(success=True, label="NOT_APPLICABLE_SUCCESS")
    assert ev(rows)["verdict"] == "NOT_ESTABLISHED"
    rows = established_rows()                                        # one direct success is still allowed
    rows[0].update(success=True, label="NOT_APPLICABLE_SUCCESS", contacts=0)
    assert ev(rows)["verdict"] == "ESTABLISHED"


def test_R03_direct_failure_with_a_non_physical_label_is_not_established():
    rows = established_rows()
    rows[0]["label"] = "CONTROLLER_EXECUTION_FAILURE"
    r = ev(rows)
    assert r["verdict"] == "NOT_ESTABLISHED" and any("PHYSICAL_GRASP_INTERFERENCE" in x for x in r["reasons"])


def test_R04_relocation_must_succeed_twice_and_physical_label_needs_saved_contact_evidence():
    rows = established_rows()
    rows[2].update(success=False, label="CONTROLLER_EXECUTION_FAILURE")
    assert any("relocation not 2/2" in x for x in ev(rows)["reasons"])
    rows = established_rows()
    for r in rows[:4]:
        r["contact_events"] = 0
    assert any("contact evidence" in x for x in ev(rows)["reasons"])


def test_R05_offset_controls_must_succeed_directly_and_never_show_physical_interference():
    rows = established_rows()
    ctl = [r for r in rows if r["config_id"] == "V2_A_OFFSET_CONTROL" and r["route"] == "direct"]
    ctl[0].update(success=False, label="CONTROLLER_EXECUTION_FAILURE")
    assert any("control direct not 2/2" in x for x in ev(rows)["reasons"])
    rows = established_rows()
    [r for r in rows if r["config_id"] == "V2_B_OFFSET_CONTROL" and r["route"] == "relocation"][0].update(success=False, label="PHYSICAL_GRASP_INTERFERENCE")
    assert any("physical interference labelled in a control" in x for x in ev(rows)["reasons"])


def test_R06_engineering_failure_rate_limit_is_12_5_percent():
    rows = established_rows()
    rows[4]["engineering_failure"] = rows[5]["engineering_failure"] = True                     # 2/16 = 12.5 % is allowed
    assert not any("ENGINEERING" in x for x in ev(rows)["reasons"])
    rows[6]["engineering_failure"] = True                                                      # 3/16 is not
    r = ev(rows)
    assert r["verdict"] == "NOT_ESTABLISHED" and any("ENGINEERING_FAILURE_RATE" in x for x in r["reasons"])


@pytest.mark.parametrize("pair,reach,needle", [(False, True, "PAIRED_RESTORE_INVALID"), (True, False, "CONTRACT_REACHABLE")])
def test_R07_invalid_pairing_or_unreachable_route_is_not_established(pair, reach, needle):
    r = ev(established_rows(), pair, reach)
    assert r["verdict"] == "NOT_ESTABLISHED" and any(needle in x for x in r["reasons"])


def test_R08_perception_or_planner_cause_and_incomplete_waves_are_not_established():
    rows = established_rows()
    rows[0]["label"] = "PERCEPTION_VERIFIER_MISMATCH"
    assert any("PERCEPTION_OR_PLANNER" in x for x in ev(rows)["reasons"])
    assert ev(established_rows()[:15])["verdict"] == "NOT_ESTABLISHED"


def _branch(exit_, status, held_pub, held_hidden, contacts, pre=None):
    qa = {"tag": "after", "hidden_truth": {"target": [0.0, 0.0, 0.85 if not held_hidden else 1.0], "interferer": [0.1, 0.1, 0.845], "eef_pos": [0.0, 0.0, 0.99 if held_hidden else 0.9],
                                           "gripper_qpos": [0.02, -0.02] if held_hidden else [0.04, -0.04], "table_top_z": 0.825, "container": [0.5, 0.5, 0.83]}}
    facts = [{"fact_id": k, "value": v} for k, v in {"p:Held:target": held_pub, "p:GripperEmpty": "TRUE", "p:OnTable:target": "TRUE", "p:Held:interferer": "FALSE",
                                                     "p:Inside:target:container": "FALSE", "p:Open:container": "TRUE", "p:OnTable:interferer": "TRUE", "p:Inside:interferer:container": "FALSE",
                                                     "p:AtBuffer:target:buffer": "FALSE", "p:AtBuffer:interferer:buffer": "FALSE"}.items()]
    pairs = [{"cat1": "finger", "cat2": "interferer", "distance": 0.0, "normal_force": 3.0, "geom1": "a", "geom2": "b"}] if contacts else []
    a = {"dir": "x", "name": "action_00", "candidate_id": "a:PICK:target:v1", "controller_exit": exit_, "facts": facts, "planner": {"status": status} if status else None, "evaluator": {},
         "snapshot": {"candidate_ids": ["a:PLACE:target:container:v1"], "candidate_mask": [False]}, "trace": [], "qa": [qa], "perception": [],
         "contacts": [{"phase": "DESCEND", "step": 1, "pairs": pairs}], "files": {}}
    return {"branch": {"branch_id": "b"}, "result": {"task_success": False, "termination_reason": "x"}, "actions": [a]}


@pytest.fixture(scope="module")
def contracts():
    import yaml
    from pathlib import Path
    from cp_disr.platforms.libero.tp_sr_runtime import ground_task_contracts
    root = Path(__file__).resolve().parents[1]
    return ground_task_contracts(root / "configs/runtime/tp_sr_v2_contract_registry.yaml", {"PICK": 9.0, "PLACE": 8.0, "PLACE_BUFFER": 8.0})


def test_R08b_physical_label_needs_contact_state_and_an_explained_stop(contracts):
    no_plan = P.classify_prototype_branch(_branch("NORMAL_TERMINATION", "NO_PLAN", "FALSE", False, True), contracts)
    assert no_plan["label"] == "PHYSICAL_GRASP_INTERFERENCE" and no_plan["checks"]["explanation_path"].startswith("NO_PLAN")
    abnormal = P.classify_prototype_branch(_branch("INCOMPLETE_MOTION", None, "FALSE", False, True), contracts)
    assert abnormal["label"] == "PHYSICAL_GRASP_INTERFERENCE" and abnormal["checks"]["explanation_path"].startswith("ABNORMAL")
    no_contact = P.classify_prototype_branch(_branch("NORMAL_TERMINATION", "NO_PLAN", "FALSE", False, False), contracts)
    assert no_contact["label"] == "CONTROLLER_EXECUTION_FAILURE"                         # NO_PLAN alone never means physical
    mismatch = P.classify_prototype_branch(_branch("NORMAL_TERMINATION", "NO_PLAN", "FALSE", True, True), contracts)
    assert mismatch["label"] == "PERCEPTION_VERIFIER_MISMATCH"
