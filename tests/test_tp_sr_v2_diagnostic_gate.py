"""D01-D08: Wave D registration, budget, gate and root-cause rules."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cp_disr.analysis import s4_tp_sr_pilot_v2 as m
from cp_disr.analysis import s4_tp_sr_pilot_v2_diag as d

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/final_master/s4_tp_sr_pilot_v2.yaml"


def test_D01_D02_D03_exactly_four_branches_four_charges_no_fifth(tmp_path):
    out = tmp_path / "o"
    for sub in ("inventory", "spec", "diagnostics"):
        (out / sub).mkdir(parents=True)
    (out / "stage_manifest.json").write_text(json.dumps({"phases_done": []}))
    m.init_ledger(out, m.load_config(CFG))
    res = m.prepare_diagnostics(ROOT, CFG, out)
    assert len(res["branches"]) == 4 and {b[0] for b in res["branches"]} == {"T_P_SR_pool_33", "T_P_SR_pool_57"}
    reg = json.loads((out / "diagnostics/witnesses/e4_branch_registration.json").read_text())
    assert {b["route"] for b in reg["branches"]} == {"direct", "relocation"} and len({b["branch_id"] for b in reg["branches"]}) == 4
    for i in range(4):                                                  # D02: four attempt charges = four environment resets
        m.charge_many(out, ["wave_d_branch_attempts", "total_branch_attempts", "environment_resets"], ref=str(i))
    led = m.rd(out / "budget_ledger.json")
    assert led["wave_d_branch_attempts"]["used"] == led["environment_resets"]["used"] == 4
    with pytest.raises(m.StopRun):                                       # D03: no fifth diagnostic attempt
        m.charge_many(out, ["wave_d_branch_attempts", "total_branch_attempts", "environment_resets"], ref="5")
    assert m.rd(out / "budget_ledger.json")["total_branch_attempts"]["used"] == 4      # nothing consumed by the refused charge
    with pytest.raises(m.StopRun):
        m.prepare_diagnostics(ROOT, CFG, out)                            # registration is one-shot


def _facts(**over):
    base = {"p:GripperEmpty": "TRUE", "p:Held:target": "FALSE", "p:Held:interferer": "FALSE", "p:OnTable:target": "TRUE", "p:OnTable:interferer": "TRUE",
            "p:Open:container": "TRUE", "p:Inside:target:container": "FALSE", "p:Inside:interferer:container": "FALSE", "p:AtBuffer:target:buffer": "FALSE",
            "p:AtBuffer:interferer:buffer": "FALSE"}
    base.update(over)
    return [{"fact_id": k, "value": v, "reason": "r", "evidence_ids": [], "capture_time": 0, "available_time": 0, "last_confirmed_value": v, "last_confirmed_time": 0} for k, v in base.items()]


def _qa(target_z, eef=(0, 0, 1.0), grip=(0.04, -0.04), inside=False):
    tgt = [0.03, 0.15, 0.83] if inside else [0.0, 0.0, target_z]
    return {"tag": "after", "hidden_truth": {"target": tgt, "interferer": [0.1, 0, 0.845], "table_top_z": 0.825, "eef_pos": list(eef), "gripper_qpos": list(grip),
                                              "container": [0.03, 0.15, 0.829]}}


def _branch(*, exit_="NORMAL_TERMINATION", facts=None, planner=None, qa=None, contacts=None, cand="a:PICK:target:v1", success=False, term="NO_PLAN"):
    trace = [{"kind": "event", "event": "skill_begin", "candidate_id": cand}, {"kind": "event", "event": "skill_end", "controller_exit": exit_, "steps": 1}, {"kind": "step"}]
    a = {"dir": Path("/nonexistent"), "name": "action_00", "candidate_id": cand, "controller_exit": exit_, "reported_steps": 1, "step_records": 1, "trace": trace,
         "facts": facts if facts is not None else _facts(), "planner": planner, "evaluator": {"terminated": False}, "qa": [qa] if qa else [], "contacts": contacts or [],
         "snapshot": {"candidate_ids": ["a:PLACE:target:container:v1"], "candidate_mask": [True]}, "perception": [], "files": {}}
    return {"branch": {"branch_id": "b", "case_id": "c"}, "result": {"task_success": success, "termination_reason": term}, "actions": [a]}


def _contracts():
    import yaml
    from cp_disr.platforms.libero.tp_sr_runtime import ground_task_contracts
    man = yaml.safe_load((ROOT / "runs/final_master/S4/tp_soft_relocation_design/20260929T084850Z_b598cfd0/spec/runtime_manifest.yaml").read_text())
    return ground_task_contracts(ROOT / "configs/runtime/tp_sr_v2_contract_registry.yaml", man["runtime"]["skill_timeouts"]["T_P_SR"])


NO_PLAN = {"status": "NO_PLAN"}
CONTACT = [{"step": 3, "phase": "DESCEND", "sim_time": 1.0, "pairs": [{"geom1": "gripper0_finger1_pad_collision", "geom2": "interferer_g0", "cat1": "finger", "cat2": "interferer",
                                                                     "distance": -0.0002, "normal_force": 3.0}]}]


def test_D07_D08_no_plan_alone_never_labels_physical_and_physical_needs_contact_and_state():
    c = _contracts()
    only_no_plan = d.classify_branch(_branch(planner=NO_PLAN, qa=None), c)
    assert only_no_plan["label"] == "UNRESOLVED"                                       # missing hidden QA: NO_PLAN alone decides nothing
    no_contact = d.classify_branch(_branch(planner=NO_PLAN, qa=_qa(0.845)), c)
    assert no_contact["label"] == "CONTROLLER_EXECUTION_FAILURE"                       # not held, no interferer contact
    with_contact = d.classify_branch(_branch(planner=NO_PLAN, qa=_qa(0.845), contacts=CONTACT), c)
    assert with_contact["label"] == "PHYSICAL_GRASP_INTERFERENCE" and with_contact["interferer_contacts"]
    held_mismatch = d.classify_branch(_branch(planner=NO_PLAN, qa=_qa(0.99, eef=(0, 0, 1.0), grip=(0.02, -0.02)), contacts=CONTACT), c)
    assert held_mismatch["label"] == "PERCEPTION_VERIFIER_MISMATCH"                    # hidden says held, public says FALSE


def test_planner_input_error_and_symbolic_labels():
    c = _contracts()
    held_true = _facts(**{"p:Held:target": "TRUE", "p:GripperEmpty": "FALSE", "p:OnTable:target": "FALSE"})
    lab = d.classify_branch(_branch(planner=NO_PLAN, facts=held_true, qa=_qa(0.99, grip=(0.02, -0.02))), c)
    assert lab["label"] == "PLANNER_FACT_INPUT_ERROR"
    sym = d.classify_branch(_branch(cand="a:PICK:interferer:v1", planner={"status": "GOAL_ALREADY_SATISFIED"}, facts=_facts(**{"p:Inside:target:container": "TRUE"}),
                                    qa=_qa(0.845), term="SYMBOLIC_EVALUATOR_MISMATCH"), c)
    assert sym["label"] == "FALSE_SYMBOLIC_GOAL_FROM_PUBLIC_FACTS"


def _gate_fixture(tmp, *, unresolved=False, drop_rgb=False, drop_facts=False):
    from tests.test_tp_sr_v2_d7_amendment import build_identity_fixture
    out, cfg, reg = build_identity_fixture(tmp)
    phys = out / "diagnostics"
    for i, b in enumerate(reg["branches"]):
        q = np.zeros(6)
        (phys / "witnesses/initial_state_checks" / f"{b['branch_id']}.json").write_text(json.dumps({"status": "OK", "compare_key": {"k": i // 2}, "candidate_mask": [True],
                                                                                                    "physical": {"qpos": q.tolist(), "qvel": q.tolist()}}))
    return out, cfg, reg


def _mk_branch(out, b, *, unresolved=False, drop_rgb=False, drop_facts=False):
    a = {"dir": out, "name": "action_00", "candidate_id": "a:PICK:target:v1", "controller_exit": "NORMAL_TERMINATION", "reported_steps": 2, "step_records": 2,
         "trace": [{"event": "phase_begin"}], "facts": [] if drop_facts else _facts(), "evaluator": {"terminated": True}, "planner": None, "snapshot": None, "perception": [],
         "files": {n: True for n in d.REQUIRED_ACTION_FILES}, "qa": [], "contacts": []}
    if drop_rgb:
        a["files"]["after_rgb.png"] = False
    return {"branch": b, "result": {"execution_status": "TERMINATED", "task_success": True, "branch_id": b["branch_id"], "env_counts": {"reset_calls": 1},
                                    "global_perception_mask_unchanged": True, "provider_module_loaded": False, "recorder_errors": 0}, "actions": [a]}


@pytest.mark.parametrize("kw,gate", [({}, None), ({"drop_rgb": True}, "D3"), ({"drop_facts": True}, "D4")])
def test_D04_D05_D06_gate_blocks_on_unresolved_missing_rgb_or_facts(tmp_path, kw, gate):
    out, cfg, reg = _gate_fixture(tmp_path)
    brs = [_mk_branch(out, b, **kw) for b in reg["branches"]]
    roots = {"c0|direct": {"label": "UNRESOLVED"}, "c1|relocation": {"label": "UNRESOLVED"}}
    g = d.wave_d_gate(out, cfg, reg, brs, {"T_P_SR_pool_33|direct": {"label": "UNRESOLVED"}, "T_P_SR_pool_57|relocation": {"label": "UNRESOLVED"}},
                      {"frames_with_fact_differences": 0, "frames_not_reproducing_online": 0})
    assert g["wave_d_status"] == "FAIL" and g["wave_p_released"] is False and g["next_action"] == "EXPLICIT_RESEARCH_DECISION"
    assert g["gates"]["D8"]["pass"] is False and g["gates"]["D9"]["pass"] is False           # D04 unresolved blocks
    if gate:
        assert g["gates"][gate]["pass"] is False
    resolved = {"T_P_SR_pool_33|direct": {"label": "CONTROLLER_EXECUTION_FAILURE"}, "T_P_SR_pool_57|relocation": {"label": "FALSE_SYMBOLIC_GOAL_FROM_PUBLIC_FACTS"}}
    g2 = d.wave_d_gate(out, cfg, reg, brs, resolved, {"frames_with_fact_differences": 0, "frames_not_reproducing_online": 0})
    if not kw:
        assert g2["wave_d_status"] == "PASS" and g2["wave_p_released"] is True, g2["gates"]
    else:
        assert g2["wave_d_status"] == "FAIL"
