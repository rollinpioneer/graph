from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cp_disr.analysis import s4_family_a_soft_ordering_mvp as m


BRANCHES = (
    {"branch_id": "tech-a", "wave": "technical"},
    {"branch_id": "tech-b", "wave": "technical"},
)
REQUIRED = (
    "before_rgb.png",
    "before_depth.npy",
    "after_rgb.png",
    "after_depth.npy",
    "controller_trace.jsonl",
    "facts.json",
    "evaluator.json",
    "snapshot.json",
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_action(out, bid, index, *, terminal=False, planner=True, missing=(), planner_call=False):
    action = out / "captures" / bid / f"action_{index:02d}"
    action.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        if name not in missing:
            path = action / name
            if name.endswith(".png") or name.endswith(".npy"):
                path.write_bytes(b"fixture")
            elif name == "controller_trace.jsonl":
                rows = [{
                    "event": "skill_begin",
                    "candidate_id": f"a:SKILL:{index}:v1",
                }]
                if planner_call:
                    rows.append({"event": "planner_call"})
                path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            elif name == "facts.json":
                path.write_text("[]")
            elif name == "evaluator.json":
                _write_json(path, {
                    "task_success": terminal,
                    "terminated": terminal,
                    "truncated": False,
                    "reason": "TASK_SUCCESS" if terminal else "CONTINUE",
                })
            elif name == "snapshot.json":
                _write_json(path, {"candidate_mask": [True]})
    if planner:
        _write_json(action / "planner.json", {
            "input_facts": {},
            "candidate_ids": [],
            "status": "PLAN_FOUND",
            "plan": ["a:SKILL"],
            "expanded_nodes": 1,
            "cpu_seconds": 0.001,
        })


def _make_branch(out, bid, terminal_index=3, *, final_planner=False, terminal_missing=(), terminal_planner_call=False, result_success=True, extra=False):
    trace = []
    for index in range(terminal_index + (2 if extra else 1)):
        terminal = index == terminal_index
        _make_action(
            out,
            bid,
            index,
            terminal=terminal,
            planner=(not terminal) or final_planner,
            missing=terminal_missing if terminal else (),
            planner_call=terminal_planner_call if terminal else False,
        )
        trace.append({"candidate_id": f"a:SKILL:{index}:v1"})
    if extra:
        _make_action(out, bid, terminal_index + 1, terminal=False, planner=True)
    _write_json(out / "physical" / "branch_results" / f"{bid}.json", {
        "task_success": result_success,
        "trace": trace,
    })


def test_T6_A1_terminal_success_without_planner_passes(tmp_path):
    _make_branch(tmp_path, "tech-a")
    result = m._check_terminal_aware_action_records(tmp_path, BRANCHES[0])
    assert result["complete"]
    assert result["actions"][-1]["planner_record_status"] == "NOT_APPLICABLE_TERMINAL_SUCCESS"
    assert not (tmp_path / "captures/tech-a/action_03/planner.json").exists()


def test_T6_A2_nonterminal_continue_without_planner_fails(tmp_path):
    _make_branch(tmp_path, "tech-a")
    (tmp_path / "captures/tech-a/action_01/planner.json").unlink()
    result = m._check_terminal_aware_action_records(tmp_path, BRANCHES[0])
    assert not result["complete"]


def test_T6_A3_terminal_success_with_real_planner_call_fails(tmp_path):
    _make_branch(tmp_path, "tech-a", terminal_planner_call=True)
    assert not m._check_terminal_aware_action_records(tmp_path, BRANCHES[0])["complete"]


def test_T6_A4_terminal_action_missing_facts_fails(tmp_path):
    _make_branch(tmp_path, "tech-a", terminal_missing=("facts.json",))
    assert not m._check_terminal_aware_action_records(tmp_path, BRANCHES[0])["complete"]


def test_T6_A5_terminal_action_missing_evaluator_fails(tmp_path):
    _make_branch(tmp_path, "tech-a", terminal_missing=("evaluator.json",))
    assert not m._check_terminal_aware_action_records(tmp_path, BRANCHES[0])["complete"]


def test_T6_A6_terminal_action_missing_controller_trace_fails(tmp_path):
    _make_branch(tmp_path, "tech-a", terminal_missing=("controller_trace.jsonl",))
    assert not m._check_terminal_aware_action_records(tmp_path, BRANCHES[0])["complete"]


def test_T6_A7_branch_result_mismatch_fails(tmp_path):
    _make_branch(tmp_path, "tech-a", result_success=False)
    assert not m._check_terminal_aware_action_records(tmp_path, BRANCHES[0])["complete"]


def test_T6_A8_action_after_terminal_fails(tmp_path):
    _make_branch(tmp_path, "tech-a", extra=True)
    assert not m._check_terminal_aware_action_records(tmp_path, BRANCHES[0])["complete"]


def test_T6_A9_original_gate_bytes_are_not_changed(tmp_path):
    gate = tmp_path / "physical/technical_wave_check.json"
    stop = tmp_path / "decision/stop_state.json"
    reg = tmp_path / "physical/witnesses/e4_branch_registration.json"
    _write_json(gate, {
        "technical_wave": "FAIL",
        "checks": {
            "T0_technical_wave_size": True,
            "T1_both_terminal": True,
            "T2_paired_initial_state_identical": True,
            "T3_qpos_qvel_max_diff_le_1e-9": True,
            "T4_both_task_success": True,
            "T5_exactly_four_skills": True,
            "T6_per_action_records_complete": False,
            "T7_one_attempt_and_one_reset_each": True,
            "T8_no_duplicate_branch": True,
            "T9_no_global_perception_residue": True,
            "T10_provider_rl_optimizer_zero": True,
        },
    })
    _write_json(stop, {"status": "STOPPED"})
    _write_json(reg, {"branches": list(BRANCHES)})
    for branch in BRANCHES:
        _make_branch(tmp_path, branch["branch_id"])
    before = hashlib.sha256(gate.read_bytes()).hexdigest()
    result = m.amend_technical_gate(Path.cwd(), "unused", tmp_path)
    assert result["technical_wave"] == "PASS"
    assert hashlib.sha256(gate.read_bytes()).hexdigest() == before


def test_T6_A10_amendment_does_not_charge_budget(tmp_path):
    _write_json(tmp_path / "physical/technical_wave_check.json", {
        "technical_wave": "FAIL",
        "checks": {"T6_per_action_records_complete": False},
    })
    _write_json(tmp_path / "decision/stop_state.json", {"status": "STOPPED"})
    _write_json(tmp_path / "physical/witnesses/e4_branch_registration.json", {"branches": list(BRANCHES)})
    for branch in BRANCHES:
        _make_branch(tmp_path, branch["branch_id"])
    result = m.amend_technical_gate(Path.cwd(), "unused", tmp_path)
    assert result["new_attempts"] == 0
    assert result["new_resets"] == 0


def test_T6_A11_new_rule_accepts_what_old_rule_rejected(tmp_path):
    _make_branch(tmp_path, "tech-a")
    old_rule = all(
        (tmp_path / "captures/tech-a" / f"action_{index:02d}" / "planner.json").is_file()
        for index in range(4)
    )
    new_rule = m._check_terminal_aware_action_records(tmp_path, BRANCHES[0])["complete"]
    assert old_rule is False
    assert new_rule is True
