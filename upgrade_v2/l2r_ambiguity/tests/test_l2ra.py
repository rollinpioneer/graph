from __future__ import annotations

import json
import csv
from pathlib import Path

from upgrade_v2.l2r_ambiguity.confirm import confirm
from upgrade_v2.l2r_ambiguity.evaluate import candidate_sequence
from upgrade_v2.l2r_ambiguity.reference_events import extract_probe_reference_events, extract_reference_events
from upgrade_v2.l2r_ambiguity.event_memory import (
    FALSE,
    TRUE,
    UNKNOWN,
    c1_effective_guard,
    run_memory,
    tri_and,
    tri_not,
    tri_or,
)


def obs(index, *, contact, closed, stable, slip="false", failed=None, **extra):
    failed = ("true" if closed == "true" and contact == "false" else "false") if failed is None else failed
    predicates = {
        "contact_present": contact,
        "gripper_command_closed": closed,
        "gripper_command_open": "false" if closed == "true" else "true" if closed == "false" else "unknown",
        "stable_hold_observed": stable,
        "contact_recently_lost": slip,
        "slip_observed": slip,
        "grasp_failed_observed": failed,
    }
    predicates.update(extra)
    return {"frame_index": index, "time": index * 0.1, "predicates": predicates}


def write_reference_rollout(root: Path, *, event: str, event_time: float, weld_before: int, intervention: bool = False, commanded_release: bool = True):
    root.mkdir(parents=True)
    with (root / "actions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["action_index", "action", "start_time", "end_time", "gripper_command", "contact_present", "termination_reason"])
        writer.writeheader()
        writer.writerow({"action_index": 0, "action": "close_gripper", "start_time": 0.0, "end_time": 0.8, "gripper_command": "closed", "contact_present": 1, "termination_reason": ""})
        writer.writerow({"action_index": 1, "action": "open_gripper" if event == "contact_lost" and weld_before and commanded_release else "recover", "start_time": 0.8, "end_time": 1.2, "gripper_command": "open" if event == "contact_lost" and weld_before and commanded_release else "closed", "contact_present": 0, "termination_reason": ""})
    with (root / "oracle_timeline.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_index", "time", "weld_state", "contact_present"])
        writer.writeheader()
        writer.writerow({"frame_index": 0, "time": 0.8, "weld_state": weld_before, "contact_present": 1})
        writer.writerow({"frame_index": 1, "time": 1.2, "weld_state": 0, "contact_present": 0})
    (root / "events.jsonl").write_text(json.dumps({"event": event, "time": event_time, "observable_online": True}) + "\n", encoding="utf-8")
    (root / "termination.json").write_text(json.dumps({"done": True, "termination_type": "diagnostic_complete"}), encoding="utf-8")
    (root / "metadata.json").write_text(json.dumps({"observation_intervention": intervention}), encoding="utf-8")


def test_frozen_rule_overlap_and_priority_control_preserves_it():
    item = {"observations": [obs(0, contact="true", closed="true", stable="true"), obs(1, contact="false", closed="true", stable="false", slip="true")], "reference": {"history_complete": True}}
    priority = candidate_sequence(item, "D_priority_only")
    assert priority["any_conflict"] is True
    assert priority["selected_action"] == "recover_object"
    assert c1_effective_guard(item["observations"][1]["predicates"]) == {"retry_grasp": FALSE, "recover_object": TRUE}


def test_causal_prefix_does_not_depend_on_future():
    prefix = [obs(0, contact="false", closed="false", stable="false"), obs(1, contact="false", closed="true", stable="false")]
    future_a = prefix + [obs(2, contact="true", closed="true", stable="true")]
    future_b = prefix + [obs(2, contact="false", closed="true", stable="false")]
    assert run_memory(prefix) == run_memory(future_a)[: len(prefix)]
    assert run_memory(prefix) == run_memory(future_b)[: len(prefix)]


def test_unknown_three_valued_logic():
    assert tri_not(UNKNOWN) == UNKNOWN
    assert tri_and(UNKNOWN, FALSE) == FALSE
    assert tri_and(UNKNOWN, TRUE) == UNKNOWN
    assert tri_or(UNKNOWN, TRUE) == TRUE
    assert tri_or(UNKNOWN, FALSE) == UNKNOWN


def test_miss_loss_touch_and_release_are_distinct():
    miss = run_memory([obs(0, contact="false", closed="false", stable="false"), obs(1, contact="false", closed="true", stable="false")])
    assert miss[-1]["selected_action"] == "retry_grasp"
    loss = run_memory([obs(0, contact="false", closed="false", stable="false"), obs(1, contact="true", closed="true", stable="true"), obs(2, contact="false", closed="true", stable="false", slip="true")])
    assert loss[-1]["selected_action"] == "recover_object"
    touch = run_memory([obs(0, contact="false", closed="false", stable="false"), obs(1, contact="true", closed="true", stable="false"), obs(2, contact="false", closed="true", stable="false", slip="true")])
    assert touch[-1]["selected_action"] == "needs_observation"
    release = run_memory([obs(0, contact="false", closed="false", stable="false"), obs(1, contact="true", closed="true", stable="true"), obs(2, contact="false", closed="false", stable="false")])
    assert release[-1]["semantic_events"]["release_expected"] == TRUE
    assert release[-1]["selected_action"] == "none"


def test_long_gap_recovery_and_new_attempt_cleanup():
    rows = [obs(0, contact="false", closed="false", stable="false"), obs(1, contact="true", closed="true", stable="true"), obs(2, contact="false", closed="true", stable="false", slip="true")]
    rows.extend(obs(i, contact="false", closed="true", stable="false", slip="false") for i in range(3, 9))
    long_gap = run_memory(rows)
    assert all(row["selected_action"] == "recover_object" for row in long_gap[2:])
    recovered = run_memory(rows + [obs(9, contact="true", closed="true", stable="true")])
    assert recovered[-1]["semantic_events"]["recovery_achieved_observed"] == TRUE
    assert recovered[-1]["selected_action"] == "none"
    new_attempt = run_memory(rows + [obs(9, contact="false", closed="false", stable="false"), obs(10, contact="false", closed="true", stable="false")])
    assert new_attempt[-1]["selected_action"] == "retry_grasp"


def test_missed_grasp_clears_after_confirmed_stable_hold():
    rows = [
        obs(0, contact="false", closed="false", stable="false"),
        obs(1, contact="false", closed="true", stable="false"),
        obs(2, contact="true", closed="true", stable="true"),
        obs(3, contact="true", closed="true", stable="true"),
    ]
    outputs = run_memory(rows)
    assert outputs[1]["selected_action"] == "retry_grasp"
    assert outputs[2]["semantic_events"]["recovery_achieved_observed"] == TRUE
    assert outputs[2]["selected_action"] == "none"
    assert outputs[3]["selected_action"] == "none"


def test_release_marker_is_scoped_to_the_current_attempt():
    rows = [
        obs(0, contact="true", closed="true", stable="true"),
        obs(1, contact="false", closed="false", stable="false"),
        obs(2, contact="false", closed="true", stable="false"),
        obs(3, contact="true", closed="true", stable="true"),
        obs(4, contact="false", closed="true", stable="false", slip="true"),
    ]
    outputs = run_memory(rows)
    assert outputs[1]["semantic_events"]["release_expected"] == TRUE
    assert outputs[4]["selected_action"] == "recover_object"


def test_forbidden_metadata_and_action_names_do_not_change_online_output():
    base = [obs(0, contact="false", closed="false", stable="false"), obs(1, contact="false", closed="true", stable="false")]
    tainted = json.loads(json.dumps(base))
    for row in tainted:
        row.update({"scenario": "slip_then_recover", "future_outcome": True, "action": "recover", "action_code": 999})
        row["predicates"].update({"gold_mode": "recover", "qpos": [1, 2, 3]})
    assert run_memory(base) == run_memory(tainted)


def test_new_guard_fields_do_not_change_raw_denominator():
    observations = [obs(0, contact="true", closed="true", stable="true"), obs(1, contact="false", closed="true", stable="false", slip="true")]
    before = sum(len(row["predicates"]) for row in observations)
    outputs = run_memory(observations)
    assert sum(len(row["predicates"]) for row in observations) == before
    assert len(outputs) == len(observations)
    assert candidate_sequence({"observations": observations, "reference": {"history_complete": True}}, "D_priority_only")["any_conflict"] is True


def test_confirmation_gate_refuses_unselected_candidate_without_consuming(tmp_path: Path):
    run_root = tmp_path / "run"
    lock_path = run_root / "locks/selection_lock.json"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(json.dumps({"status": "DEVELOPMENT_NOT_READY", "selected_candidate_id": None}), encoding="utf-8")
    result1 = confirm(lock_path, {}, {}, run_root, tmp_path / "frozen")
    first = json.loads((run_root / "rounds/l2ra_4_fresh_confirmation/confirmation_consumption.json").read_text())
    result2 = confirm(lock_path, {}, {}, run_root, tmp_path / "frozen")
    second = json.loads((run_root / "rounds/l2ra_4_fresh_confirmation/confirmation_consumption.json").read_text())
    assert result1["status"] == result2["status"] == "BLOCKED"
    assert first == second
    assert first["started_once"] is False


def test_online_module_has_no_reference_event_dependency():
    source = Path(__file__).parents[1].joinpath("event_memory.py").read_text(encoding="utf-8")
    assert "reference_events" not in source
    for forbidden in ("scenario", "future_outcome", "action_code", "qpos", "qvel", "weld_state"):
        assert forbidden not in source


def test_reference_event_label_does_not_depend_on_stratum(tmp_path: Path):
    root = tmp_path / "release"
    write_reference_rollout(root, event="contact_lost", event_time=1.0, weld_before=1)
    base = {"rollout_id": "r0", "root_family_id": "f0", "rollout_path": str(root), "stratum": "commanded_release"}
    first = extract_probe_reference_events(base)[0]
    base["stratum"] = "loss_after_observed_hold"
    second = extract_probe_reference_events(base)[0]
    assert first["reference_event_type"] == second["reference_event_type"] == "release_expected"
    assert first["reference_action_class"] == second["reference_action_class"] == "none"


def test_reference_event_distinguishes_transient_touch_and_hidden_history(tmp_path: Path):
    touch = tmp_path / "touch"
    touch.mkdir()
    (touch / "actions.csv").write_text("action_index,action,start_time,end_time\n0,touch_contact,0,1\n", encoding="utf-8")
    (touch / "oracle_timeline.csv").write_text("frame_index,time,weld_state,contact_present\n0,1,0,1\n", encoding="utf-8")
    (touch / "events.jsonl").write_text(json.dumps({"event": "transient_contact", "time": 1.0}) + "\n", encoding="utf-8")
    (touch / "metadata.json").write_text(json.dumps({"observation_intervention": False}), encoding="utf-8")
    touch_record = extract_probe_reference_events({"rollout_id": "touch", "root_family_id": "f", "rollout_path": str(touch)})[0]
    assert touch_record["reference_event_type"] == "touch_without_stable_hold"
    assert touch_record["reference_action_class"] == "none"

    hidden = tmp_path / "hidden"
    write_reference_rollout(hidden, event="contact_lost", event_time=1.0, weld_before=1, intervention=True, commanded_release=False)
    hidden_record = extract_probe_reference_events({"rollout_id": "hidden", "root_family_id": "f", "rollout_path": str(hidden)})[0]
    assert hidden_record["reference_event_type"] == "held_object_loss_recovery_required"
    assert hidden_record["reference_action_class"] == "needs_observation"
    assert hidden_record["reference_observable_at_decision"] is False


def test_reference_event_without_oracle_is_unresolved(tmp_path: Path):
    root = tmp_path / "no_oracle"
    root.mkdir()
    (root / "actions.csv").write_text("action_index,action,start_time,end_time\n0,recover,0,1\n", encoding="utf-8")
    (root / "events.jsonl").write_text(json.dumps({"event": "contact_lost", "time": 1.0}) + "\n", encoding="utf-8")
    record = extract_reference_events({"rollout_id": "r", "root_family_id": "f", "rollout_path": str(root)})[0]
    assert record["reference_event_type"] == "needs_observation"
    assert record["reference_action_class"] == "needs_observation"
    assert record["reference_label_status"] == "reference_unresolved"


def test_reference_event_uses_prefix_only_for_decision_frame(tmp_path: Path):
    root = tmp_path / "prefix"
    write_reference_rollout(root, event="contact_lost", event_time=1.0, weld_before=1)
    record = extract_probe_reference_events({"rollout_id": "r", "root_family_id": "f", "rollout_path": str(root)})[0]
    assert record["decision_frame_index"] == 0
    assert record["onset_interval_start"] == 0.8
    assert record["onset_interval_end"] == 1.2
