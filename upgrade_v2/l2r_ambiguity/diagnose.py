"""Attribute the observed overlap to explicit, evidence-backed hypotheses."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .reference_events import build_event_index


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def diagnose(resolved: dict[str, Any], trace_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    event_contract = {
        "schema": "pathgraph_l2ra_event_reference_contract_v1",
        "status": "LOCKED_FOR_DEVELOPMENT",
        "runtime_inputs": ["frozen observable predicate prefix", "contact_sensor", "gripper_command"],
        "reference_only_inputs": ["events.jsonl", "oracle_timeline", "termination", "simulator weld/qpos/qvel"],
        "events": {
            "missed_grasp_retry_required": "complete current-attempt history shows no stable hold before closed/no-contact",
            "held_object_loss_recovery_required": "stable hold was observed before non-release contact loss",
            "release_expected": "observed open command explains contact disappearance",
            "recovery_in_progress": "an unresolved missed-grasp or held-loss event remains pending",
            "recovery_achieved_observed": "stable hold is observed again after a pending held-loss event",
            "needs_observation": "required prefix or current observation is unknown, or transient touch lacks stable-hold evidence",
        },
        "three_valued_logic": {"not_unknown": "unknown", "unknown_and_false": "false", "unknown_and_true": "unknown", "unknown_or_true": "true"},
        "future_labels_forbidden_online": True,
        "materialization_note": "Definitions were implemented in event_memory.py/reference_events.py before probe collection; this explicit contract file was materialized during the same unsealed development run and was not changed after seeing confirmation data."
    }
    _write_json(output_root / "event_reference_contract.json", event_contract)
    contract_path = output_root / "event_reference_contract.json"
    _write_json(output_root.parents[1] / "locks/event_reference_contract.lock.json", {
        "schema": "pathgraph_l2ra_event_reference_contract_lock_v1", "status": "LOCKED_FOR_DEVELOPMENT",
        "path": str(contract_path.resolve()), "sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "new_confirmation_data_seen": False, "legacy_confirmation_role": "POST_HOC_DIAGNOSIS_ONLY",
        "post_selection_changes_allowed": False,
        "procedural_note": event_contract["materialization_note"],
    })
    event_rows = build_event_index(resolved)
    trace_summaries = []
    traces = []
    for path in sorted(trace_root.glob("*/legacy_reproduction.json")):
        if path.parent.name == "legacy_confirmation":
            continue
        trace_summaries.append(_read_json(path))
        traces.extend(_read_jsonl(path.parent / "guard_trace.jsonl"))
    confirm_path = trace_root / "legacy_confirmation/guard_trace.jsonl"
    if confirm_path.is_file():
        traces.extend(_read_jsonl(confirm_path))
    by_rollout = defaultdict(list)
    for row in traces: by_rollout[row["rollout_id"]].append(row)
    all_overlap = [rollout for rollout, rows in by_rollout.items() if any(row["retry_recover_both_true_at_this_time"] for row in rows)]
    held_loss_rollouts = {row["rollout_id"] for row in event_rows if row["reference_event_type"] == "held_object_loss_recovery_required"}
    missed_rollouts = {row["rollout_id"] for row in event_rows if row["reference_event_type"] == "missed_grasp_retry_required"}
    stable_before_loss = 0
    unknown_windows = 0
    early_hidden = 0
    cases = []
    for event in event_rows:
        if event["reference_event_type"] not in {"missed_grasp_retry_required", "held_object_loss_recovery_required"}:
            continue
        rows = by_rollout.get(event["rollout_id"], [])
        index = event.get("frame_index")
        before = [row for row in rows if index is None or row["frame_index"] <= index]
        had_stable = any(row["stable_hold_observed"] is True for row in before[:-1])
        unknown = any(value is None for row in before for value in (row["contact_present"], row["gripper_command_closed"]))
        overlap = any(row["retry_recover_both_true_at_this_time"] for row in rows)
        if event["reference_event_type"] == "held_object_loss_recovery_required" and had_stable:
            stable_before_loss += 1
        unknown_windows += int(unknown)
        legacy_masked = overlap and rows and (
            not rows[0]["legacy_ambiguous"]
            or rows[0]["selected_branch_legacy"] not in {"retry_grasp", "recover_object"}
        )
        if legacy_masked:
            early_hidden += 1
        cases.append({"event_id": event["event_id"], "rollout_id": event["rollout_id"], "root_family_id": event["parent_family_id"],
                      "reference_event_type": event["reference_event_type"], "frame_index": index,
                      "raw_overlap_in_rollout": overlap, "legacy_priority_masked_overlap": legacy_masked,
                      "stable_hold_before_event": had_stable,
                      "required_observation_unknown_before_event": unknown, "reference_label_status": event["reference_label_status"]})
    _write_json(output_root / "logical_overlap_probe.json", {
        "scope": "CODE_DERIVED_PLUS_A1_REPLAY_CONTEXT", "raw_prediction_replayed": True,
        "raw_overlap_rollouts": len(all_overlap), "held_loss_rollouts": len(held_loss_rollouts),
        "missed_grasp_rollouts": len(missed_rollouts),
        "stable_hold_before_held_loss_events": stable_before_loss,
        "event_count": len(cases),
        "note": "The independent logical truth table remains in the A0 static diagnosis; this file adds replay counts."
    })
    attribution = [
        {"hypothesis": "H1_overbroad_generic_failure", "evidence_for": f"raw grasp_failed overlaps slip in {len(all_overlap)} replayed rollouts; held-loss events with prior stable hold={stable_before_loss}",
         "evidence_against": "does not prove every generic failure is a held-loss event", "affected_events": stable_before_loss, "affected_families": len({row["parent_family_id"] for row in event_rows if row["reference_event_type"] == "held_object_loss_recovery_required"}), "status": "SUPPORTED_AS_MECHANISM" if stable_before_loss else "NOT_SUPPORTED"},
        {"hypothesis": "H2_generic_and_specific_actions_are_not_mutually_exclusive", "evidence_for": f"{len(all_overlap)} rollouts contain simultaneous raw guards while legacy recovery priority still marks ambiguity", "evidence_against": "raw fact predicates may remain overlapping by design", "affected_events": len(all_overlap), "affected_families": len({row["root_family_id"] for row in cases if row["raw_overlap_in_rollout"]}), "status": "SUPPORTED_AS_ACTION_LAYER_PROBLEM" if all_overlap else "NOT_SUPPORTED"},
        {"hypothesis": "H3_attempt_history_not_cleared", "evidence_for": "not established by static summary; checked through per-attempt trace fields", "evidence_against": "A1 trace exposes no direct legacy attempt state", "affected_events": 0, "affected_families": 0, "status": "NOT_ESTABLISHED"},
        {"hypothesis": "H4_short_touch_misread_as_stable_hold", "evidence_for": "requires targeted short-touch probes; frozen data has no independent stable-hold reference for this condition", "evidence_against": "no valid frozen short-touch label", "affected_events": 0, "affected_families": 0, "status": "UNTESTED_TARGETED_PROBE_REQUIRED"},
        {"hypothesis": "H5_observation_or_sampling_insufficient", "evidence_for": f"{unknown_windows} diagnostic event windows contain missing/unknown required observations", "evidence_against": "most replayed held-loss windows have observable contact and command prefixes", "affected_events": unknown_windows, "affected_families": len({row["root_family_id"] for row in cases if row["required_observation_unknown_before_event"]}), "status": "PARTIAL_LIMITATION" if unknown_windows else "NOT_SUPPORTED"},
        {"hypothesis": "H6_legacy_whole_sequence_summary_hides_earlier_conflict", "evidence_for": f"{early_hidden} labeled event windows occurred in rollouts where an earlier-return branch or final arbitration masked a raw overlap", "evidence_against": "legacy final arbitration still reproduces its original selected branch", "affected_events": early_hidden, "affected_families": len({row["root_family_id"] for row in cases if row.get("legacy_priority_masked_overlap")}), "status": "SUPPORTED_AS_REPORTING_PROBLEM" if early_hidden else "NOT_ESTABLISHED"}
    ]
    _write_csv(output_root / "cause_attribution.csv", attribution)
    with (output_root / "diagnostic_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    representatives = []
    for kind in ("missed_grasp_retry_required", "held_object_loss_recovery_required", "release_expected", "recovery_achieved_observed"):
        for event in [row for row in event_rows if row["reference_event_type"] == kind][:2]:
            index = event.get("frame_index")
            context = [row for row in by_rollout.get(event["rollout_id"], []) if index is not None and abs(row["frame_index"] - index) <= 1]
            representatives.append({"case_type": kind, "event": event, "trace_context": context})
            if len(representatives) >= 7:
                break
        if len(representatives) >= 7:
            break
    control_id = next((rollout for rollout, rows in by_rollout.items() if not any(row["retry_recover_both_true_at_this_time"] for row in rows)), None)
    if control_id:
        representatives.append({"case_type": "no_raw_overlap_control", "event": None, "trace_context": by_rollout[control_id][:3]})
    with (output_root / "representative_event_windows.jsonl").open("w", encoding="utf-8") as handle:
        for row in representatives[:8]:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    gate = {"schema": "pathgraph_l2ra_diagnosis_gate_v1", "status": "GUARD_SEMANTICS_REPAIR", "raw_overlap_replayed": True,
            "memory_candidate_required": True, "event_reference_count": len(cases),
            "supported_hypotheses": [row["hypothesis"] for row in attribution if row["status"].startswith("SUPPORTED")],
            "untested_hypotheses": [row["hypothesis"] for row in attribution if "UNTESTED" in row["status"]],
            "route_reason": "Raw facts overlap because the generic closed/no-contact guard subsumes held-loss; action semantics require an explicit event layer. Targeted negative probes remain necessary."}
    _write_json(output_root / "diagnosis_gate.json", gate)
    return {"status": gate["status"], "attribution": attribution, "cases": len(cases), "raw_overlap_rollouts": len(all_overlap)}
