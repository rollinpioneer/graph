"""R15-B fair cache re-evaluation over the existing 32 repair rollouts.

O tier runs through the frozen Round-10 repaired interface
``run_repaired_interface``; S tier runs the corrected causal state machine.
Labels are joined only in the reporting layer.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from upgrade_v2.l2r_task_context.reference import build_reference as build_legacy_reference

from .adapters import (
    DATA_VALID,
    FORBIDDEN_CANDIDATE_INPUTS,
    REQUIRED_ROLLOUT_FILES,
    attach_request_provenance,
    build_o_samples,
    build_s_samples,
    load_rollout,
)
from .metrics import (
    CORRECT,
    MISSED_REQUIRED_ACTION,
    REFERENCE_UNRESOLVED,
    WRONG_ACTION,
    classify_outcome,
    summarize_method,
)
from .o_interface import (
    O_METHOD_TO_CANDIDATE,
    load_round10_comparison,
    parity_metrics,
    parity_rows,
    run_o_method,
)
from .state_machine import METHOD_G_HR, S_METHODS, GeometryEventStateMachine
from .weld_audit import (
    contact_lost_time,
    drift_audit,
    event_transition_relation,
    read_oracle_rows,
    relative_at_order,
    weld_transition,
)

EVALUATION_VERSION = "l2rar2_geometry_events_r15b_v2"
O_METHODS = tuple(O_METHOD_TO_CANDIDATE)
LOSS_CASES = ("K4_regular_hold_loss", "K5_brief_hold_loss", "K6_long_gap_after_loss")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _discover_rollouts(data_root: Path) -> list[Path]:
    return sorted(path for path in (Path(data_root) / "rollouts").glob("*/*") if path.is_dir())


def _reference_window(meta: dict[str, Any]) -> dict[str, Any] | None:
    try:
        reference = build_legacy_reference(dict(meta))
    except Exception as error:
        return {"status": "reference_unresolved", "reason": f"{type(error).__name__}: {error}"}
    if reference.get("status") != "reference_labeled" or not reference.get("events"):
        return {"status": "reference_unresolved", "reason": reference.get("reason")}
    event = reference["events"][0]
    return {
        "status": reference["status"],
        "expected_action": event.get("expected_action"),
        "start": float(event["decision_window_start"]),
        "end": float(event["decision_window_end"]),
    }


def _first_action_in_window(rows: list[dict[str, Any]], start: float, end: float, key: str) -> dict[str, Any]:
    for row in rows:
        time_value = row.get("time")
        if time_value is None:
            continue
        time_value = float(time_value)
        if time_value < start - 1e-9 or time_value > end + 1e-9:
            continue
        action = row.get(key)
        if action in {"retry_grasp", "recover_object"}:
            return {"selected_action": action, "selected_time": time_value, "reason": row.get("reason_code")}
    return {"selected_action": "none", "selected_time": None, "reason": None}


def _anomaly_flags(rows: list[dict[str, Any]], window: dict[str, Any] | None, key: str) -> dict[str, Any]:
    flags = {
        "premature_emergency": False,
        "premature_retry": False,
        "premature_recovery": False,
        "post_window_emergency": False,
    }
    if window is None:
        return flags
    for row in rows:
        action = row.get(key)
        if action not in {"retry_grasp", "recover_object"} or row.get("time") is None:
            continue
        time_value = float(row["time"])
        if time_value < window["start"] - 1e-9:
            flags["premature_emergency"] = True
            flags["premature_retry"] = flags["premature_retry"] or action == "retry_grasp"
            flags["premature_recovery"] = flags["premature_recovery"] or action == "recover_object"
        elif time_value > window["end"] + 1e-9:
            flags["post_window_emergency"] = True
    return flags


def _projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("hold_state", "ever_held", "lift_confirmed", "loss_source", "selected_action", "reason_code")
    return [{key: row.get(key) for key in keys} for row in rows]


def _prefix_causality(samples: list[dict[str, Any]], method_id: str, protocol: dict[str, Any], meta: dict[str, Any]) -> list[str]:
    full_machine = GeometryEventStateMachine(method_id, protocol, meta.get("rollout_id"), meta.get("root_family_id"))
    full = _projection(full_machine.run(samples))
    mismatches = []
    for length in range(1, len(samples) + 1):
        machine = GeometryEventStateMachine(method_id, protocol, meta.get("rollout_id"), meta.get("root_family_id"))
        prefix = _projection(machine.run(samples[:length]))
        if prefix != full[:length]:
            mismatches.append(f"method={method_id} prefix_length={length}")
            break
    return mismatches


def run_cache_evaluation_v2(
    data_root: Path,
    protocol_path: Path,
    round10_root: Path,
    resource_report_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    data_root = Path(data_root)
    output_root = Path(output_root)
    if output_root.exists():
        raise SystemExit("REFUSING_EXISTING_OUTPUT_ROOT: " + str(output_root))
    output_root.mkdir(parents=True)

    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    resource_report = json.loads(Path(resource_report_path).read_text(encoding="utf-8"))
    parameters = protocol["parameters"]

    rollout_dirs = _discover_rollouts(data_root)
    o_traces: list[dict[str, Any]] = []
    o_decisions: list[dict[str, Any]] = []
    s_traces: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    weld_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    causality: dict[str, Any] = {"schema": "l2rar2_r15b_prefix_causality_v1", "checked": 0, "mismatches": []}
    leakage = {
        "schema": "l2rar2_r15b_input_leakage_v1",
        "o_samples_checked": 0,
        "s_samples_checked": 0,
        "violations": [],
    }
    per_rollout: list[dict[str, Any]] = []

    for rollout_dir in rollout_dirs:
        loaded = load_rollout(rollout_dir)
        if loaded.get("status") != "OK":
            raise SystemExit("ROLLOUT_INCOMPLETE " + str(rollout_dir))
        meta = loaded["metadata"]
        rollout_id = meta.get("rollout_id")
        case_id = meta.get("case_id")
        window = _reference_window(meta)
        labeled = window is not None and window.get("status") == "reference_labeled"

        o_samples = build_o_samples(loaded["observations_dense"])
        for sample in o_samples:
            leakage["o_samples_checked"] += 1
            if any(key in sample for key in ("object_xyz", "gripper_xyz", "weld_state")):
                leakage["violations"].append({"rollout_id": rollout_id, "tier": "O", "reason": "oracle_field_in_o_sample"})

        for method_id in O_METHODS:
            result = run_o_method(meta, method_id)
            for row in result["trace"]:
                o_traces.append(row)
            decision = result["decision"]
            expected = window.get("expected_action") if labeled else None
            selected = decision["selected_action"]
            outcome = classify_outcome(expected, selected, labeled)
            flags = _anomaly_flags(result["trace"], window if labeled else None, "selected_action")
            o_decisions.append(
                {
                    "method_id": method_id,
                    "candidate_id": decision["candidate_id"],
                    "rollout_id": rollout_id,
                    "root_family_id": meta.get("root_family_id"),
                    "case_id": case_id,
                    "selected_action": selected,
                    "selected_time": decision["selected_time"],
                    "correct": decision["correct"],
                    "selected_reason": decision["selected_reason"],
                }
            )
            decision_rows.append(
                {
                    "method_id": method_id,
                    "input_tier": "O",
                    "rollout_id": rollout_id,
                    "root_family_id": meta.get("root_family_id"),
                    "case_id": case_id,
                    "legacy_reference_status": window.get("status") if window else "reference_unresolved",
                    "legacy_expected_action": expected,
                    "legacy_action_agreement": outcome,
                    "selected_action": selected,
                    "selected_time": decision["selected_time"],
                    "outcome": outcome,
                    "independent_physical_loss_status": "UNRESOLVED" if case_id in LOSS_CASES else None,
                    "physical_loss_recall": "NOT_ESTIMABLE" if case_id in LOSS_CASES else None,
                    "hold_transition_event_count": 0,
                    "hold_entry_routes": None,
                    "unknown_rows": 0,
                    "rows": len(result["trace"]),
                    **flags,
                }
            )

        s_samples = build_s_samples(loaded["observations_dense"], loaded["oracle_timeline"])
        attach_request_provenance(s_samples, loaded["controller_requests"])
        for sample in s_samples:
            leakage["s_samples_checked"] += 1
            leaked = [key for key in FORBIDDEN_CANDIDATE_INPUTS if key in sample]
            if leaked:
                leakage["violations"].append({"rollout_id": rollout_id, "tier": "S", "fields": leaked})

        for method_id in S_METHODS:
            machine = GeometryEventStateMachine(method_id, protocol, rollout_id, meta.get("root_family_id"))
            records = machine.run(s_samples)
            for record in records:
                s_traces.append(record)
            causality["checked"] += 1
            causality["mismatches"].extend(_prefix_causality(s_samples, method_id, protocol, meta))
            selected = _first_action_in_window(records, window["start"], window["end"], "selected_action") if labeled else {"selected_action": "none", "selected_time": None, "reason": None}
            expected = window.get("expected_action") if labeled else None
            outcome = classify_outcome(expected, selected["selected_action"], labeled)
            flags = _anomaly_flags(records, window if labeled else None, "selected_action")
            transitions = [record for record in records if record.get("hold_transition_event")]
            decision_rows.append(
                {
                    "method_id": method_id,
                    "input_tier": "S",
                    "rollout_id": rollout_id,
                    "root_family_id": meta.get("root_family_id"),
                    "case_id": case_id,
                    "legacy_reference_status": window.get("status") if window else "reference_unresolved",
                    "legacy_expected_action": expected,
                    "legacy_action_agreement": outcome,
                    "selected_action": selected["selected_action"],
                    "selected_time": selected["selected_time"],
                    "outcome": outcome,
                    "independent_physical_loss_status": "UNRESOLVED" if case_id in LOSS_CASES else None,
                    "physical_loss_recall": "NOT_ESTIMABLE" if case_id in LOSS_CASES else None,
                    "hold_transition_event_count": len(transitions),
                    "hold_entry_routes": ",".join(record["hold_entry_route"] for record in transitions),
                    "unknown_rows": sum(1 for record in records if record.get("current_quality") != DATA_VALID),
                    "rows": len(records),
                    **flags,
                }
            )

        if case_id in LOSS_CASES:
            oracle_rows = read_oracle_rows(rollout_dir)
            transition = weld_transition(oracle_rows)
            event_time = contact_lost_time(rollout_dir)
            weld_rows.append(
                {
                    "rollout_id": rollout_id,
                    "root_family_id": meta.get("root_family_id"),
                    "case_id": case_id,
                    **transition,
                    "event_contact_lost_time": event_time,
                    "event_vs_weld_transition_relation": event_transition_relation(event_time, transition),
                }
            )
            last_on_relative = relative_at_order(s_samples, transition.get("last_on_capture_order"), transition.get("last_on_time"))
            for method_id in (S_METHODS[1], METHOD_G_HR):
                machine = GeometryEventStateMachine(method_id, protocol, rollout_id, meta.get("root_family_id"))
                records = machine.run(s_samples)
                triggered = any(record.get("loss_source") for record in records)
                anchors = drift_audit(
                    s_samples,
                    machine.r_hold_anchor,
                    machine.hold_anchor_time,
                    machine.hold_anchor_capture_order,
                    "candidate_anchor",
                )
                last_on = drift_audit(
                    s_samples,
                    last_on_relative,
                    transition.get("last_on_time"),
                    transition.get("last_on_capture_order"),
                    "last_on_anchor",
                )
                drift_rows.append(
                    {
                        "rollout_id": rollout_id,
                        "root_family_id": meta.get("root_family_id"),
                        "case_id": case_id,
                        "method_id": method_id,
                        **anchors,
                        **last_on,
                        "relative_loss_gate_m": float(parameters["relative_loss_drift_m"]),
                        "candidate_relative_exit_triggered": triggered,
                        "event_contact_lost_time": event_time,
                    }
                )

    round10_rows = load_round10_comparison(Path(round10_root) / "interface_repair_comparison.csv")
    parity = parity_rows(round10_rows, o_decisions)
    parity_summary = parity_metrics(parity)

    method_summary = {}
    for method_id in list(O_METHODS) + list(S_METHODS):
        rows = [row for row in decision_rows if row["method_id"] == method_id]
        summary = summarize_method(rows)
        if method_id in O_METHODS:
            summary["candidate_id"] = O_METHOD_TO_CANDIDATE[method_id]
            summary["interface"] = "l2rar2_online_interface_repair_v1"
        method_summary[method_id] = summary

    engineering_status = "R15B_CACHE_REEVALUATION_COMPLETE"
    if not parity_summary["all_matched"]:
        engineering_status = "BLOCKED_O_INTERFACE_PARITY_FAILED"
    elif causality["mismatches"]:
        engineering_status = "BLOCKED_CAUSALITY_FAILED"
    elif leakage["violations"]:
        engineering_status = "BLOCKED_INPUT_LEAKAGE"
    elif resource_report.get("status") not in {"PASS", "PASS_WITH_WARNINGS"}:
        engineering_status = "BLOCKED_RESOURCE_INTEGRITY"

    _write_json(output_root / "protocol_lock.json", protocol)
    _write_json(
        output_root / "source_contract.json",
        {
            "schema": "l2rar2_r15b_source_contract_v1",
            "entry_commit": protocol.get("entry_commit"),
            "base_branch": protocol.get("base_branch"),
            "proposed_branch": protocol.get("proposed_branch"),
            "data_root": str(data_root),
            "round10_root": str(round10_root),
            "resource_report": str(resource_report_path),
            "resource_report_status": resource_report.get("status"),
            "rollouts": len(rollout_dirs),
            "required_rollout_files": list(REQUIRED_ROLLOUT_FILES),
        },
    )
    _write_json(
        output_root / "method_contract.json",
        {
            "schema": "l2rar2_r15b_method_contract_v1",
            "methods": {
                "O_B2": {"input_tier": "O", "hold_feature": "B_count2", "interface": "l2rar2_online_interface_repair_v1"},
                "O_C3": {"input_tier": "O", "hold_feature": "C3_vector_rho035", "interface": "l2rar2_online_interface_repair_v1"},
                "S_G_H": {"input_tier": "S", "entry": ["HEIGHT_THRESHOLD"], "exit": ["height_return_loss_signal"]},
                "S_G_R": {"input_tier": "S", "entry": ["RELATIVE_CO_MOTION"], "exit": ["contact_true_to_false", "relative_detach_sustained"]},
                "S_G_HR": {"input_tier": "S", "entry": ["LIFT_COUPLED", "LOW_HEIGHT_CO_MOTION"], "exit": ["contact_true_to_false", "relative_detach_sustained"]},
            },
            "parameters_unchanged_from_r15a": True,
            "s_tier_label": "STATE_ASSISTED_DIAGNOSTIC",
        },
    )
    _write_json(
        output_root / "input_contract.json",
        {
            "schema": "l2rar2_r15b_input_contract_v1",
            "o_tier": ["time", "capture_order", "frame_index", "2d centroids", "confidence", "size", "contact_present", "gripper_command", "attempt lifecycle", "request provenance"],
            "s_tier_adds": ["oracle_timeline.object_xyz", "oracle_timeline.gripper_xyz"],
            "forbidden_candidate_inputs": list(FORBIDDEN_CANDIDATE_INPUTS),
            "weld_state_usage": "reference audit layer only",
        },
    )
    _write_json(
        output_root / "parameter_lock.json",
        {
            "schema": "l2rar2_r15b_parameter_lock_v1",
            "parameters": parameters,
            "matches_r15a": True,
            "parameter_search_allowed": False,
            "old_reference": protocol.get("old_reference"),
        },
    )
    _write_csv(
        output_root / "round10_interface_parity.csv",
        parity,
        [
            "method_id",
            "candidate_id",
            "rollout_id",
            "round10_selected_action",
            "v2_selected_action",
            "round10_selected_time",
            "v2_selected_time",
            "round10_selected_reason",
            "v2_selected_reason",
            "round10_correct",
            "v2_correct",
            "matched",
            "mismatch_fields",
        ],
    )
    _write_json(output_root / "round10_interface_parity.json", {**parity_summary, "rows_detail": "round10_interface_parity.csv"})
    _write_csv(
        output_root / "weld_transition_audit.csv",
        weld_rows,
        [
            "rollout_id",
            "root_family_id",
            "case_id",
            "transition_found",
            "last_on_time",
            "last_on_capture_order",
            "first_off_time",
            "first_off_capture_order",
            "transition_interval_start",
            "transition_interval_end",
            "transition_order_status",
            "event_contact_lost_time",
            "event_vs_weld_transition_relation",
        ],
    )
    drift_fields = ["rollout_id", "root_family_id", "case_id", "method_id"]
    for prefix in ("candidate_anchor", "last_on_anchor"):
        drift_fields += [
            f"{prefix}_anchor_time",
            f"{prefix}_anchor_capture_order",
            f"{prefix}_max_drift_m",
            f"{prefix}_window_start",
            f"{prefix}_window_end",
            f"{prefix}_sample_count",
        ]
    drift_fields += ["relative_loss_gate_m", "candidate_relative_exit_triggered", "event_contact_lost_time"]
    _write_csv(output_root / "relative_drift_anchor_audit.csv", drift_rows, drift_fields)
    _write_jsonl(output_root / "prediction_trace.jsonl", o_traces + s_traces)
    _write_csv(
        output_root / "feature_trace.csv",
        [
            {
                "method": row.get("method"),
                "input_tier": row.get("input_tier"),
                "rollout_id": row.get("rollout_id"),
                "time": row.get("time"),
                "capture_order": row.get("capture_order"),
                "h": row.get("h"),
                "e": row.get("e"),
                "cos": row.get("cos"),
                "rho": row.get("rho"),
                "evidence_valid": row.get("evidence_valid"),
                "evidence_duration": row.get("evidence_duration"),
            }
            for row in s_traces
        ],
        ["method", "input_tier", "rollout_id", "time", "capture_order", "h", "e", "cos", "rho", "evidence_valid", "evidence_duration"],
    )
    episode_fields = [
        "method_id",
        "input_tier",
        "rollout_id",
        "root_family_id",
        "case_id",
        "legacy_reference_status",
        "legacy_expected_action",
        "legacy_action_agreement",
        "selected_action",
        "selected_time",
        "outcome",
        "independent_physical_loss_status",
        "physical_loss_recall",
        "hold_transition_event_count",
        "hold_entry_routes",
        "premature_emergency",
        "premature_retry",
        "premature_recovery",
        "post_window_emergency",
        "unknown_rows",
        "rows",
    ]
    _write_csv(output_root / "per_episode_decisions.csv", decision_rows, episode_fields)
    metric_rows = []
    for method_id, summary in method_summary.items():
        metric_rows.append({"method_id": method_id, **{key: _csv_value(value) for key, value in summary.items()}})
    _write_csv(output_root / "metrics_v2.csv", metric_rows, ["method_id"] + sorted({key for row in metric_rows for key in row if key != "method_id"}))
    _write_json(output_root / "metrics_v2.json", {"schema": "l2rar2_r15b_metrics_v2", "methods": method_summary})
    _write_json(output_root / "method_summary_v2.json", method_summary)
    _write_json(output_root / "prefix_causality_audit.json", causality)
    _write_json(output_root / "input_leakage_audit.json", leakage)
    _write_json(
        output_root / "decision.json",
        {
            "engineering_status": engineering_status,
            "scientific_status": "L2RAR2_PARTIAL_KEEP_G1",
            "selected_candidate_id": None,
            "confirmation_run": False,
            "l3_entry_allowed": False,
            "physical_executions": 0,
            "training_jobs": 0,
            "model_api_calls": 0,
            "api_key_reads": 0,
            "round10_o_interface_parity": "PASS" if parity_summary["all_matched"] else "FAIL",
            "independent_physical_loss_status": "UNRESOLVED",
            "r14_status": "NOT_RUN_AUTHORIZATION_ABSENT",
        },
    )
    _write_json(
        output_root / "next_stage_handoff.json",
        {
            "entry_commit": protocol.get("entry_commit"),
            "implemented_version": EVALUATION_VERSION,
            "r14_application_prepared": True,
            "r14_executed": False,
            "next_research_decision": "independent physical loss confirmation still outstanding",
        },
    )
    (output_root / "final_report.md").write_text(
        _final_report(engineering_status, method_summary, parity_summary, weld_rows, drift_rows),
        encoding="utf-8",
    )
    (output_root / "actual_commands.txt").write_text(
        "\n".join(
            [
                "python -B -m upgrade_v2.l2r_geometry_events.cli cache-evaluate-v2 \\",
                f"  --data-root {data_root} \\",
                f"  --protocol {protocol_path} \\",
                f"  --round10-root {round10_root} \\",
                f"  --resource-report {resource_report_path} \\",
                f"  --output-root {output_root}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output_root / "external_artifacts.tsv").write_text(
        "artifact\tpath\tstatus\n"
        f"data_root\t{data_root}\texternal\n"
        f"round10_root\t{round10_root}\texternal\n"
        f"r15a_resource_report\t{resource_report_path}\texternal\n",
        encoding="utf-8",
    )
    _write_json(
        output_root / "run_manifest.json",
        {
            "schema": "l2rar2_r15b_run_manifest_v1",
            "evaluation_version": EVALUATION_VERSION,
            "entry_commit": protocol.get("entry_commit"),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "rollouts_loaded": len(rollout_dirs),
            "methods": list(O_METHODS) + list(S_METHODS),
            "outputs": sorted(path.name for path in output_root.iterdir()),
            "physical_executions": 0,
            "training_jobs": 0,
            "model_api_calls": 0,
            "api_key_reads": 0,
        },
    )
    return {
        "status": "CACHE_REEVALUATED_V2",
        "output_root": str(output_root),
        "engineering_status": engineering_status,
        "rollouts_loaded": len(rollout_dirs),
        "round10_parity": parity_summary,
        "prefix_causality_mismatches": len(causality["mismatches"]),
        "input_leakage_violations": len(leakage["violations"]),
        "method_summary": method_summary,
    }


def _final_report(
    engineering_status: str,
    method_summary: dict[str, Any],
    parity: dict[str, Any],
    weld_rows: list[dict[str, Any]],
    drift_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# R15-B geometry-event fair re-evaluation - final report",
        "",
        f"Engineering status: `{engineering_status}`",
        "Scientific status: `L2RAR2_PARTIAL_KEEP_G1`; selected candidate = null; L3 closed.",
        "",
        "## O-layer parity with the frozen Round-10 interface",
        "",
        f"- rows compared: {parity['rows']}; matched: {parity['matched']}; mismatched: {parity['mismatched']}",
        "",
        "## Method summary",
        "",
        "| method | tier | correct | accuracy | K1 recall | negative false emergency | hold transitions | routes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for method_id, summary in method_summary.items():
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                method_id,
                summary.get("input_tier"),
                summary.get("legacy_action_correct_count"),
                summary.get("legacy_action_accuracy"),
                summary.get("K1_recall"),
                summary.get("negative_window_false_emergency_rate"),
                summary.get("hold_transition_events"),
                summary.get("hold_entry_route_counts"),
            )
        )
    lines += [
        "",
        "## Weld transition audit (reference layer only)",
        "",
        "| rollout | case | transition | order status | event relation |",
        "|---|---|---|---|---|",
    ]
    for row in weld_rows:
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (row["rollout_id"], row["case_id"], row["transition_found"], row["transition_order_status"], row["event_vs_weld_transition_relation"])
        )
    lines += [
        "",
        "## Drift anchors",
        "",
        "Candidate-anchor drift and last-weld-on-anchor drift are reported separately in",
        "`relative_drift_anchor_audit.csv`; `candidate_relative_exit_triggered` only means",
        "the frozen candidate threshold fired on saved positions.",
        "",
    ]
    for row in drift_rows:
        lines.append(
            "- %s / %s / %s: candidate anchor drift %.4f m, last-on anchor drift %s m, exit triggered %s"
            % (
                row["rollout_id"],
                row["case_id"],
                row["method_id"],
                row.get("candidate_anchor_max_drift_m") or 0.0,
                row.get("last_on_anchor_max_drift_m"),
                row.get("candidate_relative_exit_triggered"),
            )
        )
    lines += [
        "",
        "## Limits",
        "",
        "- S tier uses saved oracle 3D positions: state-assisted diagnostic, not RGB.",
        "- No candidate exit under the frozen thresholds says nothing about whether the",
        "  physical loss occurred; independent physical loss stays UNRESOLVED.",
        "- R14 was not executed (authorization absent).",
        "",
    ]
    return "\n".join(lines)
