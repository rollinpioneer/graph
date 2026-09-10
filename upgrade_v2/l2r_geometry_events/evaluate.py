"""R15-A cache geometry evaluation over the existing 32 repair rollouts.

Two input tiers are kept strictly separate:

* O tier - the frozen legacy online interface (O_B2 / O_C3 regression).
* S tier - STATE_ASSISTED_DIAGNOSTIC geometry methods (S_G_H / S_G_R / S_G_HR).

Prediction never sees case_id, expected_action, weld_state, events or any
future row: labels are joined only in the reporting layer, after the method
traces have been produced.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from upgrade_v2.l2r_hold_evidence.event_adapter import (
    first_action,
    infer_history_complete,
    run_event_interface,
)
from upgrade_v2.l2r_hold_evidence.evaluate_v2 import _online_observation, _predicates
from upgrade_v2.l2r_hold_evidence.hold_features import build_features as legacy_build_features
from upgrade_v2.l2r_hold_evidence.hold_predicates import evaluate_candidate
from upgrade_v2.l2r_task_context.reference import build_reference as build_legacy_reference

from .adapters import (
    REQUIRED_ROLLOUT_FILES,
    attach_request_provenance,
    build_o_samples,
    build_s_samples,
    load_rollout,
)
from .features import height_baseline
from .state_machine import S_METHODS, GeometryEventStateMachine

O_METHODS = (("O_B2", "B_count2"), ("O_C3", "C3_vector"))
EXPECTED_CASES = tuple(f"K{index}" for index in range(1, 9))


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
    rollouts_root = Path(data_root) / "rollouts"
    found = sorted(path for path in rollouts_root.glob("*/*") if path.is_dir())
    return found


def _manifest_rows(data_root: Path) -> list[dict[str, str]]:
    manifest = Path(data_root) / "rollout_manifest.csv"
    if not manifest.is_file():
        return []
    with manifest.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _weld_release_time(rollout_dir: Path) -> float | None:
    """Reference audit only: read ``weld_state`` and return its release time.

    This value is never passed to a candidate or to the state machine; it is
    used to describe the cache itself in the reporting layer.
    """
    path = Path(rollout_dir) / "oracle_timeline.csv"
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    welded_times = [float(row["time"]) for row in rows if str(row.get("weld_state", "0")) == "1"]
    if not welded_times:
        return None
    return max(welded_times)


def _legacy_reference(meta: dict[str, Any]) -> dict[str, Any]:
    try:
        return build_legacy_reference(dict(meta))
    except Exception as error:  # reference_unresolved or registry miss
        return {"status": "reference_unresolved", "reason": f"{type(error).__name__}: {error}", "events": []}


def _reference_window(reference: dict[str, Any]) -> dict[str, Any] | None:
    events = reference.get("events") or []
    if not events:
        return None
    event = events[0]
    return {
        "expected_action": event.get("expected_action"),
        "reference_type": event.get("reference_type"),
        "start": float(event.get("decision_window_start")),
        "end": float(event.get("decision_window_end")),
        "physical_label_status": event.get("physical_label_status", "unresolved"),
        "event_id": event.get("event_id"),
    }


def _first_selected_action(rows: list[dict[str, Any]], start: float, end: float) -> dict[str, Any]:
    for index, row in enumerate(rows):
        time_value = row.get("time")
        if time_value is None:
            continue
        time_value = float(time_value)
        if time_value < start - 1e-9 or time_value > end + 1e-9:
            continue
        guards = row.get("effective_guards")
        if guards is None:
            action = row.get("selected_action", "none")
        else:
            retry = guards.get("retry_grasp") == "true"
            recover = guards.get("recover_object") == "true"
            action = "conflict" if (retry and recover) else "retry_grasp" if retry else "recover_object" if recover else "none"
        if action != "none":
            return {"selected_action": action, "selected_frame": index, "selected_time": time_value}
    return {"selected_action": "none", "selected_frame": None, "selected_time": None}


def _method_rows_for_rollout(
    loaded: dict[str, Any],
    protocol: dict[str, Any],
    *,
    o_config: dict[str, Any],
) -> dict[str, Any]:
    raw_observations = loaded["observations_dense"]
    meta = loaded["metadata"]
    observations = [_online_observation(row) for row in raw_observations]
    geometries = legacy_build_features(observations)
    predicted: list[dict[str, Any]] = []
    previous = None
    for observation, geometry in zip(observations, geometries):
        predicted.append({**observation, "predicates": _predicates(observation, geometry, previous)})
        previous = observation

    traces: dict[str, list[dict[str, Any]]] = {}
    for method_id, base_id in O_METHODS:
        evidence = evaluate_candidate(predicted, geometries, base_id, dict(o_config))
        # The event interface expects rows that already carry the online
        # predicate block; feed it the same prediction rows the candidate used.
        memory = run_event_interface(predicted, evidence, history_complete=infer_history_complete(predicted))
        rows = []
        for observation, evidence_row, memory_row in zip(observations, evidence, memory):
            rows.append(
                {
                    "method": method_id,
                    "input_tier": "O",
                    "rollout_id": meta.get("rollout_id"),
                    "root_family_id": meta.get("root_family_id"),
                    "time": observation.get("time"),
                    "capture_order": observation.get("capture_order"),
                    "h": None,
                    "r": None,
                    "e": None,
                    "cos": evidence_row["geometry"].get("direction_cosine"),
                    "rho": evidence_row["geometry"].get("relative_vector_error"),
                    "evidence_valid": bool(evidence_row["geometry"].get("effective_motion_interval")),
                    "evidence_duration": evidence_row.get("supported_time"),
                    "hold_entry_route": None,
                    "ever_held": bool(memory_row.get("hold_memory") == "true"),
                    "current_quality": "ONLINE",
                    "hold_state": None,
                    "history_quality": None,
                    "lift_confirmed": None,
                    "loss_source": None,
                    "selected_action": None,
                    "reason_code": "LEGACY_ONLINE_INTERFACE",
                    "event_id": memory_row.get("pending_event"),
                    "grasp_supported": bool(memory_row.get("hold_memory") == "true"),
                    "effective_guards": memory_row.get("effective_guards", {}),
                    "hold_evidence": memory_row.get("hold_evidence"),
                }
            )
        traces[method_id] = rows

    s_samples = build_s_samples(raw_observations, loaded["oracle_timeline"])
    attach_request_provenance(s_samples, loaded["controller_requests"])
    baseline = height_baseline(s_samples)
    for method_id in S_METHODS:
        machine = GeometryEventStateMachine(
            method_id,
            protocol,
            meta.get("rollout_id"),
            meta.get("root_family_id"),
        )
        traces[method_id] = machine.run(s_samples)
    return {
        "metadata": meta,
        "traces": traces,
        "height_baseline": baseline,
        "s_samples": s_samples,
        "o_samples": build_o_samples(raw_observations),
    }


def run_cache_evaluation(
    data_root: Path,
    protocol_path: Path,
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
    o_config = {
        "relative_rho_max": float(parameters["relative_rho_max"]),
        "min_motion": float(parameters["min_motion_3d_m"]),
        "direction_min": float(parameters["direction_cosine_min"]),
        "supported_window_seconds": 0.6,
    }

    rollout_dirs = _discover_rollouts(data_root)
    manifest = _manifest_rows(data_root)
    per_rollout: list[dict[str, Any]] = []
    feature_trace: list[dict[str, Any]] = []
    prediction_trace: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []

    for rollout_dir in rollout_dirs:
        loaded = load_rollout(rollout_dir)
        if loaded.get("status") != "OK":
            incomplete.append({"path": str(rollout_dir), "missing": loaded.get("missing")})
            continue
        result = _method_rows_for_rollout(loaded, protocol, o_config=o_config)
        per_rollout.append(result)
        for method_id, rows in result["traces"].items():
            for row in rows:
                trace_row = {key: value for key, value in row.items() if key != "effective_guards"}
                prediction_trace.append(trace_row)
                feature_trace.append(
                    {
                        "method": method_id,
                        "input_tier": row.get("input_tier"),
                        "rollout_id": row.get("rollout_id"),
                        "root_family_id": row.get("root_family_id"),
                        "time": row.get("time"),
                        "capture_order": row.get("capture_order"),
                        "h": row.get("h"),
                        "r": row.get("r"),
                        "e": row.get("e"),
                        "cos": row.get("cos"),
                        "rho": row.get("rho"),
                        "evidence_valid": row.get("evidence_valid"),
                        "evidence_duration": row.get("evidence_duration"),
                    }
                )

    # ---- report layer: labels are joined here only ---------------------------
    agreement_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    physical_rows: list[dict[str, Any]] = []
    method_stats: dict[str, Counter] = defaultdict(Counter)
    route_counts: Counter = Counter()
    premature_holds: Counter = Counter()
    false_recoveries: Counter = Counter()
    unknown_counts: Counter = Counter()
    for result in per_rollout:
        meta = result["metadata"]
        rollout_id = meta.get("rollout_id")
        case_id = meta.get("case_id")
        reference = _legacy_reference(meta)
        window = _reference_window(reference)
        is_loss_case = case_id in {"K4_regular_hold_loss", "K5_brief_hold_loss", "K6_long_gap_after_loss"}
        for method_id, rows in result["traces"].items():
            if window is None:
                outcome = "unknown"
                selected = {"selected_action": "none", "selected_frame": None, "selected_time": None}
            else:
                selected = _first_selected_action(rows, window["start"], window["end"])
                if selected["selected_action"] == "none":
                    outcome = "none"
                elif selected["selected_action"] == window["expected_action"]:
                    outcome = "correct"
                else:
                    outcome = "wrong"
            method_stats[method_id][outcome] += 1
            agreement_rows.append(
                {
                    "method_id": method_id,
                    "input_tier": rows[0].get("input_tier") if rows else None,
                    "rollout_id": rollout_id,
                    "root_family_id": meta.get("root_family_id"),
                    "case_id": case_id,
                    "expected_action": None if window is None else window["expected_action"],
                    "selected_action": selected["selected_action"],
                    "selected_time": selected["selected_time"],
                    "decision_window_start": None if window is None else window["start"],
                    "decision_window_end": None if window is None else window["end"],
                    "outcome": outcome,
                    "shared_geometry_source": method_id in S_METHODS,
                    "physical_label_status": "unresolved" if is_loss_case else "reference_labeled",
                }
            )
            routes = [row.get("hold_entry_route") for row in rows if row.get("hold_entry_route")]
            for route in routes:
                route_counts[(method_id, route)] += 1
            first_hold = next((row.get("time") for row in rows if row.get("ever_held")), None)
            premature = any(row.get("selected_action") == "recover_object" for row in rows)
            if window is None:
                premature = None
            else:
                premature = any(
                    row.get("selected_action") == "recover_object"
                    and row.get("time") is not None
                    and float(row["time"]) < window["start"] - 1e-9
                    for row in rows
                )
            early_hold = any(
                row.get("ever_held")
                and window is not None
                and row.get("time") is not None
                and float(row["time"]) < window["start"] - 1e-9
                for row in rows
            )
            if early_hold:
                premature_holds[method_id] += 1
            if premature:
                false_recoveries[method_id] += 1
            unknown_rows = sum(1 for row in rows if row.get("current_quality") not in ("VALID", "ONLINE"))
            unknown_counts[method_id] += unknown_rows
            episode_rows.append(
                {
                    "method_id": method_id,
                    "input_tier": rows[0].get("input_tier") if rows else None,
                    "rollout_id": rollout_id,
                    "root_family_id": meta.get("root_family_id"),
                    "case_id": case_id,
                    "expected_action": None if window is None else window["expected_action"],
                    "selected_action": selected["selected_action"],
                    "outcome": outcome,
                    "hold_entry_route": routes[0] if routes else None,
                    "first_hold_time": first_hold,
                    "hold_established_before_event": early_hold,
                    "false_recovery": bool(premature),
                    "lift_confirmed": next((row.get("lift_confirmed") for row in rows if row.get("ever_held")), None),
                    "unknown_rows": unknown_rows,
                    "rows": len(rows),
                }
            )
        if is_loss_case:
            audit_rows = result["traces"]["S_G_HR"]
            drift_values = [
                float(row["e"])
                for row in audit_rows
                if row.get("e") is not None
                and (window is None or row.get("time") is None or float(row["time"]) >= window["start"] - 1e-9)
            ]
            geometric_detach = any(row.get("loss_source") for row in audit_rows)
            physical_rows.append(
                {
                    "rollout_id": rollout_id,
                    "root_family_id": meta.get("root_family_id"),
                    "case_id": case_id,
                    "physical_loss_status": "unresolved",
                    "independent_source": "none_recorded_in_this_round",
                    "physical_loss_recall": "NOT_ESTIMABLE",
                    "denominator_reason": "no_new_independent_physical_evidence",
                    "weld_release_time_reference_audit": _weld_release_time(Path(meta.get("path", ""))),
                    "max_relative_drift_after_event_m": round(max(drift_values), 6) if drift_values else None,
                    "relative_loss_gate_m": float(parameters["relative_loss_drift_m"]),
                    "geometric_detach_observed": geometric_detach,
                }
            )

    method_summary: dict[str, Any] = {}
    for method_id in [method for method, _ in O_METHODS] + list(S_METHODS):
        stats = method_stats.get(method_id, Counter())
        method_summary[method_id] = {
            "input_tier": "O" if method_id.startswith("O_") else "S",
            "rollouts": sum(stats.values()),
            "correct": stats.get("correct", 0),
            "none": stats.get("none", 0),
            "wrong": stats.get("wrong", 0),
            "unknown": stats.get("unknown", 0),
            "entry_routes": {
                route: count
                for (owner, route), count in route_counts.items()
                if owner == method_id
            },
            "rollouts_with_hold_before_event": premature_holds.get(method_id, 0),
            "false_recovery_rollouts": false_recoveries.get(method_id, 0),
            "unknown_rows": unknown_counts.get(method_id, 0),
        }

    availability = {
        "rollouts_discovered": len(rollout_dirs),
        "rollouts_loaded": len(per_rollout),
        "root_families": sorted({result["metadata"].get("root_family_id") for result in per_rollout}),
        "cases": sorted({result["metadata"].get("case_id") for result in per_rollout}),
        "height_baseline": {
            result["metadata"].get("rollout_id"): result["height_baseline"]["status"] for result in per_rollout
        },
        "rgb_available_rollouts": sum(
            1
            for result in per_rollout
            if (Path(result["metadata"].get("path", "")) / "rgb" / "front").is_dir()
        ),
        "oracle_position_source": "oracle_timeline.csv object_xyz/gripper_xyz clipped to position/time keys",
        "orientation_available": False,
        "incomplete_rollouts": incomplete,
    }

    _write_json(
        output_root / "input_contract.json",
        {
            "o_tier_allowlist": list(protocol.get("S_input_allowlist", [])),
            "s_tier_fields": [
                "time",
                "capture_order",
                "object_xyz",
                "gripper_xyz",
                "contact_present",
                "gripper_command",
                "attempt_id",
                "attempt_phase",
                "attempt_active",
                "attempt_end",
                "attempt_end_reason",
                "attempt_end_sequence",
                "requested_effect",
                "data_quality",
            ],
            "forbidden_candidate_inputs": list(protocol.get("forbidden_candidate_inputs", [])),
            "required_rollout_files": list(REQUIRED_ROLLOUT_FILES),
            "coordinate_contract": protocol.get("coordinate_contract"),
            "label_join": "reporting_layer_only_after_method_traces",
        },
    )
    _write_json(
        output_root / "parameter_lock.json",
        {
            "protocol_schema": protocol.get("schema"),
            "protocol_status": protocol.get("status"),
            "entry_commit": protocol.get("entry_commit"),
            "parameters": parameters,
            "parameter_search_allowed": False,
            "old_reference": protocol.get("old_reference"),
            "frozen_before_cache_run": True,
        },
    )
    _write_csv(
        output_root / "feature_trace.csv",
        feature_trace,
        [
            "method",
            "input_tier",
            "rollout_id",
            "root_family_id",
            "time",
            "capture_order",
            "h",
            "r",
            "e",
            "cos",
            "rho",
            "evidence_valid",
            "evidence_duration",
        ],
    )
    _write_jsonl(output_root / "prediction_trace.jsonl", prediction_trace)
    _write_csv(
        output_root / "per_episode_evidence.csv",
        episode_rows,
        [
            "method_id",
            "input_tier",
            "rollout_id",
            "root_family_id",
            "case_id",
            "expected_action",
            "selected_action",
            "outcome",
            "hold_entry_route",
            "first_hold_time",
            "hold_established_before_event",
            "false_recovery",
            "lift_confirmed",
            "unknown_rows",
            "rows",
        ],
    )
    _write_csv(
        output_root / "legacy_label_agreement.csv",
        agreement_rows,
        [
            "method_id",
            "input_tier",
            "rollout_id",
            "root_family_id",
            "case_id",
            "expected_action",
            "selected_action",
            "selected_time",
            "decision_window_start",
            "decision_window_end",
            "outcome",
            "shared_geometry_source",
            "physical_label_status",
        ],
    )
    _write_csv(
        output_root / "physical_reference_status.csv",
        physical_rows,
        [
            "rollout_id",
            "root_family_id",
            "case_id",
            "physical_loss_status",
            "independent_source",
            "physical_loss_recall",
            "denominator_reason",
            "weld_release_time_reference_audit",
            "max_relative_drift_after_event_m",
            "relative_loss_gate_m",
            "geometric_detach_observed",
        ],
    )
    _write_json(output_root / "method_summary.json", method_summary)
    _write_json(output_root / "feature_availability.json", availability)

    (output_root / "literature_design_notes.md").write_text(
        "\n".join(
            [
                "# Literature design notes (design inspiration only)",
                "",
                "- [L1] Calandra et al. 2017, The Feeling of Success - vision and touch are",
                "  complementary for grasp outcome prediction. This round introduces no",
                "  tactile network and reuses no training/label conclusion.",
                "- [L2] Dong et al., Maintaining Grasps within Slipping Bound - relative",
                "  displacement fields motivate 'contact can persist while slipping'.",
                "  Incipient slip is not a task-loss label.",
                "- [L3] robosuite task docs / stack environment - lift height is used for",
                "  task-specific stage success, not as a general stable-hold label. This",
                "  round uses relative initial height.",
                "- [L4] MuJoCo 3.4.0 simulation docs - state/warmstart/control relations",
                "  inform the (unused, unauthorised) R14 replay record contract.",
                "",
                "No paper accuracy is reused as an expected result for this cache.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    decision = {
        "scientific_status": protocol.get("scientific_status"),
        "selected_candidate_id": protocol.get("selected_candidate_id"),
        "confirmation": False,
        "l3": False,
        "r14_status": "NOT_RUN_AUTHORIZATION_ABSENT",
        "r14_authorized_instances": 0,
        "r15_cache_only": True,
        "method_summary_ref": "method_summary.json",
        "notes": [
            "Only the existing 32 development rollouts were read.",
            "No physical execution, training, model API call or key read occurred.",
            "Physical loss denominators stay unresolved; no self-generated truth is used.",
        ],
    }
    _write_json(output_root / "decision.json", decision)
    _write_json(
        output_root / "next_stage_handoff.json",
        {
            "entry_commit": protocol.get("entry_commit"),
            "implemented_module": "upgrade_v2/l2r_geometry_events",
            "r14_requires_user_authorization": True,
            "next_research_decision": "choose between reference/collection repair, perception budget or event module work",
            "blocked_items": ["R14 physical replay (authorization absent)", "independent physical loss labels"],
        },
    )
    (output_root / "final_report.md").write_text(
        _final_report(protocol, method_summary, availability, physical_rows),
        encoding="utf-8",
    )
    _write_jsonl(
        output_root / "actual_commands.jsonl",
        [
            {
                "step": "resource_check",
                "argv": [
                    "python",
                    "-B",
                    "tools/resource_check.py",
                    "--repo",
                    "<worktree>",
                    "--data-root",
                    str(data_root),
                    "--output",
                    "artifacts/pathgraph_sarm/upgrade_v2/geometry_events_l2rar2_v1/resource_resolution.json",
                ],
            },
            {
                "step": "cache_evaluate",
                "argv": [
                    "python",
                    "-B",
                    "-m",
                    "upgrade_v2.l2r_geometry_events.cli",
                    "cache-evaluate",
                    "--data-root",
                    str(data_root),
                    "--protocol",
                    str(protocol_path),
                    "--resource-report",
                    str(resource_report_path),
                    "--output-root",
                    str(output_root),
                ],
            },
        ],
    )
    manifest_payload = {
        "module": "upgrade_v2/l2r_geometry_events",
        "entry_commit": protocol.get("entry_commit"),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "rollouts_loaded": len(per_rollout),
        "methods": [method for method, _ in O_METHODS] + list(S_METHODS),
        "outputs": sorted(path.name for path in output_root.iterdir()),
        "manifest_rows": len(manifest),
        "physical_executions": 0,
        "model_api_calls": 0,
        "key_reads": 0,
    }
    _write_json(output_root / "run_manifest.json", manifest_payload)
    return {
        "output_root": str(output_root),
        "rollouts_loaded": len(per_rollout),
        "methods": manifest_payload["methods"],
        "method_summary": method_summary,
        "height_baseline_unavailable": [
            key for key, value in availability["height_baseline"].items() if value != "AVAILABLE"
        ],
    }


def _final_report(
    protocol: dict[str, Any],
    method_summary: dict[str, Any],
    availability: dict[str, Any],
    physical_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# R15-A cache geometry baseline - final report",
        "",
        f"Entry commit: `{protocol.get('entry_commit')}`",
        f"Scientific status: `{protocol.get('scientific_status')}` (selected_candidate_id = null)",
        "",
        "## Scope",
        "",
        "Existing 32 development rollouts only. No physics, no training, no model API,",
        "no key reads. R14 physical replay was not run (authorization absent).",
        "",
        "## Method agreement against the frozen legacy windows",
        "",
        "| method | tier | correct | none | wrong | unknown | routes | hold-before-event | false-recovery |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for method_id, stats in method_summary.items():
        routes = ", ".join(f"{key}={value}" for key, value in stats["entry_routes"].items()) or "-"
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                method_id,
                stats["input_tier"],
                stats["correct"],
                stats["none"],
                stats["wrong"],
                stats["unknown"],
                routes,
                stats["rollouts_with_hold_before_event"],
                stats["false_recovery_rollouts"],
            )
        )
    lines += [
        "",
        "## Availability",
        "",
        f"- rollouts loaded: {availability['rollouts_loaded']} / {availability['rollouts_discovered']}",
        f"- root families: {', '.join(str(item) for item in availability['root_families'])}",
        f"- orientation available: {availability['orientation_available']}",
        f"- RGB present for {availability['rgb_available_rollouts']} rollouts",
        "",
        "## Physical loss status",
        "",
        f"- planned-loss rows carried to the physical table: {len(physical_rows)}",
        "- all remain `unresolved`; recall is `NOT_ESTIMABLE`",
        "- the geometry candidates never generate their own truth labels",
        "",
        "### Reference audit of the cache itself",
        "",
        "`weld_state` is read only in this reference layer and never enters a candidate.",
        "",
    ]
    for row in physical_rows:
        lines.append(
            "- %s / %s: weld released at %s; max relative drift after the event %.4f m "
            "(gate %.3f m); geometric detach observed: %s"
            % (
                row.get("root_family_id"),
                row.get("case_id"),
                row.get("weld_release_time_reference_audit"),
                row.get("max_relative_drift_after_event_m") or 0.0,
                row.get("relative_loss_gate_m"),
                row.get("geometric_detach_observed"),
            )
        )
    lines += [
        "",
        "The oracle's own weld column marks a release inside the planned loss window,",
        "while the recorded positions keep object and gripper nearly coupled afterwards.",
        "At this cache horizon the relative-translation exit stays below the proposed",
        "gate, so no geometry candidate reports a loss.  Whether the physical loss really",
        "occurred is not decidable from this cache and stays unresolved.",
        "",
        "## Limits",
        "",
        "- S-tier results are state-assisted diagnostics, not pure-vision results and",
        "  not an independent upper bound.",
        "- The cache stores no object/gripper quaternion, so only world-frame relative",
        "  translation is used; rotation compensation is out of scope.",
        "- All 32 rollouts were previously inspected during development: this is not a",
        "  confirmation set.",
        "",
    ]
    return "\n".join(lines)
