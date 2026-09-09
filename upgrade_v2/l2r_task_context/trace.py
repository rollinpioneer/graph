from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from upgrade_v2.l2r_hold_evidence.evaluate_v2 import _load_rollout
from upgrade_v2.l2r_hold_evidence.event_adapter import infer_history_complete, run_event_interface
from upgrade_v2.l2r_hold_evidence.hold_predicates import evaluate_candidate

from .io import read_csv, read_json, write_csv, write_json


def _first_emergency(rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any] | None, bool]:
    for row in rows:
        guards = row.get("effective_guards", {})
        names = [name for name in ("retry_grasp", "recover_object") if guards.get(name) == "true"]
        if names:
            return ("conflict" if len(names) > 1 else names[0]), row, len(names) > 1
    return "none", None, False


def trace_historical(inputs_path: Path, methods: list[str], data_role: str, workers: int, output_root: Path) -> dict[str, Any]:
    del workers  # Replays are lightweight; deterministic serial ordering simplifies audit.
    if set(methods) != {"B_count1", "B_count2"}:
        raise ValueError("T1 is frozen to B_count1 and B_count2")
    if data_role != "historical_r1_7_diagnosis":
        raise ValueError("historical cache cannot be promoted to new confirmation")
    inputs = read_json(inputs_path)
    cache = Path(inputs["historical_cache_root"])
    manifest_path = cache / "data/new_development/dev_select_rollout_manifest.csv"
    metadata = read_csv(manifest_path)
    traces: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for meta in metadata:
        meta = dict(meta)
        predictions, geometries, reference = _load_rollout(meta, "control_tick_20hz")
        for method in methods:
            evidence = evaluate_candidate(predictions, geometries, method)
            decisions = run_event_interface(predictions, evidence, history_complete=infer_history_complete(predictions))
            references = reference["decision_reference_events"] or [{
                "event_id": f"{meta['rollout_id']}_none", "reference_type": "no_reference_event",
                "expected_action": "none", "decision_window_start": min(float(row["time"]) for row in predictions),
                "decision_window_end": max(float(row["time"]) for row in predictions), "physical_label_status": "reference_labeled",
            }]
            for event in references:
                start, end = float(event["decision_window_start"]), float(event["decision_window_end"])
                window = [row for row in decisions if start - 1e-9 <= float(row.get("time", -1)) <= end + 1e-9]
                action, selected, conflict = _first_emergency(window)
                selected_time = selected.get("time") if selected else None
                before = [row for row in decisions if float(row.get("time", -1)) <= start + 1e-9]
                state = (selected or (before[-1] if before else decisions[-1]))
                row = {
                    "rollout_id": meta["rollout_id"],
                    "root_family_id": meta["root_family_id"],
                    "stratum": meta["stratum"],
                    "method": method,
                    "data_role": data_role,
                    "event_id": event.get("event_id"),
                    "reference_type": event.get("reference_type"),
                    "expected_action": event.get("expected_action"),
                    "selected_action": action,
                    "selected_time": selected_time,
                    "conflict": conflict,
                    "attempt_id": state.get("attempt_id"),
                    "attempt_phase": state.get("attempt_phase"),
                    "attempt_active": state.get("attempt_active"),
                    "attempt_end": state.get("attempt_end"),
                    "attempt_end_reason": state.get("attempt_end_reason"),
                    "hold_evidence_before_event": any(item.get("hold_evidence_in_current_attempt") == "true" for item in before),
                    "pending_event": state.get("pending_event"),
                    "request_context_available": False,
                    "request_context_status": "MISSING_IN_HISTORICAL_CACHE_NOT_RECONSTRUCTED",
                    "online_trace_source": "recomputed_at_maintenance_ref_from_20hz_cache",
                }
                traces.append(row)
                if event.get("reference_type") == "touch_without_stable_hold" and action != "none":
                    conflicts.append({**row, "contract_conflict": "generic_attempt_end_does_not_encode_requested_effect"})
    write_csv(output_root / "negative_trace.csv", traces)
    write_csv(output_root / "negative_attribution.csv", conflicts)
    counts = Counter((row["method"], row["selected_action"], row["attempt_end_reason"] or "") for row in conflicts)
    write_csv(output_root / "error_by_action_and_end_reason.csv", [
        {"method": key[0], "selected_action": key[1], "attempt_end_reason": key[2], "events": value}
        for key, value in sorted(counts.items())
    ])
    write_csv(output_root / "reference_contract_conflicts.csv", [{
        "rollout_id": row["rollout_id"],
        "method": row["method"],
        "archived_reference": row["expected_action"],
        "observed_action": row["selected_action"],
        "status": "TASK_PURPOSE_NOT_RECORDED_IN_HISTORICAL_CACHE",
        "resolution": "retain old label; test explicit pre-action requests on new physical data",
    } for row in conflicts])
    representatives = [row for row in conflicts if row["method"] == "B_count2"][:8]
    write_csv(output_root / "representative_window_index.csv", [{
        "rollout_id": row["rollout_id"],
        "event_id": row["event_id"],
        "selected_time": row["selected_time"],
        "selected_action": row["selected_action"],
        "attempt_end_reason": row["attempt_end_reason"],
        "source_rollout_path": str(cache / "data/new_development/rollouts" / row["root_family_id"] / f"rollout_{int(row['rollout_id'].rsplit('_r', 1)[1]):02d}"),
        "role": "historical_diagnosis_only; RGB remains external",
    } for row in representatives])
    touch = [row for row in traces if row["reference_type"] == "touch_without_stable_hold"]
    missed = [row for row in traces if row["reference_type"] == "missed_grasp_retry_required"]
    report = {
        "schema": "pathgraph_l2rar2_negative_attribution_v1",
        "status": "TASK_CONTEXT_HYPOTHESIS_SUPPORTED_FOR_NEW_TEST",
        "historical_rollouts_replayed": len(metadata),
        "methods": methods,
        "touch_events": len(touch),
        "touch_retry_or_recover": sum(row["selected_action"] in {"retry_grasp", "recover_object", "conflict"} for row in touch),
        "missed_grasp_events": len(missed),
        "missed_grasp_retry": sum(row["selected_action"] == "retry_grasp" for row in missed),
        "per_method_touch_action": {
            method: dict(Counter(row["selected_action"] for row in touch if row["method"] == method))
            for method in methods
        },
        "historical_requested_effect_available": False,
        "new_physical_rollouts": 0,
        "scientific_boundary": "historical diagnosis only; does not validate M1 online",
    }
    write_json(output_root / "attribution_summary.json", report)
    (output_root / "cause_report.md").write_text(
        "# L2RA-R2 negative attribution\n\n"
        f"- Replayed {len(metadata)} maintenance dev-select rollouts with B_count1 and B_count2.\n"
        f"- Touch-reference decisions: {len(touch)}; emergency decisions: {report['touch_retry_or_recover']}.\n"
        f"- Missed-grasp decisions: {len(missed)}; retry decisions: {report['missed_grasp_retry']}.\n"
        "- The historical cache has lifecycle edges but no pre-action requested-effect record. The trace therefore supports testing the task-context hypothesis but cannot retroactively validate it.\n"
        "- No historical label was changed and no old confirmation was consumed.\n",
        encoding="utf-8",
    )
    return report
