from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from upgrade_v2.l2r_ambiguity.replay import trace_rollout
from upgrade_v2.l2r_ambiguity.reference_events import extract_probe_reference_events
from upgrade_v2.visual_refine_l2.io import read_jsonl
from upgrade_v2.visual_refine_l2.vision import detect_frame

from .hold_features import build_features
from .inputs import read_csv, read_json, write_csv, write_json


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _record_for_old_replay(record: dict[str, Any]) -> dict[str, Any]:
    return {**record, "prediction": {"prediction_path": record["prediction_path"]}, "scenario_reference_only": record.get("scenario_reference_only", "unknown")}


def _geometry_for_record(record: dict[str, Any], predictions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    root = Path(record["rollout_path"])
    manifest = read_csv(root / "frame_manifest.csv")
    detections = []
    missing = 0
    for row in manifest:
        path = Path(row.get("front_path", ""))
        if path.is_file():
            detection = detect_frame(path)
        else:
            detection = {"width": 640, "height": 480, "object_centroid": None, "gripper_centroid": None, "object_confidence": 0.0, "gripper_confidence": 0.0}
            missing += 1
        detection["frame_index"] = int(row["frame_index"])
        detection["time"] = float(row["time"])
        detections.append(detection)
    quality = "RECOMPUTED_FROM_FROZEN_OBSERVATIONS" if detections and missing == 0 else "REFERENCE_BLOCKED"
    return build_features(detections), quality


def inspect_errors(resolved: dict[str, Any], protocol: dict[str, Any], output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    graph = None
    for item in resolved["frozen_sources"]:
        if item["logical_id"] == "frozen:graphs/G2_evidence_refined.json" and item["exists"]:
            graph = Path(item["resolved_path"])
            break
    if graph is None:
        raise FileNotFoundError("frozen G2 graph is required for old endpoint reproduction")
    all_cases, bridge, attribution = [], [], []
    reproduction = {}
    for split in ("dev_fit", "dev_select"):
        records = [r for r in resolved["rollouts"] if r["split"] == split]
        summaries, traces = [], []
        for record in records:
            adapted = _record_for_old_replay(record)
            adapted["prediction"]["prediction_path"] = record["prediction_path"]
            rows, summary = trace_rollout(adapted, read_json(graph))
            predictions = read_jsonl(Path(record["prediction_path"]))
            geometry, quality = _geometry_for_record(record, predictions)
            for row, geo in zip(rows, geometry):
                row.update(geo)
                row["capture_phase"] = "action_end"
                row["observation_quality"] = quality
            traces.extend(rows)
            summaries.append(summary)
            references = extract_probe_reference_events(adapted)
            for event in references:
                event_rows = [row for row in rows if row["frame_index"] <= (event.get("decision_frame_index") if event.get("decision_frame_index") is not None else -1)]
                hold_seen = any(row.get("stable_hold_observed") is True for row in event_rows)
                source_time = event.get("source_event_time")
                case = {
                    "case_id": event["event_id"], "root_family_id": record["root_family_id"], "content_group_id": record.get("content_group_sha256"), "split": split,
                    "old_candidate_id": "C2_attempt_scoped_event_memory", "reference_event_id": event["event_id"], "source_event_time": source_time,
                    "source_action_interval": {"start": event.get("onset_interval_start"), "end": event.get("onset_interval_end")},
                    "reference_event_type": event["reference_event_type"], "reference_action": event["reference_action_class"],
                    "frames": event_rows, "raw_hold_seen_before_event": hold_seen, "reference_source": event.get("reference_source"),
                    "reference_observable_at_decision": event.get("reference_observable_at_decision"), "observation_quality": quality,
                }
                all_cases.append(case)
            bridge.append({"rollout_id": record["rollout_id"], "split": split, "legacy_selected_action": summary["selected_branch_legacy"], "legacy_any_overlap": summary["any_raw_overlap"], "prefix_first_decision": next((row.get("selected_action") for row in rows if row.get("grasp_failed_observed") or row.get("slip_observed")), "none"), "difference_source": "legacy_final_arbitration_vs_first_observed_prefix"})
        reproduction[split] = {"rollouts": len(summaries), "families": len({r["root_family_id"] for r in summaries}), "branch_accuracy": sum(bool(r["branch_correct"]) for r in summaries) / len(summaries) if summaries else None, "raw_overlap_rollouts": sum(bool(r["any_raw_overlap"]) for r in summaries), "replayed": True}
        _write_jsonl(output_root / f"{split}/case_trace.jsonl", traces)
    # Attribution is deliberately conservative: geometry is recomputed, but the old sparse stream cannot prove undersampling.
    attribution = [
        {"hypothesis": "STATIC_CO_MOTION", "evidence_for": "Old length-only co-motion is present in frozen predicate code; recomputed cases retain no direction field in the old cache.", "evidence_against": "No zero-motion causal separation established from sparse action-end observations.", "status": "UNRESOLVED_NEEDS_PAIRED_RECORDING"},
        {"hypothesis": "DIRECTION_DISCARDED", "evidence_for": "The old formula uses non-negative displacement norms only.", "evidence_against": "Old cached frames do not contain direction-aware causal decisions.", "status": "MECHANISM_CODE_SUPPORTED_CASE_LEVEL_UNRESOLVED"},
        {"hypothesis": "BRIEF_HOLD_UNDERSAMPLED", "evidence_for": "No control-tick image stream is present in the entry cache.", "evidence_against": "Cannot infer a missed short hold without paired recording.", "status": "UNRESOLVED_NEEDS_PAIRED_RECORDING"},
        {"hypothesis": "REFERENCE_OR_EVENT_ALIGNMENT", "evidence_for": "The bridge records first-prefix and legacy-final decisions separately.", "evidence_against": "No claim is made until timestamp alignment is independently checked.", "status": "CHECKED_WITH_BRIDGE_NOT_RESOLVED"},
    ]
    write_json(output_root / "old_endpoint_reproduction.json", {"schema": "pathgraph_l2rar1_old_endpoint_reproduction_v1", "candidates": ["C2_attempt_scoped_event_memory_h1_l1", "C2_attempt_scoped_event_memory_h2_l1"], "reproduction": reproduction, "source_role": "historical_diagnosis_only"})
    _write_jsonl(output_root / "case_trace.jsonl", all_cases)
    write_csv(output_root / "case_attribution.csv", [{"case_id": c["case_id"], "reference_event_type": c["reference_event_type"], "observation_quality": c["observation_quality"], "raw_hold_seen_before_event": c["raw_hold_seen_before_event"], "status": "TRACE_ONLY"} for c in all_cases])
    write_csv(output_root / "legacy_to_prefix_bridge.csv", bridge)
    write_csv(output_root / "case_mechanism_attribution.csv", attribution)
    report = "# R1 mechanism report\n\n" + "\n".join(f"- `{row['hypothesis']}`: **{row['status']}**. {row['evidence_for']}" for row in attribution) + "\n\nThe action-end cache supports code-level direction-loss as a mechanism, but does not identify every historical false positive or missed event. A paired control-tick recording is required before a sampling or feature gain claim.\n"
    (output_root / "mechanism_report.md").write_text(report, encoding="utf-8")
    return {"status": "R1_COMPLETE_WITH_UNRESOLVED_PAIRED_RECORDING", "case_count": len(all_cases), "reproduction": reproduction, "attribution": attribution}
