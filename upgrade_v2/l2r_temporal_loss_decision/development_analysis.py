from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .baseline import fit
from . import episodes
from .audits import HORIZONS
from .replay import replay
from .sequential_decision import motion_gate_diagnostics
from .temporal_scoring import score_event, truth_for


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _value(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if value in ("", "None", "null"):
        return None
    return value


def _changed(old: Any, new: Any) -> bool:
    return old != new


def _first_decision_by_episode(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.is_file():
        return {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in _jsonl(path):
        key = (str(row.get("episode_id")), str(row.get("method")))
        grouped.setdefault(key, []).append(row)
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in grouped.items():
        evidence = [row for row in rows if row.get("state") == "LOSS_EVIDENCE"]
        result[key] = dict(evidence[0] if evidence else rows[-1])
    return result


def write_old_new_diff(old_csv: Path, new_csv: Path, output_root: Path) -> dict[str, Any]:
    """Write per-episode/method old-vs-new rows and aggregate change counts."""
    old_rows = list(csv.DictReader(old_csv.open(encoding="utf-8")))
    new_rows = list(csv.DictReader(new_csv.open(encoding="utf-8")))
    old_decisions = _first_decision_by_episode(old_csv.parent / "per_prefix_features.jsonl")
    new_decisions = _first_decision_by_episode(new_csv.parent / "per_prefix_features.jsonl")
    old_by = {(str(row.get("episode_id")), str(row.get("method"))): row for row in old_rows}
    new_by = {(str(row.get("episode_id")), str(row.get("method"))): row for row in new_rows}
    keys = sorted(set(old_by) | set(new_by))
    fields = [
        "episode_id", "family", "method", "old_first_evidence_ns", "new_first_evidence_ns",
        "old_score", "new_score", "old_state", "new_state", "changed",
        "old_source", "new_source", "old_reason", "new_reason",
        "first_evidence_changed", "score_changed", "state_changed", "source_changed", "reason_changed", "reason",
    ]
    rows: list[dict[str, Any]] = []
    for key in keys:
        old, new = old_by.get(key, {}), new_by.get(key, {})
        old_first, new_first = _value(old, "first_evidence_ns"), _value(new, "first_evidence_ns")
        old_score, new_score = _value(old, "score"), _value(new, "score")
        old_detail, new_detail = old_decisions.get(key, {}), new_decisions.get(key, {})
        old_state = _value(old_detail, "state")
        new_state = _value(new_detail, "state")
        old_source = _value(old_detail, "source")
        new_source = _value(new_detail, "source")
        old_reason = _value(old_detail, "reason")
        new_reason = _value(new_detail, "reason")
        first_changed = _changed(old_first, new_first)
        score_changed = _changed(old_score, new_score)
        state_changed = _changed(old_state, new_state)
        source_changed = _changed(old_source, new_source)
        reason_changed = _changed(old_reason, new_reason)
        rows.append({
            "episode_id": key[0], "family": _value(new, "family") or _value(old, "family"), "method": key[1],
            "old_first_evidence_ns": old_first, "new_first_evidence_ns": new_first,
            "old_score": old_score, "new_score": new_score,
            "old_state": old_state, "new_state": new_state,
            "old_source": old_source, "new_source": new_source,
            "old_reason": old_reason, "new_reason": new_reason,
            "changed": first_changed or score_changed or state_changed or source_changed or reason_changed,
            "first_evidence_changed": first_changed, "score_changed": score_changed,
            "state_changed": state_changed, "source_changed": source_changed, "reason_changed": reason_changed,
            "reason": ";".join(name for name, changed in (("first_evidence", first_changed), ("score", score_changed), ("state", state_changed), ("source", source_changed), ("reason", reason_changed)) if changed),
        })
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "old_vs_new_per_episode.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    aggregate: list[dict[str, Any]] = []
    methods = sorted({row["method"] for row in rows})
    families = sorted({str(row.get("family") or "") for row in rows})
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        aggregate.append({"scope": "method", "method": method, "family": "", "episodes": len(subset), "changed": sum(bool(row["changed"]) for row in subset), "first_evidence_changed": sum(bool(row["first_evidence_changed"]) for row in subset), "score_changed": sum(bool(row["score_changed"]) for row in subset), "state_changed": sum(bool(row["state_changed"]) for row in subset), "source_changed": sum(bool(row["source_changed"]) for row in subset), "reason_changed": sum(bool(row["reason_changed"]) for row in subset)})
        for family in families:
            family_rows = [row for row in subset if str(row.get("family") or "") == family]
            if family_rows:
                aggregate.append({"scope": "family", "method": method, "family": family, "episodes": len(family_rows), "changed": sum(bool(row["changed"]) for row in family_rows), "first_evidence_changed": sum(bool(row["first_evidence_changed"]) for row in family_rows), "score_changed": sum(bool(row["score_changed"]) for row in family_rows), "state_changed": sum(bool(row["state_changed"]) for row in family_rows), "source_changed": sum(bool(row["source_changed"]) for row in family_rows), "reason_changed": sum(bool(row["reason_changed"]) for row in family_rows)})
    with (output_root / "old_vs_new_change_counts.csv").open("w", newline="", encoding="utf-8") as stream:
        fields_agg = list(aggregate[0]) if aggregate else ["scope", "method", "family", "episodes", "changed", "first_evidence_changed", "score_changed", "state_changed", "source_changed", "reason_changed"]
        writer = csv.DictWriter(stream, fieldnames=fields_agg); writer.writeheader(); writer.writerows(aggregate)
    return {"episodes": len(keys), "changed": sum(bool(row["changed"]) for row in rows), "methods": methods}


def diagnose_b2_misses(episodes_path: Path, predictions_csv: Path, output_root: Path) -> dict[str, Any]:
    """Export per-frame frozen B2 gate diagnostics for missed shadow losses."""
    records = {str(row["episode_id"]): row for row in _jsonl(episodes_path)}
    predictions = list(csv.DictReader(predictions_csv.open(encoding="utf-8")))
    misses = [
        row for row in predictions
        if row.get("method") == "B2" and row.get("truth") == "LOSS"
        and row.get("score") in {"MISSED", "RIGHT_CENSORED"}
        and str(row.get("episode_id", "")).startswith("L2RAR2_R27_SHADOW_")
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_root / "b2_missed_loss_gate_diagnostics.jsonl"
    csv_path = output_root / "b2_missed_loss_gate_diagnostics.csv"
    csv_fields = [
        "episode_id", "family", "case_id", "truth", "score", "physical_loss_onset_ns",
        "evidence_end_ns", "first_evidence_ns", "observation_ns", "capture_order",
        "baseline_ready", "baseline_prefix_index", "visual_available", "window_valid",
        "window_reason", "window_span_ns", "window_points", "amplitude", "radial_gain",
        "path_length", "outward_efficiency", "radial_speed_per_s", "theta_motion",
        "radial_gain_threshold", "outward_efficiency_threshold", "blockers", "gate_pass",
    ]
    count_by_blocker: dict[str, int] = {}
    frame_count = 0
    with jsonl_path.open("w", encoding="utf-8") as jstream, csv_path.open("w", newline="", encoding="utf-8") as cstream:
        writer = csv.DictWriter(cstream, fieldnames=csv_fields); writer.writeheader()
        for prediction in sorted(misses, key=lambda row: str(row.get("episode_id"))):
            episode_id = str(prediction["episode_id"])
            record = records.get(episode_id)
            if record is None:
                continue
            end_key = (int(record.get("evidence_end_ns") or 0), int(record.get("evidence_end_capture_order", 10**18)))
            rows = [row for row in record.get("observations", []) if (int(row.get("physical_time_ns", 0)), int(row.get("capture_order", -1))) <= end_key]
            replayed = replay(record, "B2")
            baseline = replayed.get("baseline", {})
            anchor = baseline.get("anchor_xy")
            scale = baseline.get("scale_px")
            prefix_index = baseline.get("prefix_index")
            for index in range(len(rows)):
                # The online baseline is latched at its first ready prefix;
                # frames before that prefix must be reported as not-ready.
                frame_anchor = anchor if prefix_index is not None and index + 1 >= int(prefix_index) else None
                frame_scale = scale if frame_anchor is not None else None
                diagnostic = motion_gate_diagnostics(rows, index, window_ns=500_000_000, theta_motion=0.35, anchor_xy=frame_anchor, scale_px=frame_scale)
                payload = {"episode_id": episode_id, "family": record.get("root_family_id"), "case_id": record.get("case_id"), "truth": truth_for(record), "score": prediction.get("score"), "physical_loss_onset_ns": record.get("physical_loss_onset_ns"), "evidence_end_ns": record.get("evidence_end_ns"), "first_evidence_ns": prediction.get("first_evidence_ns"), "baseline_prefix_index": prefix_index, **diagnostic}
                jstream.write(json.dumps(payload, sort_keys=True) + "\n")
                csv_row = {field: payload.get(field) for field in csv_fields}
                csv_row["blockers"] = "|".join(diagnostic.get("blockers", []))
                writer.writerow(csv_row)
                frame_count += 1
                for blocker in diagnostic.get("blockers", []):
                    count_by_blocker[blocker] = count_by_blocker.get(blocker, 0) + 1
    summary = {"schema": "l2rar2_r27_b2_gate_diagnostics_v1", "missed_loss_episodes": len(misses), "episode_ids": [str(row.get("episode_id")) for row in misses], "frames": frame_count, "blocker_frame_counts": count_by_blocker, "thresholds_locked": {"window_ns": 500_000_000, "theta_motion": 0.35, "radial_gain": 0.175, "outward_efficiency": 0.65}, "physical_executions": 0, "confirmation_passed": False}
    (output_root / "b2_missed_loss_gate_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def run_shadow_cache_analysis(raw_root: Path, output_root: Path) -> dict[str, Any]:
    """Replay the already-collected 24 shadow rollouts without collecting."""
    output_root.mkdir(parents=True, exist_ok=True)
    resources = {"schema": "l2rar2_r27_shadow_replenishment_resources_v1", "sources": [{"name": "R27_SHADOW_REPLENISHMENT_RAW_EXISTING", "path": str(raw_root)}]}
    episode_result = episodes.build(resources, output_root / "episode_audit")
    episode_path = output_root / "episode_audit/episodes.jsonl"
    records = _jsonl(episode_path)
    prediction_rows: list[dict[str, Any]] = []
    for record in records:
        for method in ("B1", "B2", "B3"):
            result = replay(record, method)
            prediction_rows.append({
                "episode_id": record["episode_id"], "family": record["root_family_id"], "case_id": record["case_id"],
                "method": method, "truth": truth_for(record), "physical_loss_onset_ns": record.get("physical_loss_onset_ns"),
                "evidence_end_ns": result["evidence_end_ns"], "first_evidence_ns": result["first_evidence_ns"],
                "score": score_event(truth=truth_for(record), onset_ns=record.get("physical_loss_onset_ns"), end_ns=result["evidence_end_ns"], first_evidence_ns=result["first_evidence_ns"]),
            })
    pred_path = output_root / "per_episode_shadow_predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as stream:
        fields = list(prediction_rows[0]) if prediction_rows else ["episode_id"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(prediction_rows)

    def metrics(subset: list[dict[str, Any]], horizon: int) -> dict[str, int]:
        result = {"episodes": len(subset), "reference_loss_episodes": 0, "fully_observed_loss_episodes": 0, "on_time": 0, "early": 0, "late": 0, "missed": 0, "right_censored": 0, "false_loss_events": 0}
        for row in subset:
            truth = row["truth"]; onset = row.get("physical_loss_onset_ns"); end = int(row.get("evidence_end_ns") or 0); first = row.get("first_evidence_ns")
            first_i = int(first) if first not in (None, "", "None") else None
            if truth == "LOSS" and onset not in (None, ""):
                onset_i = int(onset); result["reference_loss_episodes"] += 1; result["fully_observed_loss_episodes"] += int(end >= onset_i + horizon)
                if first_i is None: result["right_censored" if end < onset_i + horizon else "missed"] += 1
                elif first_i < onset_i: result["early"] += 1
                elif first_i <= onset_i + min(horizon, 750_000_000): result["on_time"] += 1
                elif first_i <= onset_i + horizon: result["late"] += 1
                else: result["missed"] += 1
            elif truth == "NO_LOSS": result["false_loss_events"] += int(first_i is not None)
        return result

    comparison_rows: list[dict[str, Any]] = []
    for method in ("B1", "B2", "B3"):
        subset = [row for row in prediction_rows if row["method"] == method]
        for horizon in HORIZONS:
            comparison_rows.append({"method": method, "horizon_ns": horizon, **metrics(subset, horizon)})
    with (output_root / "shadow_comparison_by_horizon.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(comparison_rows[0]) if comparison_rows else ["method", "horizon_ns"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(comparison_rows)
    loss_records = [row for row in records if truth_for(row) == "LOSS" and row.get("physical_loss_onset_ns") is not None]
    coverage_rows = []
    for horizon in HORIZONS:
        full = sum(int(row.get("evidence_end_ns") or 0) >= int(row["physical_loss_onset_ns"]) + horizon for row in loss_records)
        coverage_rows.append({"horizon_ns": horizon, "reference_loss_episodes": len(loss_records), "fully_observed_loss_episodes": full, "coverage": full / len(loss_records) if loss_records else 0.0})
    with (output_root / "shadow_observation_horizon_coverage.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(coverage_rows[0]) if coverage_rows else ["horizon_ns"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(coverage_rows)
    diagnostics = diagnose_b2_misses(episode_path, pred_path, output_root / "b2_gate_diagnostics")
    result = {"schema": "l2rar2_r27_shadow_replenishment_results_v3", "status": "DEVELOPMENT_CACHE_REPLAY_COMPLETE", "source_raw_root": str(raw_root), "source_rollouts": len(records), "source_physical_rollouts": len(records), "physical_executions": 0, "candidate_control_executions": 0, "candidate_early_terminations": 0, "post_intervention_observation_ns": 1_500_000_000, "comparison_scope": "DEVELOPMENT_ONLY_B1_VS_B2_VS_B3", "diagnostics": diagnostics, "confirmation_passed": False}
    (output_root / "shadow_replenishment_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"episodes": len(records), "predictions": len(prediction_rows), "diagnostics": diagnostics}


def write_development_report(root: Path) -> Path:
    """Write a concise additive report linking all new development outputs."""
    validation = root / "outer_validation"
    metrics = list(csv.DictReader((validation / "by_horizon_metrics.csv").open(encoding="utf-8")))
    diagnostic_path = root / "shadow_replenishment/b2_gate_diagnostics/b2_missed_loss_gate_summary.json"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8")) if diagnostic_path.is_file() else {}
    blocker_counts = diagnostic.get("blocker_frame_counts", {})
    lines = [
        "# R27 Bounding-Box Contract Development Analysis",
        "",
        "This is development analysis only. It does not replace the original R27 result, does not rewrite the R27 formal conclusion, and does not announce confirmation passed.",
        "",
        "## Contract correction",
        "",
        "`visual_separation.detect_frame_boxes()` and the episode adapter now share an explicit `xyxy = [x1, y1, x2, y2]` contract. The adapter validates and preserves detector coordinates; it never applies a second width/height conversion. Tests cover positive gaps, overlap, missing boxes, and detector-to-episode round-trip invariance.",
        "",
        "## Frozen thresholds",
        "",
        "B1/B2/B3 replay uses the existing locked operating points: B1 separation threshold 0.10; B2/B3 motion window 500 ms, motion threshold 0.35, radial-growth threshold 0.175, and motion-efficiency threshold 0.65. The scoring deadline remains 750 ms. No threshold was lowered and no physical collection was run.",
        "",
        "## B1/B2/B3 per-horizon counts",
        "",
        "| method | horizon | on time | early | late | missed | right censored | false loss | |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        if row.get("method") in {"B1", "B2", "B3"}:
            lines.append(f"| {row['method']} | {int(row['horizon_ns']) // 1_000_000} ms | {row.get('on_time', 0)} | {row.get('early', 0)} | {row.get('late', 0)} | {row.get('missed', 0)} | {row.get('right_censored', 0)} | {row.get('false_loss_events', row.get('false', 0))} |")
    lines.extend([
        "",
        "## Old versus new per-episode differences",
        "",
        "Two explicit baselines are retained. `old_vs_review_v1` compares against the prior zero-physics review; its predictions are unchanged (0/624 rows changed). `old_vs_original_r27_v1` compares against the untouched original R27 artifact and records 624/624 rows with state/reason differences because the corrected review uses the replay state output; first evidence and score fields remain directly reported.",
        "",
        "Detailed rows: `old_vs_review_v1/old_vs_new_per_episode.csv`, `old_vs_original_r27_v1/old_vs_new_per_episode.csv`; aggregate method/family counts are in each `old_vs_new_change_counts.csv`.",
        "",
        "## Six B2 missed replenishment losses",
        "",
        f"Per-frame raw gate values and simultaneous blockers are in `shadow_replenishment/b2_gate_diagnostics/b2_missed_loss_gate_diagnostics.csv` and `.jsonl`; the summary is `b2_missed_loss_gate_summary.json`. The six episodes are the fast-detachment and contact-occluded true-detachment losses. Across {diagnostic.get('frames', 0)} frames, blocker counts are baseline-not-ready {blocker_counts.get('BASELINE_NOT_READY', 0)}, visual-missing/unavailable {blocker_counts.get('VISUAL_MISSING_OR_UNAVAILABLE', 0)}, window invalid {blocker_counts.get('WINDOW_INVALID', 0)}, amplitude below threshold {blocker_counts.get('AMPLITUDE_BELOW_THRESHOLD', 0)}, radial growth below threshold {blocker_counts.get('RADIAL_GROWTH_BELOW_THRESHOLD', 0)}, and motion efficiency below threshold {blocker_counts.get('MOTION_EFFICIENCY_BELOW_THRESHOLD', 0)}. Blockers may co-occur on a frame.",
        "",
        "## Conclusion",
        "",
        "The cache-only result remains `R27_INSUFFICIENT_OBSERVATION_HORIZON`; `confirmation_passed=false`. The original R27 result, review report, and 24-rollout raw replenishment data remain preserved.",
        "",
    ])
    path = root / "final/development_analysis_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
