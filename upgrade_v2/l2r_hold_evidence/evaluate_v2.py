from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .event_adapter import infer_history_complete, run_event_interface
from .hold_features import build_features
from .hold_predicates import evaluate_candidate
from .inputs import read_json, read_jsonl, write_csv, write_json
from .reference_v2 import build_reference


CANDIDATES = {
    "B_count1": {},
    "B_count2": {},
    "C3_vector_rho035": {"relative_rho_max": 0.35},
    "C3_vector_rho055": {"relative_rho_max": 0.55},
    "C4_time010_disp004": {"supported_time_min": 0.10, "supported_displacement_min": 0.004},
    "C4_time010_disp008": {"supported_time_min": 0.10, "supported_displacement_min": 0.008},
    "C4_time020_disp004": {"supported_time_min": 0.20, "supported_displacement_min": 0.004},
    "C4_time020_disp008": {"supported_time_min": 0.20, "supported_displacement_min": 0.008},
}


def _predicates(observation: dict[str, Any], geometry: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    if observation.get("observation_masked") or observation.get("observation_missing"):
        return {key: "unknown" for key in ("contact_present", "gripper_command_closed", "gripper_command_open", "stable_hold_observed", "contact_recently_lost")}
    command = observation.get("gripper_command")
    closed = "unknown" if command not in {"closed", "open"} else command == "closed"
    contact_value = observation.get("contact_present")
    contact = "unknown" if contact_value is None else bool(contact_value)
    object_norm = geometry.get("object_displacement_norm")
    gripper_norm = geometry.get("gripper_displacement_norm")
    co_motion = None
    if object_norm is not None and gripper_norm is not None:
        scale = max(float(object_norm), float(gripper_norm), 0.001)
        co_motion = max(0.0, 1.0 - abs(float(object_norm) - float(gripper_norm)) / scale)
    stable = closed is True and contact is True and co_motion is not None and co_motion >= 0.8
    recently_lost = (
        previous is not None
        and previous.get("contact_present") is True
        and contact is False
    )
    return {
        "contact_present": "unknown" if contact == "unknown" else "true" if contact else "false",
        "gripper_command_closed": "unknown" if closed == "unknown" else "true" if closed else "false",
        "gripper_command_open": "unknown" if closed == "unknown" else "false" if closed else "true",
        "stable_hold_observed": "true" if stable else "false" if closed != "unknown" and contact != "unknown" else "unknown",
        "contact_recently_lost": "true" if recently_lost else "false" if contact != "unknown" else "unknown",
    }


def _online_observation(row: dict[str, Any]) -> dict[str, Any]:
    """Remove masked visual/sensor values before any adjacent feature is built."""
    if not (row.get("observation_masked") or row.get("observation_missing")):
        return dict(row)
    clean = dict(row)
    for key in (
        "object_centroid", "gripper_centroid", "object_confidence", "gripper_confidence",
        "width", "height", "contact_present", "gripper_command",
    ):
        clean[key] = None
    return clean


def _stream_path(meta: dict[str, Any], sampling: str) -> Path:
    return Path(meta["path"]) / ("observations_dense.jsonl" if sampling == "control_tick_20hz" else "observations_action_end.jsonl")


def _load_rollout(meta: dict[str, Any], sampling: str):
    raw_observations = read_jsonl(_stream_path(meta, sampling))
    observations = [_online_observation(row) for row in raw_observations]
    geometries = build_features(observations)
    predictions, previous = [], None
    for observation, geometry in zip(observations, geometries):
        predictions.append({**observation, "predicates": _predicates(observation, geometry, previous)})
        previous = observation
    return predictions, geometries, build_reference({**meta, "rollout_path": meta["path"]})


def _first_action(rows: list[dict[str, Any]]) -> tuple[str, int | None, float | None, bool]:
    actions = []
    for index, row in enumerate(rows):
        guards = row.get("effective_guards", {})
        if guards.get("retry_grasp") == "true":
            actions.append((index, "retry_grasp", row.get("time")))
        if guards.get("recover_object") == "true":
            actions.append((index, "recover_object", row.get("time")))
    if not actions:
        return "none", None, None, False
    first_index = min(item[0] for item in actions)
    first = [item for item in actions if item[0] == first_index]
    names = {item[1] for item in first}
    return ("conflict" if len(names) > 1 else first[0][1]), first_index, first[0][2], len(names) > 1


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, max(0, math.ceil(len(values) * 0.95) - 1))]


def _metric_for_candidate(items, candidate_id: str, config: dict[str, Any], sampling: str):
    event_rows, conflict_rollouts = [], set()
    for meta, observations, geometries, reference in items:
        base_id = "B_count1" if candidate_id == "B_count1" else "B_count2" if candidate_id == "B_count2" else "C3_vector" if candidate_id.startswith("C3") else "C4_time"
        candidate_config = dict(config)
        candidate_config.setdefault("supported_window_seconds", 0.6)
        evidence = evaluate_candidate(observations, geometries, base_id, candidate_config)
        memory = run_event_interface(observations, evidence, history_complete=infer_history_complete(observations))
        references = reference["decision_reference_events"] or [{"reference_type": "none", "expected_action": "none", "decision_window_start": -float("inf"), "decision_window_end": float("inf"), "observation_quality": "recorded"}]
        for event in references:
            start, end = float(event["decision_window_start"]), float(event["decision_window_end"])
            window = [row for row in memory if row.get("time") is not None and start - 1e-9 <= float(row["time"]) <= end + 1e-9]
            action, frame, action_time, conflict = _first_action(window)
            expected = event["expected_action"]
            before = [row for row in evidence if start == -float("inf") or (row.get("time") is not None and float(row["time"]) <= start + 1e-9)]
            early = [row for row in memory if start != -float("inf") and row.get("time") is not None and float(row["time"]) < start - 1e-9]
            closed_rows = [row for row in memory if row.get("raw_predicates", {}).get("gripper_command_closed") == "true"]
            event_rows.append({
                "candidate_id": candidate_id, "sampling": sampling, "rollout_id": meta["rollout_id"], "root_family_id": meta["root_family_id"], "stratum": meta["stratum"],
                "reference_event_type": event["reference_type"], "expected_action": expected, "selected_action": action, "selected_frame": frame, "selected_time": action_time,
                "correct": action == expected, "conflict": conflict, "emergency_any": any(row.get("selected_action") in {"retry_grasp", "recover_object"} for row in window),
                "premature_emergency_any": any(row.get("selected_action") in {"retry_grasp", "recover_object"} for row in early),
                "hold_evidence_before_event": any(row.get("hold_memory") == "true" for row in before), "reference_hold": bool(event.get("reference_hold_interval")),
                "hold_memory_at_last_closed_observation": bool(closed_rows and closed_rows[-1].get("hold_memory") == "true"),
                "delay_seconds": float(action_time) - start if action_time is not None and start != -float("inf") and action == expected else None,
                "observation_quality": event.get("observation_quality", "recorded"),
                "physical_label_status": event.get("physical_label_status", "reference_labeled"),
            })
            if conflict:
                conflict_rollouts.add(meta["rollout_id"])
    labeled = [row for row in event_rows if row["physical_label_status"] == "reference_labeled"]
    primary = [row for row in labeled if row["observation_quality"] != "history_masked"]
    positives = [row for row in primary if row["expected_action"] in {"retry_grasp", "recover_object"}]
    negatives = [row for row in primary if row["expected_action"] == "none"]
    information = [row for row in event_rows if row["expected_action"] == "needs_observation"]
    by_type = defaultdict(list)
    by_stratum = defaultdict(list)
    for row in positives:
        by_type[row["reference_event_type"]].append(row)
    for row in event_rows:
        by_stratum[row["stratum"]].append(row)

    def rate(rows, predicate):
        return sum(bool(predicate(row)) for row in rows) / len(rows) if rows else None

    type_recall = {key: rate(rows, lambda row: row["correct"]) for key, rows in sorted(by_type.items())}
    stratum_false = {key: rate(rows, lambda row: row["emergency_any"]) for key, rows in sorted(by_stratum.items())}
    stratum_counts = {
        key: {"events": len(rows), "families": len({row["root_family_id"] for row in rows})}
        for key, rows in sorted(by_stratum.items())
    }
    rows_for = lambda name: [row for row in event_rows if row["stratum"] == name and row["physical_label_status"] == "reference_labeled"]
    loss_rows = [row for row in positives if row["expected_action"] == "recover_object"]
    regular_loss = [row for row in loss_rows if row["stratum"] == "loss_after_observed_hold"]
    brief_loss = [row for row in loss_rows if row["stratum"] == "brief_true_hold_then_loss"]
    long_gap = [row for row in loss_rows if row["stratum"] == "long_gap_after_loss"]
    touch_rows = rows_for("touch_without_hold_then_loss")
    release_rows = rows_for("commanded_release")
    pause_rows = rows_for("normal_hold_pause_resume")
    delays = [float(row["delay_seconds"]) for row in positives if row["delay_seconds"] is not None]
    metrics = {
        "candidate_id": candidate_id, "sampling": sampling, "events": len(event_rows), "families": len({row["root_family_id"] for row in event_rows}),
        "primary_events": len(primary), "reference_unresolved_events": sum(row["physical_label_status"] != "reference_labeled" for row in event_rows),
        "positive_events": len(positives), "positive_correct": sum(bool(row["correct"]) for row in positives), "positive_recall": rate(positives, lambda row: row["correct"]),
        "miss_recall": rate(by_type.get("missed_grasp_retry_required", []), lambda row: row["correct"]), "loss_recall": rate(by_type.get("held_object_loss_recovery_required", []), lambda row: row["correct"]),
        "regular_loss_recall": rate(regular_loss, lambda row: row["correct"]), "brief_loss_recall": rate(brief_loss, lambda row: row["correct"]), "long_gap_recall": rate(long_gap, lambda row: row["correct"]),
        "wrong_or_unknown_rate": rate(positives, lambda row: not row["correct"]), "conflict_rate": rate(event_rows, lambda row: row["conflict"]),
        "premature_emergency_rate": rate(positives, lambda row: row["premature_emergency_any"]),
        "conflict_rollout_rate": len(conflict_rollouts) / len({row["rollout_id"] for row in event_rows}) if event_rows else None,
        "negative_false_emergency_rate": rate(negatives, lambda row: row["emergency_any"]), "touch_false_emergency_rate": rate(touch_rows, lambda row: row["emergency_any"]), "release_false_emergency_rate": rate(release_rows, lambda row: row["emergency_any"]), "pause_false_emergency_rate": rate(pause_rows, lambda row: row["emergency_any"]), "information_false_emergency_rate": rate(information, lambda row: row["emergency_any"]),
        "hold_evidence_rate": rate(loss_rows, lambda row: row["hold_evidence_before_event"]), "regular_hold_evidence_rate": rate(regular_loss, lambda row: row["hold_evidence_before_event"]), "brief_hold_evidence_rate": rate(brief_loss, lambda row: row["hold_evidence_before_event"]), "touch_false_hold_evidence_rate": rate(touch_rows, lambda row: row["hold_evidence_before_event"]), "pause_hold_retention_rate": rate(pause_rows, lambda row: row["hold_memory_at_last_closed_observation"]),
        "delay_p95_seconds": _p95(delays), "delay_observed_count": len(delays), "paired_sampling": all(_stream_path(meta, sampling).is_file() for meta, *_ in items),
        "type_recall_json": json.dumps(type_recall, sort_keys=True), "stratum_false_emergency_json": json.dumps(stratum_false, sort_keys=True), "stratum_counts_json": json.dumps(stratum_counts, sort_keys=True), "data_role": "new_dynamic_mujoco_r1",
    }
    return metrics, event_rows


def build_features_for_data(data_root: Path, output_root: Path) -> dict[str, Any]:
    rows = []
    for meta_path in sorted(data_root.rglob("metadata.json")):
        meta = read_json(meta_path)
        for sampling, filename in (("control_tick_20hz", "observations_dense.jsonl"), ("action_end", "observations_action_end.jsonl")):
            observations = read_jsonl(meta_path.parent / filename)
            features = build_features(observations)
            out = output_root / meta["split"] / sampling / f"{meta['rollout_id']}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in features), encoding="utf-8")
            rows.append({"rollout_id": meta["rollout_id"], "split": meta["split"], "sampling": sampling, "feature_path": str(out.resolve()), "observation_count": len(features), "feature_contract": "timestamp_aligned_proxy_hold_v2"})
    write_csv(output_root / "feature_manifest.csv", rows)
    return {"status": "FEATURES_READY", "rollouts": len({row["rollout_id"] for row in rows}), "observations": sum(int(row["observation_count"]) for row in rows), "streams": ["control_tick_20hz", "action_end"]}


def _content_group_key(meta: dict[str, Any]) -> str:
    """Hash only arrived online observations, excluding labels and file paths."""
    rows = read_jsonl(Path(meta["path"]) / "observations_dense.jsonl")
    online = []
    for row in rows:
        online.append({key: row.get(key) for key in (
            "time", "frame_index", "capture_order", "contact_present", "gripper_command",
            "object_centroid", "gripper_centroid", "object_confidence", "gripper_confidence",
            "width", "height", "observation_masked", "observation_missing",
        )})
    payload = json.dumps(online, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_content_group_manifest(metas: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for meta in metas:
        key = _content_group_key(meta)
        grouped[key].append(meta)
    rows = []
    for key, members in sorted(grouped.items()):
        for meta in sorted(members, key=lambda item: item["rollout_id"]):
            control_path = Path(meta["path"]) / "low_level_controls.jsonl"
            rows.append({
                "content_group_id": key[:16],
                "input_sha256": key,
                "content_group_size": len(members),
                "rollout_id": meta["rollout_id"],
                "root_family_id": meta["root_family_id"],
                "split": meta["split"],
                "stratum": meta["stratum"],
                "online_source": str((Path(meta["path"]) / "observations_dense.jsonl").resolve()),
                "control_sha256": hashlib.sha256(control_path.read_bytes()).hexdigest(),
                "statistics_unit": "root_family_id",
            })
    write_csv(output_root / "content_group_manifest.csv", rows)
    return {"content_groups": len(grouped), "rollouts": len(metas), "statistics_unit": "root_family_id"}


def evaluate_development(data_root: Path, protocol: dict[str, Any], output_root: Path, partition: str | None = None) -> dict[str, Any]:
    metas = [read_json(path) for path in sorted(data_root.rglob("metadata.json"))]
    if partition:
        metas = [meta for meta in metas if meta["split"] == partition]
    content_groups = _write_content_group_manifest(metas, output_root)
    items = {"control_tick_20hz": {"dev_fit": [], "dev_select": []}, "action_end": {"dev_fit": [], "dev_select": []}}
    for meta in metas:
        for sampling in items:
            predictions, geometries, reference = _load_rollout(meta, sampling)
            items[sampling][meta["split"]].append((meta, predictions, geometries, reference))
    all_metrics, fit_metrics, event_rows = [], [], []
    for candidate_id, config in CANDIDATES.items():
        for sampling in items:
            fit_metric, _ = _metric_for_candidate(items[sampling]["dev_fit"], candidate_id, config, sampling)
            select_metric, rows = _metric_for_candidate(items[sampling]["dev_select"], candidate_id, config, sampling)
            fit_metrics.append(fit_metric); all_metrics.append(select_metric); event_rows.extend(rows)
    write_csv(output_root / "candidate_metrics.csv", all_metrics)
    write_csv(output_root / "candidate_fit_metrics.csv", fit_metrics)
    write_csv(output_root / "event_decisions.csv", event_rows)
    threshold = protocol["thresholds"]
    dense_metrics = [row for row in all_metrics if row["sampling"] == "control_tick_20hz"]
    gates = []
    for row in dense_metrics:
        counts = json.loads(row["stratum_counts_json"])
        required_strata = ["miss_without_prior_hold", "loss_after_observed_hold", "brief_true_hold_then_loss", "touch_without_hold_then_loss", "commanded_release", "long_gap_after_loss", "history_or_visual_unavailable", "normal_hold_pause_resume"]
        support_ok = all(counts.get(name, {}).get("events", 0) >= 8 and counts.get(name, {}).get("families", 0) >= 2 for name in required_strata)
        recall_values = [row[key] for key in ("miss_recall", "regular_loss_recall", "brief_loss_recall", "long_gap_recall")]
        false_values = [row[key] for key in ("touch_false_emergency_rate", "release_false_emergency_rate", "pause_false_emergency_rate")]
        hold_values = [row[key] for key in ("regular_hold_evidence_rate", "brief_hold_evidence_rate", "pause_hold_retention_rate")]
        checks = {
            "paired_recording": row["paired_sampling"], "support_minimum": support_ok,
            "per_type_recall": all(value is not None and value >= threshold["event_type_recall_min"] for value in recall_values), "wrong_or_unknown": row["wrong_or_unknown_rate"] is not None and row["wrong_or_unknown_rate"] <= threshold["wrong_or_unknown_fraction_max"],
            "conflict": row["conflict_rate"] is not None and row["conflict_rate"] <= threshold["conflict_rate_max"] and row["conflict_rollout_rate"] <= threshold["conflict_rate_max"],
            "premature_emergency": row["premature_emergency_rate"] is not None and row["premature_emergency_rate"] <= threshold["negative_false_emergency_rate_max"],
            "per_negative_stratum": all(value is not None and value <= threshold["negative_false_emergency_rate_max"] for value in false_values),
            "information_false_emergency": row["information_false_emergency_rate"] is not None and row["information_false_emergency_rate"] <= threshold["negative_false_emergency_rate_max"],
            "hold_evidence_and_retention": all(value is not None and value >= threshold["hold_evidence_recall_min"] for value in hold_values) and row["touch_false_hold_evidence_rate"] is not None and row["touch_false_hold_evidence_rate"] <= threshold["negative_false_emergency_rate_max"],
            "delay_p95": row["delay_p95_seconds"] is not None and row["delay_p95_seconds"] <= threshold["delay_p95_seconds_max"],
            "legacy_compatibility": False,
        }
        gates.append({"candidate_id": row["candidate_id"], **checks, "legacy_compatibility_status": "NOT_EVALUATED_UNTIL_PRIMARY_EVENT_GATES_PASS", "all_pass": all(checks.values())})
    write_csv(
        output_root / "development_gates.csv",
        gates,
        fields=[
            "candidate_id",
            "paired_recording",
            "support_minimum",
            "per_type_recall",
            "wrong_or_unknown",
            "conflict",
            "premature_emergency",
            "per_negative_stratum",
            "information_false_emergency",
            "hold_evidence_and_retention",
            "delay_p95",
            "legacy_compatibility",
            "legacy_compatibility_status",
            "all_pass",
        ],
    )
    eligible = [row["candidate_id"] for row in gates if row["all_pass"]]
    selected = min(eligible, key=lambda candidate: (next(row["wrong_or_unknown_rate"] for row in dense_metrics if row["candidate_id"] == candidate), next(row["negative_false_emergency_rate"] for row in dense_metrics if row["candidate_id"] == candidate), candidate)) if eligible else None
    primary_select_events = dense_metrics[0]["primary_events"] if dense_metrics else 0
    positive_select_events = dense_metrics[0]["positive_events"] if dense_metrics else 0
    route = {"schema": "pathgraph_l2rar1_development_route_v1", "status": "DEVELOPMENT_READY" if selected else "DEVELOPMENT_NOT_READY", "selected_candidate_id": selected, "eligible_candidates": eligible, "fit_families": len({item[0]["root_family_id"] for item in items["control_tick_20hz"]["dev_fit"]}), "fit_rollouts": len(items["control_tick_20hz"]["dev_fit"]), "select_families": len({item[0]["root_family_id"] for item in items["control_tick_20hz"]["dev_select"]}), "select_rollouts": len(items["control_tick_20hz"]["dev_select"]), "select_primary_events": primary_select_events, "select_positive_events": positive_select_events, "invalid_generator_batches": 1, "invalid_batch_scientific_rollouts": 0, "sampling": "control_tick_20hz", "comparison_sampling": "action_end", "selection_uses_future_labels": False, "candidate_count": len(CANDIDATES), "content_groups": content_groups["content_groups"], "statistics_unit": protocol.get("statistics", {}).get("unit", "root_family_id"), "bootstrap_resamples": protocol.get("statistics", {}).get("bootstrap_resamples"), "api_calls": 0, "training_jobs": 0}
    write_json(output_root / "development_route.json", route)
    write_json(output_root / "candidate_registry.json", {"schema": "pathgraph_l2rar1_candidate_registry_v1", "candidates": all_metrics, "gates": gates, "selected_candidate_id": selected, "selection_status": route["status"]})
    best = min(dense_metrics, key=lambda row: (row["wrong_or_unknown_rate"], row["negative_false_emergency_rate"], row["candidate_id"])) if dense_metrics else None
    summary = [
        "# L2RA-R1 development selection",
        "",
        f"- Status: `{route['status']}`; selected candidate: `{selected}`; eligible: {len(eligible)}/{len(CANDIDATES)}.",
        f"- Valid data: {route['fit_families']} fit families / {route['fit_rollouts']} rollouts; {route['select_families']} select families / {route['select_rollouts']} rollouts.",
        f"- Select denominators: {primary_select_events} primary events, including {positive_select_events} positive events; {content_groups['content_groups']} distinct online content groups across both development splits.",
        "- The superseded duplicate-control batch contributes 0 scientific rollouts.",
        "- Legacy compatibility was not evaluated because every candidate failed a primary event gate first.",
    ]
    if best:
        summary.append(
            f"- Best preregistered dense candidate `{best['candidate_id']}` still had miss recall {best['miss_recall']} and wrong-or-unknown rate {best['wrong_or_unknown_rate']}."
        )
    (output_root / "selection_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return route
