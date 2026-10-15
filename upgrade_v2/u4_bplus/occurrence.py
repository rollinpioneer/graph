"""Episode-local transition extraction and event occurrence records."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .io import read_csv, read_json, read_jsonl, write_csv, write_jsonl, safe_float
from .simulator_adapter import collect_episode
from upgrade_v2.u2.dataset import load_episode
from upgrade_v2.u2.event_schema import EVENT_NAMES
from upgrade_v2.u3_grounding.evaluate_graphs import _assign_cluster, _reference_clusters


def predicates(observation: list[float] | np.ndarray) -> list[str]:
    values = list(observation)
    result = []
    if len(values) >= 17:
        result.append("contact_present" if safe_float(values[12]) >= .5 else "contact_absent")
        result.append("collision_detected" if safe_float(values[13]) >= .5 else "collision_absent")
        result.append("object_inside_goal" if safe_float(values[14]) >= .5 else "object_outside_goal")
        result.append("object_moving" if np.hypot(safe_float(values[6]), safe_float(values[7])) >= .06 else "object_stationary")
    return result or ["progress_unknown"]


def _legacy_occurrences(input_index: dict[str, Any]) -> list[dict[str, Any]]:
    source = input_index.get("legacy_segments") or input_index.get("segment_event_summary")
    if source:
        source = __import__("pathlib").Path(source)
    if not source or not source.is_file():
        return []
    rows = read_jsonl(source)
    output = []
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[row["episode_id"]].append(row)
    for episode_id, segments in by_episode.items():
        segments.sort(key=lambda x: (int(x.get("start_t", 0)), int(x.get("end_t", 0))))
        for idx, (left, right) in enumerate(zip(segments, segments[1:])):
            if left.get("split") != right.get("split"):
                continue
            pair = [int(left["cluster_id"]), int(right["cluster_id"])]
            posterior = left.get("event_posterior_mean", left.get("event_posterior", []))
            ranked = sorted(range(min(len(posterior), len(EVENT_NAMES))), key=lambda i: safe_float(posterior[i]), reverse=True)
            all_events = [EVENT_NAMES[i] for i in ranked[:3] if EVENT_NAMES[i] != "none" and safe_float(posterior[i]) >= .10]
            proposed = sorted({
                "failure_event" if event == "contact_off_failure" else
                "recovery_attempt" if event == "recovery_start" else
                "recovery_achieved" if event == "contact_reestablished" else
                "progress" if event in {"transport_on", "goal_enter"} else
                "dwell" if event == "stagnation_onset" else
                "terminal_success" if event == "stable_success" else
                "terminal_failure" if event == "terminal_failure" else "unknown"
                for event in all_events
            } - {"unknown"})
            output.append({
                "occurrence_id": f"legacy:{episode_id}:{idx}", "episode_id": episode_id, "root_family_id": left.get("root_family_id", ""), "split": left.get("split", ""), "provenance": "legacy_segment_cache",
                "before_state_index": int(left.get("end_t", 0)), "action_index": int(left.get("end_t", 0)), "after_state_index": int(right.get("start_t", 0)),
                "src_cluster_id": pair[0], "dst_cluster_id": pair[1], "source_segment_id": left.get("segment_id", ""), "destination_segment_id": right.get("segment_id", ""),
                "observable_predicates_before": predicates(left.get("raw_mean", [])), "observable_predicates_after": predicates(right.get("raw_mean", [])),
                "observable_loss_count": int(left.get("observable_contact_loss_count", 0) or 0), "proposed_semantics": proposed,
                "evaluator_event_set": [], "evaluator_label_origin": "not_available_in_legacy_cache", "terminated": False, "truncated": False, "terminal_reason": "", "known_state_budget": True,
                "is_incoming_transition": True, "original_edge_id": f"T{pair[0]}_{pair[1]}", "boundary_source_id": left.get("source_method", "legacy"), "mapper_sha": "legacy_reference_mapper",
                "transition_pair": pair, "legacy_events": all_events,
            })
    return output


def episode_occurrences(episode: dict[str, Any], split: str, refs: dict[int, dict] | None = None, boundary_source_id: str = "frozen_rule_fallback_with_evaluator_reference") -> list[dict[str, Any]]:
    rows = []
    family = episode["family"]
    for t, event in enumerate(episode["events"]):
        before = episode["observations"][t]
        after = episode["observations"][t + 1]
        all_events = set(event.get("all_events", []))
        semantics = []
        if "contact_off_failure" in all_events: semantics.extend(["failure_event", "recovery_attempt"])
        if "recovery_start" in all_events: semantics.append("recovery_attempt")
        if "contact_reestablished" in all_events: semantics.append("recovery_achieved")
        if "transport_on" in all_events: semantics.append("progress")
        if "stagnation_onset" in all_events: semantics.append("dwell")
        if "stable_success" in all_events: semantics.append("terminal_success")
        if event.get("terminal_reason") == "horizon": semantics.append("horizon")
        src = _assign_cluster({"raw_mean": before}, refs) if refs else None
        dst = _assign_cluster({"raw_mean": after}, refs) if refs else None
        rows.append({
            "occurrence_id": f"{episode['episode_id']}:t{t}", "episode_id": episode["episode_id"], "root_family_id": episode["root_family_id"], "split": split,
            "provenance": "explicit_state_stochastic_simulator", "before_state_index": t, "action_index": t, "after_state_index": t + 1,
            "src_cluster_id": src, "dst_cluster_id": dst, "source_segment_id": f"{episode['episode_id']}:state:{t}", "destination_segment_id": f"{episode['episode_id']}:state:{t+1}",
            "observable_predicates_before": predicates(before), "observable_predicates_after": predicates(after), "observable_loss_count": int("contact_off_failure" in all_events),
            "proposed_semantics": sorted(set(semantics)), "evaluator_event_set": sorted(all_events), "evaluator_label_origin": "simulator_info.events",
            "terminated": bool(event.get("terminal_reason") and event.get("terminal_reason") != "horizon"), "truncated": event.get("terminal_reason") == "horizon", "terminal_reason": event.get("terminal_reason", ""),
            "known_state_budget": True, "is_incoming_transition": True, "original_edge_id": "", "boundary_source_id": boundary_source_id, "mapper_sha": "legacy_reference_mapper_train_only",
            "family_scenario_for_analysis_only": family["scenario"], "event": event["event"], "event_id": event["event_id"], "success": episode["success"],
            "transition_pair": [src, dst] if src is not None and dst is not None else [],
        })
    return rows


def build_occurrences(input_index: dict[str, Any], splits: list[str], output, manifest) -> dict[str, Any]:
    rows = _legacy_occurrences(input_index)
    write_jsonl(output, rows)
    write_csv(manifest, [{"source": str(input_index.get("legacy_segments", "")), "rows": len(rows), "splits": ",".join(splits), "status": "PASS" if rows else "PARTIAL"}])
    return {"status": "PASS" if rows else "PARTIAL", "occurrence_count": len(rows), "legacy": True}


def build_new_occurrences(rollout_root, output, split_map: dict[str, str], repository=None, boundary_predictions=None, boundary_key: str = "auto_boundary", boundary_source_id: str = "frozen_rule_fallback_with_evaluator_reference") -> list[dict[str, Any]]:
    from pathlib import Path
    repository = Path(repository or Path.cwd())
    refs = _reference_clusters(repository)
    prediction_index = {}
    if boundary_predictions:
        prediction_index = {row["episode_id"]: row for row in read_jsonl(Path(boundary_predictions))}
    rows = []
    for path in sorted(rollout_root.glob("*.json")):
        episode = read_json(path)
        episode_rows = episode_occurrences(episode, split_map.get(episode["root_family_id"], ""), refs, boundary_source_id)
        if prediction_index:
            prediction = prediction_index.get(episode["episode_id"])
            if prediction is None or len(prediction.get(boundary_key, [])) != episode["n_steps"]:
                raise ValueError(f"missing or misaligned {boundary_key} prediction for {episode['episode_id']}")
            episode_rows = [row for row in episode_rows if prediction[boundary_key][int(row["action_index"])]]
        rows.extend(episode_rows)
    write_jsonl(output, rows)
    return rows
