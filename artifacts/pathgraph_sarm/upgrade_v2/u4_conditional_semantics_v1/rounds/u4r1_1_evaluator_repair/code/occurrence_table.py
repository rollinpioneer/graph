"""Episode-local occurrence extraction with explicit terminal classification."""
from __future__ import annotations

import hashlib
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import read_json, read_jsonl, write_csv, write_jsonl


SEMANTICS = {
    "contact_off_failure": "failure_event",
    "recovery_start": "recovery_attempt",
    "contact_reestablished": "recovery_achieved",
    "transport_on": "progress",
    "goal_enter": "progress",
    "detour_start": "alternative",
    "stagnation_onset": "dwell",
    "stable_success": "terminal_success",
}

EVENT_NAMES = {
    0: "none", 1: "contact_on", 2: "transport_on", 3: "contact_off_failure",
    4: "recovery_start", 5: "contact_reestablished", 6: "detour_start",
    7: "goal_enter", 8: "stable_success", 9: "terminal_failure", 10: "stagnation_onset",
}


def _cluster_refs(repo: Path) -> dict[int, dict[str, Any]]:
    source = repo / "artifacts/pathgraph_sarm/upgrade_v2/u3_candidate_graph/inputs_v1/segment_event_summary_train.jsonl"
    if not source.is_file():
        return {}
    rows = read_jsonl(source)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        try:
            grouped[int(row["cluster_id"])].append(row)
        except (KeyError, TypeError, ValueError):
            continue
    refs = {}
    for cid, items in grouped.items():
        raw = []
        for i in range(max((len(x.get("raw_mean", [])) for x in items), default=0)):
            values = [float(x.get("raw_mean", [])[i]) for x in items if i < len(x.get("raw_mean", []))]
            raw.append(sum(values) / len(values) if values else 0.0)
        refs[cid] = {"raw": raw}
    return refs


def assign_cluster(observation: list[float], refs: dict[int, dict[str, Any]]) -> int | None:
    if not refs:
        return None
    best = None
    for cid, ref in refs.items():
        length = min(len(observation), len(ref["raw"]))
        distance = sum((float(observation[i]) - float(ref["raw"][i])) ** 2 for i in range(length)) ** 0.5
        key = (1.0 / (1.0 + distance), -cid)
        if best is None or key > best[0]:
            best = (key, cid)
    return best[1] if best else None


def terminal_status(event: dict[str, Any], episode: dict[str, Any], index: int) -> tuple[str, bool, bool]:
    reason = str(event.get("terminal_reason") or "")
    is_horizon = reason == "horizon"
    # Horizon is censoring, even though the simulator's event priority calls it
    # terminal_failure. It is never a physical failure label.
    if is_horizon:
        return "censored_unknown", False, True
    if reason or (index == len(episode.get("events", [])) - 1 and episode.get("success")):
        if episode.get("success") or "stable_success" in event.get("all_events", []):
            return "success_terminal", True, False
        return "failure_terminal", True, False
    return "nonterminal", False, False


def observable_context(before: list[float], after: list[float], action: list[float], history: list[dict[str, Any]]) -> dict[str, Any]:
    contact_before = bool(len(before) > 12 and float(before[12]) >= 0.5)
    contact_after = bool(len(after) > 12 and float(after[12]) >= 0.5)
    collision = bool(len(after) > 13 and float(after[13]) >= 0.5)
    goal_before = bool(len(before) > 14 and float(before[14]) >= 0.5)
    goal_after = bool(len(after) > 14 and float(after[14]) >= 0.5)
    before_goal = (float(before[4]) ** 2 + float(before[5]) ** 2) ** 0.5 if len(before) > 5 else 0.0
    after_goal = (float(after[4]) ** 2 + float(after[5]) ** 2) ** 0.5 if len(after) > 5 else before_goal
    delta = after_goal - before_goal
    object_speed = (float(after[6]) ** 2 + float(after[7]) ** 2) ** 0.5 if len(after) > 7 else 0.0
    recent_names = {name for item in history[-8:] for name in item.get("all_events", [])}
    context = {
        "contact_before": contact_before,
        "contact_after": contact_after,
        "contact_present": contact_after,
        "contact_recently_lost": (not contact_before and "contact_off_failure" in recent_names),
        "collision_detected": collision,
        "object_inside_goal": goal_after,
        "stagnation_detected": "stagnation_onset" in recent_names,
        "goal_distance_delta_sign": "negative" if delta < -1e-9 else ("positive" if delta > 1e-9 else "zero"),
        "object_speed_bin": "moving" if object_speed >= 0.06 else ("slow" if object_speed >= 0.007 else "still"),
        "recent_recovery_attempt": "recovery_start" in recent_names,
        "object_moving": object_speed >= 0.06,
        "action_norm": (float(action[0]) ** 2 + float(action[1]) ** 2) ** 0.5 if len(action) >= 2 else 0.0,
        "history_event_count": len(history),
    }
    # These names are useful for model inputs but are not added to the gold
    # label set or to scenario/phase metadata.
    context["goal_entered_before"] = goal_before
    return context


def build_occurrences(rollout_root: Path, output: Path, repo: Path, split: str, predictions: Path | None = None) -> dict[str, Any]:
    refs = _cluster_refs(repo)
    prediction_index = {row["episode_id"]: row for row in read_jsonl(predictions)} if predictions else {}
    rows: list[dict[str, Any]] = []
    for path in sorted(rollout_root.glob("*.json")):
        episode = read_json(path)
        observations = episode.get("observations", [])
        actions = episode.get("actions", [])
        events = episode.get("events", [])
        if len(observations) != len(actions) + 1 or len(events) != len(actions):
            raise ValueError(f"episode-local alignment failed: {path}")
        history: list[dict[str, Any]] = []
        for index, event in enumerate(events):
            prediction = prediction_index.get(episode["episode_id"])
            if prediction is not None:
                boundary = prediction.get("auto_boundary", [])
                if len(boundary) != len(actions):
                    raise ValueError(f"boundary prediction alignment failed: {path}")
                if not boundary[index]:
                    continue
            before = observations[index]
            after = observations[index + 1]
            all_events = sorted(set(event.get("all_events", [])))
            status, terminated, truncated = terminal_status(event, episode, index)
            semantics = sorted({SEMANTICS[name] for name in all_events if name in SEMANTICS})
            if status == "failure_terminal":
                semantics.append("terminal_failure")
            if status == "censored_unknown":
                semantics.append("censored_unknown")
            context = observable_context(before, after, actions[index], history)
            row = {
                "occurrence_id": f"{episode['episode_id']}:t{index}",
                "episode_id": episode["episode_id"],
                "root_family_id": episode["root_family_id"],
                "split": split,
                "before_state_index": index,
                "action_index": index,
                "after_state_index": index + 1,
                "src_cluster_id": assign_cluster(before, refs),
                "dst_cluster_id": assign_cluster(after, refs),
                "source_segment_id": f"{episode['episode_id']}:state:{index}",
                "destination_segment_id": f"{episode['episode_id']}:state:{index + 1}",
                "observable_context": context,
                "observable_predicates_before": [key for key, value in context.items() if isinstance(value, bool) and value],
                "observable_predicates_after": [],
                "evaluator_event_set": all_events,
                "evaluator_semantics": semantics,
                "evaluator_label_origin": "simulator_info.events_diagnostic_only",
                "terminal_status": status,
                "terminated": terminated,
                "truncated": truncated,
                "terminal_reason": str(event.get("terminal_reason") or ""),
                "known_state_budget": True,
                "online_feature_fields": sorted(key for key in context if key != "goal_entered_before"),
                "hidden_or_future_features_used": False,
                "boundary_source_id": (prediction or {}).get("boundary_source_id", "u4b_torch_recovery_v2_locked_or_historical_cache"),
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "transition_pair": [assign_cluster(before, refs), assign_cluster(after, refs)],
                "family_scenario_for_analysis_only": episode.get("family", {}).get("scenario"),
                "terminal_failure_event": status == "failure_terminal",
                "stable_success_event": status == "success_terminal",
                "horizon_censored": status == "censored_unknown",
            }
            rows.append(row)
            history.append(event)
    write_jsonl(output, rows)
    return {"status": "PASS", "rows": len(rows), "episodes": len({r["episode_id"] for r in rows}), "split": split}


def write_occurrence_summary(rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("src_cluster_id"), row.get("dst_cluster_id"))].append(row)
    summary = []
    for (src, dst), items in sorted(grouped.items(), key=str):
        labels = sorted({label for item in items for label in item.get("evaluator_semantics", [])})
        summary.append({"src_cluster_id": src, "dst_cluster_id": dst, "occurrence_count": len(items), "family_count": len({x["root_family_id"] for x in items}), "evaluator_semantics": labels})
    write_csv(output, summary)
    return {"status": "PASS", "pair_count": len(summary)}


def build_u2_train_occurrences(dataset_root: Path, output: Path, repo: Path) -> dict[str, Any]:
    """Reconstruct train occurrences from the committed U2 NPZ contract.

    U2 stores post-transition observations, so row ``t`` is aligned to
    ``(observations[t-1], actions[t-1], observations[t])``.  Gold event ids
    are diagnostic labels only; they never enter ``observable_context``.
    """
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        return {"status": "BLOCKED", "reason": f"numpy unavailable: {exc}"}

    manifest_path = dataset_root / "episode_manifest.csv"
    episode_root = dataset_root / "episodes"
    if not manifest_path.is_file() or not episode_root.is_dir():
        return {"status": "BLOCKED", "reason": "U2 formal episode manifest or episodes directory missing"}
    refs = _cluster_refs(repo)
    rows: list[dict[str, Any]] = []
    episode_count = 0
    for meta in csv.DictReader(manifest_path.open(encoding="utf-8", newline="")):
        if meta.get("split") != "train":
            continue
        path = repo / meta["npz_path"]
        if not path.is_file():
            path = dataset_root / "episodes" / f"{meta['episode_id']}.npz"
        if not path.is_file():
            return {"status": "BLOCKED", "reason": f"U2 train episode missing: {meta['episode_id']}"}
        with np.load(path, allow_pickle=False) as data:
            observations = np.asarray(data["observations"], dtype=np.float32)
            actions = np.asarray(data["actions"], dtype=np.float32)
            event_ids = np.asarray(data["gold_event_id"], dtype=np.int64)
            boundaries = np.asarray(data["gold_boundary"], dtype=np.int64)
        if observations.ndim != 2 or observations.shape[1] != 17 or len(observations) != len(actions):
            return {"status": "BLOCKED", "reason": f"U2 NPZ alignment invalid: {meta['episode_id']}"}
        if len(event_ids) != len(observations) or len(boundaries) != len(observations):
            return {"status": "BLOCKED", "reason": f"U2 event alignment invalid: {meta['episode_id']}"}
        episode_count += 1
        history: list[dict[str, Any]] = []
        for t in range(1, len(observations)):
            event_name = EVENT_NAMES.get(int(event_ids[t]), "none")
            event_set = [] if event_name == "none" else [event_name]
            semantics = []
            if event_name in SEMANTICS:
                semantics.append(SEMANTICS[event_name])
            if event_name == "terminal_failure":
                semantics.append("terminal_failure")
            status = "success_terminal" if event_name == "stable_success" else ("failure_terminal" if event_name == "terminal_failure" else "nonterminal")
            before = observations[t - 1].tolist(); after = observations[t].tolist(); action = actions[t - 1].tolist()
            context = observable_context(before, after, action, history)
            src = assign_cluster(before, refs); dst = assign_cluster(after, refs)
            rows.append({
                "occurrence_id": f"{meta['episode_id']}:t{t}", "episode_id": meta["episode_id"],
                "root_family_id": meta["root_family_id"], "split": "train", "before_state_index": t - 1,
                "action_index": t - 1, "after_state_index": t, "src_cluster_id": src, "dst_cluster_id": dst,
                "source_segment_id": f"{meta['episode_id']}:state:{t - 1}", "destination_segment_id": f"{meta['episode_id']}:state:{t}",
                "observable_context": context,
                "observable_predicates_before": [key for key, value in context.items() if isinstance(value, bool) and value],
                "observable_predicates_after": [], "evaluator_event_set": event_set,
                "evaluator_semantics": sorted(set(semantics)), "evaluator_label_origin": "u2_npz.gold_event_id_diagnostic_only",
                "terminal_status": status, "terminated": status in {"failure_terminal", "success_terminal"}, "truncated": False,
                "terminal_reason": meta.get("terminal_reason", "") if status != "nonterminal" else "", "known_state_budget": True,
                "online_feature_fields": sorted(context), "hidden_or_future_features_used": False,
                "boundary_source_id": "u2_gold_boundary_diagnostic_only", "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "transition_pair": [src, dst], "family_scenario_for_analysis_only": meta.get("scenario_for_analysis_only"),
                "terminal_failure_event": status == "failure_terminal", "stable_success_event": status == "success_terminal",
                "horizon_censored": False, "gold_boundary_diagnostic": bool(boundaries[t]),
            })
            history.append({"all_events": event_set})
    write_jsonl(output, rows)
    return {"status": "PASS", "rows": len(rows), "episodes": episode_count, "split": "train", "source": str(manifest_path.resolve()), "label_source": "u2_npz.gold_event_id_diagnostic_only", "hidden_or_future_features_used": False}
