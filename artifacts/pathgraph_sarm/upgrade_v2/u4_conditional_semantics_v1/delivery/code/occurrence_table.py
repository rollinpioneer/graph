"""Episode-local occurrence extraction with explicit terminal classification."""
from __future__ import annotations

import hashlib
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


def observable_context(before: list[float], action: list[float], history: list[dict[str, Any]]) -> dict[str, Any]:
    context = {
        "contact_present": bool(len(before) > 12 and float(before[12]) >= 0.5),
        "collision_detected": bool(len(before) > 13 and float(before[13]) >= 0.5),
        "object_inside_goal": bool(len(before) > 14 and float(before[14]) >= 0.5),
        "object_moving": bool(len(before) > 7 and (float(before[6]) ** 2 + float(before[7]) ** 2) ** 0.5 >= 0.06),
        "action_norm": (float(action[0]) ** 2 + float(action[1]) ** 2) ** 0.5 if len(action) >= 2 else 0.0,
    }
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
            context = observable_context(before, actions[index], history)
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
                "evaluator_label_origin": "simulator_info.events_and_explicit_terminal_reason",
                "terminal_status": status,
                "terminated": terminated,
                "truncated": truncated,
                "terminal_reason": str(event.get("terminal_reason") or ""),
                "known_state_budget": True,
                "online_feature_fields": sorted(context),
                "hidden_or_future_features_used": False,
                "boundary_source_id": (prediction or {}).get("boundary_source_id", "u4b_torch_recovery_v2_locked_or_historical_cache"),
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "transition_pair": [assign_cluster(before, refs), assign_cluster(after, refs)],
                "family_scenario_for_analysis_only": episode.get("family", {}).get("scenario"),
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
