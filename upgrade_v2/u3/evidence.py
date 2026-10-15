"""Build compact, split-pure evidence for U3 prompts."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .common import read_csv, read_json, read_jsonl, write_json, write_jsonl


EVENT_NAMES = {
    0: "none", 1: "contact_on", 2: "transport_on", 3: "contact_off_failure", 4: "recovery_start",
    5: "contact_reestablished", 6: "detour_start", 7: "goal_enter", 8: "stable_success",
    9: "terminal_failure", 10: "stagnation_onset",
}


def build_task_contract(fallback_policy: Path) -> dict[str, Any]:
    policy = read_json(fallback_policy)
    observable_fields = [
        "agent_to_object_position_x", "agent_to_object_position_y", "agent_velocity_x", "agent_velocity_y",
        "object_to_goal_position_x", "object_to_goal_position_y", "object_velocity_x", "object_velocity_y",
        "agent_to_obstacle_position_x", "agent_to_obstacle_position_y", "object_to_obstacle_position_x", "object_to_obstacle_position_y",
        "contact_sensor", "collision_sensor", "object_in_goal_sensor", "previous_action_x", "previous_action_y",
    ]
    return {
        "schema": "u3_task_contract_v1", "scope": "stochastic_simulator_only",
        "objective": "A two-dimensional agent approaches and contacts an object, transports it to the observable goal region, and must account for collision, contact loss, recovery, detour, and stagnation branches.",
        "object_roles": {"agent": "controlled two-dimensional pusher", "object": "transported object", "goal": "fixed target region", "obstacle": "fixed obstacle with observable relative geometry"},
        "success_observable": {"predicate": "object_inside_goal", "field": "object_in_goal_sensor", "stable_variant": "stable_goal_occupancy"},
        "terminal_failure_observable": {"predicate": "collision_detected_or_stagnation_detected", "fields": ["collision_sensor", "object_to_goal_position_x", "object_to_goal_position_y", "object_velocity_x", "object_velocity_y"], "unknown_when_unresolved": True},
        "action": {"dimension": 2, "fields": ["action_x", "action_y"], "range": [-1.0, 1.0], "causal_timing": "observations[t] are x_t and previous_action fields contain a_(t-1) that enters x_t"},
        "sensor_fields": observable_fields,
        "forbidden_fields": ["gold_event", "gold_mode", "scenario", "future_outcome", "root_family_id", "episode_id"],
        "boundary_fallback": {"automatic_boundary_supported": False, "fallback_required": True, "disclosure": policy["fallback_source"]},
        "unknown_handling": "retain unknown conditions and unresolved questions; do not infer hidden state as fact",
    }


def _top_predicates(rows: list[dict[str, Any]]) -> list[str]:
    contact = sum(bool(row.get("observable_contact_history")) for row in rows)
    loss = sum(int(row.get("observable_contact_loss_count", 0) or 0) > 0 for row in rows)
    unknown = sum(float(row.get("unknown_fraction", row.get("uncertainty", 0.0)) or 0.0) > 0.25 for row in rows)
    predicates = []
    if contact: predicates.append("contact_present")
    if loss: predicates.append("contact_recently_lost")
    if unknown: predicates.append("progress_unknown")
    return predicates or ["progress_unknown"]


def _representatives(rows: list[dict[str, Any]], max_representatives: int) -> list[str]:
    ordered = sorted(rows, key=lambda row: (float(row.get("unknown_fraction", row.get("uncertainty", 0.0)) or 0.0), row["segment_id"]))
    if not ordered:
        return []
    choices = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    seen = {str(row.get("root_family_id")) for row in choices}
    for row in ordered:
        if len(choices) >= max_representatives:
            break
        if str(row.get("root_family_id")) not in seen:
            choices.append(row); seen.add(str(row.get("root_family_id")))
    result: list[str] = []
    for row in choices:
        if row["segment_id"] not in result:
            result.append(row["segment_id"])
    return result[:max_representatives]


def _event_posterior(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    arrays = [row.get("event_posterior", row.get("event_posterior_mean", [])) for row in rows]
    arrays = [np.asarray(value, dtype=float) for value in arrays if value]
    if not arrays:
        return []
    mean = np.stack(arrays).mean(axis=0)
    return [{"event_id": int(index), "event": EVENT_NAMES.get(int(index), "unknown"), "posterior": float(value)} for index, value in sorted(enumerate(mean), key=lambda item: (-item[1], item[0]))[:3]]


def build_compact_evidence(train_segments: Path, train_prototypes: Path, train_transitions: Path, fallback_clips: Path, dataset_root: Path, output_dir: Path, max_representatives: int = 4, max_pairs: int = 60) -> dict[str, Any]:
    segments = read_jsonl(train_segments)
    if not segments or any(row.get("split") != "train" for row in segments):
        raise ValueError("compact evidence requires non-empty train-only segments")
    prototypes = read_json(train_prototypes)
    transitions = read_csv(train_transitions)
    if any(row.get("split") != "train" for row in transitions):
        raise ValueError("transition input is not train-only")
    by_cluster: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in segments:
        by_cluster[int(row.get("cluster_id", 0))].append(row)
    transition_groups: dict[tuple[int, int], dict[str, Any]] = {}
    for row in transitions:
        key = (int(row["from_cluster_id"]), int(row["to_cluster_id"]))
        item = transition_groups.setdefault(key, {"from_cluster_id": key[0], "to_cluster_id": key[1], "observation_count": 0, "root_families": set(), "example_from_segment_ids": [], "example_to_segment_ids": []})
        item["observation_count"] += 1; item["root_families"].add(row["root_family_id"])
        if len(item["example_from_segment_ids"]) < 3: item["example_from_segment_ids"].append(row["from_segment_id"]); item["example_to_segment_ids"].append(row["to_segment_id"])
    pairs = []
    for item in transition_groups.values():
        pairs.append({"from_cluster_id": item["from_cluster_id"], "to_cluster_id": item["to_cluster_id"], "observation_count": item["observation_count"], "support_root_families": len(item["root_families"]), "example_from_segment_ids": item["example_from_segment_ids"], "example_to_segment_ids": item["example_to_segment_ids"]})
    pairs.sort(key=lambda row: (-row["support_root_families"], -row["observation_count"], row["from_cluster_id"], row["to_cluster_id"]))
    outgoing: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        outgoing[row["from_cluster_id"]].append({key: row[key] for key in ("to_cluster_id", "observation_count", "support_root_families")})
    clusters = []
    for cluster_id, values in sorted(by_cluster.items()):
        proto = prototypes.get(str(cluster_id), {})
        clusters.append({"cluster_id": cluster_id, "n_segments": len(values), "support_root_families": len({row["root_family_id"] for row in values}), "mean_duration": proto.get("mean_duration"), "mean_unknown_fraction": proto.get("mean_unknown_fraction"), "top_observable_predicates": _top_predicates(values), "top_event_posterior": _event_posterior(values), "top_outgoing_transitions": outgoing.get(cluster_id, [])[:8], "representative_segment_ids": _representatives(values, max_representatives)})
    clip_rows = [row for row in read_csv(fallback_clips) if row.get("split") == "train"]
    if len(clip_rows) != 30:
        raise ValueError(f"expected 30 train fallback clips, found {len(clip_rows)}")
    manifest = {row["episode_id"]: row for row in read_csv(dataset_root / "episode_manifest.csv")}
    compact_fallback: list[dict[str, Any]] = []
    for clip in clip_rows:
        if clip["episode_id"] not in manifest or manifest[clip["episode_id"]]["split"] != "train":
            raise ValueError(f"fallback episode is not train: {clip['episode_id']}")
        episode_path = dataset_root / "episodes" / f"{clip['episode_id']}.npz"
        with np.load(episode_path) as payload:
            obs = np.asarray(payload["observations"], dtype=float)
        start, end = int(clip["start_t"]), int(clip["end_t"])
        compact_fallback.append({"clip_id": clip["clip_id"], "episode_id": clip["episode_id"], "root_family_id": clip["root_family_id"], "start_t": start, "end_t": end, "observable_before": obs[max(start - 1, 0)].round(6).tolist(), "confirmed_event": EVENT_NAMES[int(clip["event_id"])], "observable_after": obs[min(end, len(obs) - 1)].round(6).tolist(), "confirmation_source": "simulator_gold_train_clip", "status": "confirmed_fragment_not_validated_graph"})
    registry = {"schema": "u3_evidence_registry_v1", "input_split": "train", "segment_ids": sorted(row["segment_id"] for row in segments), "cluster_ids": sorted(by_cluster), "transition_pairs": [{"from_cluster_id": row["from_cluster_id"], "to_cluster_id": row["to_cluster_id"]} for row in pairs], "fallback_clip_ids": sorted(row["clip_id"] for row in compact_fallback), "fallback_episode_ids": sorted(row["episode_id"] for row in compact_fallback)}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "cluster_evidence_compact.json", {"schema": "u3_cluster_evidence_compact_v1", "input_split": "train", "clusters": clusters})
    write_json(output_dir / "transition_evidence_compact.json", {"schema": "u3_transition_evidence_compact_v1", "input_split": "train", "transitions": pairs[:max_pairs], "all_transition_pair_count": len(pairs)})
    write_jsonl(output_dir / "fallback_confirmations_compact.jsonl", compact_fallback)
    write_json(output_dir / "evidence_id_registry.json", registry)
    return {"status": "PASS", "cluster_count": len(clusters), "transition_pair_count": len(pairs), "fallback_confirmation_count": len(compact_fallback), "train_segment_count": len(segments)}


def supervision_ledger(fallback_policy: Path) -> dict[str, Any]:
    policy = read_json(fallback_policy)
    calibration = policy["shared_calibration_supervision"]; fallback = policy["additional_fallback"]
    return {"instruction_only": {"shared_calibration_families": 0, "shared_calibration_frames": 0, "additional_fallback_clips": 0, "additional_fallback_frames": 0}, "instruction_plus_auto_train_segments": {"shared_calibration_families": calibration["family_count"], "shared_calibration_frames": calibration["frame_count"], "additional_fallback_clips": 0, "additional_fallback_frames": 0}, "instruction_plus_budgeted_train_fallback": {"shared_calibration_families": calibration["family_count"], "shared_calibration_frames": calibration["frame_count"], "additional_fallback_clips": fallback["clip_count"], "additional_fallback_frames": fallback["frame_count"]}}
