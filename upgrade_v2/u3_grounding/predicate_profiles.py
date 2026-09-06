"""Build train-only cluster profiles used by deterministic grounding."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import read_json, read_jsonl, safe_float, write_csv, write_json

EVENT_NAMES = [
    "none", "contact_on", "transport_on", "contact_off_failure",
    "recovery_start", "contact_reestablished", "detour_start", "goal_enter",
    "stable_success", "terminal_failure", "stagnation_onset",
]


def _predicates(row: dict[str, Any]) -> set[str]:
    top = set(row.get("top_observable_predicates", []))
    mean = row.get("raw_mean", [])
    if len(mean) >= 17:
        if safe_float(mean[12]) >= 0.5: top.add("contact_present")
        else: top.add("contact_absent")
        if safe_float(mean[14]) >= 0.5: top.add("object_inside_goal")
        if safe_float(mean[13]) >= 0.5: top.add("collision_detected")
        if abs(safe_float(mean[6])) + abs(safe_float(mean[7])) < 0.06: top.add("object_stationary")
        else: top.add("object_moving")
    return top


def build_cluster_profiles(*, cluster_catalog: Path, predicate_vocabulary: Path, train_segments: Path, output: Path, table: Path, report: Path) -> dict[str, Any]:
    catalog = read_json(cluster_catalog)["clusters"]
    vocabulary = read_json(predicate_vocabulary)
    allowed_predicates = {item["name"] for item in vocabulary.get("allowed_predicates", [])}
    by_cluster: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(train_segments):
        if row.get("split", "train") == "train": by_cluster[int(row["cluster_id"])].append(row)
    profiles = []
    rows = []
    for c in catalog:
        cid = int(c["raw_cluster_id"]); segments = by_cluster[cid]; counts: dict[str, int] = defaultdict(int); events = [0.0] * len(EVENT_NAMES)
        for s in segments:
            for pred in _predicates(s): counts[pred] += 1
            posterior = s.get("event_posterior_mean", [])
            for i, value in enumerate(posterior[:len(events)]): events[i] += safe_float(value)
        denom = max(1, len(segments)); pred_probs = {key: round(value / denom, 8) for key, value in sorted(counts.items())}
        top_events = [{"event": EVENT_NAMES[i], "posterior": round(value / denom, 8)} for i, value in sorted(enumerate(events), key=lambda x: -x[1])[:5]]
        unknown_predicates = set(pred_probs) - allowed_predicates
        if unknown_predicates:
            raise ValueError(f"predicate vocabulary mismatch: {sorted(unknown_predicates)}")
        profile = {"cluster_handle": c["handle"], "raw_cluster_id": cid, "support_root_families": c.get("support_root_families", 0), "mean_duration": c.get("mean_duration", 0), "unknown_fraction": c.get("mean_unknown_fraction", 1), "predicate_probabilities": pred_probs, "top_predicates": [key for key, _ in sorted(pred_probs.items(), key=lambda x: -x[1])[:8]], "top_event_posterior": top_events, "incoming_transition_handles": [], "outgoing_transition_handles": []}
        profiles.append(profile); rows.append({"cluster_handle": c["handle"], "raw_cluster_id": cid, "support_root_families": profile["support_root_families"], "top_predicates": ";".join(profile["top_predicates"]), "top_event": top_events[0]["event"] if top_events else "none", "mean_duration": profile["mean_duration"], "unknown_fraction": profile["unknown_fraction"]})
    transition_path = cluster_catalog.parent / "transition_handles.json"
    if transition_path.is_file():
        transitions = read_json(transition_path)["transitions"]
        by_id = {p["raw_cluster_id"]: p for p in profiles}
        for transition in transitions:
            src = by_id[int(transition["from_cluster_id"])]
            dst = by_id[int(transition["to_cluster_id"])]
            src["outgoing_transition_handles"].append(transition["handle"])
            dst["incoming_transition_handles"].append(transition["handle"])
        for profile in profiles:
            profile["incoming_transition_handles"].sort()
            profile["outgoing_transition_handles"].sort()
    write_json(output, {"schema": "u3_cluster_predicate_profiles_v1", "input_split": "train", "profiles": profiles})
    write_csv(table, rows)
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text("# Cluster predicate profiles\n\n- split: `train`\n- cluster_count: `%d`\n- test/val gold used: `false`\n" % len(profiles), encoding="utf-8")
    return {"status": "PASS", "cluster_count": len(profiles), "train_only": True}
