"""Map verbose train evidence to a finite deterministic handle catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import family_from_id, read_json, read_jsonl, write_csv, write_json


def build_evidence_handles(*, clusters: Path, transitions: Path, fallback: Path, registry: Path, output_dir: Path, manifest: Path) -> dict[str, Any]:
    cluster_rows = sorted(read_json(clusters)["clusters"], key=lambda row: int(row["cluster_id"]))
    transition_rows = sorted(read_json(transitions)["transitions"], key=lambda row: (-int(row.get("support_root_families", 0)), -int(row.get("observation_count", 0)), int(row["from_cluster_id"]), int(row["to_cluster_id"])))
    fallback_rows = sorted(read_jsonl(fallback), key=lambda row: str(row.get("clip_id", "")))
    registry_value = read_json(registry)
    if registry_value.get("input_split") != "train":
        raise ValueError("evidence registry is not train-only")
    registry_pairs = {(int(x["from_cluster_id"]), int(x["to_cluster_id"])) for x in registry_value.get("transition_pairs", [])}
    source_pairs = {(int(x["from_cluster_id"]), int(x["to_cluster_id"])) for x in transition_rows}
    if registry_pairs != source_pairs:
        raise ValueError("transition registry does not match compact evidence")
    registry_fallback = set(registry_value.get("fallback_clip_ids", []))
    source_fallback = {str(x.get("clip_id")) for x in fallback_rows}
    if registry_fallback != source_fallback:
        raise ValueError("fallback registry does not match compact evidence")
    cluster_handles = []
    raw: dict[str, Any] = {"clusters": {}, "transitions": {}, "fallback": {}}
    cmap: dict[int, str] = {}
    for i, row in enumerate(cluster_rows):
        handle = f"C{i:02d}"; cid = int(row["cluster_id"]); cmap[cid] = handle
        item = {"handle": handle, "raw_cluster_id": cid, "support_root_families": row.get("support_root_families", 0), "n_segments": row.get("n_segments", 0), "mean_duration": row.get("mean_duration", 0), "mean_unknown_fraction": row.get("mean_unknown_fraction", 1), "top_observable_predicates": row.get("top_observable_predicates", []), "top_event_posterior": row.get("top_event_posterior", []), "top_outgoing_transitions": row.get("top_outgoing_transitions", [])}
        cluster_handles.append(item); raw["clusters"][handle] = row
    transition_handles = []
    for i, row in enumerate(transition_rows):
        handle = f"T{i:03d}"
        item = {"handle": handle, "from_cluster_handle": cmap[int(row["from_cluster_id"])], "to_cluster_handle": cmap[int(row["to_cluster_id"])], "from_cluster_id": int(row["from_cluster_id"]), "to_cluster_id": int(row["to_cluster_id"]), "transition_count": row.get("observation_count", 0), "support_root_families": row.get("support_root_families", 0), "source_transition_pair": {"from_cluster_id": int(row["from_cluster_id"]), "to_cluster_id": int(row["to_cluster_id"])}, "top_event_posterior_before": row.get("top_event_posterior_before", []), "top_event_posterior_after": row.get("top_event_posterior_after", []), "example_from_segment_ids": row.get("example_from_segment_ids", []), "example_to_segment_ids": row.get("example_to_segment_ids", [])}
        transition_handles.append(item); raw["transitions"][handle] = row
    fallback_handles = []
    for i, row in enumerate(fallback_rows):
        handle = f"F{i:03d}"; item = {"handle": handle, "clip_id": row.get("clip_id"), "episode_id": row.get("episode_id"), "root_family_id": row.get("root_family_id") or family_from_id(str(row.get("episode_id", ""))), "start_t": row.get("start_t"), "end_t": row.get("end_t"), "confirmed_event": row.get("confirmed_event"), "status": row.get("status", "confirmed_fragment_not_validated_graph")}
        fallback_handles.append(item); raw["fallback"][handle] = row
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "cluster_handles.json", {"schema": "u3_cluster_handle_catalog_v1", "input_split": "train", "clusters": cluster_handles})
    write_json(output_dir / "transition_handles.json", {"schema": "u3_transition_handle_catalog_v1", "input_split": "train", "transitions": transition_handles})
    write_json(output_dir / "fallback_handles.json", {"schema": "u3_fallback_handle_catalog_v1", "input_split": "train", "fallback": fallback_handles})
    write_json(output_dir / "handle_to_raw_evidence.json", {"schema": "u3_handle_raw_lookup_v1", **raw})
    write_csv(manifest, [{"handle": x["handle"], "kind": "cluster", "raw_id": x["raw_cluster_id"], "split": "train", "support_root_families": x["support_root_families"]} for x in cluster_handles] + [{"handle": x["handle"], "kind": "transition", "raw_id": f"{x['from_cluster_id']}->{x['to_cluster_id']}", "split": "train", "support_root_families": x["support_root_families"]} for x in transition_handles] + [{"handle": x["handle"], "kind": "fallback", "raw_id": x["clip_id"], "split": "train", "support_root_families": 1} for x in fallback_handles])
    return {"status": "EVIDENCE_CATALOG_AND_DATA_GRAPH_READY", "cluster_count": len(cluster_handles), "transition_count": len(transition_handles), "fallback_count": len(fallback_handles)}
