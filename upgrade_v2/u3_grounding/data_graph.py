"""Construct an observed train-only transition graph, preserving rare edges."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import read_json, write_csv, write_json


def build_data_only_graph(*, cluster_profiles: Path, transition_catalog: Path, minimum_transition_families: int, minimum_cluster_families: int, merge_predicate_jaccard: float, merge_transition_jaccard: float, output: Path, edit_log: Path, report: Path) -> dict[str, Any]:
    profiles = read_json(cluster_profiles)["profiles"]; transitions = read_json(transition_catalog)["transitions"]
    by_id = {int(p["raw_cluster_id"]): p for p in profiles}
    start_id = max(by_id, key=lambda cid: ("contact_on" in {e["event"] for e in by_id[cid].get("top_event_posterior", [])[:3]}, -cid)) if by_id else 0
    success_id = max(by_id, key=lambda cid: (max((e["posterior"] for e in by_id[cid].get("top_event_posterior", []) if e["event"] in {"goal_enter", "stable_success"}), default=0.0), -cid)) if by_id else 0
    nodes = []
    for p in sorted(profiles, key=lambda x: int(x["raw_cluster_id"])):
        cid = int(p["raw_cluster_id"]); role = "intermediate"
        if cid == start_id: role = "start"
        if cid == success_id and success_id != start_id: role = "success_terminal"
        event_names = {e["event"] for e in p.get("top_event_posterior", [])[:3]}
        if {"terminal_failure", "stagnation_onset", "contact_off_failure"} & event_names: role = "failure_terminal" if cid != start_id and cid != success_id else role
        nodes.append({"id": p["cluster_handle"], "raw_cluster_id": cid, "role": role, "status": "observed" if int(p.get("support_root_families", 0)) >= minimum_cluster_families else "unresolved", "observable_predicates": p.get("top_predicates", []), "n_segments": int(p.get("n_segments", 0)), "support_root_families": p.get("support_root_families", 0), "unknown_fraction": p.get("unknown_fraction", 1)})
    edges = []; edits = []
    for t in transitions:
        pair = (int(t["from_cluster_id"]), int(t["to_cluster_id"])); rare = int(t.get("support_root_families", 0)) < minimum_transition_families
        edges.append({"id": t["handle"], "src": t["from_cluster_handle"], "dst": t["to_cluster_handle"], "raw_pair": list(pair), "transition_count": t.get("transition_count", 0), "support_root_families": t.get("support_root_families", 0), "status": "unresolved" if rare else "observed", "unresolved_reason": "unresolved_rare_edge" if rare else ""})
        if rare: edits.append({"action": "retain_rare_edge", "handle": t["handle"], "reason": "support below threshold; not deleted"})
    graph = {"schema": "u3_data_only_transition_graph_v1", "graph_id": "data_only_transition_graph", "source_candidate_id": "data_only", "scope": "same stochastic simulator family only", "input_split": "train", "nodes": nodes, "edges": edges, "start_cluster_handle": next((n["id"] for n in nodes if n["role"] == "start"), ""), "success_cluster_handle": next((n["id"] for n in nodes if n["role"] == "success_terminal"), ""), "all_statuses": ["observed", "unresolved"], "physical_generalization_eligible": False}
    write_json(output, graph); write_csv(edit_log, edits); report.parent.mkdir(parents=True, exist_ok=True); report.write_text("# Data-only transition graph\n\n" + "\n".join([f"- nodes: `{len(nodes)}`", f"- transitions: `{len(edges)}`", f"- start: `{graph['start_cluster_handle']}`", f"- success: `{graph['success_cluster_handle']}`", f"- rare edges retained: `{len(edits)}`", "- status policy: `observed|unresolved`; no empirically_validated labels"]) + "\n", encoding="utf-8")
    return {"status": "PASS", "node_count": len(nodes), "edge_count": len(edges), "start": graph["start_cluster_handle"], "success": graph["success_cluster_handle"]}
