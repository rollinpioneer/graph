"""Evidence-bounded graph edits and development-only graph selection."""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from .evaluate import selection_score
from .io import bool_value, now_iso, read_csv, read_json, sha256_file, write_csv, write_json, write_report


EDIT_DEFINITIONS = {
    "unnecessary_manipulation": ("E2", "goal_verified_stop", "Add a no-action stop branch when the task is already stably satisfied."),
    "missed_grasp_unhandled": ("E4", "missed_grasp_retry", "Add retry_grasp after an observable closed-without-contact event."),
    "contact_loss_unhandled": ("E5", "contact_loss_recovery", "Add recover_object after observable recent contact loss."),
    "target_blocked_unhandled": ("E6", "target_blocked_branch", "Add clear_target or clarification before placement when occupancy is observed."),
    "visual_ambiguity_unhandled": ("E7", "visual_unknown_clarification", "Retain insufficient visual evidence as unknown and request clarification."),
    "occlusion_unhandled": ("E1", "visual_unknown_clarification", "Add an observation/clarification node for persistent front-view unknown."),
}


EDIT_NODES = {
    "E1": {"id": "visual_unknown", "state": "visual_unknown", "status": "development_evidence"},
    "E4": {"id": "grasp_failed", "state": "grasp_failed", "status": "development_evidence"},
    "E5": {"id": "recovery_required", "state": "recovery_required", "status": "development_evidence"},
    "E6": {"id": "target_blocked", "state": "target_blocked", "status": "development_evidence"},
}


VISUAL_UNKNOWN_NODE = EDIT_NODES["E1"]
PERSISTENT_UNKNOWN = {"op": "CONSECUTIVE", "n": 2, "arg": "visual_unknown"}


EDIT_EDGES = {
    "E4": {"id": "retry_after_missed_grasp", "src": "grasp_failed", "dst": "grasp_candidate", "action": "retry_grasp", "condition": "grasp_failed_observed"},
    "E5": {"id": "recover_after_contact_loss", "src": "recovery_required", "dst": "grasp_candidate", "action": "recover_object", "condition": "slip_observed"},
    "E6": {"id": "clear_blocked_target", "src": "target_blocked", "dst": "scene_unverified", "action": "clear_target", "condition": "target_occupied"},
    "E7": {"id": "clarify_persistent_unknown", "src": "visual_unknown", "dst": "scene_unverified", "action": "request_clarification", "condition": PERSISTENT_UNKNOWN},
}


def propose_refinements(base_graph: Path, errors_path: Path, allowed_edits: set[str], max_edits: int, minimum_families: int, output: Path, edit_log: Path, report: Path) -> dict[str, Any]:
    graph = read_json(base_graph)
    errors = [row for row in read_csv(errors_path) if row["graph_id"] == graph["graph_id"]]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in errors:
        grouped[row["error_type"]].append(row)
    proposals = []
    capabilities = list(graph.get("capabilities", []))
    for error_type, rows in sorted(grouped.items(), key=lambda item: (-len({row["root_family_id"] for row in item[1]}), item[0])):
        if error_type not in EDIT_DEFINITIONS:
            continue
        edit_id, capability, description = EDIT_DEFINITIONS[error_type]
        families = sorted({row["root_family_id"] for row in rows})
        accepted = edit_id in allowed_edits and len(families) >= minimum_families and len([row for row in proposals if row["accepted"]]) < max_edits
        proposals.append({
            "edit_id": edit_id, "edit_type": edit_id, "capability": capability, "description": description,
            "dev_fit_error_count": len(rows), "independent_family_count": len(families), "family_ids": ";".join(families),
            "coarse_elements": "scene_unverified;grasp_candidate;goal_stability_candidate",
            "expected_metric": "branch_accuracy;failure_recall;recovery_recall;coverage",
            "known_risk": "Rule may over-trigger under visual uncertainty; confirmation remains frozen and one-shot.",
            "accepted": accepted, "acceptance_reason": "Meets frozen minimum-family support and edit budget." if accepted else "Insufficient independent-family support or duplicate capability.",
            "selection_split": "dev_fit",
        })
        if accepted and capability not in capabilities:
            capabilities.append(capability)
    accepted_edits = [row["edit_id"] for row in proposals if row["accepted"]]
    nodes = list(graph.get("nodes", []))
    edges = [dict(edge) for edge in graph.get("edges", [])]
    for edit_id in accepted_edits:
        node = EDIT_NODES.get(edit_id)
        if node and not any(existing["id"] == node["id"] for existing in nodes):
            nodes.append(node)
        edge = EDIT_EDGES.get(edit_id)
        if edge and not any(existing["id"] == edge["id"] for existing in edges):
            edges.append(edge)
    if {"E1", "E7"}.intersection(accepted_edits):
        if not any(existing["id"] == VISUAL_UNKNOWN_NODE["id"] for existing in nodes):
            nodes.append(VISUAL_UNKNOWN_NODE)
        for edge in edges:
            if edge["id"] == "observe_unknown":
                edge["dst"] = "visual_unknown"
                edge["condition"] = {
                    "op": "AND",
                    "args": ["visual_unknown", {"op": "NOT", "arg": PERSISTENT_UNKNOWN}],
                }
    if "E7" in accepted_edits:
        for edge in edges:
            if edge["id"] == "manipulate":
                edge["condition"] = {"op": "AND", "args": [edge["condition"], {"op": "NOT", "arg": "visual_unknown"}]}
    refined = {
        **graph,
        "graph_id": "G2_evidence_refined",
        "status": "development_candidate",
        "base_graph_sha256": sha256_file(base_graph),
        "capabilities": capabilities,
        "nodes": nodes,
        "edges": edges,
        "accepted_edits": accepted_edits,
        "edit_selection_split": "dev_fit",
        "max_edits": max_edits,
    }
    write_json(output, refined); write_csv(edit_log, proposals)
    write_report(report, "G2 Evidence Refinement", [("status", "PASS"), ("proposals", len(proposals)), ("accepted edits", len(refined["accepted_edits"])), ("selection split", "dev_fit")])
    return {"status": "PASS", "proposals": len(proposals), "accepted_edits": len(refined["accepted_edits"]), "output": str(output)}


def build_g3(g2_path: Path, output: Path, query_threshold: float = 1.0) -> dict[str, Any]:
    graph = read_json(g2_path)
    graph.update(graph_id="G3_active_second_view", status="development_candidate", active_second_view=True, active_view_trigger="visual_unknown == true", query_threshold=query_threshold, topology_same_as_g2=True)
    write_json(output, graph)
    return graph


def select_refined_graph(metrics_path: Path, per_family_path: Path, edit_log: Path, graph_paths: dict[str, Path], predicate_thresholds: Path, camera_config: Path, family_generation_lock: Path, active_view_query_max: float, output: Path, lock: Path, report: Path) -> dict[str, Any]:
    metrics = {row["graph_id"]: row for row in read_csv(metrics_path)}
    for row in metrics.values():
        for key, value in list(row.items()):
            if key not in {"graph_id"} and value not in {"", None}:
                try: row[key] = float(value)
                except ValueError: pass
    def qualifies(row: dict[str, Any]) -> bool:
        return row.get("branch_accuracy", 0) >= .80 and row.get("goal_precision", 0) >= .85 and row.get("failure_recall", 0) >= .65 and row.get("recovery_recall", 0) >= .65 and row.get("graph_completion_coverage", 0) >= .80 and row.get("false_ready_rate", 1) <= .15
    if "G2_evidence_refined" in metrics and qualifies(metrics["G2_evidence_refined"]):
        selected = "G2_evidence_refined"
    elif "G3_active_second_view" in metrics and qualifies(metrics["G3_active_second_view"]) and metrics["G3_active_second_view"].get("second_view_query_rate", 1) <= active_view_query_max:
        selected = "G3_active_second_view"
    else:
        selected = "G1_predicate_bound"
    shutil.copy2(graph_paths[selected], output)
    accepted = sum(row["accepted"] == "True" for row in read_csv(edit_log))
    lock_payload = {
        "schema": "pathgraph_l2r_graph_selection_lock_v1", "status": "LOCKED_FOR_FRESH_CONFIRMATION", "created_at": now_iso(),
        "selected_graph_id": selected, "selected_graph_path": str(output.resolve()), "selected_graph_sha256": sha256_file(output),
        "source_graph_path": str(graph_paths[selected].resolve()), "source_graph_sha256": sha256_file(graph_paths[selected]),
        "predicate_thresholds_path": str(predicate_thresholds.resolve()), "predicate_thresholds_sha256": sha256_file(predicate_thresholds),
        "camera_config_path": str(camera_config.resolve()), "camera_config_sha256": sha256_file(camera_config),
        "family_generation_lock_path": str(family_generation_lock.resolve()), "family_generation_lock_sha256": sha256_file(family_generation_lock),
        "active_view_query_max": active_view_query_max, "evaluation_metrics": list(metrics[selected]),
        "selection_split": "dev_select", "fresh_confirmation_used": False, "accepted_edits": accepted,
        "graph_source_path_by_id": {graph_id: str(path.resolve()) for graph_id, path in sorted(graph_paths.items())},
        "graph_sha256_by_id": {graph_id: sha256_file(path) for graph_id, path in sorted(graph_paths.items())},
    }
    write_json(lock, lock_payload)
    write_report(report, "Refined Graph Selection", [("status", lock_payload["status"]), ("selected graph", selected), ("accepted edits", accepted), ("selection split", "dev_select"), ("fresh confirmation used", False)])
    return lock_payload
