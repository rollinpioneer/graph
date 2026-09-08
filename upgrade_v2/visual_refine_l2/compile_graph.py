"""Compile the canonical L1V skeleton into direct and predicate-bound graphs."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .io import read_json, sha256_file, write_csv, write_json


def compile_graphs(canonical_graph: Path, thresholds: Path, predicate_schema: Path, output_g0: Path, output_g1: Path, binding_table: Path, unbound: Path) -> dict[str, Any]:
    canonical = read_json(canonical_graph)
    schema = read_json(predicate_schema)
    known = set(schema["predicates"])
    g0 = {
        "schema": "pathgraph_l2r_executable_graph_v1", "graph_id": "G0_coarse_direct", "status": "frozen_baseline",
        "source_sha256": sha256_file(canonical_graph), "topology": canonical,
        "capabilities": ["fixed_manipulation_path"], "online_predicates": [], "active_second_view": False,
        "selection_data": "none", "numeric_reward_or_cost": None,
    }
    bindings = [
        {"coarse_state": "scene_unverified", "predicate": "visual_unknown", "binding": "observation_or_unknown"},
        {"coarse_state": "task_already_satisfied_candidate", "predicate": "goal_verified", "binding": "temporal_visual_plus_open_gripper"},
        {"coarse_state": "stable_hold_candidate", "predicate": "stable_hold_observed", "binding": "temporal_visual_plus_contact_action"},
        {"coarse_state": "object_in_transport", "predicate": "transport_observed", "binding": "temporal_visual_plus_contact_action"},
        {"coarse_state": "object_above_target", "predicate": "object_above_target", "binding": "single_view_geometry_plus_contact"},
        {"coarse_state": "object_on_target_candidate", "predicate": "placement_candidate", "binding": "single_view_geometry_plus_contact"},
        {"coarse_state": "object_released", "predicate": "gripper_command_open", "binding": "allowed_command_sensor"},
        {"coarse_state": "goal_stability_candidate", "predicate": "goal_verified", "binding": "temporal_visual_plus_open_gripper"},
    ]
    missing = sorted({row["predicate"] for row in bindings} - known)
    if missing:
        raise ValueError(f"predicate schema missing bindings: {missing}")
    g1 = {
        "schema": "pathgraph_l2r_executable_graph_v1", "graph_id": "G1_predicate_bound", "status": "development_frozen",
        "source_sha256": sha256_file(canonical_graph), "predicate_thresholds_path": str(thresholds.resolve()),
        "predicate_thresholds_sha256": sha256_file(thresholds), "capabilities": ["fixed_manipulation_path", "goal_verified_stop", "visual_unknown_observe"],
        "online_predicates": sorted({row["predicate"] for row in bindings}), "active_second_view": False,
        "nodes": canonical["nodes"],
        "edges": [
            {"id": "already_done", "action": "stop_no_action", "condition": "goal_verified"},
            {"id": "observe_unknown", "action": "observe_scene", "condition": "visual_unknown"},
            {"id": "manipulate", "action": "grasp_object", "condition": {"op": "AND", "args": ["object_visible", "target_visible", {"op": "NOT", "arg": "goal_verified"}]}},
        ],
        "selection_data": "predicate bindings only; no dynamic structural edits", "numeric_reward_or_cost": None,
    }
    write_json(output_g0, g0); write_json(output_g1, g1)
    write_csv(binding_table, bindings)
    write_csv(unbound, [{"coarse_condition": "physical stability or reachability language", "resolution": "UNKNOWN", "reason": "not directly observable under the frozen online interface"}])
    return {"status": "PASS", "g0": str(output_g0), "g1": str(output_g1), "bindings": len(bindings), "unbound": 1}
