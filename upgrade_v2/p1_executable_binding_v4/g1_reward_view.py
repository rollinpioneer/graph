"""Executable G1 reward view. Human bindings are not original LLM edges."""
from __future__ import annotations
from typing import Any

G1_ORDER = [
    "scene_unverified", "grasp_candidate", "stable_hold_candidate", "object_in_transport",
    "object_above_target", "object_on_target_candidate", "object_released",
    "goal_stability_candidate", "goal_verified",
]
HUMAN_EDGES = [
    ("scene_unverified", "grasp_candidate", "manipulate_or_approach", "ORIGINAL_OR_HUMAN"),
    ("grasp_candidate", "stable_hold_candidate", "hold_established", "HUMAN_BINDING"),
    ("stable_hold_candidate", "object_in_transport", "transport_started", "HUMAN_BINDING"),
    ("object_in_transport", "object_above_target", "above_target", "HUMAN_BINDING"),
    ("object_above_target", "object_on_target_candidate", "on_target", "HUMAN_BINDING"),
    ("object_on_target_candidate", "object_released", "released", "HUMAN_BINDING"),
    ("object_released", "goal_stability_candidate", "stability_wait", "HUMAN_BINDING"),
    ("goal_stability_candidate", "goal_verified", "goal_verified", "HUMAN_BINDING"),
    ("scene_unverified", "goal_verified", "already_done", "ORIGINAL_G1_EDGE"),
]


def g1_node(facts: dict[str, Any], action: str, distance: float | None) -> str:
    hold = facts.get("hold_current") is True
    goal = facts.get("goal_verified") is True
    released = facts.get("release_intent") is True and not hold
    if goal and action in ("verify", "stop") and not hold:
        if action == "stop" or facts.get("hold_established_earlier") is False:
            return "goal_verified"
        return "goal_verified"
    if goal and released:
        return "goal_stability_candidate"
    if released:
        return "object_released"
    if hold and distance is not None and distance < 0.05:
        return "object_on_target_candidate"
    if hold and distance is not None and distance < 0.25:
        return "object_above_target"
    if hold and action in ("lift", "transport_to_target", "align", "lower"):
        return "object_in_transport"
    if hold:
        return "stable_hold_candidate"
    if action in ("approach_object", "close_gripper"):
        return "grasp_candidate"
    return "scene_unverified"


def reward_view_graph() -> dict[str, Any]:
    nodes = [{"id": n} for n in G1_ORDER]
    edges = []
    for i, (s, d, name, origin) in enumerate(HUMAN_EDGES):
        edges.append(dict(id=name, src=s, dst=d, type="forward", base_step_cost=1,
                          origin=origin, structural_extension=(origin == "HUMAN_BINDING")))
    return dict(graph_id="G1_RW_NORMAL_PATH_V4", nodes=nodes, edges=edges,
                success_nodes=["goal_verified"], terminal_failure_nodes=[],
                original_edges=["already_done", "observe_unknown", "manipulate"],
                controller_driven_by_graph=False,
                missing_failure_recovery=["STRUCTURAL_EXTENSION_REQUIRED"],
                extension_name_if_added="G1_RW_EXTENDED_REFERENCE_V4")