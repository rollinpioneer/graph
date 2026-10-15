"""Frozen high-level action programs used only to collect the benchmark."""

from __future__ import annotations


VERIFY_WINDOW = ["verify"] * 5
BASE = ["observe_scene", "approach_object", "close_gripper", "lift", "transport_to_target", "align", "lower", "open_gripper", *VERIFY_WINDOW]


def action_program(scenario: str, rollout_index: int) -> list[str]:
    if scenario == "already_satisfied_stable":
        return ["observe_scene", *VERIFY_WINDOW, "stop"]
    if scenario == "already_on_target_offcenter":
        return BASE
    if scenario == "missed_grasp_then_retry":
        return ["observe_scene", "approach_object", "close_gripper", "retry", "lift", "transport_to_target", "align", "lower", "open_gripper", *VERIFY_WINDOW]
    if scenario == "slip_then_recover":
        return ["observe_scene", "approach_object", "close_gripper", "lift", "transport_to_target", "recover", "lift", "transport_to_target", "align", "lower", "open_gripper", *VERIFY_WINDOW]
    if scenario == "target_occupied":
        return ["observe_scene", "clear_target", *BASE[1:]]
    if scenario == "distractor_object_ambiguity" and rollout_index == 3:
        return ["observe_scene", "verify", "stop"]
    if scenario == "primary_view_occlusion":
        return ["observe_scene", *BASE[1:]]
    return list(BASE)
