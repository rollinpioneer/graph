from __future__ import annotations

from typing import Any

import numpy as np

STRATA = (
    "miss_without_prior_hold", "loss_after_observed_hold", "brief_true_hold_then_loss",
    "touch_without_hold_then_loss", "commanded_release", "long_gap_after_loss",
    "history_or_visual_unavailable", "normal_hold_pause_resume",
)


def program_for_stratum(stratum: str) -> list[str]:
    if stratum == "miss_without_prior_hold":
        return ["observe_scene", "approach_object", "close_gripper", "retry", "lift", "verify"]
    if stratum in {"loss_after_observed_hold", "brief_true_hold_then_loss"}:
        return ["observe_scene", "approach_object", "close_gripper", "lift", "transport_to_target", "verify", "recover", "lift"]
    if stratum == "touch_without_hold_then_loss":
        return ["observe_scene", "approach_object", "touch_contact", "separate_touch", "verify", "verify"]
    if stratum == "commanded_release":
        return ["observe_scene", "approach_object", "close_gripper", "lift", "open_gripper", "verify", "verify"]
    if stratum in {"long_gap_after_loss", "history_or_visual_unavailable"}:
        return ["observe_scene", "approach_object", "close_gripper", "lift", "transport_to_target", "verify", "verify", "verify", "verify", "verify", "verify", "recover", "lift"]
    if stratum == "normal_hold_pause_resume":
        return ["observe_scene", "approach_object", "close_gripper", "lift", "verify", "verify", "transport_to_target", "verify", "open_gripper", "verify"]
    raise ValueError(f"unknown R1 stratum: {stratum}")


CONTROL_VARIANTS = (
    {"variant_id": "v0_short", "lift_delta_z": 0.18, "lift_controls": 3, "touch_controls": 1, "separation_y": -0.20, "separation_controls": 2},
    {"variant_id": "v1_medium", "lift_delta_z": 0.20, "lift_controls": 4, "touch_controls": 2, "separation_y": -0.24, "separation_controls": 3},
    {"variant_id": "v2_long", "lift_delta_z": 0.22, "lift_controls": 5, "touch_controls": 3, "separation_y": -0.28, "separation_controls": 4},
    {"variant_id": "v3_wide", "lift_delta_z": 0.24, "lift_controls": 4, "touch_controls": 4, "separation_y": -0.32, "separation_controls": 5},
)


def control_variant(index: int) -> dict[str, Any]:
    return dict(CONTROL_VARIANTS[index % len(CONTROL_VARIANTS)])


def _finish_custom(sim: Any, action: str, before: float) -> dict[str, Any]:
    result = {"action_index": sim.action_index, "action": action, "start_time": round(before, 6),
              "end_time": round(float(sim.data.time), 6), "gripper_command": "closed" if sim.gripper_closed else "open",
              "contact_present": sim.contact_sensor(), "termination_reason": None,
              "low_level_control_sequence": sim._active_control_sequence,
              "attempt_lifecycle": sim.attempt_lifecycle.snapshot()}
    callback = getattr(sim, "_r1_action_end_callback", None)
    if callback is not None:
        callback(sim, action, int(result["action_index"]), result)
    return result


def perform(sim: Any, action: str, variant_index: int = 0) -> dict[str, Any]:
    variant = control_variant(variant_index)
    if action not in {"touch_contact", "separate_touch", "lift"}:
        return sim.perform(action)
    if action == "lift":
        sim.action_index += 1
        sim._active_control_sequence = []
        sim.lifecycle_before_action(action)
        before = float(sim.data.time)
        sim._advance(
            sim.data.mocap_pos[0] + np.array([0.0, 0.0, float(variant["lift_delta_z"])]),
            controls=int(variant["lift_controls"]),
        )
        sim.lifecycle_after_action(action)
        return _finish_custom(sim, action, before)
    sim.action_index += 1
    sim._active_control_sequence = []
    sim.lifecycle_before_action(action)
    before = float(sim.data.time)
    if action == "touch_contact":
        sim.gripper_closed = True
        sim._advance(
            sim.object_xyz + np.array([0.0, 0.0, 0.13]),
            controls=int(variant["touch_controls"]),
        )
        sim._record_event("transient_contact", observable=True)
    else:
        sim._advance(
            sim.data.mocap_pos[0] + np.array([0.0, float(variant["separation_y"]), 0.0]),
            controls=int(variant["separation_controls"]),
        )
        sim._record_event("transient_contact_lost", observable=True)
    sim.lifecycle_after_action(action)
    return _finish_custom(sim, action, before)
