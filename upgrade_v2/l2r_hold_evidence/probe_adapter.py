from __future__ import annotations

from typing import Any

import numpy as np

from upgrade_v2.l2r_ambiguity.probes import _custom_perform


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


def perform(sim: Any, action: str) -> dict[str, Any]:
    if action not in {"touch_contact", "separate_touch"}:
        return sim.perform(action)
    return _custom_perform(sim, action)
