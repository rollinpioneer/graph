from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from upgrade_v2.l2r_forced_drop.intervention import make_force_pulse, run_force_pulse
from upgrade_v2.l2r_forced_drop.physical_reference import evaluate_loss_trace
from upgrade_v2.l2r_forced_drop.protocol import PulseLevel
from upgrade_v2.l2r_rgb_temporal_confirmation.rgb_capture import (
    PHYSICS_HZ, R17Tabletop, RolloutCapture, _observe, _prehold,
)
from upgrade_v2.visual_refine_l2.dynamic_simulator import family_spec

from .io_utils import write_json
from .protocol import LEVELS
from .r18_registry import CASES, FAMILIES, TransportCase

PULSE_DURATION_S = 0.20


class R18Capture(RolloutCapture):
    def _jitter(self, now: float) -> float:
        if self.case_id.startswith("T2_") and self.phase.startswith(("transport", "observation")):
            if self.jitter_start_time is None:
                self.jitter_start_time = now
            return 1.5 * math.sin(2.0 * math.pi * (now - self.jitter_start_time) / 0.40)
        return 0.0


def _pulse(capture: R18Capture, level_name: str) -> dict[str, Any]:
    sim = capture.sim
    level = LEVELS[level_name]
    pulse_level = PulseLevel(level_name, tuple(level["delta_v_local_mps"]))
    object_body = int(sim.mujoco.mj_name2id(sim.model, sim.mujoco.mjtObj.mjOBJ_BODY, "object"))
    gripper_body = int(sim.mujoco.mj_name2id(sim.model, sim.mujoco.mjtObj.mjOBJ_BODY, "gripper"))
    pulse = make_force_pulse(float(sim.model.body_mass[object_body]),
                             sim.data.xmat[gripper_body].reshape(3, 3), pulse_level,
                             duration_s=PULSE_DURATION_S)

    def event(current: R17Tabletop, name: str) -> None:
        if name == "post_weld_off_pre_force":
            capture.set_phase(name, "intervention")
            capture.add_event_reference(name)
        elif name == "force_pulse_started":
            capture.force_start_time = float(current.data.time)
            capture.set_phase(name, "intervention")
            if capture.case_id.startswith("T11_"):
                capture.drop_times = tuple(capture.force_start_time + value for value in (0.10, 0.15, 0.20))
            capture.add_event_reference(name)
        elif name == "physics_step":
            capture.set_phase("force_pulse", "intervention")
        elif name == "force_pulse_ended":
            capture.set_phase(name, "intervention")
            capture.add_event_reference(name)

    run_force_pulse(sim, pulse, trace=event)
    capture.capture_frame(action_end=True)
    for row in capture.frame_rows:
        if row.get("frame_missing") and capture.case_id.startswith("T11_"):
            row["frame_missing_reason"] = "T11_FORCE_RELATIVE_RGB_DROPOUT"
    return {"level_id": level_name, "duration_s": pulse.duration_s,
            "target_delta_v_local_mps": list(pulse.target_delta_v_local_mps),
            "force_local_n": list(pulse.force_local_n), "force_world_n": list(pulse.force_world_n),
            "force_start_time": capture.force_start_time}


def _transport(capture: R18Capture, case: TransportCase) -> dict[str, Any]:
    sim = capture.sim
    sim.action_index += 1
    sim._active_control_sequence = []
    sim.lifecycle_before_action("transport_to_target")
    target = np.array([sim.spec.target_x, sim.spec.target_y, 0.80])
    midway = (sim.data.mocap_pos[0] + target) / 2.0
    capture.set_phase("transport_pre_event", "transport_to_target")
    sim._advance(midway, controls=4)
    capture.capture_frame(action_end=False)
    intervention: dict[str, Any] = {"level_id": None, "force_start_time": None,
                                     "phase_offset_ms": case.phase_offset_ms}
    if case.commanded_release:
        capture.commanded_release = True
        capture.perform("open_gripper", "commanded_release_during_transport")
    elif case.level:
        for _ in range((case.phase_offset_ms or 0) // 10):
            capture.set_phase("transport_phase_offset", "transport_to_target")
            sim.physics_step()
        intervention = _pulse(capture, case.level)
        intervention["phase_offset_ms"] = case.phase_offset_ms
    capture.set_phase("transport_post_event", "transport_to_target")
    sim._advance(target, controls=4)
    sim.lifecycle_after_action("transport_to_target")
    capture.capture_frame(action_end=True)
    return intervention


def collect_rollout(root: Path, *, family_id: str, family_seed: int, rollout_seed: int,
                    case: TransportCase) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    sim = R17Tabletop(family_spec(family_id, "normal_pick_place", family_seed, rollout_seed), rollout_seed)
    capture = R18Capture(sim, root, family_id=family_id, family_seed=family_seed,
                         rollout_seed=rollout_seed, case_id=case.case_id,
                         requested_effect="RELEASE_OBJECT" if case.commanded_release else "HOLD_OBJECT")
    try:
        pre = _prehold(capture, 50)
        if not pre["pre_hold_verified"]:
            raise RuntimeError("PREHOLD_UNVERIFIED")
        intervention = _transport(capture, case)
        _observe(capture, 100)
        capture.write()
    finally:
        capture.close()

    rows = capture.physics_rows
    numeric_pass = bool(rows) and all(row.get("numeric_health", {}).get("passed", False) for row in rows)
    last_prehold = max(index for index, row in enumerate(rows) if row.get("phase") == "pre_hold")
    tail = rows[last_prehold + 1:]
    if case.commanded_release:
        outcome = {"state": "COMMANDED_RELEASE", "physical_loss_confirmed": False,
                   "reference_action": "none", "resolvable": True}
    elif case.level is None:
        stable = bool(tail) and all(row.get("weld_active") and row.get("inside_capture") for row in tail)
        outcome = {"state": "WELD_SUPPORTED_HOLD" if stable else "INCIPIENT_OR_UNRESOLVED",
                   "physical_loss_confirmed": False, "reference_action": "none", "resolvable": stable}
    else:
        outcome = evaluate_loss_trace(tail, pre_hold_verified=pre["pre_hold_verified"],
                                      force_start_time=capture.force_start_time)
        outcome["reference_action"] = "recover_object" if outcome.get("physical_loss_confirmed") else "none"
        outcome["resolvable"] = outcome.get("state") not in {
            "TRACE_INCOMPLETE", "NUMERICAL_INVALID", "PREHOLD_UNVERIFIED", "INCIPIENT_OR_UNRESOLVED"}
    prehold_rows = [row for row in rows if row.get("phase") == "pre_hold"]
    anchor = np.asarray(prehold_rows[-1]["object_in_gripper_position"], dtype=float)
    separations = [float(np.linalg.norm(np.asarray(row["object_in_gripper_position"], dtype=float) - anchor))
                   for row in tail if row.get("object_in_gripper_position") is not None]
    reference = {
        "schema": "l2rar2_r18_physical_reference_v1", "family_id": family_id,
        "family_seed": family_seed, "rollout_seed": rollout_seed, "case_id": case.case_id,
        "requested_effect": capture.requested_effect, "pre_hold": pre,
        "pre_hold_verified": bool(pre["pre_hold_verified"]), "trace_complete": bool(rows),
        "frame_manifest_complete": bool(capture.frame_rows), "numeric_health_pass": numeric_pass,
        "detector_job_complete": False, "commanded_release": case.commanded_release,
        "force_start_time": capture.force_start_time, "phase_offset_ms": case.phase_offset_ms,
        "peak_relative_separation_m": max(separations, default=None), **outcome,
    }
    write_json(root / "reference/pre_hold_summary.json", pre)
    write_json(root / "reference/intervention.json", intervention)
    write_json(root / "reference/numerical_health.json", {"passed": numeric_pass, "samples": len(rows)})
    write_json(root / "reference/physical_reference.json", reference)
    write_json(root / "metadata.json", {key: reference[key] for key in (
        "family_id", "family_seed", "rollout_seed", "case_id", "requested_effect")})
    write_json(root / "termination.json", {"status": "COMPLETE", "time": float(sim.data.time)})
    return reference


def collect_confirmation(output_root: Path) -> list[dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for family_id, family_seed, seed_base in FAMILIES:
        for index, case in enumerate(CASES):
            rollout_root = output_root / f"{family_id}__{case.case_id}"
            completed = rollout_root / "reference/physical_reference.json"
            if completed.is_file() and (rollout_root / "termination.json").is_file():
                results.append(json.loads(completed.read_text(encoding="utf-8")))
                continue
            if rollout_root.exists():
                raise RuntimeError(f"INCOMPLETE_ROLLOUT_REQUIRES_QUARANTINE:{rollout_root}")
            results.append(collect_rollout(rollout_root, family_id=family_id, family_seed=family_seed,
                                           rollout_seed=seed_base + index, case=case))
    return results
