from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from upgrade_v2.l2r_forced_drop.intervention import make_force_pulse, run_force_pulse
from upgrade_v2.l2r_forced_drop.physical_reference import evaluate_loss_trace
from upgrade_v2.l2r_forced_drop.protocol import PulseLevel
from upgrade_v2.l2r_forced_drop.simulator import ControlledForcedDropTabletop
from upgrade_v2.l2r_forced_drop.trace_recorder_v2 import snapshot
from upgrade_v2.visual_refine_l2.dynamic_simulator import family_spec
from upgrade_v2.visual_refine_l2.renderer import TabletopRenderer

from .case_registry import CASES, CONFIRMATION_FAMILIES, jitter_deg
from .io_utils import sha256_file, write_csv, write_json, write_jsonl
from .protocol import CAPTURE_EVERY, HEIGHT, JPEG_QUALITY, PHYSICS_HZ, PULSE_DURATION_S, WIDTH


class R17Tabletop(ControlledForcedDropTabletop):
    """R16 physics with a read-only hook after every genuine ``mj_step``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.after_physics_step = None
        super().__init__(*args, **kwargs)

    def _advance(self, target: np.ndarray | None = None, controls: int = 4) -> None:
        start = self.data.mocap_pos[0].copy()
        for control in range(controls):
            control_start = float(self.data.time)
            alpha = (control + 1) / controls
            if target is not None:
                self.data.mocap_pos[0] = start * (1 - alpha) + target * alpha
            commanded_position = self.data.mocap_pos[0].copy()
            for _ in range(5):
                if self.attached:
                    self._set_object_xyz(self.data.mocap_pos[0] + np.array([0.0, 0.0, -0.13]))
                self.mujoco.mj_step(self.model, self.data)
                if self.after_physics_step is not None:
                    self.after_physics_step(self)
            self._active_control_sequence.append({
                "control_step": len(self._active_control_sequence),
                "start_time": round(control_start, 6),
                "end_time": round(float(self.data.time), 6),
                "mocap_position": [round(float(value), 9) for value in commanded_position],
                "gripper_command": "closed" if self.gripper_closed else "open",
                "physics_steps": 5,
            })
            if self._r1_control_callback is not None:
                self._r1_control_callback(self, self._active_control_sequence[-1])

    def physics_step(self) -> None:
        super().physics_step()
        if self.after_physics_step is not None:
            self.after_physics_step(self)


class RolloutCapture:
    def __init__(self, sim: R17Tabletop, root: Path, *, family_id: str, family_seed: int,
                 rollout_seed: int, case_id: str, requested_effect: str) -> None:
        self.sim = sim
        self.root = root
        self.family_id = family_id
        self.family_seed = family_seed
        self.rollout_seed = rollout_seed
        self.case_id = case_id
        self.requested_effect = requested_effect
        self.renderer = TabletopRenderer(sim.model, width=WIDTH, height=HEIGHT)
        self.physics_rows: list[dict[str, Any]] = []
        self.frame_rows: list[dict[str, Any]] = []
        self.contact_rows: list[dict[str, Any]] = []
        self.lifecycle_rows: list[dict[str, Any]] = []
        self.command_rows: list[dict[str, Any]] = []
        self.request_rows: list[dict[str, Any]] = []
        self.physics_step_index = 0
        self.capture_order = 0
        self.phase = "initial"
        self.action: str | None = None
        self.force_start_time: float | None = None
        self.jitter_start_time: float | None = None
        self.drop_times: tuple[float, ...] = ()
        self.commanded_release = False
        self.pre_hold_verified_final = False
        sim.after_physics_step = self._after_step

    def _reference_row(self, *, phase: str | None = None, verified: bool | None = None) -> dict[str, Any]:
        row = snapshot(self.sim, step=self.physics_step_index, phase=phase or self.phase,
                       action=self.action, case_id=self.case_id, family_id=self.family_id,
                       seed=self.rollout_seed,
                       pre_hold_verified=self.pre_hold_verified_final if verified is None else verified,
                       commanded_release=self.commanded_release)
        row["family_seed"] = self.family_seed
        row["rollout_seed"] = self.rollout_seed
        return row

    def add_event_reference(self, phase: str) -> None:
        self.physics_rows.append(self._reference_row(phase=phase))

    def _after_step(self, unused: R17Tabletop) -> None:
        self.physics_step_index += 1
        # Raw pre-hold rows never claim verification before the summary exists.
        verified = False if self.phase == "pre_hold" else None
        self.physics_rows.append(self._reference_row(verified=verified))
        if self.physics_step_index % CAPTURE_EVERY == 0:
            self.capture_frame(action_end=False)

    def set_phase(self, phase: str, action: str | None = None) -> None:
        self.phase, self.action = phase, action

    def perform(self, action: str, phase: str | None = None) -> dict[str, Any]:
        self.set_phase(phase or action, action)
        result = self.sim.perform(action)
        self.capture_frame(action_end=True)
        return result

    def _lifecycle(self) -> dict[str, Any]:
        state = self.sim.attempt_lifecycle.snapshot()
        return {
            "attempt_id": int(state.get("attempt_id") or 1),
            "attempt_phase": state.get("attempt_phase", state.get("phase", "unknown")),
            "attempt_active": bool(state.get("attempt_active", False)),
            "attempt_end": bool(state.get("attempt_end", False)),
            "attempt_end_reason": state.get("attempt_end_reason"),
            "attempt_end_sequence": int(state.get("attempt_end_sequence") or 0),
        }

    def _jitter(self, now: float) -> float:
        if self.case_id.startswith("C4_") and self.phase == "observation":
            if self.jitter_start_time is None:
                self.jitter_start_time = now
            return jitter_deg(now, self.jitter_start_time)
        return 0.0

    def capture_frame(self, *, action_end: bool) -> None:
        now = float(self.sim.data.time)
        jitter = self._jitter(now)
        missing = (not action_end and any(abs(now - value) <= 0.006 for value in self.drop_times))
        rel = Path("online_raw/rgb/front") / f"{self.capture_order:06d}.jpg"
        path = self.root / rel
        raw_hash = jpeg_hash = None
        if not missing:
            rgb = self.renderer.render(self.sim.data, "front", jitter=jitter)
            raw_hash = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
            self.renderer.save_jpeg(rgb, path)
            jpeg_hash = sha256_file(path)
        lifecycle = self._lifecycle()
        frame = {
            "capture_order": self.capture_order, "time": now, "phase": self.phase,
            "action": self.action, "physics_step_index": self.physics_step_index,
            "action_end": bool(action_end), "camera_jitter_deg": jitter,
            "frame_missing": missing,
            "frame_missing_reason": "C9_FORCE_RELATIVE_DROPOUT" if missing else None,
            "raw_rgb_sha256": raw_hash, "jpeg_sha256": jpeg_hash,
            "jpeg_path": str(rel) if not missing else None,
        }
        self.frame_rows.append(frame)
        common = {"capture_order": self.capture_order, "time": now}
        self.contact_rows.append({**common, "contact_present": bool(self.sim.contact_sensor())})
        self.lifecycle_rows.append({**common, **lifecycle})
        self.command_rows.append({**common, "gripper_command": "closed" if self.sim.gripper_closed else "open"})
        self.request_rows.append({**common, "request_id": "request_1", "requested_effect": self.requested_effect,
                                  "context_valid": True})
        self.capture_order += 1

    def close(self) -> None:
        self.sim.after_physics_step = None
        self.renderer.close()

    def write(self) -> None:
        raw = self.root / "online_raw"
        write_csv(raw / "frame_manifest.csv", self.frame_rows)
        write_jsonl(raw / "contact_proxy.jsonl", self.contact_rows)
        write_jsonl(raw / "attempt_lifecycle.jsonl", self.lifecycle_rows)
        write_jsonl(raw / "gripper_commands.jsonl", self.command_rows)
        write_jsonl(raw / "request_context.jsonl", self.request_rows)
        reference = self.root / "reference"
        write_jsonl(reference / "physics_trace.jsonl", self.physics_rows)
        write_jsonl(reference / "contact_trace.jsonl", self.physics_rows)
        write_jsonl(reference / "capture_volume_trace.jsonl", self.physics_rows)
        write_jsonl(reference / "force_trace.jsonl", self.physics_rows)


def _prehold(capture: RolloutCapture, steps: int) -> dict[str, Any]:
    z0 = float(capture.sim.object_xyz[2])
    capture.capture_frame(action_end=True)
    capture.perform("approach_object")
    capture.perform("close_gripper")
    capture.perform("lift")
    capture.set_phase("pre_hold_settle", "hold")
    for _ in range(10):
        capture.sim.physics_step()
    capture.set_phase("pre_hold", "hold")
    start = len(capture.physics_rows)
    for _ in range(steps):
        capture.sim.physics_step()
    rows = capture.physics_rows[start:]
    anchor = np.asarray(rows[0]["object_in_gripper_position"], dtype=float) if rows else None
    drift = max((float(np.linalg.norm(np.asarray(row["object_in_gripper_position"]) - anchor))
                 for row in rows), default=float("inf")) if anchor is not None else float("inf")
    rise = float(capture.sim.object_xyz[2] - z0)
    verified = bool(len(rows) == steps and all(row["weld_active"] and row["numeric_health"]["passed"] for row in rows)
                    and rise >= 0.03 and drift <= 0.01)
    capture.pre_hold_verified_final = verified
    capture.sim.pre_hold_verified = verified
    return {"pre_hold_verified": verified, "sample_count": len(rows),
            "duration_s": steps / PHYSICS_HZ, "height_rise_m": rise,
            "max_relative_drift_m": drift}


def _pulse(capture: RolloutCapture, level: dict[str, Any]) -> dict[str, Any]:
    sim = capture.sim
    pulse_level = PulseLevel(str(level["level_id"]), tuple(float(v) for v in level["target_delta_v_local_mps"]))
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
            if capture.case_id.startswith("C9_"):
                capture.drop_times = tuple(capture.force_start_time + value for value in (0.10, 0.15, 0.20))
            capture.add_event_reference(name)
        elif name == "physics_step":
            capture.set_phase("force_pulse", "intervention")
        elif name == "force_pulse_ended":
            capture.set_phase(name, "intervention")
            capture.add_event_reference(name)

    run_force_pulse(sim, pulse, trace=event)
    capture.capture_frame(action_end=True)
    return {"level_id": pulse.level_id, "duration_s": pulse.duration_s,
            "target_delta_v_local_mps": list(pulse.target_delta_v_local_mps),
            "force_local_n": list(pulse.force_local_n), "force_world_n": list(pulse.force_world_n),
            "force_start_time": capture.force_start_time}


def _observe(capture: RolloutCapture, steps: int = 100) -> None:
    capture.set_phase("observation", "observe")
    for _ in range(steps):
        capture.sim.physics_step()
    capture.capture_frame(action_end=True)


def collect_rollout(root: Path, *, family_id: str, family_seed: int, rollout_seed: int,
                    case: Any, difficulty: dict[str, Any]) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    scenario = "missed_grasp_then_retry" if case.case_id.startswith("C1_") else "normal_pick_place"
    variant = "brief_hold" if case.case_id.startswith("C10_") else "default"
    sim = R17Tabletop(family_spec(family_id, scenario, family_seed, rollout_seed,
                                  probe_variant=variant), rollout_seed)
    capture = RolloutCapture(sim, root, family_id=family_id, family_seed=family_seed,
                             rollout_seed=rollout_seed, case_id=case.case_id,
                             requested_effect=case.requested_effect)
    pre = {"pre_hold_verified": False, "sample_count": 0, "duration_s": 0.0}
    intervention = {"level_id": None, "force_start_time": None}
    try:
        if case.case_id.startswith("C1_"):
            capture.capture_frame(action_end=True)
            capture.perform("approach_object")
            capture.perform("close_gripper")
            capture.perform("verify", "hold_attempt_end")
            _observe(capture, 50)
        elif case.case_id.startswith("C2_"):
            capture.capture_frame(action_end=True)
            capture.perform("approach_object")
            capture.perform("verify", "touch_attempt_end")
            _observe(capture, 50)
        else:
            pre = _prehold(capture, 10 if case.case_id.startswith("C10_") else 50)
            if not pre["pre_hold_verified"]:
                raise RuntimeError("PREHOLD_UNVERIFIED")
            if case.case_id.startswith(("C3_", "C4_")):
                pass
            elif case.case_id.startswith("C5_"):
                sim.disable_weld_for_intervention(); sim.forward()
                capture.add_event_reference("post_weld_off_pre_force")
            elif case.case_id.startswith("C6_"):
                intervention = _pulse(capture, difficulty["weak"])
            elif case.case_id.startswith("C7_"):
                intervention = _pulse(capture, difficulty["medium"])
            elif case.case_id.startswith(("C8_", "C9_", "C10_")):
                intervention = _pulse(capture, difficulty["strong"])
            elif case.case_id.startswith("C11_"):
                capture.perform("transport_to_target", "transport")
                intervention = _pulse(capture, difficulty["strong"])
            elif case.case_id.startswith("C12_"):
                capture.commanded_release = True
                capture.perform("open_gripper", "commanded_release")
            _observe(capture, 100)
        capture.write()
    finally:
        capture.close()

    numeric_pass = all(row.get("numeric_health", {}).get("passed", False) for row in capture.physics_rows)
    outcome = evaluate_loss_trace(capture.physics_rows, pre_hold_verified=pre["pre_hold_verified"],
                                  force_start_time=capture.force_start_time)
    if case.case_id.startswith("C1_"):
        outcome = {"state": "MISSED_HOLD_RESOLVED", "physical_loss_confirmed": False,
                   "reference_action": "retry_grasp", "resolvable": True}
    elif case.case_id.startswith("C2_"):
        outcome = {"state": "TOUCH_ONLY_RESOLVED", "physical_loss_confirmed": False,
                   "reference_action": "none", "resolvable": True}
    elif case.case_id.startswith("C12_"):
        outcome = {"state": "COMMANDED_RELEASE", "physical_loss_confirmed": False,
                   "reference_action": "none", "resolvable": True}
    else:
        outcome["reference_action"] = "recover_object" if outcome.get("physical_loss_confirmed") else "none"
        outcome["resolvable"] = outcome.get("state") not in {"TRACE_INCOMPLETE", "NUMERICAL_INVALID", "PREHOLD_UNVERIFIED"}
    prehold_rows = [row for row in capture.physics_rows if row.get("phase") == "pre_hold"]
    separation_anchor = (np.asarray(prehold_rows[-1]["object_in_gripper_position"], dtype=float)
                         if prehold_rows else None)
    separations = [float(np.linalg.norm(np.asarray(row["object_in_gripper_position"], dtype=float) - separation_anchor))
                   for row in capture.physics_rows if separation_anchor is not None
                   and row.get("phase") != "pre_hold" and row.get("object_in_gripper_position") is not None]
    reference = {
        "schema": "l2rar2_r17_physical_reference_v1", "family_id": family_id,
        "family_seed": family_seed, "rollout_seed": rollout_seed, "case_id": case.case_id,
        "requested_effect": case.requested_effect, "pre_hold": pre,
        "pre_hold_verified": bool(pre["pre_hold_verified"]), "trace_complete": bool(capture.physics_rows),
        "frame_manifest_complete": bool(capture.frame_rows), "numeric_health_pass": numeric_pass,
        "detector_job_complete": False, "commanded_release": case.case_id.startswith("C12_"),
        "force_start_time": capture.force_start_time,
        "peak_relative_separation_m": max(separations, default=None),
        **outcome,
    }
    write_json(root / "reference/pre_hold_summary.json", pre)
    write_json(root / "reference/intervention.json", intervention)
    write_json(root / "reference/numerical_health.json", {"passed": numeric_pass,
                                                            "samples": len(capture.physics_rows)})
    write_json(root / "reference/physical_reference.json", reference)
    write_json(root / "metadata.json", {k: reference[k] for k in (
        "family_id", "family_seed", "rollout_seed", "case_id", "requested_effect")})
    write_json(root / "termination.json", {"status": "COMPLETE", "time": float(sim.data.time)})
    return reference


def collect_confirmation(output_root: Path, difficulty: dict[str, Any]) -> list[dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for family_id, family_seed, rollout_seed_base in CONFIRMATION_FAMILIES:
        for index, case in enumerate(CASES):
            rollout_seed = rollout_seed_base + index
            rollout_id = f"{family_id}__{case.case_id}"
            rollout_root = output_root / rollout_id
            completed = rollout_root / "reference/physical_reference.json"
            termination = rollout_root / "termination.json"
            if completed.is_file() and termination.is_file():
                rows.append(json.loads(completed.read_text(encoding="utf-8")))
                continue
            if rollout_root.exists():
                raise RuntimeError(f"INCOMPLETE_ROLLOUT_REQUIRES_QUARANTINE:{rollout_root}")
            rows.append(collect_rollout(rollout_root, family_id=family_id,
                                        family_seed=family_seed, rollout_seed=rollout_seed,
                                        case=case, difficulty=difficulty))
    return rows
