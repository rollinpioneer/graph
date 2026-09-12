from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from upgrade_v2.l2r_forced_drop.intervention import make_force_pulse, run_force_pulse
from upgrade_v2.l2r_forced_drop.physical_reference import evaluate_loss_trace
from upgrade_v2.l2r_forced_drop.protocol import PulseLevel
from upgrade_v2.l2r_rgb_temporal_confirmation.protocol import HEIGHT, JPEG_QUALITY, PHYSICS_HZ, WIDTH
from upgrade_v2.l2r_rgb_temporal_confirmation.rgb_capture import R17Tabletop, RolloutCapture, _observe, _prehold
from upgrade_v2.visual_refine_l2.dynamic_simulator import family_spec

from .io_utils import sha256, write_csv, write_json, write_jsonl
from .protocol import LEVELS
from .registry import CASES, ConfirmationCase, load_families

PULSE_DURATION_S = 0.20
FROZEN_REGISTRY_RANGE = (886000, 886005, 88700000, 88700500)


class R20Capture(RolloutCapture):
    def __init__(self, *args: Any, case: ConfirmationCase, **kwargs: Any) -> None:
        self.case = case
        self.transport_start_time: float | None = None
        self.transport_midpoint_time: float | None = None
        self.transport_end_time: float | None = None
        self.phase_grid_time: float | None = None
        self.requested_phase_offset_ms: int | None = case.phase_offset_ms
        self.actual_phase_offset_ms: float | None = None
        self.dropout_thresholds = (0.10, 0.15, 0.20) if case.rgb_dropout else ()
        self.dropout_assigned: set[float] = set()
        self.logical_rows: list[dict[str, Any]] = []
        super().__init__(*args, **kwargs)

    def _jitter(self, now: float) -> float:
        if self.case.camera_jitter and self.phase.startswith(("transport", "observation")):
            if self.jitter_start_time is None: self.jitter_start_time = now
            return 1.5 * math.sin(2.0 * math.pi * (now - self.jitter_start_time) / 0.40)
        return 0.0

    def _planned_dropout(self, now: float, action_end: bool) -> tuple[bool, float | None]:
        if action_end or self.force_start_time is None: return False, None
        elapsed = now - self.force_start_time
        target = next((value for value in self.dropout_thresholds
                       if value not in self.dropout_assigned and elapsed + 1e-12 >= value), None)
        if target is None: return False, None
        self.dropout_assigned.add(target)
        return True, target

    def capture_frame(self, *, action_end: bool) -> None:
        now = float(self.sim.data.time)
        jitter = self._jitter(now)
        missing, requested_offset = self._planned_dropout(now, action_end)
        rel = Path("online_raw/rgb/front") / f"{self.capture_order:06d}.jpg"
        path = self.root / rel
        raw_hash = jpeg_hash = None
        if not missing:
            rgb = self.renderer.render(self.sim.data, "front", jitter=jitter)
            raw_hash = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
            self.renderer.save_jpeg(rgb, path)
            jpeg_hash = sha256(path)
        lifecycle = self._lifecycle()
        kind = "action_end" if action_end else "periodic"
        frame = {"capture_order": self.capture_order, "time": now, "phase": self.phase,
                 "action": self.action, "physics_step_index": self.physics_step_index,
                 "capture_kind": kind, "action_end": bool(action_end), "camera_jitter_deg": jitter,
                 "frame_missing": missing,
                 "frame_missing_reason": "T11_SUCCESSOR_PERIODIC_RGB_DROPOUT" if missing else None,
                 "dropout_requested_offset_s": requested_offset,
                 "dropout_actual_offset_s": None if not missing else now - float(self.force_start_time),
                 "raw_rgb_sha256": raw_hash, "jpeg_sha256": jpeg_hash,
                 "jpeg_path": str(rel) if not missing else None}
        self.frame_rows.append(frame)
        common = {"capture_order": self.capture_order, "time": now}
        contact = bool(self.sim.contact_sensor())
        self.contact_rows.append({**common, "contact_present": contact})
        self.lifecycle_rows.append({**common, **lifecycle})
        self.command_rows.append({**common, "gripper_command": "closed" if self.sim.gripper_closed else "open"})
        self.request_rows.append({**common, "request_id": "request_1", "requested_effect": self.requested_effect,
                                  "context_valid": True})
        self.logical_rows.append({"source_capture_order": self.capture_order,
                                  "candidate_capture_order": self.capture_order, "time": now,
                                  "physics_step_index": self.physics_step_index, "capture_kind": kind,
                                  "phase": self.phase, "synthetic_duplicate": False,
                                  "jpeg_sha256": jpeg_hash})
        self.capture_order += 1

    def write(self) -> None:
        super().write()
        write_csv(self.root / "online_raw/logical_observation_manifest.csv", self.logical_rows)
        write_json(self.root / "online_raw/fault_injection_manifest.json",
                   {"schema": "l2rar2_r20_fault_injection_manifest_v1", "status": "NOT_YET_APPLIED",
                    "case_id": self.case_id, "faults": []})


def _pulse(capture: R20Capture, level_name: str) -> dict[str, Any]:
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
            capture.set_phase(name, "intervention"); capture.add_event_reference(name)
        elif name == "force_pulse_started":
            capture.force_start_time = float(current.data.time)
            capture.actual_phase_offset_ms = 1000.0 * (capture.force_start_time - float(capture.phase_grid_time))
            capture.set_phase(name, "intervention"); capture.add_event_reference(name)
        elif name == "physics_step": capture.set_phase("force_pulse", "intervention")
        elif name == "force_pulse_ended":
            capture.set_phase(name, "intervention"); capture.add_event_reference(name)

    run_force_pulse(sim, pulse, trace=event)
    capture.capture_frame(action_end=True)
    return {"level_id": level_name, "duration_s": pulse.duration_s,
            "target_delta_v_local_mps": list(pulse.target_delta_v_local_mps),
            "force_local_n": list(pulse.force_local_n), "force_world_n": list(pulse.force_world_n),
            "phase_grid_time": capture.phase_grid_time,
            "requested_phase_offset_ms": capture.requested_phase_offset_ms,
            "actual_phase_offset_ms": capture.actual_phase_offset_ms,
            "force_start_time": capture.force_start_time}


def _transport(capture: R20Capture) -> dict[str, Any]:
    sim, case = capture.sim, capture.case
    sim.action_index += 1; sim._active_control_sequence = []; sim.lifecycle_before_action("transport_to_target")
    capture.transport_start_time = float(sim.data.time)
    target = np.array([sim.spec.target_x, sim.spec.target_y, 0.80])
    midway = (sim.data.mocap_pos[0] + target) / 2.0
    capture.set_phase("transport_pre_event", "transport_to_target")
    sim._advance(midway, controls=4)
    capture.transport_midpoint_time = float(sim.data.time)
    while capture.physics_step_index % 5:
        capture.set_phase("transport_grid_alignment", "transport_to_target"); sim.physics_step()
    capture.phase_grid_time = float(sim.data.time)
    intervention: dict[str, Any] = {"level_id": None, "phase_grid_time": capture.phase_grid_time,
                                    "requested_phase_offset_ms": case.phase_offset_ms,
                                    "actual_phase_offset_ms": None, "force_start_time": None}
    if case.phase_offset_ms is not None:
        for _ in range(case.phase_offset_ms // 10):
            capture.set_phase("transport_phase_offset", "transport_to_target"); sim.physics_step()
    if case.commanded_release:
        capture.actual_phase_offset_ms = 1000.0 * (float(sim.data.time) - float(capture.phase_grid_time))
        capture.commanded_release = True
        capture.perform("open_gripper", "commanded_release_during_transport")
        intervention["actual_phase_offset_ms"] = capture.actual_phase_offset_ms
    elif case.level:
        intervention = _pulse(capture, case.level)
    capture.set_phase("transport_post_event", "transport_to_target")
    sim._advance(target, controls=4); sim.lifecycle_after_action("transport_to_target")
    capture.transport_end_time = float(sim.data.time); capture.capture_frame(action_end=True)
    intervention.update({"transport_start_time": capture.transport_start_time,
                         "transport_midpoint_time": capture.transport_midpoint_time,
                         "transport_end_time": capture.transport_end_time})
    return intervention


def collect_rollout(root: Path, family_id: str, family_seed: int, rollout_seed: int,
                    case: ConfirmationCase) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    sim = R17Tabletop(family_spec(family_id, "normal_pick_place", family_seed, rollout_seed), rollout_seed)
    capture = R20Capture(sim, root, family_id=family_id, family_seed=family_seed,
                         rollout_seed=rollout_seed, case_id=case.case_id,
                         requested_effect="RELEASE_OBJECT" if case.commanded_release else "HOLD_OBJECT", case=case)
    try:
        pre = _prehold(capture, 50)
        if not pre["pre_hold_verified"]: raise RuntimeError("PREHOLD_UNVERIFIED")
        intervention = _transport(capture); _observe(capture, 100); capture.write()
    except Exception as exc:
        write_json(root / "termination.json", {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
                                                 "time": float(sim.data.time), "physics_steps": capture.physics_step_index})
        raise
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
        outcome["resolvable"] = outcome.get("state") not in {"TRACE_INCOMPLETE", "NUMERICAL_INVALID",
                                                                 "PREHOLD_UNVERIFIED", "INCIPIENT_OR_UNRESOLVED"}
    prehold_rows = [row for row in rows if row.get("phase") == "pre_hold"]
    anchor = np.asarray(prehold_rows[-1]["object_in_gripper_position"], dtype=float)
    separations = [float(np.linalg.norm(np.asarray(row["object_in_gripper_position"], dtype=float) - anchor))
                   for row in tail if row.get("object_in_gripper_position") is not None]
    reference = {"schema": "l2rar2_r20_physical_reference_v1", "family_id": family_id,
                 "family_seed": family_seed, "rollout_seed": rollout_seed, "case_id": case.case_id,
                 "requested_effect": capture.requested_effect, "pre_hold": pre,
                 "pre_hold_verified": bool(pre["pre_hold_verified"]), "trace_complete": bool(rows),
                 "frame_manifest_complete": bool(capture.frame_rows), "numeric_health_pass": numeric_pass,
                 "detector_job_complete": False, "commanded_release": case.commanded_release,
                 "force_start_time": capture.force_start_time,
                 "phase_grid_time": capture.phase_grid_time,
                 "requested_phase_offset_ms": case.phase_offset_ms,
                 "actual_phase_offset_ms": capture.actual_phase_offset_ms,
                 "peak_relative_separation_m": max(separations, default=None), **outcome}
    write_json(root / "reference/pre_hold_summary.json", pre)
    write_json(root / "reference/intervention.json", intervention)
    write_json(root / "reference/numerical_health.json", {"passed": numeric_pass, "samples": len(rows)})
    write_json(root / "reference/physical_reference.json", reference)
    write_json(root / "metadata.json", {key: reference[key] for key in
                                         ("family_id", "family_seed", "rollout_seed", "case_id", "requested_effect")})
    write_json(root / "termination.json", {"status": "COMPLETE", "time": float(sim.data.time),
                                             "physics_steps": capture.physics_step_index})
    return reference


def _run_task(args: tuple[str, int, int, ConfirmationCase, str]) -> dict[str, Any]:
    family_id, family_seed, rollout_seed, case, output = args
    return collect_rollout(Path(output), family_id, family_seed, rollout_seed, case)


def collect(registry_path: Path, output_root: Path, workers: int = 2) -> dict[str, Any]:
    if workers < 1 or workers > 2: raise ValueError("WORKERS_MUST_BE_1_OR_2")
    output_root.mkdir(parents=True, exist_ok=True)
    families = load_families(registry_path)
    if (families[0][1], families[-1][1], families[0][2], families[-1][2]) != FROZEN_REGISTRY_RANGE:
        raise RuntimeError("FROZEN_REGISTRY_RANGE_MISMATCH")
    tasks = []
    for family_id, family_seed, seed_base in families:
        for index, case in enumerate(CASES):
            root = output_root / f"{family_id}__{case.case_id}"
            if (root / "termination.json").is_file() and json.loads((root / "termination.json").read_text())["status"] == "COMPLETE":
                continue
            if root.exists(): raise RuntimeError(f"INCOMPLETE_ROLLOUT_PRESENT:{root}")
            tasks.append((family_id, family_seed, seed_base + index, case, str(root)))
    completed, failures = [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        pending: dict[concurrent.futures.Future, tuple] = {}
        iterator = iter(tasks)
        for _ in range(min(workers, len(tasks))):
            task = next(iterator, None)
            if task: pending[pool.submit(_run_task, task)] = task
        stop = False
        while pending:
            done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                try:
                    result = future.result(); completed.append(result)
                except Exception as exc:
                    failures.append({"rollout": Path(task[-1]).name, "error": f"{type(exc).__name__}: {exc}"}); stop = True
                if not stop:
                    next_task = next(iterator, None)
                    if next_task: pending[pool.submit(_run_task, next_task)] = next_task
    all_complete = len(list(output_root.glob("*/termination.json"))) == 72 and all(
        json.loads(path.read_text())["status"] == "COMPLETE" for path in output_root.glob("*/termination.json"))
    result = {"schema": "l2rar2_r20_collection_result_v1", "status": "PASS" if all_complete else "FAIL",
              "newly_completed": len(completed), "failures": failures, "all_72_complete": all_complete}
    write_json(output_root / "collection_status.json", result)
    return result
