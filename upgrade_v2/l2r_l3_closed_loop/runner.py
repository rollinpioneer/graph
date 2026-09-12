from __future__ import annotations

import concurrent.futures
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from upgrade_v2.l2r_canonical_time_confirmation.collector import CanonicalTimeCapture
from upgrade_v2.l2r_canonical_time_confirmation.guard import apply_guard
from upgrade_v2.l2r_logical_clock_confirmation.io_utils import write_json, write_jsonl
from upgrade_v2.l2r_logical_clock_confirmation.registry import ConfirmationCase
from upgrade_v2.l2r_rgb_temporal_confirmation.candidate_inputs import validate_online_row
from upgrade_v2.l2r_rgb_temporal_confirmation.evaluation import _run_o
from upgrade_v2.l2r_rgb_temporal_confirmation.rgb_capture import R17Tabletop, _prehold
from upgrade_v2.visual_refine_l2.dynamic_simulator import family_spec
from upgrade_v2.visual_refine_l2.vision import detect_frame

from . import CLP3_ID, METHODS
from .registry import CASES, FAMILIES, L3Case

RECOVERY_ACTIONS = {"retry_grasp", "recover_object"}
HOLD_VERIFY_STEPS = 50
RECOVERY_LOOP_LIMIT = 2


class L3ClosedLoopTabletop(R17Tabletop):
    """Use the weld and MuJoCo steps for transport; never teleport the object."""

    def _advance(self, target: np.ndarray | None = None, controls: int = 4) -> None:
        start = self.data.mocap_pos[0].copy()
        for control in range(controls):
            control_start = float(self.data.time)
            alpha = (control + 1) / controls
            if target is not None: self.data.mocap_pos[0] = start * (1 - alpha) + target * alpha
            commanded = self.data.mocap_pos[0].copy()
            for _ in range(5):
                self.mujoco.mj_step(self.model, self.data)
                if self.after_physics_step is not None: self.after_physics_step(self)
            self._active_control_sequence.append({
                "control_step": len(self._active_control_sequence), "start_time": round(control_start, 6),
                "end_time": round(float(self.data.time), 6),
                "mocap_position": [round(float(value), 9) for value in commanded],
                "gripper_command": "closed" if self.gripper_closed else "open", "physics_steps": 5,
            })

    def perform(self, action: str) -> dict[str, Any]:
        if action != "open_gripper": return super().perform(action)
        self.action_index += 1; self._active_control_sequence = []; self.lifecycle_before_action(action)
        before = float(self.data.time); self.gripper_closed = False
        if self.attached: self._detach("released")
        self._advance(controls=10); self.lifecycle_after_action(action)
        return {"action_index": self.action_index, "action": action, "start_time": round(before, 6),
                "end_time": round(float(self.data.time), 6), "gripper_command": "open",
                "contact_present": self.contact_sensor(), "termination_reason": None,
                "low_level_control_sequence": self._active_control_sequence,
                "attempt_lifecycle": self.attempt_lifecycle.snapshot()}


class L3Capture(CanonicalTimeCapture):
    def __init__(self, *args: Any, method: str, l3_case: L3Case, **kwargs: Any) -> None:
        self.method = method
        self.l3_case = l3_case
        self.online_rows: list[dict[str, Any]] = []
        self.decision_rows: list[dict[str, Any]] = []
        self.first_requested_action: dict[str, Any] | None = None
        self.action_execution_active = False
        self.transport_periodic_count = 0
        self.contact_fault_injected = False
        super().__init__(*args, **kwargs)

    def capture_frame(self, *, action_end: bool) -> None:
        super().capture_frame(action_end=action_end)
        frame = self.frame_rows[-1]
        detection = {
            "time": float(frame["time"]), "capture_order": int(frame["capture_order"]),
            "object_centroid": None, "object_area": 0.0, "object_component_count": 0,
            "object_confidence": 0.0, "gripper_centroid": None, "gripper_area": 0.0,
            "gripper_confidence": 0.0, "width": 192, "height": 144,
        }
        if not frame["frame_missing"]:
            result = detect_frame(self.root / str(frame["jpeg_path"]))
            detection.update({key: result.get(key) for key in (
                "object_centroid", "object_area", "object_component_count", "object_confidence",
                "gripper_centroid", "gripper_area", "gripper_confidence", "width", "height",
            )})
        contact = dict(self.contact_rows[-1])
        if self.phase.startswith("transport") and not action_end:
            self.transport_periodic_count += 1
            if self.l3_case.contact_dropout and self.transport_periodic_count == 4:
                contact["contact_present"] = False
                self.contact_fault_injected = True
        merged = {
            **detection, **contact, **self.lifecycle_rows[-1], **self.command_rows[-1],
            **self.request_rows[-1],
        }
        row = validate_online_row(merged)
        row["physical_time_ns"] = int(self.contact_rows[-1]["physical_time_ns"])
        self.online_rows.append(row)
        base = [{key: value for key, value in item.items() if key != "physical_time_ns"}
                for item in self.online_rows]
        raw = _run_o(base, "O_C3")
        records = apply_guard(self.online_rows, raw) if self.method in {CLP3_ID, "RECOVERY_DISABLED"} else raw
        latest = records[-1]
        decision = {
            "time": row["time"], "physical_time_ns": row["physical_time_ns"],
            "capture_order": row["capture_order"], "method": self.method,
            "selected_action": latest.get("selected_action"), "reason_code": latest.get("reason_code"),
            "guard_state": latest.get("guard_state"), "execution_suppressed": self.method == "RECOVERY_DISABLED",
        }
        self.decision_rows.append(decision)
        if (not self.action_execution_active and self.first_requested_action is None
                and decision["selected_action"] in RECOVERY_ACTIONS):
            self.first_requested_action = decision

    def write_l3(self) -> None:
        self.write()
        write_jsonl(self.root / "candidate_input/live_observations.jsonl", self.online_rows)
        write_jsonl(self.root / "candidate_output/live_decisions.jsonl", self.decision_rows)


def _case_adapter(case: L3Case) -> ConfirmationCase:
    return ConfirmationCase(case.case_id, level="strong" if case.force_loss else None,
                            phase_offset_ms=20 if (case.force_loss or case.commanded_release) else None,
                            commanded_release=case.commanded_release)


def _verify_hold(capture: L3Capture) -> dict[str, Any]:
    sim = capture.sim
    capture.set_phase("recovery_hold_settle", "hold")
    for _ in range(20): sim.physics_step()
    capture.set_phase("recovery_hold_verification", "hold")
    start = len(capture.physics_rows)
    for _ in range(HOLD_VERIFY_STEPS): sim.physics_step()
    rows = capture.physics_rows[start:]
    anchor = np.asarray(rows[0]["object_in_gripper_position"], dtype=float) if rows else None
    drift = max((float(np.linalg.norm(np.asarray(row["object_in_gripper_position"]) - anchor))
                 for row in rows), default=float("inf")) if anchor is not None else float("inf")
    passed = bool(len(rows) == HOLD_VERIFY_STEPS and all(
        row["weld_active"] and row["contact_present"] and row["numeric_health"]["passed"] for row in rows)
        and drift <= 0.01)
    return {"passed": passed, "samples": len(rows), "duration_s": HOLD_VERIFY_STEPS / 100,
            "max_relative_drift_m": drift}


def _controlled_move(capture: L3Capture, action: str, target: np.ndarray, controls: int) -> None:
    sim = capture.sim
    sim.action_index += 1; sim._active_control_sequence = []; sim.lifecycle_before_action(action)
    capture.set_phase(action, action)
    sim._advance(target, controls=controls)
    sim.lifecycle_after_action(action); capture.capture_frame(action_end=True)


def _finish_task(capture: L3Capture) -> None:
    sim = capture.sim
    _controlled_move(capture, "lift", sim.data.mocap_pos[0] + np.array([0.0, 0.0, 0.12]), 20)
    _controlled_move(capture, "transport_to_target", np.array([sim.spec.target_x, sim.spec.target_y, 0.80]), 40)
    _controlled_move(capture, "lower", np.array([sim.spec.target_x, sim.spec.target_y,
                                                   0.5 + sim.spec.object_radius * 2.1]), 20)
    capture.perform("open_gripper", "task_release")
    capture.perform("verify", "task_verify")


def _execute_recovery(capture: L3Capture, action: str) -> dict[str, Any]:
    sim = capture.sim
    started = float(sim.data.time)
    loops: list[dict[str, Any]] = []
    capture.action_execution_active = True
    try:
        for loop in range(1, RECOVERY_LOOP_LIMIT + 1):
            controller_action = "retry" if action == "retry_grasp" else "recover"
            sim.action_index += 1; sim._active_control_sequence = []; sim.lifecycle_before_action(controller_action)
            capture.set_phase(f"{action}_relocate", controller_action)
            relocation_error = float("inf")
            for _ in range(40):
                target = sim.object_xyz + np.array([0.0, 0.0, 0.10])
                sim._advance(target, controls=1)
                relocation_error = float(np.linalg.norm(sim.data.mocap_pos[0] - target))
                contact_distance = float(np.linalg.norm(sim.object_xyz - (sim.data.mocap_pos[0] + np.array([0.0, 0.0, -0.13]))))
                if relocation_error <= 0.02 and contact_distance < sim.spec.object_radius + 0.045:
                    sim.gripper_closed = True
                    if not sim.attached: sim._attach()
                    sim.recovered = True; sim._record_event("recovery_achieved")
                    break
            relocation_ok = bool(sim.attached and relocation_error <= 0.02)
            sim.lifecycle_after_action(controller_action); capture.capture_frame(action_end=True)
            if relocation_ok:
                _controlled_move(capture, "lift", sim.data.mocap_pos[0] + np.array([0.0, 0.0, 0.12]), 20)
            hold = _verify_hold(capture) if relocation_ok else {"passed": False, "samples": 0,
                                                                  "duration_s": 0.0,
                                                                  "max_relative_drift_m": None}
            loops.append({"loop": loop, "relocation_error_m": relocation_error,
                          "relocation_ok": relocation_ok, "hold_verification": hold})
            if relocation_ok and hold["passed"]:
                _finish_task(capture)
                return {"success": True, "loops": loops, "loop_count": loop,
                        "duration_s": float(sim.data.time) - started, "failure_stage": None}
        stage = "RELOCATION_ERROR" if loops and not loops[-1]["relocation_ok"] else "REGRASP_ERROR"
        return {"success": False, "loops": loops, "loop_count": len(loops),
                "duration_s": float(sim.data.time) - started, "failure_stage": stage}
    finally:
        capture.action_execution_active = False


def _transport(capture: L3Capture, case: L3Case) -> None:
    sim = capture.sim
    sim.action_index += 1; sim._active_control_sequence = []; sim.lifecycle_before_action("transport_to_target")
    capture.set_phase("transport_pre_event", "transport_to_target")
    target = np.array([sim.spec.target_x, sim.spec.target_y, 0.80])
    midway = (sim.data.mocap_pos[0] + target) / 2.0
    sim._advance(midway, controls=20)
    if case.commanded_release:
        capture.commanded_release = True
        capture.perform("open_gripper", "commanded_release_during_transport")
    elif case.force_loss:
        capture.phase_grid_time = float(sim.data.time)
        capture.set_phase("transport_contact_loss", "transport_to_target")
        sim.disable_weld_for_intervention(); capture.add_event_reference("post_weld_off_pre_force")
        for _ in range(20):
            sim.physics_step()
            if capture.first_requested_action is not None: break
        if capture.first_requested_action is not None:
            sim.lifecycle_after_action("transport_to_target"); capture.capture_frame(action_end=True)
            return
    capture.set_phase("transport_post_event", "transport_to_target")
    sim._advance(target, controls=20)
    sim.lifecycle_after_action("transport_to_target")
    capture.capture_frame(action_end=True)


def run_rollout(root: Path, family_id: str, family_seed: int, rollout_seed: int,
                case: L3Case, method: str) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    sim = L3ClosedLoopTabletop(family_spec(family_id, case.scenario, family_seed, rollout_seed), rollout_seed)
    adapter = _case_adapter(case)
    capture = L3Capture(sim, root, family_id=family_id, family_seed=family_seed,
                        rollout_seed=rollout_seed, case_id=case.case_id,
                        requested_effect="RELEASE_OBJECT" if case.commanded_release else "HOLD_OBJECT",
                        case=adapter, method=method, l3_case=case)
    recovery = {"success": False, "loops": [], "loop_count": 0, "duration_s": None,
                "failure_stage": None}
    try:
        if case.expected_action == "retry_grasp":
            capture.capture_frame(action_end=True)
            capture.perform("approach_object")
            capture.perform("close_gripper")
            capture.perform("verify", "hold_attempt_end")
        else:
            pre = _prehold(capture, 50)
            if not pre["pre_hold_verified"]: raise RuntimeError("PREHOLD_UNVERIFIED")
            _transport(capture, case)

        requested = capture.first_requested_action
        expected = case.expected_action
        trigger_correct = ((expected == "none" and requested is None)
                           or (requested is not None and requested["selected_action"] == expected))
        executed_action = None
        if requested is not None and method != "RECOVERY_DISABLED":
            executed_action = requested["selected_action"]
            recovery = _execute_recovery(capture, executed_action)
        elif expected in RECOVERY_ACTIONS:
            recovery["failure_stage"] = "TRIGGERING_ERROR"
        elif not case.commanded_release:
            _finish_task(capture)

        if case.commanded_release:
            final_task_success = bool(capture.commanded_release and executed_action is None)
        else:
            final_task_success = bool(sim.oracle_snapshot()["goal_stable"])
        false_action = expected == "none" and executed_action in RECOVERY_ACTIONS
        failure_stage = recovery.get("failure_stage")
        if not trigger_correct or false_action: failure_stage = "TRIGGERING_ERROR"
        elif expected in RECOVERY_ACTIONS and recovery.get("success") and not final_task_success:
            failure_stage = "TASK_RECOVERY_ERROR"
        overall_success = bool(final_task_success and trigger_correct and not false_action
                               and (expected == "none" or recovery.get("success")))
        outcome = {
            "schema": "l2rar2_r23_closed_loop_outcome_v1", "family_id": family_id,
            "family_seed": family_seed, "rollout_seed": rollout_seed, "case_id": case.case_id,
            "method": method, "expected_action": expected,
            "candidate_requested_action": requested["selected_action"] if requested else None,
            "candidate_request_time": requested["time"] if requested else None,
            "candidate_request_physical_time_ns": requested["physical_time_ns"] if requested else None,
            "executed_action": executed_action, "trigger_correct": trigger_correct,
            "false_executed_action": false_action, "contact_fault_injected": capture.contact_fault_injected,
            "recovery": recovery, "final_task_success": final_task_success,
            "overall_success": overall_success, "failure_stage": failure_stage,
        }
        capture.write_l3()
        write_json(root / "outcome.json", outcome)
        write_json(root / "metadata.json", {key: outcome[key] for key in (
            "family_id", "family_seed", "rollout_seed", "case_id", "method", "expected_action")})
        write_json(root / "termination.json", {"status": "COMPLETE", "time": float(sim.data.time),
                                                  "physics_steps": capture.physics_step_index})
        return outcome
    except Exception as exc:
        write_json(root / "termination.json", {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
                                                  "time": float(sim.data.time),
                                                  "physics_steps": capture.physics_step_index})
        raise
    finally:
        capture.close()


def _task(args: tuple[Any, ...]) -> dict[str, Any]:
    return run_rollout(Path(args[0]), *args[1:])


def collect(output: Path, workers: int = 2) -> dict[str, Any]:
    if workers not in (1, 2): raise ValueError("WORKERS_MUST_BE_1_OR_2")
    output.mkdir(parents=True, exist_ok=True)
    tasks = []
    for family, family_seed, seed_base in FAMILIES:
        for case_index, case in enumerate(CASES):
            rollout_seed = seed_base + case_index
            for method in METHODS:
                root = output / f"{family}__{case.case_id}__{method}"
                if (root / "termination.json").is_file() and json.loads((root / "termination.json").read_text())["status"] == "COMPLETE":
                    continue
                if root.exists(): raise RuntimeError(f"INCOMPLETE_ROLLOUT_PRESENT:{root}")
                tasks.append((str(root), family, family_seed, rollout_seed, case, method))
    completed, failures = [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        pending, iterator, stop = {}, iter(tasks), False
        for _ in range(min(workers, len(tasks))):
            item = next(iterator, None)
            if item: pending[pool.submit(_task, item)] = item
        while pending:
            done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                item = pending.pop(future)
                try: completed.append(future.result())
                except Exception as exc:
                    failures.append({"rollout": Path(item[0]).name, "error": f"{type(exc).__name__}: {exc}"}); stop = True
                if not stop:
                    nxt = next(iterator, None)
                    if nxt: pending[pool.submit(_task, nxt)] = nxt
    terms = list(output.glob("*/termination.json"))
    passed = len(terms) == 60 and all(json.loads(path.read_text())["status"] == "COMPLETE" for path in terms)
    result = {"schema": "l2rar2_r23_collection_v1", "status": "PASS" if passed else "FAIL",
              "all_60_complete": passed, "newly_completed": len(completed), "failures": failures}
    write_json(output / "collection_status.json", result)
    return result
