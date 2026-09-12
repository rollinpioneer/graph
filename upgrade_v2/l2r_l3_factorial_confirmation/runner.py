from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import Any

import numpy as np

from upgrade_v2.l2r_canonical_time_confirmation.collector import CanonicalTimeCapture
from upgrade_v2.l2r_canonical_time_confirmation.guard import apply_guard
from upgrade_v2.l2r_l3_closed_loop.runner import L3Capture, L3ClosedLoopTabletop, _execute_recovery, _finish_task
from upgrade_v2.l2r_loss_observability_fusion import LossObservabilityFusionV1
from upgrade_v2.l2r_logical_clock_confirmation.io_utils import write_json, write_jsonl
from upgrade_v2.l2r_logical_clock_confirmation.registry import ConfirmationCase
from upgrade_v2.l2r_recovery_supervisor import RecoverySupervisorV2
from upgrade_v2.l2r_rgb_temporal_confirmation.candidate_inputs import validate_online_row
from upgrade_v2.l2r_rgb_temporal_confirmation.evaluation import _run_o
from upgrade_v2.l2r_rgb_temporal_confirmation.rgb_capture import _prehold
from upgrade_v2.visual_refine_l2.dynamic_simulator import family_spec
from upgrade_v2.visual_refine_l2.vision import detect_frame

from . import CLP3_ID
from .registry import ARMS, CASES, FAMILIES, Case, FactorArm


class R25Sim(L3ClosedLoopTabletop):
    def __init__(self, *args: Any, fault: str, **kwargs: Any) -> None:
        self.r25_fault = fault
        self.recovery_call_index = 0
        self.secondary_injected = False
        super().__init__(*args, **kwargs)

    def _capture(self):
        return getattr(getattr(self, "after_physics_step", None), "__self__", None)

    def _advance(self, target=None, controls=4):
        capture = self._capture()
        phase = getattr(capture, "phase", "")
        adjusted = target
        if (self.r25_fault == "transient_relocation" and self.recovery_call_index == 1
                and phase.endswith("_relocate") and target is not None):
            adjusted = np.asarray(target) + np.array([0.25, 0.0, 0.0])
        if (self.r25_fault == "goal_error" and self.recovery_call_index == 1
                and phase == "lower" and target is not None):
            adjusted = np.asarray(target) + np.array([0.14, 0.0, 0.0])
        if (self.r25_fault == "secondary_loss" and self.recovery_call_index == 1
                and phase == "transport_to_target" and self.recovered and not self.secondary_injected):
            self.secondary_injected = True
            if capture is not None: capture.begin_loss_episode(2)
            if self.attached: self._detach("contact_lost")
        return super()._advance(adjusted, controls)


class R25Capture(L3Capture):
    def __init__(self, *args: Any, case: Case, arm: FactorArm, **kwargs: Any) -> None:
        self.r25_case = case
        self.arm = arm
        self.online_rows: list[dict[str, Any]] = []
        self.decision_rows: list[dict[str, Any]] = []
        self.fusion_rows: list[dict[str, Any]] = []
        self.proposal_events: list[dict[str, Any]] = []
        self.fusion = LossObservabilityFusionV1()
        self.loss_episode = 0
        self.actual_false_counts: dict[int, int] = {}
        self.physical_loss_started = False
        self.first_requested_action = None
        self.action_execution_active = False
        self._last_phase = "initial"
        self._last_event_key = None
        super().__init__(*args, method=CLP3_ID,
                         l3_case=type("R25CaseAdapter", (), {"contact_dropout": False})(),
                         case=_adapter(case), **kwargs)

    def begin_loss_episode(self, episode: int) -> None:
        self.loss_episode = episode
        self.physical_loss_started = True
        self.actual_false_counts.setdefault(episode, 0)

    def capture_frame(self, *, action_end: bool) -> None:
        phase_before = self._last_phase
        CanonicalTimeCapture.capture_frame(self, action_end=action_end)
        if phase_before == "recovery_hold_verification" and self.phase != phase_before:
            self.fusion.rearm_after_verified_hold()
        self._last_phase = self.phase
        frame = self.frame_rows[-1]
        result = detect_frame(self.root / str(frame["jpeg_path"])) if not frame["frame_missing"] else {}
        detection = {
            "time": float(frame["time"]), "capture_order": int(frame["capture_order"]),
            "object_centroid": result.get("object_centroid"),
            "object_confidence": result.get("object_confidence", 0.0),
            "gripper_centroid": result.get("gripper_centroid"),
            "gripper_confidence": result.get("gripper_confidence", 0.0),
            "width": result.get("width", 192), "height": result.get("height", 144),
        }
        contact = dict(self.contact_rows[-1])
        actual = bool(contact["contact_present"])
        if self.physical_loss_started and not actual:
            self.actual_false_counts[self.loss_episode] = self.actual_false_counts.get(self.loss_episode, 0) + 1
        false_count = self.actual_false_counts.get(self.loss_episode, 0)
        if self.r25_case.fault == "contact_absence" and self.loss_episode == 1 and not actual and false_count <= 2:
            contact["contact_present"] = None
        elif self.r25_case.fault == "rgb_detach" and self.loss_episode == 1:
            contact["contact_present"] = True
        elif self.r25_case.fault == "secondary_loss" and self.loss_episode == 2 and not actual and false_count <= 2:
            contact["contact_present"] = None
        merged = {**detection, **contact, **self.lifecycle_rows[-1], **self.command_rows[-1], **self.request_rows[-1]}
        row = validate_online_row(merged)
        row["physical_time_ns"] = int(self.contact_rows[-1]["physical_time_ns"])
        self.online_rows.append(row)
        raw = _run_o([{key: value for key, value in item.items() if key != "physical_time_ns"}
                      for item in self.online_rows], "O_C3")
        clp3 = apply_guard(self.online_rows, raw)[-1]
        fallback = self.fusion.step(row)
        self.fusion_rows.append(fallback)
        clp3_action = clp3.get("selected_action")
        fallback_action = fallback.get("selected_action") if self.arm.fusion_enabled else "none"
        if clp3_action in {"retry_grasp", "recover_object"}:
            action, source, reason = clp3_action, "CLP3", clp3.get("reason_code")
        elif fallback_action in {"retry_grasp", "recover_object"}:
            action, source, reason = fallback_action, fallback.get("proposal_source"), fallback.get("reason_code")
        else:
            action, source, reason = "none", None, clp3.get("reason_code")
        decision = {
            "time": row["time"], "physical_time_ns": row["physical_time_ns"],
            "capture_order": row["capture_order"], "arm_id": self.arm.arm_id,
            "selected_action": action, "proposal_source": source, "reason_code": reason,
            "clp3_selected_action": clp3_action, "clp3_reason_code": clp3.get("reason_code"),
            "clp3_guard_state": clp3.get("guard_state"),
            "fusion_selected_action": fallback.get("selected_action"),
            "fusion_reason_code": fallback.get("reason_code"),
            "fusion_source": fallback.get("proposal_source"),
        }
        self.decision_rows.append(decision)
        if action in {"retry_grasp", "recover_object"}:
            key = (row["physical_time_ns"], row["capture_order"], source)
            if key != self._last_event_key:
                event = dict(decision)
                self.proposal_events.append(event)
                if not self.action_execution_active and self.first_requested_action is None:
                    self.first_requested_action = event
                self._last_event_key = key

    def next_proposal_after(self, physical_time_ns: int, consumed: set[tuple]) -> dict[str, Any] | None:
        for event in self.proposal_events:
            key = (event["physical_time_ns"], event["capture_order"], event["proposal_source"])
            if event["physical_time_ns"] > physical_time_ns and key not in consumed:
                consumed.add(key)
                return event
        return None

    def write_r25(self) -> None:
        self.write()
        write_jsonl(self.root / "candidate_input/live_observations.jsonl", self.online_rows)
        write_jsonl(self.root / "candidate_output/factor_decisions.jsonl", self.decision_rows)
        write_jsonl(self.root / "candidate_output/fusion_diagnostics.jsonl", self.fusion_rows)


def _adapter(case: Case) -> ConfirmationCase:
    return ConfirmationCase(case.case_id, commanded_release=case.commanded_release)


def _initial_event(capture: R25Capture, case: Case) -> None:
    sim = capture.sim
    sim.action_index += 1
    sim._active_control_sequence = []
    sim.lifecycle_before_action("transport_to_target")
    capture.set_phase("transport_to_target", "transport_to_target")
    target = np.array([sim.spec.target_x, sim.spec.target_y, 0.80])
    sim._advance((sim.data.mocap_pos[0] + target) / 2, controls=20)
    if case.commanded_release:
        capture.commanded_release = True
        capture.perform("open_gripper", "commanded_release_during_transport")
        return
    capture.begin_loss_episode(1)
    sim.disable_weld_for_intervention()
    capture.add_event_reference("physical_contact_loss_started")
    if case.fault == "partial_slip":
        for _ in range(5): sim.physics_step()
        sim._attach()
        capture.physical_loss_started = False
        capture.add_event_reference("partial_slip_contact_restored")
        for _ in range(100): sim.physics_step()
    else:
        sim.set_object_force(np.array([27.0, 27.0, -5.4]))
        for _ in range(20): sim.physics_step()
        sim.clear_object_force()
        for _ in range(150):
            sim.physics_step()
            if capture.first_requested_action is not None: break
    sim.lifecycle_after_action("transport_to_target")
    capture.capture_frame(action_end=True)


def _old_supervision(capture: R25Capture, requested: dict[str, Any]) -> dict[str, Any]:
    sim = capture.sim
    sim.recovery_call_index = 1
    capture.action_execution_active = True
    try:
        execution = _execute_recovery(capture, requested["selected_action"])
    finally:
        capture.action_execution_active = False
    goal = bool(sim.oracle_snapshot()["goal_stable"])
    stage = execution.get("failure_stage") if not execution.get("success") else (None if goal else "TASK_RECOVERY_ERROR")
    return {"success": bool(execution.get("success") and goal), "failure_stage": stage,
            "outer_cycles": 1, "execution_results": [execution],
            "goal_verifications": [goal], "rearmed_loss_episodes": 0}


def _new_supervision(capture: R25Capture, requested: dict[str, Any]) -> dict[str, Any]:
    sim = capture.sim
    consumed = {(requested["physical_time_ns"], requested["capture_order"], requested["proposal_source"])}
    starts: dict[int, int] = {}

    def execute(action: str, cycle: int) -> dict[str, Any]:
        sim.recovery_call_index = cycle
        starts[cycle] = int(capture.online_rows[-1]["physical_time_ns"])
        capture.action_execution_active = True
        try: return _execute_recovery(capture, action)
        finally: capture.action_execution_active = False

    def next_loss(cycle: int) -> str | None:
        event = capture.next_proposal_after(starts[cycle], consumed)
        return event["selected_action"] if event else None

    return RecoverySupervisorV2().run(
        initial_action=requested["selected_action"], execute_recovery=execute,
        verify_goal=lambda cycle: bool(sim.oracle_snapshot()["goal_stable"]),
        next_loss_action=next_loss,
        rearm_after_verified_hold=lambda cycle: None,
    )


def run_rollout(root: Path, family: str, family_seed: int, rollout_seed: int,
                case: Case, arm: FactorArm) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    sim = R25Sim(family_spec(family, "normal_pick_place", family_seed, rollout_seed),
                 rollout_seed, fault=case.fault)
    capture = R25Capture(sim, root, family_id=family, family_seed=family_seed,
                         rollout_seed=rollout_seed, case_id=case.case_id,
                         requested_effect="RELEASE_OBJECT" if case.commanded_release else "HOLD_OBJECT",
                         case=case, arm=arm)
    try:
        prehold = _prehold(capture, 50)
        if not prehold["pre_hold_verified"]: raise RuntimeError("PREHOLD_UNVERIFIED")
        _initial_event(capture, case)
        requested = capture.first_requested_action
        supervision = None
        if requested is not None:
            supervision = (_new_supervision(capture, requested) if arm.supervisor_v2_enabled
                           else _old_supervision(capture, requested))
        elif not case.recovery_required and not case.commanded_release:
            _finish_task(capture)
        final_success = (bool(case.commanded_release and requested is None) if case.commanded_release
                         else bool(sim.oracle_snapshot()["goal_stable"]))
        false_action = bool(not case.recovery_required and requested is not None)
        if case.recovery_required and requested is None:
            failure = "SIGNAL_NOT_OBSERVED"
        elif false_action:
            failure = "TRIGGERING_ERROR"
        elif supervision is not None and not supervision["success"]:
            failure = supervision["failure_stage"] or "TASK_RECOVERY_ERROR"
        elif case.recovery_required and not final_success:
            failure = "TASK_RECOVERY_ERROR"
        else:
            failure = None
        executions = supervision["execution_results"] if supervision else []
        outcome = {
            "schema": "l2rar2_r25_factorial_outcome_v1",
            "family_id": family, "family_seed": family_seed, "rollout_seed": rollout_seed,
            "case_id": case.case_id, "arm_id": arm.arm_id,
            "fusion_enabled": arm.fusion_enabled,
            "supervisor_v2_enabled": arm.supervisor_v2_enabled,
            "recovery_required": case.recovery_required,
            "signal_exposed": requested is not None,
            "proposal_source": requested.get("proposal_source") if requested else None,
            "proposal_time_ns": requested.get("physical_time_ns") if requested else None,
            "signal_not_observed": failure == "SIGNAL_NOT_OBSERVED",
            "candidate_error": False,
            "false_executed_action": false_action,
            "actual_recovery_success": bool(supervision and any(item.get("success") for item in executions)),
            "supervision_success": bool(supervision and supervision["success"]),
            "outer_recovery_cycles": int(supervision["outer_cycles"] if supervision else 0),
            "inner_recovery_loops": sum(int(item.get("loop_count", 0)) for item in executions),
            "rearmed_loss_episodes": int(supervision["rearmed_loss_episodes"] if supervision else 0),
            "goal_verifications": supervision["goal_verifications"] if supervision else [],
            "recovery_executions": executions,
            "final_task_success": final_success,
            "failure_stage": failure,
            "task_duration_s": float(sim.data.time),
        }
        capture.write_r25()
        write_json(root / "metadata.json", {key: outcome[key] for key in
                   ("family_id", "family_seed", "rollout_seed", "case_id", "arm_id")})
        write_json(root / "outcome.json", outcome)
        write_json(root / "termination.json", {"status": "COMPLETE", "time": float(sim.data.time)})
        return outcome
    finally:
        capture.close()


def _task(args):
    return run_rollout(Path(args[0]), *args[1:])


def collect(output: Path, workers: int = 2) -> dict[str, Any]:
    if workers not in (1, 2): raise ValueError("WORKERS_MUST_BE_1_OR_2")
    output.mkdir(parents=True, exist_ok=True)
    tasks = []
    for family, family_seed, seed_base in FAMILIES:
        for case_index, case in enumerate(CASES):
            for arm_index, arm in enumerate(ARMS):
                root = output / f"{family}__{case.case_id}__{arm.arm_id}"
                if not root.exists():
                    tasks.append((str(root), family, family_seed, seed_base + case_index * 10 + arm_index, case, arm))
    completed, failures = [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        pending, iterator, stop = {}, iter(tasks), False
        for _ in range(min(workers, len(tasks))):
            task = next(iterator, None)
            if task is not None: pending[pool.submit(_task, task)] = task
        while pending:
            done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                try: completed.append(future.result())
                except Exception as exc:
                    failures.append({"rollout": Path(task[0]).name,
                                     "error": f"{type(exc).__name__}:{exc}"})
                    stop = True
                if not stop:
                    following = next(iterator, None)
                    if following is not None: pending[pool.submit(_task, following)] = following
    count = len(list(output.glob("*/outcome.json")))
    result = {"schema": "l2rar2_r25_collection_v1",
              "status": "PASS" if count == 112 and not failures else "FAIL",
              "rollouts": count, "newly_completed": len(completed), "failures": failures}
    write_json(output / "collection_status.json", result)
    return result
