from __future__ import annotations

import concurrent.futures
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from upgrade_v2.l2r_forced_drop.physical_reference import evaluate_loss_trace
from upgrade_v2.l2r_l3_factorial_confirmation.registry import FactorArm
from upgrade_v2.l2r_l3_factorial_confirmation.runner import R25Capture, R25Sim
from upgrade_v2.l2r_logical_clock_confirmation.io_utils import write_json
from upgrade_v2.l2r_rgb_temporal_confirmation.rgb_capture import _prehold
from upgrade_v2.visual_refine_l2.dynamic_simulator import family_spec


@dataclass(frozen=True)
class ShadowCase:
    case_id: str
    mode: str
    expected_loss: bool = False
    commanded_release: bool = False
    force_world_n: tuple[float, float, float] | None = None
    force_steps: int = 0
    contact_masked: bool = False


FAMILIES = tuple(
    (f"L2RAR2_R27_SHADOW_{index:02d}_{902000 + index}", 902000 + index, 90300000 + index * 100)
    for index in range(4)
)
CASES = (
    ShadowCase("R27S1_stable_hold", "stable_hold"),
    ShadowCase("R27S2_supported_recovery_after_slip", "supported_recovery"),
    ShadowCase("R27S3_slow_detachment", "slow_detachment", True, False, (9.0, 9.0, -1.8), 60),
    ShadowCase("R27S4_fast_detachment", "fast_detachment", True, False, (27.0, 27.0, -5.4), 20),
    ShadowCase("R27S5_contact_occluded_true_detachment", "contact_occluded", True, False, (27.0, 27.0, -5.4), 20, True),
    ShadowCase("R27S6_active_release", "active_release", False, True),
)
SHADOW_ARM = FactorArm("R27_SHADOW_ONLY", False, False)
POST_INTERVENTION_STEPS = 150


def design() -> dict[str, Any]:
    return {
        "schema": "l2rar2_r27_shadow_replenishment_design_v2",
        "status": "PREREGISTERED_FOR_DEVELOPMENT_COLLECTION",
        "families": [
            {
                "family_id": family,
                "family_seed": family_seed,
                "rollout_base": rollout_base,
                "cases": [asdict(case) for case in CASES],
            }
            for family, family_seed, rollout_base in FAMILIES
        ],
        "total_rollouts": len(FAMILIES) * len(CASES),
        "post_intervention_observation_s": 1.50,
        "candidate_mode": "SHADOW_ONLY",
        "candidate_controls_recovery": False,
        "candidate_can_terminate": False,
        "seed_replacement": False,
        "confirmation_dataset": False,
        "protocol_basis": {
            "fast_force_world_n": [27.0, 27.0, -5.4],
            "fast_force_steps": 20,
            "slow_force_world_n": [9.0, 9.0, -1.8],
            "slow_force_steps": 60,
            "physics_step_s": 0.01,
            "camera": "frozen front 192x144 JPEG capture path",
        },
    }


def _reference_after_prehold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    start = next((index for index, row in enumerate(rows) if row.get("pre_hold_verified")), None)
    if start is None:
        return {"state": "PREHOLD_UNVERIFIED", "physical_loss_confirmed": False}
    return evaluate_loss_trace(rows[start:], pre_hold_verified=True)


def _run_intervention(capture: R25Capture, case: ShadowCase) -> dict[str, Any]:
    sim = capture.sim
    sim.action_index += 1
    sim._active_control_sequence = []
    sim.lifecycle_before_action("shadow_intervention")
    capture.set_phase("shadow_transport_pre_intervention", "transport_to_target")
    target = np.array([sim.spec.target_x, sim.spec.target_y, 0.80])
    sim._advance((sim.data.mocap_pos[0] + target) / 2.0, controls=20)
    start_ns = int(capture.physics_step_index) * 10_000_000

    if case.mode == "stable_hold":
        capture.set_phase("shadow_stable_hold_intervention", "shadow_intervention")
        for _ in range(20):
            sim.physics_step()
    elif case.mode == "supported_recovery":
        capture.begin_loss_episode(1)
        capture.set_phase("shadow_supported_slip", "shadow_intervention")
        sim.disable_weld_for_intervention()
        capture.add_event_reference("shadow_supported_slip_started")
        for _ in range(5):
            sim.physics_step()
        sim._attach()
        capture.physical_loss_started = False
        capture.add_event_reference("shadow_support_restored")
        for _ in range(15):
            sim.physics_step()
    elif case.force_world_n is not None:
        capture.begin_loss_episode(1)
        capture.set_phase(f"shadow_{case.mode}", "shadow_intervention")
        sim.disable_weld_for_intervention()
        capture.add_event_reference("shadow_physical_contact_loss_started")
        sim.set_object_force(np.asarray(case.force_world_n, dtype=float))
        for _ in range(case.force_steps):
            sim.physics_step()
        sim.clear_object_force()
    elif case.commanded_release:
        capture.commanded_release = True
        capture.set_phase("shadow_active_release", "shadow_intervention")
        capture.perform("open_gripper", "shadow_commanded_release")
    else:
        raise ValueError(f"UNKNOWN_SHADOW_MODE:{case.mode}")

    end_ns = int(capture.physics_step_index) * 10_000_000
    capture.add_event_reference("shadow_intervention_ended")
    capture.set_phase("shadow_post_intervention_observation", "shadow_observation")
    for _ in range(POST_INTERVENTION_STEPS):
        sim.physics_step()
    saved_end_ns = int(capture.physics_step_index) * 10_000_000
    sim.lifecycle_after_action("shadow_intervention")
    capture.capture_frame(action_end=True)
    return {
        "intervention_start_ns": start_ns,
        "intervention_end_ns": end_ns,
        "saved_observation_end_ns": saved_end_ns,
        "post_intervention_observation_ns": saved_end_ns - end_ns,
        "candidate_control_executions": 0,
        "candidate_terminated_early": False,
        "force_world_n": list(case.force_world_n) if case.force_world_n else None,
        "force_steps": case.force_steps,
    }


def run_rollout(
    root: Path, family: str, family_seed: int, rollout_seed: int, case: ShadowCase,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    sim = R25Sim(
        family_spec(family, "normal_pick_place", family_seed, rollout_seed),
        rollout_seed,
        fault="rgb_detach" if case.contact_masked else "none",
    )
    adapter = type(
        "R27ShadowCaseAdapter",
        (),
        {
            "case_id": case.case_id,
            "fault": "rgb_detach" if case.contact_masked else "none",
            "recovery_required": case.expected_loss,
            "commanded_release": case.commanded_release,
        },
    )()
    capture = R25Capture(
        sim, root, family_id=family, family_seed=family_seed,
        rollout_seed=rollout_seed, case_id=case.case_id,
        requested_effect="RELEASE_OBJECT" if case.commanded_release else "HOLD_OBJECT",
        case=adapter, arm=SHADOW_ARM,
    )
    try:
        prehold = _prehold(capture, 50)
        if not prehold["pre_hold_verified"]:
            raise RuntimeError("PREHOLD_UNVERIFIED")
        intervention = _run_intervention(capture, case)
        capture.write_r25()
        reference = (_reference_after_prehold(capture.physics_rows)
                     if not case.commanded_release
                     else {"state": "COMMANDED_RELEASE", "physical_loss_confirmed": False})
        physical_loss = bool(reference.get("physical_loss_confirmed"))
        outcome = {
            "schema": "l2rar2_r27_shadow_replenishment_outcome_v1",
            "family_id": family,
            "family_seed": family_seed,
            "rollout_seed": rollout_seed,
            "case_id": case.case_id,
            "arm_id": SHADOW_ARM.arm_id,
            "intended_loss": case.expected_loss,
            "physical_loss_exists": physical_loss,
            "commanded_release": case.commanded_release,
            "candidate_requested_action": (capture.first_requested_action or {}).get("selected_action"),
            "proposal_time_ns": (capture.first_requested_action or {}).get("physical_time_ns"),
            "recovery_required": physical_loss,
            "executed_recovery_cycles": 0,
            "recovery_executions": [],
            "candidate_control_executions": 0,
            "candidate_terminated_early": False,
            "post_intervention_observation_ns": intervention["post_intervention_observation_ns"],
            "reference_state": reference.get("state"),
        }
        write_json(root / "metadata.json", {
            "family_id": family, "family_seed": family_seed,
            "rollout_seed": rollout_seed, "case_id": case.case_id,
            "arm_id": SHADOW_ARM.arm_id,
        })
        write_json(root / "outcome.json", outcome)
        write_json(root / "reference/shadow_intervention.json", intervention)
        write_json(root / "reference/shadow_physical_reference.json", reference)
        write_json(root / "termination.json", {
            "status": "COMPLETE", "time": float(sim.data.time),
            "physics_steps": capture.physics_step_index,
        })
        return outcome
    except Exception as exc:
        write_json(root / "termination.json", {
            "status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
            "time": float(sim.data.time), "physics_steps": capture.physics_step_index,
        })
        raise
    finally:
        capture.close()


def _task(args: tuple[Any, ...]) -> dict[str, Any]:
    return run_rollout(Path(args[0]), *args[1:])


def collect(output: Path, workers: int = 2) -> dict[str, Any]:
    if workers not in (1, 2):
        raise ValueError("WORKERS_MUST_BE_1_OR_2")
    output.mkdir(parents=True, exist_ok=True)
    tasks = []
    for family, family_seed, rollout_base in FAMILIES:
        for case_index, case in enumerate(CASES):
            root = output / f"{family}__{case.case_id}__{SHADOW_ARM.arm_id}"
            termination = root / "termination.json"
            if termination.is_file() and json.loads(termination.read_text(encoding="utf-8")).get("status") == "COMPLETE":
                continue
            if root.exists():
                raise RuntimeError(f"INCOMPLETE_ROLLOUT_PRESENT:{root}")
            tasks.append((str(root), family, family_seed, rollout_base + case_index, case))

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        pending: dict[concurrent.futures.Future, tuple[Any, ...]] = {}
        iterator = iter(tasks)
        for _ in range(min(workers, len(tasks))):
            item = next(iterator, None)
            if item:
                pending[pool.submit(_task, item)] = item
        stop = False
        while pending:
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                item = pending.pop(future)
                try:
                    completed.append(future.result())
                except Exception as exc:
                    failures.append({
                        "rollout": Path(item[0]).name,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    stop = True
                if not stop:
                    following = next(iterator, None)
                    if following:
                        pending[pool.submit(_task, following)] = following

    terms = list(output.glob("*/termination.json"))
    all_complete = len(terms) == len(FAMILIES) * len(CASES) and all(
        json.loads(path.read_text(encoding="utf-8")).get("status") == "COMPLETE"
        for path in terms
    )
    outcomes = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in output.glob("*/outcome.json")
    ]
    result = {
        "schema": "l2rar2_r27_shadow_replenishment_collection_v1",
        "status": "PASS" if all_complete and not failures else "FAIL",
        "rollouts": len(terms),
        "newly_completed": len(completed),
        "physical_executions": len(terms),
        "candidate_control_executions": sum(int(row.get("candidate_control_executions", 0)) for row in outcomes),
        "candidate_early_terminations": sum(bool(row.get("candidate_terminated_early")) for row in outcomes),
        "all_post_intervention_observation_ns": sorted({
            int(row.get("post_intervention_observation_ns", 0))
            for row in outcomes
        }),
        "failures": failures,
    }
    write_json(output / "collection_status.json", result)
    return result
