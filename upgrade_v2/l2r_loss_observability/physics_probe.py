"""Bounded, read-only instrumentation replay for the R11 loss audit.

The probe executes the already locked R2 programs for one root and four
cases.  It does not add actions, steps, randomness, or a new family.  The
instrumented run is paired with an ordinary run and is usable for mechanism
claims only when their action/event/state traces agree.
"""
from __future__ import annotations

import copy
import json
import types
from pathlib import Path
from typing import Any

import numpy as np

from upgrade_v2.l2r_hold_evidence.probe_adapter import perform
from upgrade_v2.l2r_task_context.collector import CASES
from upgrade_v2.l2r_task_context.io import read_csv, read_json, read_jsonl, write_csv, write_json
from upgrade_v2.visual_refine_l2.dynamic_simulator import family_spec
from upgrade_v2.visual_refine_l2.renderer import TabletopRenderer
from upgrade_v2.visual_refine_l2.repaired_simulator import AttachRelposeDynamicTabletop

from .contracts import MAX_PHYSICAL_EXECUTIONS, PHYSICS_PROBE_CASES, PHYSICS_PROBE_VERSION


class _MujocoProxy:
    """Delegate every MuJoCo call except mj_step, which is logged after it."""

    def __init__(self, module: Any, callback: Any) -> None:
        self._module = module
        self._callback = callback

    def mj_step(self, model: Any, data: Any) -> None:
        self._module.mj_step(model, data)
        self._callback()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)


def _name(model: Any, obj_type: Any, index: int) -> str:
    try:
        return str(model.names[model.name_adr[obj_type, index]])
    except Exception:
        try:
            return str(model.joint(index))
        except Exception:
            return "unknown"


def _object_geom_ids(sim: Any) -> set[int]:
    mujoco = sim.mujoco._module if isinstance(sim.mujoco, _MujocoProxy) else sim.mujoco
    object_body = int(mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "object"))
    return {index for index in range(sim.model.ngeom) if int(sim.model.geom_bodyid[index]) == object_body}


def _contact_rows(sim: Any, object_geom_ids: set[int]) -> list[dict[str, Any]]:
    mujoco = sim.mujoco._module if isinstance(sim.mujoco, _MujocoProxy) else sim.mujoco
    rows: list[dict[str, Any]] = []
    force = np.zeros(6, dtype=np.float64)
    for contact_id in range(int(sim.data.ncon)):
        contact = sim.data.contact[contact_id]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if geom1 not in object_geom_ids and geom2 not in object_geom_ids:
            continue
        other = geom2 if geom1 in object_geom_ids else geom1
        other_name = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, other) or "unknown"
        other_body_id = int(sim.model.geom_bodyid[other])
        other_body = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_BODY, other_body_id) or "unknown"
        mujoco.mj_contactForce(sim.model, sim.data, contact_id, force)
        rows.append({
            "contact_id": contact_id,
            "object_geom": mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, geom1 if geom1 in object_geom_ids else geom2) or "object_geom",
            "other_geom": other_name,
            "other_body": other_body,
            "distance_m": float(contact.dist),
            "force_contact_frame": [float(value) for value in force],
            "is_hand_support": other_body == "gripper" or other_name in {"palm", "finger_left", "finger_right"},
            "is_external_support": other_body in {"world", "target", "obstacle"} or other_name in {"table", "floor", "target_geom"},
        })
    return rows


def _state(sim: Any, object_geom_ids: set[int], writeback_count: int) -> dict[str, Any]:
    return {
        "time": float(sim.data.time),
        "phase": "physics_step",
        "weld_active": bool(sim.data.eq_active[sim.weld_id]),
        "attached": bool(sim.attached),
        "eq_active": bool(sim.data.eq_active[sim.weld_id]),
        "object_qpos": [float(value) for value in sim.data.qpos[sim.object_qpos:sim.object_qpos + 7]],
        "object_qvel": [float(value) for value in sim.data.qvel[sim.object_dof:sim.object_dof + 6]],
        "mocap_position": [float(value) for value in sim.data.mocap_pos[0]],
        "mocap_quaternion": [float(value) for value in sim.data.mocap_quat[0]],
        "post_detach_writeback_count": int(writeback_count),
        "contacts": _contact_rows(sim, object_geom_ids),
        "ncon": int(sim.data.ncon),
    }


def _ordinary_replay(spec: Any, seed: int, program: list[str], variant_index: int, render_callbacks: bool = False) -> dict[str, Any]:
    sim = AttachRelposeDynamicTabletop(spec, seed)
    action_rows: list[dict[str, Any]] = []
    renderer = TabletopRenderer(sim.model) if render_callbacks else None
    if renderer is not None:
        sim._r1_control_callback = lambda current, control: renderer.render(current.data, "front", spec.camera_jitter)
        sim._r1_action_end_callback = lambda current, action, index, result: renderer.render(current.data, "front", spec.camera_jitter)
    try:
        for action in program:
            result = perform(sim, action, variant_index)
            action_rows.append({
                "action": action,
                "action_index": int(result["action_index"]),
                "start_time": float(result["start_time"]),
                "end_time": float(result["end_time"]),
                "contact_present": bool(result["contact_present"]),
                "controls": copy.deepcopy(result.get("low_level_control_sequence", [])),
                "events": copy.deepcopy(sim.events),
                "qpos": [float(value) for value in sim.data.qpos],
                "qvel": [float(value) for value in sim.data.qvel],
                "object_xyz": [float(value) for value in sim.object_xyz],
                "gripper_xyz": [float(value) for value in sim.data.mocap_pos[0]],
            })
    finally:
        if renderer is not None:
            renderer.close()
    return {"actions": action_rows, "events": copy.deepcopy(sim.events), "final_qpos": [float(value) for value in sim.data.qpos], "final_qvel": [float(value) for value in sim.data.qvel]}


def _instrumented_replay(spec: Any, seed: int, program: list[str], variant_index: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sim = AttachRelposeDynamicTabletop(spec, seed)
    object_geom_ids = _object_geom_ids(sim)
    trace: list[dict[str, Any]] = []
    writeback_count = 0
    active_action = "before_program"

    original_set = sim._set_object_xyz

    def counted_set(self: Any, xyz: Any) -> None:
        nonlocal writeback_count
        writeback_count += 1
        original_set(xyz)

    sim._set_object_xyz = types.MethodType(counted_set, sim)

    def after_step() -> None:
        trace.append({"action": active_action, "physics_step_index": len(trace), **_state(sim, object_geom_ids, writeback_count)})

    sim.mujoco = _MujocoProxy(sim.mujoco, after_step)
    action_rows: list[dict[str, Any]] = []
    for action in program:
        active_action = action
        result = perform(sim, action, variant_index)
        action_rows.append({
            "action": action,
            "action_index": int(result["action_index"]),
            "start_time": float(result["start_time"]),
            "end_time": float(result["end_time"]),
            "contact_present": bool(result["contact_present"]),
            "controls": copy.deepcopy(result.get("low_level_control_sequence", [])),
            "events": copy.deepcopy(sim.events),
            "qpos": [float(value) for value in sim.data.qpos],
            "qvel": [float(value) for value in sim.data.qvel],
            "object_xyz": [float(value) for value in sim.object_xyz],
            "gripper_xyz": [float(value) for value in sim.data.mocap_pos[0]],
        })
    return ({"actions": action_rows, "events": copy.deepcopy(sim.events), "final_qpos": [float(value) for value in sim.data.qpos], "final_qvel": [float(value) for value in sim.data.qvel]}, trace)


def _same_numbers(left: list[float], right: list[float], atol: float = 1e-12) -> bool:
    return bool(np.array_equal(np.asarray(left), np.asarray(right)) or np.allclose(left, right, rtol=0.0, atol=atol))


def _compare(ordinary: dict[str, Any], instrumented: dict[str, Any]) -> dict[str, Any]:
    action_equal = len(ordinary["actions"]) == len(instrumented["actions"])
    state_equal = action_equal
    event_equal = ordinary["events"] == instrumented["events"]
    if action_equal:
        for left, right in zip(ordinary["actions"], instrumented["actions"]):
            action_equal = action_equal and left["action"] == right["action"] and left["contact_present"] == right["contact_present"] and left["start_time"] == right["start_time"] and left["end_time"] == right["end_time"]
            state_equal = state_equal and _same_numbers(left["qpos"], right["qpos"]) and _same_numbers(left["qvel"], right["qvel"])
    return {"actions_equal": action_equal, "events_equal": event_equal, "states_equal": state_equal, "equivalent": bool(action_equal and event_equal and state_equal)}


def _float_equal(left: Any, right: Any, atol: float = 1e-12) -> bool:
    try:
        return bool(np.allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float), rtol=0.0, atol=atol))
    except (TypeError, ValueError):
        return left == right


def _control_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    scalar_keys = ("control_step", "physics_steps", "gripper_command")
    if any(left.get(key) != right.get(key) for key in scalar_keys):
        return False
    return (
        _float_equal(left.get("start_time"), right.get("start_time"))
        and _float_equal(left.get("end_time"), right.get("end_time"))
        and _float_equal(left.get("mocap_position"), right.get("mocap_position"), atol=2e-9)
    )


def _cached_replay_equivalence(data_root: Path, rollout_id: str, case_id: str, ordinary: dict[str, Any], render_callbacks: bool = False) -> dict[str, Any]:
    """Compare the reconstructed ordinary run to the exact cached action stream."""
    manifest_rows = read_csv(data_root / "rollout_manifest.csv")
    meta = next(row for row in manifest_rows if row["rollout_id"] == rollout_id)
    path = Path(meta["path"])
    cached_actions = read_csv(path / "actions.csv")
    cached_controls = read_jsonl(path / "low_level_controls.jsonl")
    cached_events = read_jsonl(path / "events.jsonl")
    cached_oracle = read_csv(path / "oracle_timeline.csv")
    action_equal = len(cached_actions) == len(ordinary["actions"])
    controls_equal = len(cached_controls) == len(ordinary["actions"])
    geometry_equal = True
    if action_equal:
        for generated, cached in zip(ordinary["actions"], cached_actions):
            action_equal = action_equal and (
                generated["action"] == cached["action"]
                and generated["action_index"] == int(cached["action_index"])
                and _float_equal(generated["start_time"], float(cached["start_time"]))
                and _float_equal(generated["end_time"], float(cached["end_time"]))
            )
    if controls_equal:
        for generated, cached in zip(ordinary["actions"], cached_controls):
            generated_controls = generated.get("controls", [])
            cached_controls_for_action = cached.get("controls", [])
            controls_equal = controls_equal and len(generated_controls) == len(cached_controls_for_action)
            if controls_equal:
                controls_equal = all(_control_equal(left, right) for left, right in zip(generated_controls, cached_controls_for_action))
    cached_action_end = [row for row in cached_oracle if row.get("phase") == "action_end"]
    geometry_equal = len(cached_action_end) == len(ordinary["actions"])
    if geometry_equal:
        for generated, cached in zip(ordinary["actions"], cached_action_end):
            object_xyz = json.loads(cached["object_xyz"])
            gripper_xyz = json.loads(cached["gripper_xyz"])
            geometry_equal = geometry_equal and _float_equal(generated["object_xyz"], object_xyz) and _float_equal(generated["gripper_xyz"], gripper_xyz)
    events_equal = cached_events == ordinary["events"]
    result = {
        "rollout_id": rollout_id,
        "case_id": case_id,
        "cached_action_rows": len(cached_actions),
        "generated_action_rows": len(ordinary["actions"]),
        "cached_control_rows": len(cached_controls),
        "action_stream_equal": action_equal,
        "low_level_controls_equal": controls_equal,
        "action_end_geometry_equal": geometry_equal,
        "event_stream_equal": events_equal,
        "equivalent": bool(action_equal and controls_equal and geometry_equal and events_equal),
        "comparison_tolerance": {"scalar_seconds": 1e-12, "mocap_position": 2e-9, "geometry": 1e-12},
        "source_path": str(path.resolve()),
        "render_callbacks": render_callbacks,
    }
    return result


def _mechanism_summary(trace: list[dict[str, Any]], loss_time: float | None) -> dict[str, Any]:
    if not trace:
        return {"physical_loss_status": "insufficient_data", "reason": "empty_probe_trace"}
    if loss_time is None:
        return {"physical_loss_status": "not_applicable", "physical_loss_reason": "no_contact_lost_event_in_control_program", "post_detach_hand_contact_samples": None, "post_detach_external_contact_samples": None, "post_detach_writeback_delta": None, "post_detach_separation_dynamics_observed": None, "max_post_detach_object_speed_mps": None}
    post = [row for row in trace if row["time"] >= loss_time - 1e-9]
    hand = [contact for row in post for contact in row["contacts"] if contact["is_hand_support"]]
    external = [contact for row in post for contact in row["contacts"] if contact["is_external_support"]]
    writes_at_boundary = max((row["post_detach_writeback_count"] for row in trace if row["time"] <= loss_time + 1e-9), default=0)
    writes_after = max((row["post_detach_writeback_count"] for row in trace if row["time"] > loss_time + 1e-9), default=writes_at_boundary) - writes_at_boundary
    separated = any(not row["weld_active"] and np.linalg.norm(np.asarray(row["object_qvel"][:3])) > 1e-6 for row in post)
    # Contact force is stronger evidence of continued support than a small
    # velocity spike.  A weld-off event with persistent finger/palm support
    # is therefore not accepted as a task-level loss.
    if hand:
        status = "not_verified"
        reason = "post_detach_hand_contact_support_persists_after_weld_off"
    elif separated:
        status = "verified"
        reason = "post_detach_object_dynamics_observed_after_weld_off"
    else:
        status = "insufficient_data"
        reason = "no_hand_support_or_separation_dynamics_sufficient_for_classification"
    return {
        "physical_loss_status": status,
        "physical_loss_reason": reason,
        "post_detach_hand_contact_samples": len(hand),
        "post_detach_external_contact_samples": len(external),
        "post_detach_writeback_delta": writes_after,
        "post_detach_separation_dynamics_observed": separated,
        "max_post_detach_object_speed_mps": max((float(np.linalg.norm(np.asarray(row["object_qvel"][:3]))) for row in post), default=None),
    }


def run_physics_probe(data_root: Path, output_root: Path, budget: int, allow_physical_replay: bool, generation_lock: Path, root_family_id: str | None = None) -> dict[str, Any]:
    if not allow_physical_replay:
        raise PermissionError("physical replay requires --allow-physical-replay")
    if budget <= 0 or budget > MAX_PHYSICAL_EXECUTIONS:
        raise ValueError(f"budget must be between 1 and {MAX_PHYSICAL_EXECUTIONS}")
    manifest = list(__import__("csv").DictReader((data_root / "rollout_manifest.csv").open(encoding="utf-8")))
    selected_root = root_family_id or sorted({row["root_family_id"] for row in manifest})[0]
    root_rows = {row["case_id"]: row for row in manifest if row["root_family_id"] == selected_root}
    if set(PHYSICS_PROBE_CASES) - set(root_rows):
        raise ValueError(f"root {selected_root} does not contain all probe cases")
    if budget < 2 * len(PHYSICS_PROBE_CASES):
        raise ValueError("R11 default probe requires ordinary and instrumented replay for all four cases")
    if not generation_lock.is_file():
        raise FileNotFoundError("repair_generation_lock.json is required to reconstruct the frozen family")
    lock = read_json(generation_lock)
    family = next(row for row in lock["families"] if row["root_family_id"] == selected_root)
    all_traces: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    cached_comparisons: list[dict[str, Any]] = []
    executions = 0
    for case_id in PHYSICS_PROBE_CASES:
        case = CASES[case_id]
        seed = int(family["rollout_seed_base"]) + list(CASES).index(case_id)
        spec = family_spec(selected_root, case["scenario"], int(family["family_seed"]), seed, probe_variant=case.get("probe_variant", "default"))
        ordinary = _ordinary_replay(spec, seed, case["program"], int(family["family_index"]), render_callbacks=True)
        executions += 1
        instrumented, trace = _instrumented_replay(spec, seed, case["program"], int(family["family_index"]))
        executions += 1
        comparison = _compare(ordinary, instrumented)
        cached_comparison = _cached_replay_equivalence(
            data_root, f"{selected_root}:{case_id}", case_id, ordinary,
        )
        loss_time = next((float(row["time"]) for row in instrumented["events"] if row.get("event") == "contact_lost"), None)
        mechanism = _mechanism_summary(trace, loss_time)
        summaries.append({"root_family_id": selected_root, "case_id": case_id, "loss_time": loss_time, **comparison, "cached_replay_equivalence": cached_comparison["equivalent"], **mechanism})
        cached_comparisons.append(cached_comparison)
        all_traces.extend({"root_family_id": selected_root, "case_id": case_id, **row} for row in trace)
        ledger.append({"execution_index": executions - 1, "root_family_id": selected_root, "case_id": case_id, "mode": "ordinary", "status": "COMPLETE"})
        ledger.append({"execution_index": executions, "root_family_id": selected_root, "case_id": case_id, "mode": "instrumented_read_only", "status": "COMPLETE", "equivalence": comparison["equivalent"]})
    output_root.mkdir(parents=True, exist_ok=True)
    serialised_traces = []
    for row in all_traces:
        item = dict(row)
        item["contacts"] = json.dumps(item["contacts"], sort_keys=True, separators=(",", ":"))
        serialised_traces.append(item)
    write_csv(output_root / "physics_probe_trace.csv", serialised_traces)
    write_csv(output_root / "probe_execution_ledger.csv", ledger)
    equivalence = all(row["equivalent"] for row in summaries)
    cached_equivalence = all(row["equivalent"] for row in cached_comparisons)
    write_json(output_root / "probe_cached_equivalence.json", {
        "schema": "l2rar2_r11_probe_cached_equivalence_v1",
        "all_pairs_equal": cached_equivalence,
        "pairs": cached_comparisons,
        "mechanism_claims_require": "ordinary_vs_instrumented_and_instrumented_vs_cached_equal",
    })
    probe_status = "PHYSICS_PROBE_COMPLETE" if equivalence and cached_equivalence else "PROBE_EQUIVALENCE_FAILED"
    write_json(output_root / "physics_probe_manifest.json", {
        "schema": PHYSICS_PROBE_VERSION,
        "status": probe_status,
        "root_family_id": selected_root,
        "cases": list(PHYSICS_PROBE_CASES),
        "budget": budget,
        "executions_used": executions,
        "ordinary_and_instrumented_pairs": len(PHYSICS_PROBE_CASES),
        "new_dataset_rollouts": 0,
        "training_jobs": 0,
        "api_calls": 0,
        "api_key_read": False,
        "equivalence": {"all_pairs_equal": equivalence, "pairs": summaries},
        "source_rollout_manifest": str((data_root / "rollout_manifest.csv").resolve()),
        "cached_replay_equivalence": cached_equivalence,
        "mechanism_claims_allowed": False,
    })
    return {"status": probe_status, "executions_used": executions, "equivalence": equivalence, "cached_replay_equivalence": cached_equivalence, "summaries": summaries}
