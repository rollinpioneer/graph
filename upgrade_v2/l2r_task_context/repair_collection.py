"""Frozen 4-family x K1-K8 validation collection for attach-relpose repair."""
from __future__ import annotations

import copy
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from upgrade_v2.l2r_hold_evidence.probe_adapter import control_variant, perform
from upgrade_v2.visual_refine_l2.repaired_simulator import AttachRelposeDynamicTabletop, ATTACH_RELPOSE_VERSION
from upgrade_v2.visual_refine_l2.renderer import TabletopRenderer
from upgrade_v2.visual_refine_l2.vision import detect_frame

from .collector import CASES, _lifecycle
from .io import canonical_hash, read_csv, read_json, read_jsonl, sha256, write_csv, write_json, write_jsonl


REPAIR_COLLECTION_VERSION = "l2rar2_attach_relpose_collection_v1"
ROOT_FAMILIES = 4
FAMILY_SEED_BASE = 840000
ROLLOUT_SEED_BASE = 84100000


def _family_spec_record(index: int) -> dict[str, Any]:
    from upgrade_v2.visual_refine_l2.dynamic_simulator import family_spec

    seed = FAMILY_SEED_BASE + index
    rollout_seed_base = ROLLOUT_SEED_BASE + index * 100
    spec = family_spec(f"L2RAR2_REPAIR_{index:02d}_{seed}", "normal_pick_place", seed, rollout_seed_base)
    physical = {
        "object_radius": spec.object_radius,
        "friction": spec.friction,
        "camera_jitter": spec.camera_jitter,
        "object_x": spec.object_x,
        "object_y": spec.object_y,
        "target_x": spec.target_x,
        "target_y": spec.target_y,
    }
    return {
        "family_index": index,
        "root_family_id": spec.root_family_id,
        "family_seed": seed,
        "rollout_seed_base": rollout_seed_base,
        "physical_spec": physical,
        "physical_spec_sha256": canonical_hash(physical),
        "cases": list(CASES),
    }


def plan_repair_collection(output: Path, source_commit: str, protocol_path: Path, reference_contract_path: Path) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    reference = read_json(reference_contract_path)
    if protocol.get("cases") != list(CASES):
        raise ValueError("repair collection case order differs from frozen R2 case order")
    if float(reference["hold_proxy"]["maximum_relative_position_drift_m"]) != 0.02:
        raise ValueError("repair collection requires unchanged 0.02 m reference limit")
    families = [_family_spec_record(index) for index in range(ROOT_FAMILIES)]
    lock = {
        "schema": "pathgraph_l2rar2_attach_relpose_generation_lock_v1",
        "status": "LOCKED_BEFORE_REPAIR_COLLECTION",
        "repair_version": ATTACH_RELPOSE_VERSION,
        "collection_version": REPAIR_COLLECTION_VERSION,
        "source_commit": source_commit,
        "protocol_sha256": sha256(protocol_path),
        "reference_contract_sha256": sha256(reference_contract_path),
        "families": families,
        "root_families": ROOT_FAMILIES,
        "planned_rollouts": ROOT_FAMILIES * len(CASES),
        "case_order": list(CASES),
        "candidate_comparison_locked": ["B_count2", "C3_vector_rho035"],
        "new_sampling": False,
        "training_jobs": 0,
        "api_calls": 0,
        "api_key_read": False,
    }
    write_json(output, lock)
    return {"status": lock["status"], "repair_version": ATTACH_RELPOSE_VERSION, "root_families": ROOT_FAMILIES, "physical_rollouts": lock["planned_rollouts"]}


def _collect_job(task: tuple[dict[str, Any], str, Path]) -> dict[str, Any]:
    family, case_id, data_root = task
    case = CASES[case_id]
    family_id = family["root_family_id"]
    case_index = list(CASES).index(case_id)
    rollout_seed = int(family["rollout_seed_base"]) + case_index
    from upgrade_v2.visual_refine_l2.dynamic_simulator import family_spec

    spec = family_spec(family_id, case["scenario"], int(family["family_seed"]), rollout_seed, probe_variant=case.get("probe_variant", "default"))
    sim = AttachRelposeDynamicTabletop(spec, rollout_seed)
    out = data_root / "rollouts" / family_id / case_id
    out.mkdir(parents=True, exist_ok=False)
    capture_order = 0
    dense: list[dict[str, Any]] = []
    action_end: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    current_action = "observe_scene"
    end_reason = case.get("end_reason", "segment_complete")
    request = {
        "schema": "l2rar2_controller_request_v1",
        "request_id": f"{family_id}:{case_id}:request_001",
        "attempt_id": 1,
        "target_track_id": "tracked_object_001",
        "requested_effect": case["effect"],
        "issued_time": 0.0,
        "issued_capture_order": -1,
        "received_time": 0.0,
        "source": "controller_dispatch",
        "dispatch_program_sha256": canonical_hash(case["program"]),
    }
    write_jsonl(out / "controller_requests.jsonl", [request])

    with TabletopRenderer(sim.model) as renderer:
        def capture(current: Any, stream: str, action: str) -> None:
            nonlocal capture_order
            image_path = out / "rgb/front" / f"frame_{capture_order:05d}.jpg"
            renderer.save_jpeg(renderer.render(current.data, "front", spec.camera_jitter), image_path)
            row = {
                "frame_index": capture_order,
                "time": round(float(current.data.time), 9),
                "sim_time": round(float(current.data.time), 9),
                "capture_order": capture_order,
                "phase": stream,
                "front_path": str(image_path.resolve()),
                "front_sha256": sha256(image_path),
                "contact_present": bool(current.contact_sensor()),
                "gripper_command": "closed" if current.gripper_closed else "open",
                **_lifecycle(action, case["end_action"], end_reason, stream),
            }
            (dense if stream == "control_tick" else action_end).append(row)
            snap = current.oracle_snapshot()
            oracle.append({
                "frame_index": capture_order,
                "time": row["time"],
                "capture_order": capture_order,
                "phase": stream,
                "weld_state": int(snap["weld_state"]),
                "object_xyz": json.dumps([float(value) for value in current.object_xyz]),
                "gripper_xyz": json.dumps([float(value) for value in current.data.mocap_pos[0]]),
                "object_target_distance": float(snap["object_target_distance"]),
                "repair_version": ATTACH_RELPOSE_VERSION,
            })
            capture_order += 1

        def control_callback(current: Any, control: dict[str, Any]) -> None:
            del control
            capture(current, "control_tick", current_action)

        def action_callback(current: Any, action: str, action_index: int, result: dict[str, Any]) -> None:
            del action_index, result
            capture(current, "action_end", action)

        sim._r1_control_callback = control_callback
        sim._r1_action_end_callback = action_callback
        for planned_action in case["program"]:
            current_action = planned_action
            result = perform(sim, planned_action, int(family["family_index"]))
            low = result.pop("low_level_control_sequence", [])
            actions.append({
                "action_index": result["action_index"],
                "action": planned_action,
                "start_time": result["start_time"],
                "end_time": result["end_time"],
                "planned_request_id": request["request_id"],
                "planned_attempt_id": 1,
                "planned_request_end": planned_action == case["end_action"],
                "planned_end_reason": end_reason if planned_action == case["end_action"] else "",
            })
            controls.append({"action_index": result["action_index"], "action": planned_action, "controls": low})

    for row in dense + action_end:
        detected = detect_frame(Path(row["front_path"]))
        row.update({key: detected.get(key) for key in ("object_centroid", "gripper_centroid", "object_confidence", "gripper_confidence", "width", "height")})
    by_time: dict[float, dict[str, Any]] = {}
    for row in sorted(dense + action_end, key=lambda value: (float(value["time"]), int(value["capture_order"]))):
        candidate = dict(row)
        if candidate["phase"] == "action_end":
            candidate["source_phase"] = "action_end"
            candidate["phase"] = "control_tick_final"
        by_time[round(float(row["time"]), 9)] = candidate
    merged = [by_time[key] for key in sorted(by_time)]
    write_jsonl(out / "observations_dense.jsonl", merged)
    write_jsonl(out / "observations_action_end.jsonl", action_end)
    write_csv(out / "actions.csv", actions)
    write_jsonl(out / "low_level_controls.jsonl", controls)
    write_csv(out / "oracle_timeline.csv", oracle)
    write_jsonl(out / "events.jsonl", sim.events)
    write_json(out / "termination.json", {"done": True, "termination_type": "diagnostic_program_complete", "horizon": False, "time": float(sim.data.time)})
    meta = {
        "schema": "pathgraph_l2rar2_repair_rollout_v1",
        "rollout_id": f"{family_id}:{case_id}",
        "root_family_id": family_id,
        "case_id": case_id,
        "partition": "repair_dev",
        "path": str(out.resolve()),
        "family_seed": family["family_seed"],
        "rollout_seed": rollout_seed,
        "physical_spec_sha256": family["physical_spec_sha256"],
        "control_variant": control_variant(int(family["family_index"])),
        "requested_effect": case["effect"],
        "request_issued_before_first_action": True,
        "lifecycle_binding_version": "l2rar2_preplanned_request_boundary_v1",
        "program_sha256": canonical_hash(case["program"]),
        "dense_observation_count": len(merged),
        "action_end_observation_count": len(action_end),
        "source_kind": "new_dynamic_mujoco_r2_attach_relpose_repair",
        "repair_version": ATTACH_RELPOSE_VERSION,
        "collection_version": REPAIR_COLLECTION_VERSION,
        "api_calls": 0,
        "training_jobs": 0,
    }
    write_json(out / "metadata.json", meta)
    return meta


def collect_repair(lock_path: Path, output_root: Path, workers: int = 1) -> dict[str, Any]:
    lock = read_json(lock_path)
    if lock.get("status") != "LOCKED_BEFORE_REPAIR_COLLECTION":
        raise ValueError("repair generation lock is not valid")
    families = lock["families"]
    expected = {f"{family['root_family_id']}:{case_id}": (family, case_id, output_root) for family in families for case_id in CASES}
    manifest_path = output_root / "rollout_manifest.csv"
    if manifest_path.exists():
        rows = read_csv(manifest_path)
        if {row["rollout_id"] for row in rows} != set(expected):
            raise RuntimeError("existing repair manifest differs from frozen lock")
        return {"status": "REPAIR_COLLECTION_ALREADY_COMPLETE", "root_families": len(families), "physical_rollouts": len(rows), "newly_executed_jobs": 0}
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = list(expected.values())
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            metadata = list(pool.map(_collect_job, tasks))
    else:
        metadata = [_collect_job(task) for task in tasks]
    metadata = sorted(metadata, key=lambda row: row["rollout_id"])
    if len(metadata) != len(expected):
        raise RuntimeError("repair collection did not produce all frozen rollouts")
    write_csv(manifest_path, metadata)
    result = {
        "schema": "pathgraph_l2rar2_repair_collection_summary_v1",
        "status": "REPAIR_COLLECTION_COMPLETE",
        "repair_version": lock["repair_version"],
        "collection_version": lock["collection_version"],
        "root_families": len(families),
        "physical_rollouts": len(metadata),
        "cases": len(CASES),
        "request_pre_action_all": all(row["request_issued_before_first_action"] for row in metadata),
        "api_calls": 0,
        "training_jobs": 0,
        "newly_executed_jobs": len(metadata),
    }
    write_json(output_root / "collection_summary.json", result)
    return result
