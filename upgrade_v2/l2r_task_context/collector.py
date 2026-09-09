from __future__ import annotations

import copy
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from upgrade_v2.l2r_hold_evidence.probe_adapter import control_variant, perform
from upgrade_v2.visual_refine_l2.dynamic_simulator import DynamicTabletop, family_spec
from upgrade_v2.visual_refine_l2.renderer import TabletopRenderer
from upgrade_v2.visual_refine_l2.vision import detect_frame

from .io import canonical_hash, read_json, sha256, write_csv, write_json, write_jsonl


CASES: dict[str, dict[str, Any]] = {
    "K1_hold_request_ends_without_hold": {
        "effect": "HOLD_OBJECT", "scenario": "missed_grasp_then_retry",
        "program": ["observe_scene", "approach_object", "close_gripper"], "end_action": "close_gripper",
    },
    "K2_touch_request_completes_without_hold": {
        "effect": "TOUCH_OBJECT", "scenario": "normal_pick_place",
        "program": ["observe_scene", "approach_object", "touch_contact", "separate_touch"], "end_action": "separate_touch",
    },
    "K3_normal_hold_pause_resume": {
        "effect": "HOLD_OBJECT", "scenario": "normal_pick_place",
        "program": ["observe_scene", "approach_object", "close_gripper", "lift", "verify", "verify", "transport_to_target", "verify"], "end_action": "lift",
    },
    "K4_regular_hold_loss": {
        "effect": "HOLD_OBJECT", "scenario": "slip_then_recover",
        "program": ["observe_scene", "approach_object", "close_gripper", "lift", "transport_to_target", "verify"], "end_action": "lift",
    },
    "K5_brief_hold_loss": {
        "effect": "HOLD_OBJECT", "scenario": "slip_then_recover", "probe_variant": "brief_hold",
        "program": ["observe_scene", "approach_object", "close_gripper", "lift", "transport_to_target", "verify"], "end_action": "lift",
    },
    "K6_long_gap_after_loss": {
        "effect": "HOLD_OBJECT", "scenario": "slip_then_recover",
        "program": ["observe_scene", "approach_object", "close_gripper", "lift", "transport_to_target", "verify", "verify", "verify", "verify", "verify", "verify"], "end_action": "lift",
    },
    "K7_commanded_release": {
        "effect": "RELEASE_OBJECT", "scenario": "normal_pick_place",
        "program": ["observe_scene", "approach_object", "close_gripper", "lift", "open_gripper", "verify"], "end_action": "open_gripper", "end_reason": "release",
    },
    "K8_acquisition_touch_then_continue": {
        "effect": "HOLD_OBJECT", "scenario": "normal_pick_place",
        "program": ["observe_scene", "approach_object", "touch_contact", "separate_touch", "approach_object", "close_gripper", "lift", "verify"], "end_action": "verify",
    },
}

PARTITIONS = {"dev_fit": ("fit", 4, 810000), "dev_select": ("select", 8, 820000), "confirmation": ("confirm", 8, 830000)}


def plan_families(protocol_path: Path, inputs_path: Path, contract_root: Path, output_path: Path) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    inputs = read_json(inputs_path)
    status = read_json(contract_root / "contract_status.json")
    if status.get("status") != "CONTRACT_LOCKED_FOR_NEW_COLLECTION":
        raise ValueError("task contract is not locked")
    expected_cases = protocol.get("cases")
    if expected_cases != list(CASES):
        raise ValueError("protocol case order differs from implemented frozen case order")
    families = []
    for partition, (tag, count, seed_base) in PARTITIONS.items():
        for index in range(count):
            seed = seed_base + index
            spec = family_spec(f"L2RAR2_{tag.upper()}_{index:02d}_{seed}", "normal_pick_place", seed, 83100000 + seed)
            physical = {
                "object_radius": spec.object_radius, "friction": spec.friction, "camera_jitter": spec.camera_jitter,
                "object_x": spec.object_x, "object_y": spec.object_y, "target_x": spec.target_x, "target_y": spec.target_y,
            }
            families.append({
                "partition": partition,
                "root_family_id": spec.root_family_id,
                "family_index": index,
                "family_seed": seed,
                "rollout_seed_base": (81100000 if partition == "dev_fit" else 82100000 if partition == "dev_select" else 83100000) + index * 100,
                "physical_spec": physical,
                "physical_spec_sha256": canonical_hash(physical),
                "cases": list(CASES),
            })
    if len(families) != 20 or len({row["physical_spec_sha256"] for row in families}) != 20:
        raise RuntimeError("family plan is not 20 unique physical root specifications")
    lock = {
        "schema": "pathgraph_l2rar2_generation_lock_v1",
        "status": "LOCKED_BEFORE_PHYSICAL_COLLECTION",
        "formal_main_commit": inputs["formal_main_commit"],
        "maintenance_source_commit": inputs["maintenance_source_commit"],
        "protocol_sha256": sha256(protocol_path),
        "controller_contract_sha256": sha256(contract_root / "controller_context_contract.json"),
        "reference_contract_sha256": sha256(contract_root / "reference_contract.json"),
        "case_registry": CASES,
        "families": families,
        "physical_root_families": len(families),
        "planned_physical_rollouts": len(families) * len(CASES),
        "historical_root_overlap": False,
        "historical_overlap_basis": "new L2RAR2 IDs and disjoint frozen seed ranges 810000/820000/830000",
        "variant_assignment": "family_index modulo four; shared by all cases within a root",
        "api_calls": 0,
        "training_jobs": 0,
    }
    write_json(output_path, lock)
    return {"status": lock["status"], "root_families": 20, "physical_rollouts": 160}


def _lifecycle(action: str, end_action: str, end_reason: str, phase: str) -> dict[str, Any]:
    if phase == "action_end" and action == end_action:
        return {"attempt_id": 1, "attempt_phase": "ended", "attempt_active": False, "attempt_end": True, "attempt_end_reason": end_reason, "attempt_end_sequence": 1}
    active_actions = {"approach_object", "touch_contact", "separate_touch", "close_gripper", "lift"}
    if action in active_actions and action != end_action:
        attempt_phase = "settling" if action == "close_gripper" else "post_contact_motion" if action in {"separate_touch", "lift"} else "acquiring"
        return {"attempt_id": 1, "attempt_phase": attempt_phase, "attempt_active": True, "attempt_end": False, "attempt_end_reason": None, "attempt_end_sequence": 1}
    if action == end_action:
        return {"attempt_id": 1, "attempt_phase": "post_contact_motion", "attempt_active": True, "attempt_end": False, "attempt_end_reason": None, "attempt_end_sequence": 1}
    return {"attempt_id": 1, "attempt_phase": "inactive", "attempt_active": False, "attempt_end": False, "attempt_end_reason": None, "attempt_end_sequence": 1}


def _collect_job(task: tuple[dict[str, Any], str, Path]) -> dict[str, Any]:
    family, case_id, data_root = task
    case = CASES[case_id]
    family_id = family["root_family_id"]
    case_index = list(CASES).index(case_id)
    rollout_seed = int(family["rollout_seed_base"]) + case_index
    spec = family_spec(
        family_id, case["scenario"], int(family["family_seed"]), rollout_seed,
        probe_variant=case.get("probe_variant", "default"),
    )
    sim = DynamicTabletop(spec, rollout_seed)
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
                "frame_index": capture_order, "time": row["time"], "capture_order": capture_order, "phase": stream,
                "weld_state": int(snap["weld_state"]),
                "object_xyz": json.dumps([float(value) for value in current.object_xyz]),
                "gripper_xyz": json.dumps([float(value) for value in current.data.mocap_pos[0]]),
                "object_target_distance": float(snap["object_target_distance"]),
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
                "action_index": result["action_index"], "action": planned_action,
                "start_time": result["start_time"], "end_time": result["end_time"],
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
        "schema": "pathgraph_l2rar2_rollout_v1",
        "rollout_id": f"{family_id}:{case_id}",
        "root_family_id": family_id,
        "case_id": case_id,
        "partition": family["partition"],
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
        "source_kind": "new_dynamic_mujoco_r2",
        "api_calls": 0,
        "training_jobs": 0,
    }
    write_json(out / "metadata.json", meta)
    return meta


def collect(generation_lock_path: Path, partition: str, workers: int, output_root: Path) -> dict[str, Any]:
    if partition not in PARTITIONS:
        raise ValueError(f"unknown partition: {partition}")
    lock = read_json(generation_lock_path)
    if lock.get("status") != "LOCKED_BEFORE_PHYSICAL_COLLECTION":
        raise ValueError("generation lock is not valid")
    families = [row for row in lock["families"] if row["partition"] == partition]
    expected = PARTITIONS[partition][1]
    if len(families) != expected:
        raise ValueError("partition family count differs from lock")
    expected_jobs = {
        f"{family['root_family_id']}:{case_id}": (family, case_id, output_root)
        for family in families for case_id in CASES
    }
    manifest_path = output_root / "rollout_manifest.csv"
    if manifest_path.exists():
        rows = __import__("upgrade_v2.l2r_task_context.io", fromlist=["read_csv"]).read_csv(manifest_path)
        if len(rows) != expected * len(CASES) or {row["rollout_id"] for row in rows} != set(expected_jobs):
            raise RuntimeError("existing manifest differs from the frozen generation lock")
        return {"status": "COLLECTION_ALREADY_COMPLETE", "partition": partition, "root_families": expected, "physical_rollouts": len(rows)}
    completed: list[dict[str, Any]] = []
    for rollout_id, (family, case_id, _) in expected_jobs.items():
        metadata_path = output_root / "rollouts" / family["root_family_id"] / case_id / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = read_json(metadata_path)
        if metadata.get("rollout_id") != rollout_id or metadata.get("partition") != partition or metadata.get("physical_spec_sha256") != family["physical_spec_sha256"]:
            raise RuntimeError(f"existing job does not match generation lock: {rollout_id}")
        completed.append(metadata)
    completed_ids = {row["rollout_id"] for row in completed}
    tasks = [task for rollout_id, task in expected_jobs.items() if rollout_id not in completed_ids]
    if workers > 1 and tasks:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            new_metadata = list(pool.map(_collect_job, tasks))
    else:
        new_metadata = [_collect_job(task) for task in tasks]
    metadata = sorted(completed + new_metadata, key=lambda row: row["rollout_id"])
    if len(metadata) != len(expected_jobs):
        raise RuntimeError("collection ended without all frozen jobs")
    write_csv(manifest_path, metadata)
    result = {
        "schema": "pathgraph_l2rar2_collection_summary_v1",
        "status": "COLLECTION_COMPLETE",
        "partition": partition,
        "root_families": len({row["root_family_id"] for row in metadata}),
        "physical_rollouts": len(metadata),
        "cases": len({row["case_id"] for row in metadata}),
        "request_pre_action_all": all(row["request_issued_before_first_action"] for row in metadata),
        "api_calls": 0,
        "training_jobs": 0,
        "resumed_completed_jobs": len(completed),
        "newly_executed_jobs": len(new_metadata),
    }
    write_json(output_root / "collection_summary.json", result)
    return result
