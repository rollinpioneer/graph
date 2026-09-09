from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from upgrade_v2.visual_refine_l2.dynamic_simulator import DynamicTabletop, family_spec
from upgrade_v2.visual_refine_l2.renderer import TabletopRenderer
from upgrade_v2.visual_refine_l2.vision import detect_frame

from .inputs import sha256_file, write_csv, write_json, write_jsonl
from .probe_adapter import CONTROL_VARIANTS, control_variant, perform, program_for_stratum


class ControlTickRecorder:
    """Read-only recorder for simulator control boundaries."""

    def __init__(self) -> None:
        self.dense: list[dict[str, Any]] = []
        self.action_end: list[dict[str, Any]] = []
        self.controls: list[dict[str, Any]] = []

    def capture_control_tick(self, sim: Any, phase: str, capture_order: int) -> None:
        self.dense.append({"sim_time": float(sim.data.time), "phase": phase, "capture_order": capture_order, "mocap_position": copy.deepcopy(sim.data.mocap_pos[0].tolist()), "contact_present": bool(sim.contact_sensor()), "gripper_command": "closed" if sim.gripper_closed else "open", **sim.attempt_lifecycle.snapshot()})

    def capture_action_end(self, sim: Any, action: str, capture_order: int) -> None:
        self.action_end.append({"sim_time": float(sim.data.time), "action": action, "capture_order": capture_order, "contact_present": bool(sim.contact_sensor()), "gripper_command": "closed" if sim.gripper_closed else "open", **sim.attempt_lifecycle.snapshot()})


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _payload_hash(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def merge_final_action_observations(
    dense_rows: list[dict[str, Any]], action_end_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Keep one final, post-update observation at each physical timestamp."""
    by_time: dict[float, dict[str, Any]] = {}
    for row in sorted(dense_rows + action_end_rows, key=lambda item: (float(item["time"]), int(item["capture_order"]))):
        key = round(float(row["time"]), 9)
        candidate = dict(row)
        if candidate.get("phase") == "action_end":
            candidate["source_phase"] = "action_end"
            candidate["phase"] = "control_tick_final"
        by_time[key] = candidate
    return [by_time[key] for key in sorted(by_time)]


def normalize_development_streams(data_root: Path, output_path: Path) -> dict[str, Any]:
    rows = []
    metadata = []
    for meta_path in sorted(data_root.rglob("metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rollout = meta_path.parent
        dense_path = rollout / "observations_dense.jsonl"
        action_path = rollout / "observations_action_end.jsonl"
        dense = [json.loads(line) for line in dense_path.read_text(encoding="utf-8").splitlines() if line]
        action_end = [json.loads(line) for line in action_path.read_text(encoding="utf-8").splitlines() if line]
        before_sha = sha256_file(dense_path)
        merged = merge_final_action_observations(dense, action_end)
        write_jsonl(dense_path, merged)
        meta["dense_observation_count"] = len(merged)
        meta["dense_final_observation_rule"] = "last capture_order per timestamp; action-end post-update state replaces pre-update control tick"
        write_json(meta_path, meta)
        metadata.append(meta)
        rows.append({
            "rollout_id": meta["rollout_id"],
            "split": meta["split"],
            "before_dense_sha256": before_sha,
            "after_dense_sha256": sha256_file(dense_path),
            "before_count": len(dense),
            "after_count": len(merged),
            "timestamps_strictly_increasing": all(float(left["time"]) < float(right["time"]) for left, right in zip(merged, merged[1:])),
            "physical_rollout_reexecuted": False,
        })
    for split in ("dev_fit", "dev_select"):
        write_csv(data_root / f"{split}_rollout_manifest.csv", [meta for meta in metadata if meta["split"] == split])
    write_csv(output_path.with_suffix(".csv"), rows)
    result = {
        "schema": "pathgraph_l2rar1_dense_stream_normalization_v1",
        "status": "PASS" if rows and all(row["timestamps_strictly_increasing"] for row in rows) else "FAIL",
        "rollouts": len(rows),
        "physical_rollouts_reexecuted": 0,
        "rule": "one post-update final observation per timestamp",
        "details": str(output_path.with_suffix(".csv").resolve()),
    }
    write_json(output_path, result)
    if result["status"] != "PASS":
        raise RuntimeError("dense stream normalization failed")
    return result


def audit_callback_equivalence(output_path: Path) -> dict[str, Any]:
    """Prove that read-only callbacks do not alter a same-seed trajectory."""
    spec = family_spec(
        "L2RAR1_CALLBACK_AUDIT",
        "normal_pick_place",
        510000,
        51100000,
        camera_jitter=False,
        object_size_jitter=False,
        friction_jitter=False,
    )
    actions = ["observe_scene", "approach_object", "close_gripper", "lift", "transport_to_target", "verify"]

    def execute(with_callbacks: bool) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
        sim = DynamicTabletop(spec, 51100000)
        callback_counts = {"control_tick": 0, "action_end": 0}
        if with_callbacks:
            def control_callback(current: Any, control: dict[str, Any]) -> None:
                callback_counts["control_tick"] += 1
                _ = (float(current.data.time), bool(current.contact_sensor()), copy.deepcopy(control))

            def action_callback(current: Any, action: str, action_index: int, result: dict[str, Any]) -> None:
                callback_counts["action_end"] += 1
                _ = (float(current.data.time), action, action_index, copy.deepcopy(result))

            sim._r1_control_callback = control_callback
            sim._r1_action_end_callback = action_callback
        controls = []
        for action in actions:
            result = sim.perform(action)
            controls.append(copy.deepcopy(result["low_level_control_sequence"]))
        return sim.snapshot(), controls, callback_counts

    plain_state, plain_controls, plain_counts = execute(False)
    callback_state, callback_controls, callback_counts = execute(True)
    state_equal = _jsonable(plain_state) == _jsonable(callback_state)
    controls_equal = plain_controls == callback_controls
    rng_equal = plain_state["rng_state"] == callback_state["rng_state"]
    result = {
        "schema": "pathgraph_l2rar1_callback_equivalence_v1",
        "status": "PASS" if state_equal and controls_equal and rng_equal else "FAIL",
        "family_id": spec.root_family_id,
        "rollout_seed": 51100000,
        "actions": actions,
        "final_state_equal": state_equal,
        "control_sequences_equal": controls_equal,
        "rng_state_equal": rng_equal,
        "plain_final_state_sha256": _payload_hash(plain_state),
        "callback_final_state_sha256": _payload_hash(callback_state),
        "plain_controls_sha256": _payload_hash(plain_controls),
        "callback_controls_sha256": _payload_hash(callback_controls),
        "plain_callback_counts": plain_counts,
        "observing_callback_counts": callback_counts,
        "physical_rollouts_counted_for_science": 0,
        "purpose": "software equivalence audit only",
    }
    write_json(output_path, result)
    if result["status"] != "PASS":
        raise RuntimeError("read-only callback changed simulator trajectory")
    return result


def collect_one(stratum: str, family_index: int, rollout_index: int, split: str, output_root: Path, family_seed: int, rollout_seed: int) -> dict[str, Any]:
    family_id = f"L2RAR1_{stratum[:4].upper()}_{family_index:02d}_{family_seed}"
    scenario = "missed_grasp_then_retry" if stratum == "miss_without_prior_hold" else "slip_then_recover" if stratum in {"loss_after_observed_hold", "brief_true_hold_then_loss", "long_gap_after_loss", "history_or_visual_unavailable"} else "normal_pick_place"
    spec = family_spec(
        family_id,
        scenario,
        family_seed + family_index,
        rollout_seed,
        probe_variant="brief_hold" if stratum == "brief_true_hold_then_loss" else "default",
    )
    sim = DynamicTabletop(spec, rollout_seed + rollout_index)
    out = output_root / "rollouts" / family_id / f"rollout_{rollout_index:02d}"
    out.mkdir(parents=True, exist_ok=True)
    capture_order = 0
    dense_rows, action_end_rows, oracle_rows = [], [], []
    actions, controls = [], []
    with TabletopRenderer(sim.model) as renderer:
        def capture(current: Any, phase: str, action: str | None = None, action_index: int | None = None) -> None:
            nonlocal capture_order
            path = out / "rgb/front" / f"frame_{capture_order:05d}.jpg"
            renderer.save_jpeg(renderer.render(current.data, "front", spec.camera_jitter), path)
            row = {
                "frame_index": capture_order,
                "sim_time": float(current.data.time),
                "capture_order": capture_order,
                "phase": phase,
                "front_path": str(path.resolve()),
                "front_sha256": sha256_file(path),
                "contact_present": bool(current.contact_sensor()),
                "gripper_command": "closed" if current.gripper_closed else "open",
            }
            row.update(current.attempt_lifecycle.snapshot())
            if action is not None:
                row.update({"action": action, "action_index": action_index})
            (dense_rows if phase == "control_tick" else action_end_rows).append(row)
            oracle = current.oracle_snapshot()
            oracle_rows.append({
                "frame_index": capture_order,
                "time": float(current.data.time),
                "capture_order": capture_order,
                "phase": phase,
                "weld_state": int(oracle["weld_state"]),
                "object_target_distance": float(oracle["object_target_distance"]),
                "goal_stable": int(oracle["goal_stable"]),
                "object_xyz": json.dumps([float(value) for value in current.object_xyz]),
                "gripper_xyz": json.dumps([float(value) for value in current.data.mocap_pos[0]]),
            })
            capture_order += 1

        def callback(current: Any, control: dict[str, Any]) -> None:
            capture(current, "control_tick")

        def action_end_callback(current: Any, action: str, action_index: int, result: dict[str, Any]) -> None:
            capture(current, "action_end", action, action_index)

        sim._r1_control_callback = callback
        sim._r1_action_end_callback = action_end_callback
        for action in program_for_stratum(stratum):
            result = perform(sim, action, rollout_index)
            low = result.pop("low_level_control_sequence", [])
            lifecycle = result.get("attempt_lifecycle", {})
            actions.append({"action_index": result.get("action_index"), "action": action, "start_time": result.get("start_time"), "end_time": result.get("end_time"), "gripper_command": result.get("gripper_command"), "contact_present": result.get("contact_present"), "termination_reason": result.get("termination_reason"), **lifecycle})
            controls.append({"action_index": result.get("action_index"), "action": action, "controls": low})
    for row in dense_rows + action_end_rows:
        detection = detect_frame(Path(row["front_path"]))
        row.update({"object_centroid": detection.get("object_centroid"), "gripper_centroid": detection.get("gripper_centroid"), "object_confidence": detection.get("object_confidence"), "gripper_confidence": detection.get("gripper_confidence"), "width": detection.get("width"), "height": detection.get("height"), "time": row["sim_time"], "frame_index": row["frame_index"]})
    if stratum == "history_or_visual_unavailable":
        cutoff_time = dense_rows[max(1, len(dense_rows) // 3) - 1]["sim_time"]
        for row in dense_rows:
            if float(row["sim_time"]) <= float(cutoff_time):
                row["observation_masked"] = True
        for row in action_end_rows:
            if float(row["sim_time"]) <= float(cutoff_time):
                row["observation_masked"] = True
    all_captures = dense_rows + action_end_rows
    dense_rows = merge_final_action_observations(dense_rows, action_end_rows)
    write_csv(out / "actions.csv", actions)
    write_jsonl(out / "low_level_controls.jsonl", controls)
    write_csv(out / "raw_capture_manifest.csv", [{"sim_time": row["sim_time"], "capture_order": row["capture_order"], "phase": row["phase"], "image_sha256": row["front_sha256"]} for row in all_captures])
    write_csv(out / "frame_manifest.csv", [{"frame_index": row["frame_index"], "time": row["sim_time"], "phase": row["phase"], "front_path": row["front_path"], "front_sha256": row["front_sha256"], "side_path": "", "side_sha256": ""} for row in all_captures])
    write_csv(out / "contact_sensor.csv", [{"frame_index": row["frame_index"], "time": row["sim_time"], "contact_present": int(row["contact_present"])} for row in dense_rows])
    write_csv(out / "gripper_command.csv", [{"frame_index": row["frame_index"], "time": row["sim_time"], "gripper_command": row["gripper_command"]} for row in dense_rows])
    write_csv(out / "oracle_timeline.csv", oracle_rows)
    write_jsonl(out / "observations_dense.jsonl", dense_rows)
    write_jsonl(out / "observations_action_end.jsonl", action_end_rows)
    write_jsonl(out / "oracle_dense.jsonl", oracle_rows)
    write_jsonl(out / "events.jsonl", sim.events)
    alignment = []
    for end in action_end_rows:
        prior = [row for row in dense_rows if float(row["sim_time"]) <= float(end["sim_time"]) + 1e-9]
        dense = prior[-1] if prior else None
        alignment.append({
            "action_index": end.get("action_index"),
            "action_end_frame_index": end["frame_index"],
            "action_end_time": end["sim_time"],
            "dense_frame_index": dense.get("frame_index") if dense else "",
            "dense_time": dense.get("sim_time") if dense else "",
            "delta_seconds": float(end["sim_time"]) - float(dense["sim_time"]) if dense else "",
        })
    write_csv(out / "stream_alignment.csv", alignment)
    write_json(out / "termination.json", {"schema": "pathgraph_l2rar1_termination_v1", "done": True, "termination_type": "diagnostic_complete", "horizon": False, "time": float(sim.data.time)})
    meta = {"schema": "pathgraph_l2rar1_rollout_v3", "rollout_id": f"{family_id}_r{rollout_index:02d}", "root_family_id": family_id, "stratum": stratum, "split": split, "path": str(out.resolve()), "source_kind": "new_dynamic_mujoco_r1", "observation_intervention": stratum == "history_or_visual_unavailable", "dense_observation_count": len(dense_rows), "action_end_observation_count": len(action_end_rows), "control_hz": 20, "api_calls": 0, "training_jobs": 0, "probe_variant": spec.probe_variant, "control_variant": control_variant(rollout_index), "controller_lifecycle_contract": "attempt_id/attempt_phase/attempt_active/attempt_end/attempt_end_reason; post-update aligned; one-cycle attempt_end edge", "controller_lifecycle_fields": ["attempt_id", "attempt_phase", "attempt_active", "attempt_end", "attempt_end_reason"]}
    write_json(out / "metadata.json", meta)
    return meta


def collect_development(output_root: Path, protocol: dict[str, Any], partition: str = "dev_fit", workers: int = 1) -> dict[str, Any]:
    from concurrent.futures import ProcessPoolExecutor
    dev = protocol["development"]
    start, stop = (0, 2) if partition == "dev_fit" else (2, 4)
    tasks = [(stratum, fi, ri, partition, output_root, int(protocol["new_seeds"]["development_family"]), int(protocol["new_seeds"]["development_rollout"])) for stratum in dev["strata"] for fi in range(start, stop) for ri in range(dev["rollouts_per_family"])]
    if partition == "dev_fit":
        write_json(output_root / "generator_lock.json", {
            "schema": "pathgraph_l2rar1_generator_lock_v1",
            "status": "LOCKED_BEFORE_SELECT",
            "partition": "development",
            "strata": list(dev["strata"]),
            "families_per_stratum": int(dev["families_per_stratum"]),
            "rollouts_per_family": int(dev["rollouts_per_family"]),
            "fit_family_indices": [0, 1],
            "select_family_indices": [2, 3],
            "family_seed": int(protocol["new_seeds"]["development_family"]),
            "rollout_seed": int(protocol["new_seeds"]["development_rollout"]),
            "programs": {stratum: program_for_stratum(stratum) for stratum in dev["strata"]},
            "control_variants": list(CONTROL_VARIANTS),
            "variant_assignment": "rollout_index modulo 4; fixed before recollection",
            "brief_hold_variant": "one midpoint control segment before forced loss",
            "api_calls": 0,
            "training_jobs": 0,
        })
        write_json(output_root / "reference_contract.json", {
            "schema": "pathgraph_l2rar1_reference_contract_v1",
            "status": "LOCKED_BEFORE_SELECT",
            "world_position_error_m": 0.02,
            "minimum_object_displacement_m": 0.01,
            "response_window_seconds": float(protocol["response_window_seconds"]),
            "max_gap_seconds": float(protocol["max_gap_seconds"]),
            "reference_only_sources": ["events.jsonl", "oracle_timeline.csv", "actions.csv", "termination.json"],
            "online_features_excluded": ["scenario", "stratum", "weld_state", "future_outcome", "reference_event_name"],
        })
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            metadata = list(pool.map(_collect_one_task, tasks))
    else:
        metadata = [collect_one(*task) for task in tasks]
    write_csv(output_root / f"{partition}_rollout_manifest.csv", metadata)
    if partition == "dev_fit":
        run_root = output_root.parent.parent
        collection_lock = run_root / "locks/development_collection_lock.json"
        write_json(collection_lock, {
            "schema": "pathgraph_l2rar1_development_collection_lock_v1",
            "status": "LOCKED_AFTER_COLLECTION_BEFORE_SELECT",
            "partition": "development",
            "families": len({row["root_family_id"] for row in metadata}),
            "rollouts": len(metadata),
            "rollout_manifest": {"path": str((output_root / "dev_fit_rollout_manifest.csv").resolve()), "sha256": sha256_file(output_root / "dev_fit_rollout_manifest.csv")},
            "generator_lock": {"path": str((output_root / "generator_lock.json").resolve()), "sha256": sha256_file(output_root / "generator_lock.json")},
            "reference_contract": {"path": str((output_root / "reference_contract.json").resolve()), "sha256": sha256_file(output_root / "reference_contract.json")},
            "api_calls": 0,
            "training_jobs": 0,
        })
    return {"status": "COLLECTION_COMPLETE", "partition": partition, "families": len({row["root_family_id"] for row in metadata}), "rollouts": len(metadata), "dense_observations": sum(row["dense_observation_count"] for row in metadata), "api_calls": 0, "training_jobs": 0}


def _collect_one_task(args: tuple[Any, ...]) -> dict[str, Any]:
    return collect_one(*args)
