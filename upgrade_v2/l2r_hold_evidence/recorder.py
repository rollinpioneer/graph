from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from upgrade_v2.visual_refine_l2.dynamic_simulator import DynamicTabletop, family_spec
from upgrade_v2.visual_refine_l2.renderer import TabletopRenderer
from upgrade_v2.visual_refine_l2.vision import detect_frame

from .inputs import sha256_file, write_csv, write_json, write_jsonl
from .probe_adapter import perform, program_for_stratum


class ControlTickRecorder:
    """Read-only recorder for simulator control boundaries."""

    def __init__(self) -> None:
        self.dense: list[dict[str, Any]] = []
        self.action_end: list[dict[str, Any]] = []
        self.controls: list[dict[str, Any]] = []

    def capture_control_tick(self, sim: Any, phase: str, capture_order: int) -> None:
        self.dense.append({"sim_time": float(sim.data.time), "phase": phase, "capture_order": capture_order, "mocap_position": copy.deepcopy(sim.data.mocap_pos[0].tolist()), "contact_present": bool(sim.contact_sensor()), "gripper_command": "closed" if sim.gripper_closed else "open"})

    def capture_action_end(self, sim: Any, action: str, capture_order: int) -> None:
        self.action_end.append({"sim_time": float(sim.data.time), "action": action, "capture_order": capture_order, "contact_present": bool(sim.contact_sensor()), "gripper_command": "closed" if sim.gripper_closed else "open"})


def collect_one(stratum: str, family_index: int, rollout_index: int, split: str, output_root: Path, family_seed: int, rollout_seed: int) -> dict[str, Any]:
    family_id = f"L2RAR1_{stratum[:4].upper()}_{family_index:02d}_{family_seed}"
    scenario = "missed_grasp_then_retry" if stratum == "miss_without_prior_hold" else "slip_then_recover" if stratum in {"loss_after_observed_hold", "brief_true_hold_then_loss", "long_gap_after_loss", "history_or_visual_unavailable"} else "normal_pick_place"
    spec = family_spec(family_id, scenario, family_seed + family_index, rollout_seed)
    sim = DynamicTabletop(spec, rollout_seed + rollout_index)
    out = output_root / "rollouts" / family_id / f"rollout_{rollout_index:02d}"
    out.mkdir(parents=True, exist_ok=True)
    recorder = ControlTickRecorder()
    capture_order = 0
    dense_rows, oracle_rows, frames = [], [], []
    actions, controls = [], []
    with TabletopRenderer(sim.model) as renderer:
        def callback(current: Any, control: dict[str, Any]) -> None:
            nonlocal capture_order
            path = out / "rgb/front" / f"frame_{capture_order:05d}.jpg"
            renderer.save_jpeg(renderer.render(current.data, "front", spec.camera_jitter), path)
            row = {"frame_index": capture_order, "sim_time": float(current.data.time), "capture_order": capture_order, "phase": "control_tick", "front_path": str(path.resolve()), "front_sha256": sha256_file(path), "contact_present": bool(current.contact_sensor()), "gripper_command": "closed" if current.gripper_closed else "open", "object_xyz_reference_only": current.object_xyz.tolist(), "gripper_xyz_reference_only": current.data.mocap_pos[0].tolist()}
            dense_rows.append(row)
            oracle = current.oracle_snapshot()
            oracle_rows.append({"frame_index": capture_order, "time": float(current.data.time), "weld_state": int(oracle["weld_state"]), "object_target_distance": float(oracle["object_target_distance"]), "goal_stable": int(oracle["goal_stable"])})
            capture_order += 1
        sim._r1_control_callback = callback
        for action in program_for_stratum(stratum):
            result = perform(sim, action)
            low = result.pop("low_level_control_sequence", [])
            actions.append({"action_index": result.get("action_index"), "action": action, "start_time": result.get("start_time"), "end_time": result.get("end_time"), "gripper_command": result.get("gripper_command"), "contact_present": result.get("contact_present"), "termination_reason": result.get("termination_reason")})
            controls.append({"action_index": result.get("action_index"), "action": action, "controls": low})
    for row in dense_rows:
        detection = detect_frame(Path(row["front_path"]))
        row.update({"object_centroid": detection.get("object_centroid"), "gripper_centroid": detection.get("gripper_centroid"), "object_confidence": detection.get("object_confidence"), "gripper_confidence": detection.get("gripper_confidence"), "width": detection.get("width"), "height": detection.get("height"), "time": row["sim_time"], "frame_index": row["frame_index"]})
    end_by_action = {}
    for row in actions:
        candidates = [item for item in dense_rows if float(item["sim_time"]) <= float(row["end_time"]) + 1e-9]
        if candidates:
            end_by_action[row["action_index"]] = candidates[-1]
    action_end = [dict(end_by_action[key], phase="action_end") for key in sorted(end_by_action)]
    if stratum == "history_or_visual_unavailable":
        for row in dense_rows[: max(1, len(dense_rows) // 3)]:
            row["observation_masked"] = True
    write_csv(out / "actions.csv", actions)
    write_jsonl(out / "low_level_controls.jsonl", controls)
    write_csv(out / "raw_capture_manifest.csv", [{"sim_time": row["sim_time"], "capture_order": row["capture_order"], "phase": row["phase"], "image_sha256": row["front_sha256"]} for row in dense_rows])
    write_csv(out / "frame_manifest.csv", [{"frame_index": row["frame_index"], "time": row["sim_time"], "front_path": row["front_path"], "front_sha256": row["front_sha256"], "side_path": "", "side_sha256": ""} for row in dense_rows])
    write_csv(out / "contact_sensor.csv", [{"frame_index": row["frame_index"], "time": row["sim_time"], "contact_present": int(row["contact_present"])} for row in dense_rows])
    write_csv(out / "gripper_command.csv", [{"frame_index": row["frame_index"], "time": row["sim_time"], "gripper_command": row["gripper_command"]} for row in dense_rows])
    write_csv(out / "oracle_timeline.csv", oracle_rows)
    write_jsonl(out / "observations_dense.jsonl", dense_rows)
    write_jsonl(out / "observations_action_end.jsonl", action_end)
    write_jsonl(out / "oracle_dense.jsonl", oracle_rows)
    write_jsonl(out / "events.jsonl", sim.events)
    write_csv(out / "stream_alignment.csv", [{"dense_frame_index": row["frame_index"], "dense_time": row["sim_time"], "action_end_time": next((x["sim_time"] for x in action_end if x["sim_time"] == row["sim_time"]), "")} for row in dense_rows])
    write_json(out / "termination.json", {"schema": "pathgraph_l2rar1_termination_v1", "done": True, "termination_type": "diagnostic_complete", "horizon": False, "time": float(sim.data.time)})
    meta = {"schema": "pathgraph_l2rar1_rollout_v1", "rollout_id": f"{family_id}_r{rollout_index:02d}", "root_family_id": family_id, "stratum": stratum, "split": split, "path": str(out.resolve()), "source_kind": "new_dynamic_mujoco_r1", "observation_intervention": stratum == "history_or_visual_unavailable", "dense_observation_count": len(dense_rows), "action_end_observation_count": len(action_end), "control_hz": 20, "api_calls": 0, "training_jobs": 0}
    write_json(out / "metadata.json", meta)
    return meta


def collect_development(output_root: Path, protocol: dict[str, Any], partition: str = "dev_fit", workers: int = 1) -> dict[str, Any]:
    from concurrent.futures import ProcessPoolExecutor
    dev = protocol["development"]
    start, stop = (0, 2) if partition == "dev_fit" else (2, 4)
    tasks = [(stratum, fi, ri, partition, output_root, int(protocol["new_seeds"]["development_family"]), int(protocol["new_seeds"]["development_rollout"])) for stratum in dev["strata"] for fi in range(start, stop) for ri in range(dev["rollouts_per_family"])]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            metadata = list(pool.map(lambda args: collect_one(*args), tasks))
    else:
        metadata = [collect_one(*task) for task in tasks]
    write_csv(output_root / f"{partition}_rollout_manifest.csv", metadata)
    return {"status": "COLLECTION_COMPLETE", "partition": partition, "families": len({row["root_family_id"] for row in metadata}), "rollouts": len(metadata), "dense_observations": sum(row["dense_observation_count"] for row in metadata), "api_calls": 0, "training_jobs": 0}
