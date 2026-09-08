"""Dynamic rollout collection, split locks, and dataset validation."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .dynamic_simulator import SCENARIOS, DynamicTabletop, FamilySpec, family_spec
from .io import FORBIDDEN_ONLINE_KEYS, find_forbidden_keys, now_iso, read_csv, read_json, sha256_file, write_csv, write_json, write_jsonl, write_report
from .renderer import TabletopRenderer
from .scripted_controller import action_program


def _spec_dict(spec: FamilySpec) -> dict[str, Any]:
    return {name: getattr(spec, name) for name in spec.__dataclass_fields__}


def _collect_rollout(payload: tuple[dict[str, Any], int, str, bool]) -> dict[str, Any]:
    raw_spec, rollout_index, output_text, render_side = payload
    spec = FamilySpec(**raw_spec)
    output = Path(output_text)
    rollout_id = f"{spec.root_family_id}_r{rollout_index:02d}"
    rollout_seed = spec.rollout_seed_base + rollout_index
    simulator = DynamicTabletop(spec, rollout_seed)
    actions = action_program(spec.scenario, rollout_index)
    action_rows = []
    contact_rows = []
    command_rows = []
    oracle_rows = []
    frame_rows = []
    qpos, qvel, weld, oracle_goal = [], [], [], []
    online_contact, online_gripper, online_action = [], [], []
    with TabletopRenderer(simulator.model) as renderer:
        for frame_index, action in enumerate(actions):
            result = simulator.perform(action)
            action_rows.append(result)
            contact_rows.append({"frame_index": frame_index, "time": result["end_time"], "contact_present": int(result["contact_present"])})
            command_rows.append({"frame_index": frame_index, "time": result["end_time"], "gripper_command": result["gripper_command"]})
            front_path = output / "rgb/front" / f"frame_{frame_index:04d}.jpg"
            front = renderer.render(simulator.data, "front", spec.camera_jitter)
            renderer.save_jpeg(front, front_path)
            side_path = output / "rgb/side" / f"frame_{frame_index:04d}.jpg"
            if render_side:
                side = renderer.render(simulator.data, "side", spec.camera_jitter)
                renderer.save_jpeg(side, side_path)
            oracle = simulator.oracle_snapshot()
            oracle_rows.append({
                "frame_index": frame_index, "time": result["end_time"], "goal_stable": int(oracle["goal_stable"]),
                "weld_state": int(oracle["weld_state"]), "object_target_distance": oracle["object_target_distance"], "ncon": oracle["ncon"],
            })
            qpos.append(oracle["qpos"]); qvel.append(oracle["qvel"]); weld.append(oracle["weld_state"]); oracle_goal.append(oracle["goal_stable"])
            online_contact.append(result["contact_present"]); online_gripper.append(result["gripper_command"] == "closed"); online_action.append(action)
            frame_rows.append({
                "frame_index": frame_index, "time": result["end_time"], "front_path": str(front_path.resolve()),
                "front_sha256": sha256_file(front_path), "side_path": str(side_path.resolve()) if render_side else "",
                "side_sha256": sha256_file(side_path) if render_side else "",
            })
    unresolved = spec.scenario == "distractor_object_ambiguity" and rollout_index == 3
    final_goal = bool(oracle_goal[-1]) if oracle_goal else False
    termination = {
        "schema": "pathgraph_l2r_termination_v1", "done": not unresolved,
        "termination_type": "horizon" if unresolved else "goal_terminal" if final_goal else "failure_terminal",
        "terminal_failure": bool(not unresolved and not final_goal), "horizon": unresolved, "time": action_rows[-1]["end_time"],
    }
    write_csv(output / "actions.csv", action_rows)
    write_csv(output / "contact_sensor.csv", contact_rows)
    write_csv(output / "gripper_command.csv", command_rows)
    write_json(output / "termination.json", termination)
    write_jsonl(output / "events.jsonl", simulator.events)
    write_csv(output / "oracle_timeline.csv", oracle_rows)
    write_csv(output / "frame_manifest.csv", frame_rows)
    np.savez_compressed(
        output / "online_observation.npz",
        timestamps=np.asarray([row["time"] for row in frame_rows], dtype=np.float32),
        action_code=np.asarray([action_program(spec.scenario, rollout_index).index(value) for value in online_action], dtype=np.int16),
        gripper_command_closed=np.asarray(online_gripper, dtype=np.uint8),
        contact_present=np.asarray(online_contact, dtype=np.uint8),
    )
    np.savez_compressed(
        output / "oracle_diagnostic.npz", qpos=np.asarray(qpos), qvel=np.asarray(qvel), weld_state=np.asarray(weld),
        future_outcome=np.asarray(oracle_goal), scenario=np.asarray(spec.scenario),
    )
    event_names = [row["event"] for row in simulator.events]
    metadata = {
        "schema": "pathgraph_l2r_rollout_metadata_v1", "rollout_id": rollout_id, "root_family_id": spec.root_family_id,
        "split": "diagnostic_metadata_only", "scenario": spec.scenario, "rollout_seed": rollout_seed,
        "active_object": True, "physics_hz": simulator.physics_hz, "control_hz": simulator.control_hz, "render_hz": simulator.render_hz,
        "frames": len(frame_rows), "front_frames": len(frame_rows), "side_frames": len(frame_rows) if render_side else 0,
        "final_goal_stable": final_goal, "missed_grasp": "missed_grasp" in event_names, "contact_loss": "contact_lost" in event_names,
        "recovery_attempt": any(action in {"retry", "recover"} for action in actions), "recovery_achieved": "recovery_achieved" in event_names,
        "target_blocked": spec.scenario == "target_occupied", "ambiguous_or_occluded": spec.scenario in {"distractor_object_ambiguity", "primary_view_occlusion"},
        "online_observation_keys": ["timestamps", "action_code", "gripper_command_closed", "contact_present"],
        "oracle_diagnostic_keys": ["qpos", "qvel", "weld_state", "future_outcome", "scenario"],
    }
    write_json(output / "metadata.json", metadata)
    return {**metadata, "path": str(output.resolve()), "termination_type": termination["termination_type"]}


def make_family_specs(family_count: int, scenarios: list[str], family_seed: int, rollout_seed_base: int, prefix: str) -> list[FamilySpec]:
    if family_count % len(scenarios):
        raise ValueError("family count must be balanced across scenarios")
    per_scenario = family_count // len(scenarios)
    specs = []
    for scenario_index, scenario in enumerate(scenarios):
        if scenario not in SCENARIOS:
            raise ValueError(scenario)
        for local_index in range(per_scenario):
            family_id = f"{prefix}_{scenario_index:02d}_{local_index:02d}_{family_seed}"
            seed = family_seed + scenario_index * 100 + local_index
            specs.append(family_spec(family_id, scenario, seed, rollout_seed_base + len(specs) * 100))
    return specs


def collect_dataset(split: str, family_count: int, rollouts_per_family: int, scenarios: list[str], family_seed: int, rollout_seed_base: int, output_root: Path, manifest: Path, family_split: Path | None, workers: int = 1, render_side: bool = True) -> dict[str, Any]:
    prefix = {"pilot": "L2P", "development": "L2D", "fresh_confirmation": "L2C"}.get(split, f"L2_{split}")
    specs = make_family_specs(family_count, scenarios, family_seed, rollout_seed_base, prefix)
    tasks = []
    split_rows = []
    for scenario_index, scenario in enumerate(scenarios):
        scenario_specs = [spec for spec in specs if spec.scenario == scenario]
        fit_count = (len(scenario_specs) * 2) // 3 if split == "development" else 0
        for local_index, spec in enumerate(scenario_specs):
            subset = "dev_fit" if split == "development" and local_index < fit_count else "dev_select" if split == "development" else split
            split_rows.append({"root_family_id": spec.root_family_id, "scenario": spec.scenario, "split": subset, "family_seed": family_seed + scenario_index * 100 + local_index})
            for rollout_index in range(rollouts_per_family):
                rollout_dir = output_root / spec.root_family_id / f"rollout_{rollout_index:02d}"
                tasks.append((_spec_dict(spec), rollout_index, str(rollout_dir), render_side))
    results = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_collect_rollout, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
    else:
        results = [_collect_rollout(task) for task in tasks]
    results.sort(key=lambda row: row["rollout_id"])
    split_by_family = {row["root_family_id"]: row["split"] for row in split_rows}
    for row in results:
        row["split"] = split_by_family[row["root_family_id"]]
    fields = ["rollout_id", "root_family_id", "split", "scenario", "rollout_seed", "path", "active_object", "frames", "front_frames", "side_frames", "termination_type", "final_goal_stable", "missed_grasp", "contact_loss", "recovery_attempt", "recovery_achieved", "target_blocked", "ambiguous_or_occluded"]
    write_csv(manifest, results, fields)
    if family_split:
        write_csv(family_split, sorted(split_rows, key=lambda row: row["root_family_id"]))
    return {"status": "DYNAMIC_TABLETOP_DATASET_READY", "split": split, "families": len(specs), "rollouts": len(results), "manifest": str(manifest), "workers": workers}


def verify_dynamic_simulator(output_root: Path, metrics: Path, seed: int, workers: int = 1) -> dict[str, Any]:
    manifest = output_root / "pilot_rollout_manifest.csv"
    family_split = output_root / "pilot_family_split.csv"
    collect_dataset("pilot", 8, 2, list(SCENARIOS), seed, seed * 10, output_root, manifest, family_split, workers)
    rows = read_csv(manifest)
    checks = {
        "successful_placement": any(row["final_goal_stable"] == "True" for row in rows),
        "no_move_completion": any(row["scenario"] == "already_satisfied_stable" and row["final_goal_stable"] == "True" for row in rows),
        "missed_grasp": any(row["missed_grasp"] == "True" for row in rows),
        "contact_loss": any(row["contact_loss"] == "True" for row in rows),
        "successful_recovery": any(row["recovery_achieved"] == "True" for row in rows),
        "target_blocked": any(row["target_blocked"] == "True" for row in rows),
        "ambiguous_or_occluded": any(row["ambiguous_or_occluded"] == "True" for row in rows),
    }
    result = {"schema": "pathgraph_l2r_dynamic_pilot_v1", "status": "DYNAMIC_TABLETOP_DATASET_READY" if all(checks.values()) else "DYNAMIC_SIMULATOR_FAILED", "families": 8, "rollouts": len(rows), "checks": checks}
    write_json(metrics, result)
    return result


def validate_dynamic_dataset(root: Path, manifest: Path, family_split: Path, output: Path, event_counts: Path, report: Path, expected_rollouts: int = 192, expected_families: int = 48) -> dict[str, Any]:
    rows = read_csv(manifest)
    splits = read_csv(family_split)
    failures = []
    families = {row["root_family_id"] for row in rows}
    fit = {row["root_family_id"] for row in splits if row["split"] == "dev_fit"}
    select = {row["root_family_id"] for row in splits if row["split"] == "dev_select"}
    if len(rows) != expected_rollouts:
        failures.append(f"expected {expected_rollouts} rollouts, got {len(rows)}")
    if len(families) != expected_families:
        failures.append(f"expected {expected_families} families, got {len(families)}")
    if fit & select or families != fit | select:
        failures.append("family split leakage or omission")
    counts = {"failure_occurrences": 0, "recovery_attempts": 0, "recovery_achievements": 0, "stable_goal_confirmations": 0, "horizon": 0, "terminal_failure": 0}
    image_rows = []
    rollout_rows = []
    for row in rows:
        path = Path(row["path"])
        required = [path / name for name in ("actions.csv", "contact_sensor.csv", "gripper_command.csv", "termination.json", "online_observation.npz", "oracle_diagnostic.npz", "events.jsonl", "metadata.json", "frame_manifest.csv")]
        missing = [str(item) for item in required if not item.is_file()]
        if missing:
            failures.extend(missing); continue
        frame_manifest = read_csv(path / "frame_manifest.csv")
        if not frame_manifest or any(not Path(frame["front_path"]).is_file() for frame in frame_manifest):
            failures.append(f"front image missing: {row['rollout_id']}")
        with np.load(path / "online_observation.npz") as online:
            forbidden = set(online.files) & FORBIDDEN_ONLINE_KEYS
            if forbidden:
                failures.append(f"oracle leakage {row['rollout_id']}: {sorted(forbidden)}")
            if any(not np.isfinite(online[key]).all() for key in online.files):
                failures.append(f"non-finite online observation: {row['rollout_id']}")
        metadata = read_json(path / "metadata.json")
        if metadata["missed_grasp"] or metadata["contact_loss"]:
            counts["failure_occurrences"] += 1
        counts["recovery_attempts"] += int(metadata["recovery_attempt"])
        counts["recovery_achievements"] += int(metadata["recovery_achieved"])
        counts["stable_goal_confirmations"] += int(metadata["final_goal_stable"])
        termination = read_json(path / "termination.json")
        counts["horizon"] += int(termination["horizon"])
        counts["terminal_failure"] += int(termination["terminal_failure"])
        for frame in frame_manifest:
            image_rows.append({"rollout_id": row["rollout_id"], "view": "front", "path": frame["front_path"], "sha256": frame["front_sha256"]})
            if frame["side_path"]:
                image_rows.append({"rollout_id": row["rollout_id"], "view": "side", "path": frame["side_path"], "sha256": frame["side_sha256"]})
        rollout_rows.append({"rollout_id": row["rollout_id"], "path": row["path"], "size_bytes": sum(item.stat().st_size for item in path.rglob("*") if item.is_file()), "online_npz_sha256": sha256_file(path / "online_observation.npz"), "oracle_npz_sha256": sha256_file(path / "oracle_diagnostic.npz"), "purpose": "dynamic MuJoCo primitive rollout; raw payload externalized"})
    minimums = {"failure_occurrences": 20, "recovery_attempts": 20, "recovery_achievements": 15, "stable_goal_confirmations": 20}
    for key, minimum in minimums.items():
        if counts[key] < minimum:
            failures.append(f"{key}: expected >= {minimum}, got {counts[key]}")
    write_csv(output.parent.parent / "manifests/image_manifest.tsv", image_rows, delimiter="\t")
    write_csv(output.parent.parent / "manifests/rollout_manifest.tsv", rollout_rows, delimiter="\t")
    write_csv(
        output.parent.parent / "manifests/large_file_manifest.tsv",
        [{"path": row["path"], "size_bytes": row["size_bytes"], "artifact_type": "raw_dynamic_rollout_directory", "reason_omitted": "per-frame RGB and NPZ payload externalized", "recovery_method": "rerun the locked family and rollout seeds"} for row in rollout_rows],
        ["path", "size_bytes", "artifact_type", "reason_omitted", "recovery_method"], delimiter="\t",
    )
    event_rows = [{"event": key, "count": value} for key, value in counts.items()]
    write_csv(event_counts, event_rows)
    result = {"schema": "pathgraph_l2r_dynamic_dataset_gate_v1", "status": "DYNAMIC_TABLETOP_DATASET_READY" if not failures else "DYNAMIC_DATASET_FAILED", "rollouts": len(rows), "families": len(families), "dev_fit_families": len(fit), "dev_select_families": len(select), "family_leakage": len(fit & select), "event_counts": counts, "failures": failures}
    write_json(output, result)
    write_report(report, "Dynamic Dataset Summary", [("status", result["status"]), ("rollouts", len(rows)), ("families", len(families)), ("failure occurrences", counts["failure_occurrences"]), ("recovery achievements", counts["recovery_achievements"]), ("stable goals", counts["stable_goal_confirmations"])])
    return result


def generate_fresh(generation_lock: Path, output_root: Path, manifest: Path, family_lock: Path, workers: int = 1) -> dict[str, Any]:
    lock = read_json(generation_lock)
    if lock.get("opened") is not False or lock.get("used") is not False:
        raise ValueError("fresh confirmation generation lock is not sealed and unused")
    result = collect_dataset("fresh_confirmation", lock["family_count"], lock["rollouts_per_family"], lock["scenarios"], lock["family_seed"], lock["rollout_seed_base"], output_root, manifest, None, workers)
    rows = read_csv(manifest)
    families = sorted({row["root_family_id"] for row in rows})
    write_json(family_lock, {
        "schema": "pathgraph_l2r_fresh_family_lock_v1", "status": "LOCKED", "created_at": now_iso(),
        "generation_lock_sha256": sha256_file(generation_lock), "families": families, "family_count": len(families),
        "rollouts": len(rows), "manifest_sha256": sha256_file(manifest), "opened_once": True,
    })
    return {**result, "status": "L2R_FRESH_FAMILIES_GENERATED", "family_lock": str(family_lock)}
