"""Collect the bounded six-stratum L2RA dynamic development probes."""
from __future__ import annotations

import json
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from upgrade_v2.visual_refine_l2.dynamic_simulator import DynamicTabletop, family_spec
from upgrade_v2.visual_refine_l2.predicates import infer_dataset
from upgrade_v2.visual_refine_l2.renderer import TabletopRenderer
from upgrade_v2.visual_refine_l2.io import read_csv, sha256_file, write_csv, write_json, write_jsonl


STRATA = (
    "miss_without_prior_hold",
    "loss_after_observed_hold",
    "touch_without_hold_then_loss",
    "commanded_release",
    "long_gap_after_loss",
    "history_or_visual_unavailable",
)

REFERENCE = {
    "miss_without_prior_hold": ("missed_grasp_retry_required", "retry_grasp"),
    "loss_after_observed_hold": ("held_object_loss_recovery_required", "recover_object"),
    "touch_without_hold_then_loss": ("touch_without_stable_hold", "none"),
    "commanded_release": ("release_expected", "none"),
    "long_gap_after_loss": ("held_object_loss_recovery_required", "recover_object"),
    "history_or_visual_unavailable": ("needs_observation", "needs_observation"),
}


def _program(stratum: str) -> list[str]:
    if stratum == "miss_without_prior_hold":
        return ["observe_scene", "approach_object", "close_gripper", "retry", "lift", "verify"]
    if stratum == "loss_after_observed_hold":
        return ["observe_scene", "approach_object", "close_gripper", "lift", "transport_to_target", "verify", "recover", "lift"]
    if stratum == "touch_without_hold_then_loss":
        return ["observe_scene", "approach_object", "touch_contact", "separate_touch", "verify", "verify"]
    if stratum == "commanded_release":
        return ["observe_scene", "approach_object", "close_gripper", "lift", "open_gripper", "verify", "verify"]
    if stratum in {"long_gap_after_loss", "history_or_visual_unavailable"}:
        return ["observe_scene", "approach_object", "close_gripper", "lift", "transport_to_target", *(["verify"] * 6), "recover", "lift"]
    raise ValueError(stratum)


def _custom_perform(sim: DynamicTabletop, action: str) -> dict[str, Any]:
    if action not in {"touch_contact", "separate_touch"}:
        return sim.perform(action)
    sim.action_index += 1
    sim._active_control_sequence = []
    before = float(sim.data.time)
    if action == "touch_contact":
        # Close near the object without activating the weld. This produces
        # physical sensor contact but no held-object co-motion.
        sim.gripper_closed = True
        sim._advance(sim.object_xyz + np.array([0.0, 0.0, 0.13]), controls=2)
        sim._record_event("transient_contact", observable=True)
    else:
        sim._advance(sim.data.mocap_pos[0] + np.array([0.0, -0.28, 0.0]), controls=3)
        sim._record_event("transient_contact_lost", observable=True)
    return {"action_index": sim.action_index, "action": action, "start_time": round(before, 6),
            "end_time": round(float(sim.data.time), 6), "gripper_command": "closed" if sim.gripper_closed else "open",
            "contact_present": sim.contact_sensor(), "termination_reason": None,
            "low_level_control_sequence": sim._active_control_sequence}


def _collect_one(payload: tuple[str, int, int, int, str]) -> dict[str, Any]:
    stratum, stratum_index, family_index, rollout_index, output_text = payload
    family_id = f"L2RA_D_{stratum_index:02d}_{family_index:02d}_430000"
    family_seed = 430000 + stratum_index * 100 + family_index
    rollout_seed = 43100000 + stratum_index * 10000 + family_index * 100 + rollout_index
    scenario = "missed_grasp_then_retry" if stratum == "miss_without_prior_hold" else "slip_then_recover" if stratum in {"loss_after_observed_hold", "long_gap_after_loss", "history_or_visual_unavailable"} else "normal_pick_place"
    spec = family_spec(family_id, scenario, family_seed, rollout_seed - rollout_index)
    sim = DynamicTabletop(spec, rollout_seed)
    output = Path(output_text)
    actions, controls, contacts, commands, frames, oracle_rows = [], [], [], [], [], []
    qpos, qvel, weld, future = [], [], [], []
    with TabletopRenderer(sim.model) as renderer:
        for frame_index, action in enumerate(_program(stratum)):
            result = _custom_perform(sim, action)
            low = result.pop("low_level_control_sequence")
            controls.append({"action_index": result["action_index"], "action": action, "controls": low})
            actions.append(result)
            contacts.append({"frame_index": frame_index, "time": result["end_time"], "contact_present": int(result["contact_present"])})
            commands.append({"frame_index": frame_index, "time": result["end_time"], "gripper_command": result["gripper_command"]})
            image_path = output / "rgb/front" / f"frame_{frame_index:04d}.jpg"
            renderer.save_jpeg(renderer.render(sim.data, "front", spec.camera_jitter), image_path)
            oracle = sim.oracle_snapshot()
            oracle_rows.append({"frame_index": frame_index, "time": result["end_time"], "goal_stable": int(oracle["goal_stable"]), "weld_state": int(oracle["weld_state"]), "object_target_distance": oracle["object_target_distance"], "ncon": oracle["ncon"]})
            qpos.append(oracle["qpos"]); qvel.append(oracle["qvel"]); weld.append(oracle["weld_state"]); future.append(oracle["goal_stable"])
            frames.append({"frame_index": frame_index, "time": result["end_time"], "front_path": str(image_path.resolve()), "front_sha256": sha256_file(image_path), "side_path": "", "side_sha256": ""})
    termination = {"schema": "pathgraph_l2ra_termination_v1", "done": True, "termination_type": "diagnostic_complete", "terminal_failure": False, "horizon": False, "time": actions[-1]["end_time"]}
    write_csv(output / "actions.csv", actions); write_jsonl(output / "low_level_controls.jsonl", controls)
    write_csv(output / "contact_sensor.csv", contacts); write_csv(output / "gripper_command.csv", commands)
    write_csv(output / "frame_manifest.csv", frames); write_csv(output / "oracle_timeline.csv", oracle_rows)
    write_jsonl(output / "events.jsonl", sim.events); write_json(output / "termination.json", termination)
    np.savez_compressed(output / "online_observation.npz", timestamps=np.asarray([r["time"] for r in frames], dtype=np.float32), gripper_command_closed=np.asarray([r["gripper_command"] == "closed" for r in commands], dtype=np.uint8), contact_present=np.asarray([r["contact_present"] for r in contacts], dtype=np.uint8))
    np.savez_compressed(output / "oracle_diagnostic.npz", qpos=np.asarray(qpos), qvel=np.asarray(qvel), weld_state=np.asarray(weld), future_outcome=np.asarray(future), diagnostic_stratum=np.asarray(stratum))
    event_names = [row["event"] for row in sim.events]
    metadata = {"schema": "pathgraph_l2ra_probe_rollout_v1", "rollout_id": f"{family_id}_r{rollout_index:02d}", "root_family_id": family_id,
                "split": "dev_fit" if family_index < 2 else "dev_select", "stratum": stratum, "simulator_scenario_reference_only": scenario,
                "rollout_seed": rollout_seed, "path": str(output.resolve()), "frames": len(frames), "source_kind": "new_dynamic_mujoco_probe",
                "observation_intervention": stratum == "history_or_visual_unavailable", "events": event_names}
    write_json(output / "metadata.json", metadata)
    return metadata


def _mask_history(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = json.loads(json.dumps(predictions))
    cutoff = max(1, len(output) // 3)
    for row in output[:cutoff]:
        for key in ("contact_present", "stable_hold_observed", "object_moves_with_gripper", "contact_recently_lost"):
            row["predicates"][key] = "unknown"
    return output


def _decision_frame(predictions: list[dict[str, Any]], stratum: str) -> int | None:
    if stratum == "history_or_visual_unavailable": return 0
    for i, row in enumerate(predictions):
        p = row["predicates"]
        if stratum == "commanded_release" and i > 0 and p.get("gripper_command_open") == "true" and predictions[i - 1]["predicates"].get("gripper_command_closed") == "true": return i
        if stratum == "touch_without_hold_then_loss" and i > 0 and p.get("contact_present") == "false" and predictions[i - 1]["predicates"].get("contact_present") == "true": return i
        if stratum in {"miss_without_prior_hold", "loss_after_observed_hold", "long_gap_after_loss"} and p.get("gripper_command_closed") == "true" and p.get("contact_present") == "false": return i
    return None


def collect_probes(resolved: dict[str, Any], protocol: dict[str, Any], output_root: Path, partition: str = "development", workers: int = 1) -> dict[str, Any]:
    expected = protocol["new_development"]
    if expected["family_count"] != 24 or expected["rollouts_per_family"] != 4:
        raise ValueError("unexpected protocol scale")
    data_root = output_root / "rollouts"
    tasks = []
    for si, stratum in enumerate(STRATA):
        for fi in range(4):
            for ri in range(4):
                path = data_root / f"L2RA_D_{si:02d}_{fi:02d}_430000" / f"rollout_{ri:02d}"
                tasks.append((stratum, si, fi, ri, str(path)))
    existing_metadata = sorted(data_root.rglob("metadata.json"))
    metadata = [json.loads(path.read_text(encoding="utf-8")) for path in existing_metadata] if len(existing_metadata) == 96 else []
    if not metadata:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_collect_one, task) for task in tasks]
                for future in as_completed(futures): metadata.append(future.result())
        else:
            metadata = [_collect_one(task) for task in tasks]
    metadata.sort(key=lambda row: row["rollout_id"])
    write_csv(output_root / "probe_rollout_manifest.csv", metadata)
    thresholds = Path(next(item["resolved_path"] for item in resolved["frozen_sources"] if item["logical_id"] == "frozen:locks/predicate_thresholds.json"))
    pred_root = output_root / "predictions"; pred_manifest = output_root / "prediction_manifest.csv"
    infer_dataset(data_root, thresholds, pred_root, pred_manifest, camera="front")
    pred_by_id = {row["rollout_id"]: row for row in read_csv(pred_manifest)}
    records, manifest_rows = [], []
    for meta in metadata:
        predictions = [json.loads(line) for line in Path(pred_by_id[meta["rollout_id"]]["prediction_path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
        if meta["observation_intervention"]: predictions = _mask_history(predictions)
        event_type, action = REFERENCE[meta["stratum"]]
        reference = {"event_type": event_type, "action_class": action,
                     "label_status": "observation_intervention" if meta["observation_intervention"] else "reference_labeled_from_simulator_events_and_state",
                     "source_kind": "new_dynamic_mujoco_probe", "observable_at_decision": not meta["observation_intervention"],
                     "history_complete": not meta["observation_intervention"], "decision_frame_index": _decision_frame(predictions, meta["stratum"])}
        content_group = hashlib.sha256(json.dumps(predictions, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        records.append({"probe_id": meta["rollout_id"], "root_family_id": meta["root_family_id"], "stratum": meta["stratum"], "split": meta["split"], "content_group_sha256": content_group, "observations": predictions, "reference": reference})
        manifest_rows.append({"probe_id": meta["rollout_id"], "root_family_id": meta["root_family_id"], "stratum": meta["stratum"], "split": meta["split"], "source_kind": meta["source_kind"], "observation_intervention": meta["observation_intervention"], "observation_count": len(predictions), "decision_frame_index": reference["decision_frame_index"], "content_group_sha256": content_group})
    with (output_root / "probe_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in records: handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_csv(output_root / "probe_manifest.csv", manifest_rows)
    group_rows = []
    for digest_value in sorted({row["content_group_sha256"] for row in manifest_rows}):
        members = [row for row in manifest_rows if row["content_group_sha256"] == digest_value]
        group_rows.append({"content_group_sha256": digest_value, "rollout_count": len(members),
                           "family_count": len({row["root_family_id"] for row in members}),
                           "strata": ";".join(sorted({row["stratum"] for row in members})),
                           "member_rollout_ids": ";".join(row["probe_id"] for row in members)})
    write_csv(output_root / "content_duplicate_groups.csv", group_rows)
    lock = {"schema": "pathgraph_l2ra_probe_lock_v1", "status": "LOCKED_BEFORE_CANDIDATE_SELECTION", "family_count": 24, "rollouts": 96,
            "strata": list(STRATA), "fit_families_per_stratum": 2, "select_families_per_stratum": 2,
            "new_dynamic_rollouts": 96, "observation_intervention_rollouts": 16,
            "unique_observation_content_groups": len(group_rows),
            "intervention_parent_family_rule": "masked observations remain in their physical parent family",
            "repeat_interpretation": "rollouts sharing a content hash are repeated observations, not independent units; root_family_id remains the statistical unit",
            "api_calls": 0, "training_jobs": 0}
    write_json(output_root / "probe_lock.json", lock)
    return {"status": "PROBES_READY", **{key: lock[key] for key in ("family_count", "rollouts", "new_dynamic_rollouts", "observation_intervention_rollouts")}}
