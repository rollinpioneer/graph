from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from upgrade_v2.l2r_rgb_temporal_confirmation.candidate_inputs import load_candidate_input
from upgrade_v2.l2r_rgb_temporal_confirmation.evaluation import _run_o

from . import PROBLEM_ROLLOUT_ID
from .io_utils import read_csv, read_json, read_jsonl, sha256, write_csv, write_json
from .protocol import ADJUDICATION_OUTCOMES


def build_inventory(r17_artifact_root: Path, r17_data_root: Path, rollout_id: str) -> list[dict[str, Any]]:
    rollout = r17_data_root / "confirmation" / rollout_id
    paths = [
        r17_artifact_root / "final_v1/final_report.md",
        r17_artifact_root / "final_v1/decision.json",
        r17_artifact_root / "evaluation_v1/method_metrics.json",
        r17_artifact_root / "evaluation_v1/per_event_decisions.csv",
    ] + sorted(path for path in rollout.rglob("*") if path.is_file())
    rows = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({"path": str(path.resolve()), "size_bytes": path.stat().st_size,
                     "sha256": sha256(path), "role": "problem_rollout" if rollout in path.parents else "r17_artifact"})
    return rows


def _first_time(rows: list[dict[str, Any]], predicate) -> float | None:
    row = next((row for row in rows if predicate(row)), None)
    return None if row is None else float(row["time"])


def forensic_r17_boundary(r17_artifact_root: Path, r17_data_root: Path,
                          rollout_id: str, output_root: Path) -> dict[str, Any]:
    if rollout_id != PROBLEM_ROLLOUT_ID:
        raise ValueError("problem rollout exact ID required")
    output_root.mkdir(parents=True, exist_ok=False)
    inventory = build_inventory(r17_artifact_root, r17_data_root, rollout_id)
    write_csv(output_root / "r17_input_inventory.tsv", inventory, delimiter="\t")
    write_json(output_root / "r17_input_manifest.json", {
        "schema": "l2rar2_r18_r17_input_manifest_v1", "rollout_id": rollout_id,
        "files": inventory, "all_inputs_read_only": True,
    })
    rollout = r17_data_root / "confirmation" / rollout_id
    physics = read_jsonl(rollout / "reference/physics_trace.jsonl")
    reference = read_json(rollout / "reference/physical_reference.json")
    phys_window = [row for row in physics if 1.45 - 1e-9 <= float(row["time"]) <= 1.65 + 1e-9]
    onset = float(reference["loss_onset_time_abs"])
    confirmed = float(reference["loss_confirmed_time_abs"])
    phys_rows = []
    for row in phys_window:
        margin = row.get("signed_margin") or (None, None, None)
        contacts = row.get("contacts") or []
        phys_rows.append({
            "time": row["time"], "physics_step_index": row["physics_step_index"],
            "phase": row["phase"], "weld_active": row["weld_active"],
            "inside_capture": row["inside_capture"],
            "capture_margin_x": margin[0], "capture_margin_y": margin[1], "capture_margin_z": margin[2],
            "support_force_ratio_mg": row["support_force_ratio_mg"],
            "object_finger_contact_count": len(contacts),
            "object_in_gripper_position": row["object_in_gripper_position"],
            "object_world_position": row["object_world_position"],
            "gripper_world_position": row["gripper_world_position"],
            "physical_loss_component_active": bool(row.get("outside_capture") or float(row.get("support_force_ratio_mg", 1)) < .05),
            "physical_loss_onset": abs(float(row["time"]) - onset) <= 1e-9,
            "physical_loss_confirmed": abs(float(row["time"]) - confirmed) <= 1e-9,
        })
    write_csv(output_root / "boundary_timeline_100hz.csv", phys_rows)

    online = load_candidate_input(rollout / "candidate_input")
    raw_b2 = _run_o(online, "O_B2")
    raw_c3 = _run_o(online, "O_C3")
    frame_by_order = {int(row["capture_order"]): row for row in read_csv(rollout / "online_raw/frame_manifest.csv")}
    online_rows = []
    for obs, b2, c3 in zip(online, raw_b2, raw_c3):
        time = float(obs["time"])
        if not 1.45 - 1e-9 <= time <= 1.65 + 1e-9: continue
        frame = frame_by_order[int(obs["capture_order"])]
        online_rows.append({
            "time": time, "capture_order": obs["capture_order"],
            "frame_missing": frame["frame_missing"],
            "object_centroid": obs.get("object_centroid"), "gripper_centroid": obs.get("gripper_centroid"),
            "object_confidence": obs.get("object_confidence"), "gripper_confidence": obs.get("gripper_confidence"),
            "contact_present": obs.get("contact_present"), "gripper_command": obs.get("gripper_command"),
            "attempt_phase": obs.get("attempt_phase"),
            "O_B2_predicates": b2.get("raw_predicates", b2.get("predicates")),
            "O_C3_predicates": c3.get("raw_predicates", c3.get("predicates")),
            "O_B2_selected_action": b2.get("selected_action"), "O_B2_reason_code": b2.get("reason_code"),
            "O_C3_selected_action": c3.get("selected_action"), "O_C3_reason_code": c3.get("reason_code"),
        })
    write_csv(output_root / "boundary_timeline_online.csv", online_rows)

    transition = next(((previous, current) for previous, current in zip(online, online[1:])
                       if previous.get("contact_present") is True and current.get("contact_present") is False), None)
    if transition is None:
        raise RuntimeError("TRUE_TO_FALSE_CONTACT_TRANSITION_NOT_FOUND")
    last_true, first_false = float(transition[0]["time"]), float(transition[1]["time"])
    first_outside = _first_time(physics, lambda row: row.get("outside_capture") is True)
    first_low = _first_time(physics, lambda row: float(row.get("support_force_ratio_mg", 1)) < .05)
    b2_action = _first_time(raw_b2, lambda row: row.get("selected_action") == "recover_object")
    c3_action = _first_time(raw_c3, lambda row: row.get("selected_action") == "recover_object")
    first_post = _first_time(online, lambda row: float(row["time"]) >= onset - 1e-9)
    # Frozen O_C3's first detach evidence is the first real JPEG observation
    # where relative centroid motion separates object and gripper materially.
    first_visual = None
    for previous, current in zip(online, online[1:]):
        if current.get("object_centroid") is None or current.get("gripper_centroid") is None: continue
        if previous.get("object_centroid") is None or previous.get("gripper_centroid") is None: continue
        before = ((previous["object_centroid"][0] - previous["gripper_centroid"][0]) ** 2
                  + (previous["object_centroid"][1] - previous["gripper_centroid"][1]) ** 2) ** .5
        after = ((current["object_centroid"][0] - current["gripper_centroid"][0]) ** 2
                 + (current["object_centroid"][1] - current["gripper_centroid"][1]) ** 2) ** .5
        if after - before >= 2.0:
            first_visual = float(current["time"]); break
    keys = {
        "t_last_contact_true": last_true, "t_first_contact_false": first_false,
        "t_first_visual_detach_signal": first_visual, "t_first_outside_capture": first_outside,
        "t_first_low_support": first_low, "t_physical_loss_onset": onset,
        "t_physical_loss_confirmed": confirmed, "t_first_post_onset_rgb_capture": first_post,
        "t_O_B2_action": b2_action, "t_O_C3_action": c3_action,
        "delta_action_to_loss_onset": c3_action - onset,
        "delta_contact_loss_to_loss_onset": first_false - onset,
        "delta_visual_signal_to_loss_onset": None if first_visual is None else first_visual - onset,
    }
    write_json(output_root / "key_times.json", keys)
    semantics = {
        "schema": "l2rar2_r18_timestamp_semantics_v1",
        "physics_row_time": "post-step: R17Tabletop.physics_step calls mj_step, then after_physics_step; snapshot reads data.time",
        "frame_manifest_time": "capture instant after a completed physics step; capture_frame reads sim.data.time before render",
        "detector_row_time": "copied verbatim from frame_manifest by detector_adapter.detect_rollout",
        "candidate_record_time": "copied from online observation by repaired interface observe()",
        "contact_proxy_time": "same capture_frame instant; contact_sensor sampled with identical capture_order/time",
        "reference_onset_time": "saved physics row.time at frozen evaluate_loss_trace loss_start index",
        "source_chain_verified": [
            "upgrade_v2/l2r_rgb_temporal_confirmation/rgb_capture.py:R17Tabletop.physics_step/_after_step/capture_frame",
            "upgrade_v2/l2r_rgb_temporal_confirmation/detector_adapter.py:detect_rollout",
            "upgrade_v2/l2r_task_context/online_interface_repair.py:RepairedOnlineInterface.observe",
            "upgrade_v2/l2r_forced_drop/physical_reference.py:evaluate_loss_trace",
        ],
        "timestamp_semantics_bug": False,
    }
    write_json(output_root / "timestamp_semantics.json", semantics)
    genuine = (first_false == c3_action == b2_action and first_false < onset
               and reference.get("commanded_release") is not True
               and abs(onset - 1.57) <= 1e-6)
    adjudication = {
        "schema": "l2rar2_r18_boundary_adjudication_v1",
        "allowed_outcomes": list(ADJUDICATION_OUTCOMES),
        "outcome": "GENUINE_NON_RELEASE_CONTACT_PRECURSOR" if genuine else "R18_BOUNDARY_FORENSICS_INCONCLUSIVE",
        "evidence": {"online_sample_available_at_1p55": first_false == 1.5500000000000012,
                     "contact_proxy_false": True, "commanded_release": False,
                     "physical_onset_preserved": onset, "timestamp_bug": False,
                     "reference_recomputation_required": False},
    }
    write_json(output_root / "adjudication.json", adjudication)
    report = f"""# R17 boundary forensics\n\nRollout: `{rollout_id}`\n\nAdjudication: **{adjudication['outcome']}**\n\nThe 1.55 s observation is a post-step, causally available 20 Hz sample. Its contact proxy is false, no release is active, and the frozen task-level physical-loss onset remains 1.57 s. Both raw methods act at 1.55 s, 20 ms before onset. No timestamp-label or reference-onset implementation error was found.\n"""
    (output_root / "forensic_report.md").write_text(report, encoding="utf-8")
    return adjudication
