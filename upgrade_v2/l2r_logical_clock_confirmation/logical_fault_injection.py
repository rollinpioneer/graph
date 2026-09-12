from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .io_utils import read_csv, read_json, read_jsonl, sha256, write_csv, write_json, write_jsonl

STREAM_FILES = ("detections_rgb", "contact_proxy", "attempt_lifecycle", "gripper_commands", "request_context")
FAULT_CASE_IDS = ("T3_transport_single_contact_proxy_dropout",
                  "T4_transport_same_time_contact_false_then_true")


def _hash_tree(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob("*")) if path.is_file()}


def _periodic_transport_orders(frames: list[dict[str, str]]) -> list[int]:
    return [int(row["capture_order"]) for row in frames
            if row.get("capture_kind") == "periodic" and str(row.get("phase", "")).startswith("transport")]


def _renumber(row: dict[str, Any], order: int) -> dict[str, Any]:
    value = copy.deepcopy(row); value["capture_order"] = order; return value


def build_rollout_candidate_input(rollout_root: Path) -> dict[str, Any]:
    case_id = read_json(rollout_root / "metadata.json")["case_id"]
    frames = read_csv(rollout_root / "online_raw/frame_manifest.csv")
    streams = {
        "detections_rgb": read_jsonl(rollout_root / "detector_output/detections_rgb.jsonl"),
        "contact_proxy": read_jsonl(rollout_root / "online_raw/contact_proxy.jsonl"),
        "attempt_lifecycle": read_jsonl(rollout_root / "online_raw/attempt_lifecycle.jsonl"),
        "gripper_commands": read_jsonl(rollout_root / "online_raw/gripper_commands.jsonl"),
        "request_context": read_jsonl(rollout_root / "online_raw/request_context.jsonl"),
    }
    orders = [int(row["capture_order"]) for row in streams["detections_rgb"]]
    if any([int(row["capture_order"]) for row in stream] != orders for stream in streams.values()):
        raise RuntimeError("SOURCE_STREAM_ALIGNMENT_ERROR")
    raw_contact_hash = sha256(rollout_root / "online_raw/contact_proxy.jsonl")
    reference_before = _hash_tree(rollout_root / "reference")
    periodic = _periodic_transport_orders(frames)
    target = periodic[3] if case_id.startswith(("T3_", "T4_")) and len(periodic) >= 4 else None
    faults: list[dict[str, Any]] = []
    candidate = {key: [] for key in STREAM_FILES}
    logical_rows = []
    source_frame = {int(row["capture_order"]): row for row in frames}
    for source_index, source_order in enumerate(orders):
        candidate_order = source_order + (1 if target is not None and case_id.startswith("T4_") and source_order > target else 0)
        if source_order == target and case_id.startswith("T4_"):
            if streams["contact_proxy"][source_index].get("contact_present") is not True:
                raise RuntimeError("T4_SOURCE_CONTACT_MUST_BE_TRUE")
            for key in STREAM_FILES:
                first = _renumber(streams[key][source_index], candidate_order)
                second = _renumber(streams[key][source_index], candidate_order + 1)
                if key == "contact_proxy": first["contact_present"], second["contact_present"] = False, True
                candidate[key].extend((first, second))
            frame = source_frame[source_order]
            logical_rows.extend((
                {"source_capture_order": source_order, "candidate_capture_order": candidate_order,
                 "time": float(streams["detections_rgb"][source_index]["time"]), "synthetic_duplicate": False,
                 "fault_role": "T4_FALSE_A", "jpeg_sha256": frame.get("jpeg_sha256")},
                {"source_capture_order": source_order, "candidate_capture_order": candidate_order + 1,
                 "time": float(streams["detections_rgb"][source_index]["time"]), "synthetic_duplicate": True,
                 "fault_role": "T4_TRUE_B", "jpeg_sha256": frame.get("jpeg_sha256")},
            ))
            canonical = lambda row: json.dumps({key: value for key, value in row.items() if key != "capture_order"}, sort_keys=True)
            if canonical(candidate["detections_rgb"][-2]) != canonical(candidate["detections_rgb"][-1]):
                raise RuntimeError("T4_DETECTION_DUPLICATION_MISMATCH")
            faults.append({"fault_type": "T4_SAME_TIME_FALSE_TRUE", "target_capture_order": candidate_order,
                           "source_capture_order": source_order, "source_contact": True,
                           "injected_contact": [False, True], "source_time": float(streams["detections_rgb"][source_index]["time"]),
                           "duplicate_capture_order": candidate_order + 1, "same_jpeg_sha256": frame.get("jpeg_sha256"),
                           "reference_unchanged": True})
            continue
        for key in STREAM_FILES:
            row = _renumber(streams[key][source_index], candidate_order)
            if key == "contact_proxy" and source_order == target and case_id.startswith("T3_"):
                if row.get("contact_present") is not True: raise RuntimeError("T3_SOURCE_CONTACT_MUST_BE_TRUE")
                row["contact_present"] = False
            candidate[key].append(row)
        logical_rows.append({"source_capture_order": source_order, "candidate_capture_order": candidate_order,
                             "time": float(streams["detections_rgb"][source_index]["time"]),
                             "synthetic_duplicate": False,
                             "fault_role": "T3_SINGLE_FALSE" if source_order == target and case_id.startswith("T3_") else None,
                             "jpeg_sha256": source_frame[source_order].get("jpeg_sha256")})
        if source_order == target and case_id.startswith("T3_"):
            later = next((row for row in streams["contact_proxy"][source_index + 1:]
                          if float(row["time"]) > float(streams["contact_proxy"][source_index]["time"])), None)
            if later is None or later.get("contact_present") is not True:
                raise RuntimeError("T3_NEXT_LATER_CONTACT_MUST_RESTORE_TRUE")
            faults.append({"fault_type": "T3_SINGLE_FALSE", "target_capture_order": candidate_order,
                           "source_capture_order": source_order, "source_contact": True, "injected_contact": False,
                           "source_time": float(streams["contact_proxy"][source_index]["time"]),
                           "next_later_source_time": float(later["time"]), "duplicate_capture_order": None,
                           "reference_unchanged": True})
    candidate_root = rollout_root / "candidate_input"; candidate_root.mkdir(parents=True, exist_ok=False)
    for key, rows in candidate.items(): write_jsonl(candidate_root / f"{key}.jsonl", rows)
    write_csv(rollout_root / "online_raw/logical_observation_manifest.csv", logical_rows)
    reference_after = _hash_tree(rollout_root / "reference")
    if reference_before != reference_after or raw_contact_hash != sha256(rollout_root / "online_raw/contact_proxy.jsonl"):
        raise RuntimeError("FAULT_MODIFIED_RAW_OR_REFERENCE")
    manifest = {"schema": "l2rar2_r20_fault_injection_manifest_v1", "status": "PASS",
                "case_id": case_id, "faults": faults, "reference_hashes_before": reference_before,
                "reference_hashes_after": reference_after, "raw_contact_sha256_before": raw_contact_hash,
                "raw_contact_sha256_after": sha256(rollout_root / "online_raw/contact_proxy.jsonl"),
                "candidate_fields_exclude_fault_metadata": True}
    write_json(rollout_root / "online_raw/fault_injection_manifest.json", manifest)
    write_json(candidate_root / "build_status.json", {"status": "PASS", "rows": len(candidate["detections_rgb"])})
    return {"rollout_id": rollout_root.name, "case_id": case_id, "rows": len(candidate["detections_rgb"]),
            "faults": len(faults), "status": "PASS"}


def build_candidate_inputs(confirmation_root: Path) -> dict[str, Any]:
    rows = [build_rollout_candidate_input(path.parent) for path in sorted(confirmation_root.glob("*/metadata.json"))]
    result = {"schema": "l2rar2_r20_candidate_input_build_v1",
              "status": "PASS" if len(rows) == 72 else "FAIL", "rollouts": len(rows), "rows": rows}
    write_json(confirmation_root / "candidate_input_status.json", result)
    return result
