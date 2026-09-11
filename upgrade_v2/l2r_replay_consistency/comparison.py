from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def numbers_equal(left: Any, right: Any, atol: float) -> bool:
    try:
        return bool(np.allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float), rtol=0.0, atol=atol))
    except (TypeError, ValueError):
        return left == right


def compare_actions(actual: list[dict[str, Any]], cache_root: Path, time_atol: float) -> dict[str, Any]:
    cached = read_csv(cache_root / "actions.csv")
    mismatches: list[dict[str, Any]] = []
    if len(actual) != len(cached):
        mismatches.append({"field": "row_count", "actual": len(actual), "cached": len(cached)})
    for index, (left, right) in enumerate(zip(actual, cached)):
        checks = {
            "action": left["action"] == right["action"],
            "action_index": int(left["action_index"]) == int(right["action_index"]),
            "start_time": numbers_equal(left["start_time"], right["start_time"], time_atol),
            "end_time": numbers_equal(left["end_time"], right["end_time"], time_atol),
        }
        for field, passed in checks.items():
            if not passed:
                mismatches.append({"row": index, "field": field, "actual": left.get(field), "cached": right.get(field)})
    return {"passed": not mismatches, "actual_rows": len(actual), "cached_rows": len(cached), "mismatches": mismatches}


def _control_equal(left: dict[str, Any], right: dict[str, Any], *, time_atol: float, mocap_atol: float) -> bool:
    return bool(
        left.get("control_step") == right.get("control_step")
        and left.get("physics_steps") == right.get("physics_steps")
        and left.get("gripper_command") == right.get("gripper_command")
        and numbers_equal(left.get("start_time"), right.get("start_time"), time_atol)
        and numbers_equal(left.get("end_time"), right.get("end_time"), time_atol)
        and numbers_equal(left.get("mocap_position"), right.get("mocap_position"), mocap_atol)
    )


def compare_controls(actual: list[dict[str, Any]], cache_root: Path, *, time_atol: float, mocap_atol: float) -> dict[str, Any]:
    cached = read_jsonl(cache_root / "low_level_controls.jsonl")
    mismatches: list[dict[str, Any]] = []
    if len(actual) != len(cached):
        mismatches.append({"field": "row_count", "actual": len(actual), "cached": len(cached)})
    for index, (left, right) in enumerate(zip(actual, cached)):
        if left.get("action") != right.get("action") or left.get("action_index") != right.get("action_index"):
            mismatches.append({"row": index, "field": "action_identity"})
            continue
        left_controls = left.get("controls", [])
        right_controls = right.get("controls", [])
        if len(left_controls) != len(right_controls):
            mismatches.append({"row": index, "field": "control_count", "actual": len(left_controls), "cached": len(right_controls)})
        for control_index, (left_control, right_control) in enumerate(zip(left_controls, right_controls)):
            if not _control_equal(left_control, right_control, time_atol=time_atol, mocap_atol=mocap_atol):
                mismatches.append({"row": index, "control": control_index, "field": "control"})
    return {"passed": not mismatches, "actual_rows": len(actual), "cached_rows": len(cached), "mismatches": mismatches}


def compare_events(actual: list[dict[str, Any]], cache_root: Path) -> dict[str, Any]:
    cached = read_jsonl(cache_root / "events.jsonl")
    return {"passed": actual == cached, "actual": actual, "cached": cached}


def compare_action_end_geometry(action_end_states: list[dict[str, Any]], cache_root: Path, geometry_atol: float) -> dict[str, Any]:
    cached = [row for row in read_csv(cache_root / "oracle_timeline.csv") if row.get("phase") == "action_end"]
    mismatches: list[dict[str, Any]] = []
    if len(action_end_states) != len(cached):
        mismatches.append({"field": "row_count", "actual": len(action_end_states), "cached": len(cached)})
    for index, (left, right) in enumerate(zip(action_end_states, cached)):
        cached_object = json.loads(right["object_xyz"])
        cached_gripper = json.loads(right["gripper_xyz"])
        if not numbers_equal(left["object_xyz"], cached_object, geometry_atol):
            mismatches.append({"row": index, "field": "object_xyz", "actual": left["object_xyz"], "cached": cached_object})
        if not numbers_equal(left["gripper_xyz"], cached_gripper, geometry_atol):
            mismatches.append({"row": index, "field": "gripper_xyz", "actual": left["gripper_xyz"], "cached": cached_gripper})
        if int(left["weld_state"]) != int(right["weld_state"]):
            mismatches.append({"row": index, "field": "weld_state", "actual": left["weld_state"], "cached": right["weld_state"]})
    return {"passed": not mismatches, "actual_rows": len(action_end_states), "cached_rows": len(cached), "mismatches": mismatches}


def compare_callback_order(captures: list[dict[str, Any]], cache_root: Path, time_atol: float) -> dict[str, Any]:
    cached = read_csv(cache_root / "oracle_timeline.csv")
    mismatches: list[dict[str, Any]] = []
    if len(captures) != len(cached):
        mismatches.append({"field": "row_count", "actual": len(captures), "cached": len(cached)})
    for index, (left, right) in enumerate(zip(captures, cached)):
        if left["capture_order"] != int(right["capture_order"]):
            mismatches.append({"row": index, "field": "capture_order"})
        if left["phase"] != right["phase"]:
            mismatches.append({"row": index, "field": "phase", "actual": left["phase"], "cached": right["phase"]})
        if not numbers_equal(left["time"], right["time"], time_atol):
            mismatches.append({"row": index, "field": "time", "actual": left["time"], "cached": right["time"]})
        expected_steps = ["callback_enter", "render_start", "render_end", "state_saved", "callback_exit"]
        if left.get("callback_steps") != expected_steps:
            mismatches.append({"row": index, "field": "callback_steps", "actual": left.get("callback_steps")})
    return {"passed": not mismatches, "actual_rows": len(captures), "cached_rows": len(cached), "mismatches": mismatches}


def compare_callback_to_return(action_end_states: list[dict[str, Any]], return_states: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("qpos", "qvel", "qacc_warmstart", "mocap_pos", "mocap_quat", "eq_active", "model_eq_data")
    mismatches: list[dict[str, Any]] = []
    if len(action_end_states) != len(return_states):
        mismatches.append({"field": "row_count", "callback": len(action_end_states), "return": len(return_states)})
    for index, (left, right) in enumerate(zip(action_end_states, return_states)):
        for field in fields:
            if not numbers_equal(left[field], right[field], 0.0):
                mismatches.append({"row": index, "field": field})
        if left["rng_state_sha256"] != right["rng_state_sha256"]:
            mismatches.append({"row": index, "field": "rng_state_sha256"})
    return {"passed": not mismatches, "mismatches": mismatches}


def state_health(states: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    expected_shapes: dict[str, tuple[int, ...]] = {}
    previous_time = -math.inf
    for index, row in enumerate(states):
        for field in ("qpos", "qvel", "qacc_warmstart", "mocap_pos", "mocap_quat", "eq_active", "model_eq_data"):
            array = np.asarray(row[field], dtype=float)
            if field not in expected_shapes:
                expected_shapes[field] = array.shape
            if array.shape != expected_shapes[field]:
                failures.append({"state": index, "field": field, "reason": "shape_changed", "shape": list(array.shape)})
            if not np.all(np.isfinite(array)):
                failures.append({"state": index, "field": field, "reason": "non_finite"})
        mocap_quat = np.asarray(row["mocap_quat"], dtype=float)
        if mocap_quat.size and not np.allclose(np.linalg.norm(mocap_quat, axis=-1), 1.0, rtol=0.0, atol=1e-12):
            failures.append({"state": index, "field": "mocap_quat", "reason": "not_unit"})
        eq_active = np.asarray(row["eq_active"], dtype=float)
        if not np.all(np.isin(eq_active, (0.0, 1.0))):
            failures.append({"state": index, "field": "eq_active", "reason": "not_binary"})
        if not math.isfinite(float(row["time"])):
            failures.append({"state": index, "field": "time", "reason": "non_finite"})
        elif float(row["time"]) < previous_time:
            failures.append({"state": index, "field": "time", "reason": "not_monotonic"})
        previous_time = float(row["time"])
        for field in ("object_xyz", "gripper_xyz"):
            if not np.all(np.isfinite(np.asarray(row[field], dtype=float))):
                failures.append({"state": index, "field": field, "reason": "non_finite"})
        for contact in row.get("contact_pair_summary", []):
            if not math.isfinite(float(contact["distance_m"])):
                failures.append({"state": index, "field": "contact_pair_summary", "reason": "non_finite_distance"})
    return {
        "passed": not failures,
        "state_count": len(states),
        "stable_shapes": {key: list(value) for key, value in expected_shapes.items()},
        "failures": failures,
    }
