"""O/S input adapters for the L2RA-R2 geometry-event diagnostic.

O tier: the existing online observation budget (2D centroids, size and
confidence, contact proxy, gripper command, request/attempt lifecycle).
S tier: STATE_ASSISTED_DIAGNOSTIC - adds the saved oracle world positions for
offline diagnostic use only.  Oracle records are clipped to position, time and
matching keys; ``weld_state``, ``object_target_distance``, event names and case
labels never reach a candidate.

Adapters return plain data structures.  Candidates never open files.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

REQUIRED_ROLLOUT_FILES = (
    "metadata.json",
    "observations_dense.jsonl",
    "observations_action_end.jsonl",
    "controller_requests.jsonl",
    "actions.csv",
    "low_level_controls.jsonl",
    "oracle_timeline.csv",
    "events.jsonl",
    "termination.json",
)

O_ALLOWLIST = (
    "time",
    "capture_order",
    "frame_index",
    "object_centroid",
    "gripper_centroid",
    "object_confidence",
    "gripper_confidence",
    "gripper_command",
    "contact_present",
    "attempt_id",
    "attempt_phase",
    "attempt_active",
    "attempt_end",
    "attempt_end_reason",
    "attempt_end_sequence",
    "width",
    "height",
)

ORACLE_CLIP_FIELDS = ("time", "capture_order", "object_xyz", "gripper_xyz")

FORBIDDEN_CANDIDATE_INPUTS = (
    "weld_state",
    "attached",
    "events",
    "event",
    "event_name",
    "case_id",
    "scenario",
    "expected_action",
    "future_outcome",
    "object_target_distance",
    "future_frame",
    "front_path",
    "front_sha256",
)

DATA_VALID = "VALID"
DATA_MASKED = "MISSING_OR_INVALID"
DATA_BOUNDARY = "BOUNDARY_ORDER_UNKNOWN"
DATA_MISMATCH = "ORACLE_MATCH_MISMATCH"

DEFAULT_TIME_ATOL = 1e-6


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def vector3(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        value = json.loads(text)
    try:
        items = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return items if len(items) == 3 else None


def flag(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def missing_rollout_files(rollout_dir: Path) -> list[str]:
    root = Path(rollout_dir)
    return [name for name in REQUIRED_ROLLOUT_FILES if not (root / name).is_file()]


def load_rollout(rollout_dir: Path) -> dict[str, Any]:
    root = Path(rollout_dir)
    missing = missing_rollout_files(root)
    if missing:
        return {"status": "ROLLOUT_INCOMPLETE", "missing": missing, "path": str(root)}
    return {
        "status": "OK",
        "path": str(root),
        "metadata": read_json(root / "metadata.json"),
        "observations_dense": read_jsonl(root / "observations_dense.jsonl"),
        "observations_action_end": read_jsonl(root / "observations_action_end.jsonl"),
        "controller_requests": read_jsonl(root / "controller_requests.jsonl"),
        "actions": read_csv_rows(root / "actions.csv"),
        "low_level_controls": read_jsonl(root / "low_level_controls.jsonl"),
        "oracle_timeline": read_csv_rows(root / "oracle_timeline.csv"),
        "events": read_jsonl(root / "events.jsonl"),
        "termination": read_json(root / "termination.json"),
    }


def oracle_lookup(oracle_rows: list[dict[str, Any]], time_atol: float = DEFAULT_TIME_ATOL) -> dict[str, Any]:
    """Index oracle rows by capture_order, keeping only position/time keys."""
    by_order: dict[int, dict[str, Any]] = {}
    duplicate_orders: set[int] = set()
    for row in oracle_rows:
        raw_order = row.get("capture_order")
        if raw_order is None or str(raw_order).strip() == "":
            continue
        order = int(float(raw_order))
        if order in by_order:
            duplicate_orders.add(order)
            continue
        by_order[order] = {field: row.get(field) for field in ORACLE_CLIP_FIELDS}
    return {"by_order": by_order, "duplicate_orders": sorted(duplicate_orders), "time_atol": time_atol}


def build_o_samples(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{field: row.get(field) for field in O_ALLOWLIST} for row in observations]


def build_s_samples(
    observations: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
    time_atol: float = DEFAULT_TIME_ATOL,
) -> list[dict[str, Any]]:
    lookup = oracle_lookup(oracle_rows, time_atol=time_atol)
    by_order = lookup["by_order"]
    duplicates = set(lookup["duplicate_orders"])
    samples: list[dict[str, Any]] = []
    for row in observations:
        raw_order = row.get("capture_order")
        order = int(float(raw_order)) if raw_order is not None else None
        sample: dict[str, Any] = {
            "time": row.get("time"),
            "capture_order": order,
            "contact_present": flag(row.get("contact_present")),
            "gripper_command": row.get("gripper_command"),
            "attempt_id": row.get("attempt_id"),
            "attempt_phase": row.get("attempt_phase"),
            "attempt_active": flag(row.get("attempt_active")),
            "attempt_end": flag(row.get("attempt_end")),
            "attempt_end_reason": row.get("attempt_end_reason"),
            "attempt_end_sequence": row.get("attempt_end_sequence"),
            "object_xyz": None,
            "gripper_xyz": None,
            "data_quality": DATA_MASKED,
            "data_quality_reason": "oracle_record_absent",
        }
        if order is None:
            sample["data_quality_reason"] = "missing_capture_order"
            samples.append(sample)
            continue
        if order in duplicates:
            sample["data_quality"] = DATA_BOUNDARY
            sample["data_quality_reason"] = "duplicate_oracle_capture_order"
            samples.append(sample)
            continue
        oracle = by_order.get(order)
        if oracle is None:
            samples.append(sample)
            continue
        object_xyz = vector3(oracle.get("object_xyz"))
        gripper_xyz = vector3(oracle.get("gripper_xyz"))
        if object_xyz is None or gripper_xyz is None:
            sample["data_quality_reason"] = "oracle_position_unavailable"
            samples.append(sample)
            continue
        observation_time = row.get("time")
        observation_time = float(observation_time) if observation_time is not None else None
        try:
            oracle_time = float(oracle.get("time"))
        except (TypeError, ValueError):
            oracle_time = None
        if observation_time is None or oracle_time is None:
            sample["object_xyz"] = object_xyz
            sample["gripper_xyz"] = gripper_xyz
            sample["data_quality"] = DATA_MISMATCH
            sample["data_quality_reason"] = "time_unavailable_for_crosscheck"
            samples.append(sample)
            continue
        if abs(oracle_time - observation_time) > time_atol:
            sample["object_xyz"] = object_xyz
            sample["gripper_xyz"] = gripper_xyz
            sample["data_quality"] = DATA_MISMATCH
            sample["data_quality_reason"] = "capture_order_time_mismatch"
            sample["oracle_time"] = oracle_time
            samples.append(sample)
            continue
        sample["object_xyz"] = object_xyz
        sample["gripper_xyz"] = gripper_xyz
        sample["data_quality"] = DATA_VALID
        sample["data_quality_reason"] = "capture_order_time_match"
        samples.append(sample)
    return samples


def attach_request_provenance(samples: list[dict[str, Any]], requests: list[dict[str, Any]]) -> None:
    """Attach the arrived request provenance to each sample (no future intent)."""
    ordered = sorted(
        requests,
        key=lambda item: (
            float(item.get("received_time", 0.0) or 0.0),
            str(item.get("request_id", "")),
        ),
    )
    for sample in samples:
        sample_time = sample.get("time")
        sample_time = float(sample_time) if sample_time is not None else None
        active = None
        for request in ordered:
            received = float(request.get("received_time", 0.0) or 0.0)
            if sample_time is None or received <= sample_time + 1e-9:
                active = request
        if active is None:
            sample["requested_effect"] = None
            sample["request_provenance"] = "none"
        else:
            sample["requested_effect"] = active.get("requested_effect")
            sample["request_provenance"] = active.get("source")


def assert_no_forbidden_inputs(sample: dict[str, Any]) -> None:
    leaked = sorted(key for key in FORBIDDEN_CANDIDATE_INPUTS if key in sample)
    if leaked:
        raise AssertionError("forbidden candidate input leaked: " + ",".join(leaked))


def displacement(a: list[float] | None, b: list[float] | None) -> list[float] | None:
    if a is None or b is None:
        return None
    return [float(b[i]) - float(a[i]) for i in range(3)]


def norm3(vector: list[float] | None) -> float | None:
    if vector is None:
        return None
    return math.sqrt(sum(float(item) ** 2 for item in vector))


def distance3(left: list[float] | None, right: list[float] | None) -> float | None:
    if left is None or right is None:
        return None
    return math.sqrt(sum((float(left[i]) - float(right[i])) ** 2 for i in range(3)))
