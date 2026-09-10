"""Read-only normalization of saved R11 audit rows; no replay or inference."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "l2rar2_r12_saved_state_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def extract_physics_probe_action_end(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract only rows actually present in the saved probe trace.

    The trace is instrumented-only; this function therefore returns provenance
    and explicitly does not manufacture an ordinary replay counterpart.
    """
    if not path.is_file() or path.is_symlink():
        raise ValueError("saved trace must be a regular file")
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for ordinal, row in enumerate(csv.DictReader(handle)):
            rows.append({
                "schema": SCHEMA,
                "rollout_id": f"{row.get('root_family_id', 'unknown')}:{row.get('case_id', 'unknown')}",
                "case_id": row.get("case_id", "unknown"),
                "sample_key": f"physics_step:{row.get('physics_step_index', ordinal)}",
                "phase": row.get("phase", "physics_step"),
                "sampling_point": "instrumented_after_mj_step",
                "time": float(row["time"]),
                "ordinal": ordinal,
                "source_file": str(path.resolve()),
                "source_sha256": sha256(path),
                "values": {
                    "object_qpos": _json_value(row.get("object_qpos", "null")),
                    "object_qvel": _json_value(row.get("object_qvel", "null")),
                    "mocap_position": _json_value(row.get("mocap_position", "null")),
                    "weld_active": row.get("weld_active"),
                    "attached": row.get("attached"),
                    "contacts": _json_value(row.get("contacts", "null")),
                },
            })
    return rows, {"source_file": str(path.resolve()), "source_sha256": sha256(path), "rows": len(rows), "ordinary_counterpart_saved": False}
