from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import PROTOCOL_ID

BASE_COMMIT = "54d3e95ff84cbab0e9305b08378ae7190a8e6c71"
FORMAL_MAIN_COMMIT = "234cb6dc0e2767fa62cd2dbec4a868b8d0711bb2"
CALIBRATION_DURATION_S = 0.05
CALIBRATION_LEVELS = (
    ("I1", (0.0, 0.35, -0.10)),
    ("I2", (0.0, 0.70, -0.20)),
    ("I3", (0.0, 1.05, -0.30)),
)


@dataclass(frozen=True)
class PulseLevel:
    level_id: str
    target_delta_v_local_mps: tuple[float, float, float]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = load_json(path)
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: dict[str, Any]) -> None:
    errors: list[str] = []
    if protocol.get("protocol_id") != PROTOCOL_ID:
        errors.append("protocol_id")
    if protocol.get("base_commit") != BASE_COMMIT:
        errors.append("base_commit")
    if protocol.get("formal_main_commit") != FORMAL_MAIN_COMMIT:
        errors.append("formal_main_commit")
    if protocol.get("status") != "DRAFT_NOT_AUTHORIZED":
        errors.append("status")
    if protocol.get("current_physical_authorization") != 0:
        errors.append("current_physical_authorization")
    if protocol.get("selected_candidate_id") is not None:
        errors.append("selected_candidate_id")
    levels = protocol.get("calibration_levels", {}).get("levels", [])
    expected = [{"id": key, "target_delta_v_local_mps": list(delta)} for key, delta in CALIBRATION_LEVELS]
    if levels != expected:
        errors.append("calibration_levels")
    if protocol.get("development", {}).get("rollouts") != 32:
        errors.append("development_rollouts")
    if errors:
        raise ValueError("protocol lock mismatch: " + ", ".join(errors))


def pulse_levels() -> tuple[PulseLevel, ...]:
    return tuple(PulseLevel(level_id, delta) for level_id, delta in CALIBRATION_LEVELS)
