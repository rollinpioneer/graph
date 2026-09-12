from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import read_jsonl

ALLOWED = {
    "time", "capture_order", "object_centroid", "gripper_centroid",
    "object_confidence", "gripper_confidence", "width", "height",
    "contact_present", "gripper_command", "attempt_id", "attempt_phase",
    "attempt_active", "attempt_end", "attempt_end_reason",
    "requested_effect", "context_valid",
}
FORBIDDEN = {
    "object_world_position", "gripper_world_position", "object_xyz", "gripper_xyz",
    "weld_active", "eq_active", "eq_data", "xfrc_applied", "force_level",
    "intervention_phase", "case_id", "family_id", "physical_loss_confirmed",
    "loss_onset_time", "loss_confirmed_time", "reference_action", "future_outcome",
}


class OnlineInputLeakage(RuntimeError):
    pass


def validate_online_row(row: dict[str, Any]) -> dict[str, Any]:
    leaked = sorted(FORBIDDEN.intersection(row))
    if leaked:
        raise OnlineInputLeakage("ONLINE_INPUT_LEAKAGE:" + ",".join(leaked))
    return {key: row.get(key) for key in ALLOWED if key in row}


def _by_order(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["capture_order"]): row for row in read_jsonl(path)}


def load_candidate_input(candidate_root: Path) -> list[dict[str, Any]]:
    root = Path(candidate_root).resolve()
    if root.name != "candidate_input":
        raise OnlineInputLeakage("ONLINE_INPUT_LEAKAGE:candidate_root_must_be_candidate_input")
    detections = _by_order(root / "detections_rgb.jsonl")
    contacts = _by_order(root / "contact_proxy.jsonl")
    lifecycle = _by_order(root / "attempt_lifecycle.jsonl")
    commands = _by_order(root / "gripper_commands.jsonl")
    requests = _by_order(root / "request_context.jsonl")
    orders = sorted(detections)
    if any(set(source) != set(orders) for source in (contacts, lifecycle, commands, requests)):
        raise ValueError("CANDIDATE_STREAM_ALIGNMENT_ERROR")
    rows = []
    for order in orders:
        merged = {**detections[order], **contacts[order], **lifecycle[order],
                  **commands[order], **requests[order]}
        # File paths and detector diagnostics do not enter the method.
        merged.pop("frame_path", None)
        rows.append(validate_online_row(merged))
    return rows
