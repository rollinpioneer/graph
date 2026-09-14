"""Manifest-backed sample classification. Do not pick traces by reward outcome."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Any

REQUIRED = (
    "actions.csv",
    "contact_sensor.csv",
    "gripper_command.csv",
    "online_observation.npz",
    "oracle_timeline.csv",
    "events.jsonl",
)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def event_class(row: dict[str, str]) -> list[str]:
    labels = []
    if row.get("scenario") == "normal_pick_place" and row.get("final_goal_stable") == "True":
        labels.append("normal_success")
    if row.get("scenario") == "already_satisfied_stable":
        labels.append("already_satisfied_terminal")
    if row.get("contact_loss") == "True" and row.get("recovery_achieved") == "True":
        labels.append("loss_then_recovery_success")
    if row.get("missed_grasp") == "True" and row.get("recovery_achieved") == "True":
        labels.append("missed_grasp_recovery_success")
    if row.get("termination_type") == "goal_terminal" and row.get("final_goal_stable") != "True":
        labels.append("terminal_failure")
    if "dual" in (row.get("scenario") or "").lower() or "order" in (row.get("scenario") or "").lower():
        labels.append("dual_order")
    if not labels:
        labels.append("other_pilot_variant")
    return labels


def inventory(manifest_path: Path) -> dict[str, Any]:
    rows = load_manifest(manifest_path)
    items = []
    coverage = {
        "normal_success": [],
        "loss_then_recovery_success": [],
        "missed_grasp_recovery_success": [],
        "terminal_failure": [],
        "dual_order": [],
        "already_satisfied_terminal": [],
        "other_pilot_variant": [],
    }
    for row in rows:
        path = Path(row["path"])
        missing = [name for name in REQUIRED if not (path / name).is_file()]
        labels = event_class(row)
        rec = dict(
            rollout_id=row["rollout_id"],
            root_family_id=row["root_family_id"],
            scenario=row["scenario"],
            path=str(path),
            frames=int(row["frames"]),
            termination_type=row["termination_type"],
            missed_grasp=row["missed_grasp"] == "True",
            contact_loss=row["contact_loss"] == "True",
            recovery_attempt=row["recovery_attempt"] == "True",
            recovery_achieved=row["recovery_achieved"] == "True",
            final_goal_stable=row["final_goal_stable"] == "True",
            event_classes=labels,
            missing_required_files=missing,
            eligible=not missing,
            catalog_source="pilot_rollout_manifest.csv",
        )
        items.append(rec)
        for lab in labels:
            if rec["eligible"]:
                coverage[lab].append(rec["rollout_id"])
    selected = []
    for key, prefer in (
        ("normal_success", "L2P_00_00_220701_r00"),
        ("missed_grasp_recovery_success", "L2P_03_00_220701_r00"),
        ("loss_then_recovery_success", "L2P_04_00_220701_r00"),
        ("already_satisfied_terminal", "L2P_01_00_220701_r00"),
        ("other_pilot_variant", "L2P_02_00_220701_r00"),
    ):
        ids = coverage.get(key) or []
        if prefer in ids:
            selected.append(prefer)
        elif ids:
            selected.append(ids[0])
    return dict(
        n_manifest_rows=len(rows),
        items=items,
        coverage={k: v for k, v in coverage.items()},
        selected_for_replay=selected,
        dual_order_raw="RAW_NOT_FOUND" if not coverage["dual_order"] else "FOUND",
        terminal_failure_raw="RAW_NOT_FOUND" if not coverage["terminal_failure"] else "FOUND",
        note="Classification uses the original pilot manifest, not reward outcomes. Dual-order physical traces are not in this pilot family.",
    )