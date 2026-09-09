from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .collector import collect
from .evaluate import METHODS, evaluate, paired_bootstrap
from .io import read_csv, read_json, sha256, write_csv, write_json


def confirm(selection_lock_path: Path, protocol_path: Path, workers: int, output_root: Path) -> dict[str, Any]:
    selection = read_json(selection_lock_path)
    if selection.get("status") != "LOCKED_FOR_FOCUSED_CONFIRMATION" or selection.get("selected_candidate_id") != "M1_requested_effect_gate":
        raise ValueError("valid frozen M1 selection lock is required")
    protocol = read_json(protocol_path)
    generation_lock_path = Path(selection_lock_path).with_name("generation_lock.json")
    if sha256(generation_lock_path) != selection["generation_lock_sha256"]:
        raise ValueError("generation lock hash mismatch")
    consumption_path = output_root / "confirmation_consumption.json"
    if consumption_path.exists():
        consumption = read_json(consumption_path)
        if consumption.get("status") == "COMPLETE":
            return read_json(output_root / "focused_confirmation_status.json")
        if consumption.get("status") != "STARTED" or consumption.get("selection_lock_sha256") != sha256(selection_lock_path) or consumption.get("generation_lock_sha256") != sha256(generation_lock_path):
            raise RuntimeError("incomplete confirmation record does not match the frozen batch")
    else:
        consumption = {
            "schema": "pathgraph_l2rar2_confirmation_consumption_v1",
            "status": "STARTED",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "selection_lock_sha256": sha256(selection_lock_path),
            "generation_lock_sha256": sha256(generation_lock_path),
            "planned_root_families": 8,
            "planned_physical_rollouts": 64,
            "completed_jobs": [],
        }
        write_json(consumption_path, consumption)
    data_root = output_root / "data"
    collection = collect(generation_lock_path, "confirmation", workers, data_root)
    consumption["completed_jobs"] = [row["rollout_id"] for row in read_csv(data_root / "rollout_manifest.csv")]
    consumption["status"] = "COMPLETE"
    consumption["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(consumption_path, consumption)
    evaluation_root = output_root / "evaluation"
    gates = evaluate(data_root, list(METHODS), protocol_path, output_root.parents[0] / "l2rar2_2_task_contract", evaluation_root)
    decisions = read_csv(evaluation_root / "event_decisions.csv")
    for row in decisions:
        for key in ("correct", "missed", "false_emergency", "premature_emergency", "event_window_conflict", "rollout_conflict", "hold_evidence_before_event"):
            row[key] = str(row[key]).lower() in {"true", "1"}
    effects = paired_bootstrap(decisions, resamples=int(protocol["statistics"]["paired_bootstrap_resamples"]), seed=int(protocol["statistics"]["bootstrap_seed"]))
    write_csv(output_root / "paired_effects.csv", effects)
    status = {
        "schema": "pathgraph_l2rar2_focused_confirmation_status_v1",
        "status": "FOCUSED_CONFIRMATION_PASS" if gates["all_pass"] else "FOCUSED_CONFIRMATION_FAILED",
        "all_gates_pass": gates["all_pass"],
        "failed_gates": gates["failed"],
        "root_families": collection["root_families"],
        "physical_rollouts": collection["physical_rollouts"],
        "events": len(decisions),
        "selected_candidate_id": selection["selected_candidate_id"],
        "standard_l2r_confirmation_status": "NOT_RUN_OUT_OF_SCOPE",
        "old_r4_status": "NOT_RUN_PRESERVED",
    }
    write_json(output_root / "focused_confirmation_status.json", status)
    return status
