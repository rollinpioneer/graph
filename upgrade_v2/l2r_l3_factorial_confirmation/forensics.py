from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from upgrade_v2.l2r_loss_observability_fusion.fusion import run_fusion
from upgrade_v2.l2r_logical_clock_confirmation.io_utils import read_json, read_jsonl, write_csv, write_json
from upgrade_v2.l2r_rgb_temporal_confirmation.evaluation import _run_o


CLP3 = "O_C3_CLP3_CANONICAL_TIME"


def _physics_onset(rows: list[dict[str, Any]]) -> tuple[float | None, int | None]:
    previous = None
    for row in rows:
        weld = row.get("weld_active")
        if previous is True and weld is False:
            return float(row["time"]), int(row.get("physics_step_index", 0)) * 10_000_000
        previous = weld
    return None, None


def _actions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"selected_action": row.get("selected_action"), "reason_code": row.get("reason_code"),
             "physical_time_ns": row.get("physical_time_ns"), "capture_order": row.get("capture_order")}
            for row in rows if row.get("selected_action") in {"retry_grasp", "recover_object"}]


def _fault_manifest(root: Path) -> dict[str, Any]:
    path = root / "online_raw/fault_injection_manifest.json"
    return read_json(path) if path.is_file() else {}


def build_forensics(r24_root: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    observability, recovery = [], []
    for path in sorted(r24_root.glob(f"*__{CLP3}/outcome.json")):
        root = path.parent
        outcome = read_json(path)
        case = outcome["case_id"]
        if case.startswith(("R24C2_", "R24C4_")):
            online = read_jsonl(root / "candidate_input/live_observations.jsonl")
            physics = read_jsonl(root / "reference/physics_trace.jsonl")
            decisions = read_jsonl(root / "candidate_output/live_decisions.jsonl")
            onset_s, onset_ns = _physics_onset(physics)
            post = [row for row in online if onset_ns is not None and int(row["physical_time_ns"]) >= onset_ns]
            raw = _run_o([{key: value for key, value in row.items() if key != "physical_time_ns"} for row in online], "O_C3")
            fused = run_fusion(online)
            proposal = next((row for row in fused if row["selected_action"] == "recover_object"), None)
            first_false = next((row for row in post if row.get("contact_present") is False), None)
            observability.append({
                "rollout_id": root.name,
                "family_id": outcome["family_id"],
                "case_id": case,
                "physical_loss_onset_s": onset_s,
                "physical_loss_onset_ns": onset_ns,
                "post_onset_observations": len(post),
                "post_contact_true": sum(row.get("contact_present") is True for row in post),
                "post_contact_false": sum(row.get("contact_present") is False for row in post),
                "post_contact_missing": sum(row.get("contact_present") is None for row in post),
                "first_contact_false_ns": first_false.get("physical_time_ns") if first_false else None,
                "raw_o_c3_actions": len(_actions(raw)),
                "clp3_actions": len(_actions(decisions)),
                "fusion_proposal": proposal is not None,
                "fusion_source": proposal.get("proposal_source") if proposal else None,
                "fusion_proposal_ns": proposal.get("physical_time_ns") if proposal else None,
                "fault_manifest": json.dumps(_fault_manifest(root), sort_keys=True),
                "forensic_class": "OBSERVATION_MISSING_CONTACT_MASKED_TRUE" if case.startswith("R24C2_")
                                  else "OBSERVATION_MISSING_EDGE_BROKEN_BY_LEADING_NULLS",
                "official_outcome": outcome["failure_stage"],
                "candidate_error": outcome["candidate_error"],
            })
        if ((case.startswith("R24C1_") and outcome.get("failure_stage") == "TASK_RECOVERY_ERROR")
                or case.startswith("R24C6_")
                or (case.startswith("R24C8_") and outcome.get("failure_stage") == "TASK_RECOVERY_ERROR")):
            decisions = read_jsonl(root / "candidate_output/live_decisions.jsonl")
            executions = outcome.get("recovery_executions", [])
            stage = outcome.get("failure_stage")
            if case.startswith("R24C8_"):
                layer = "SECOND_LOSS_NOT_REARMED"
            elif case.startswith("R24C6_"):
                layer = "RELOCATION_ERROR"
            else:
                layer = "POST_RECOVERY_GOAL_VERIFICATION_FAILED"
            recovery.append({
                "rollout_id": root.name,
                "family_id": outcome["family_id"],
                "case_id": case,
                "official_failure_stage": stage,
                "forensic_layer": layer,
                "signal_observed": outcome["signal_observed"],
                "candidate_action_events": len(_actions(decisions)),
                "outer_recovery_cycles": len(executions),
                "inner_recovery_loops": sum(int(item.get("loop_count", 0)) for item in executions),
                "successful_recovery_executions": sum(bool(item.get("success")) for item in executions),
                "last_execution_failure_stage": executions[-1].get("failure_stage") if executions else None,
                "hold_verification_passes": sum(bool(loop.get("hold_verification", {}).get("passed"))
                                                for item in executions for loop in item.get("loops", [])),
                "final_task_success": outcome["final_task_success"],
                "candidate_error": outcome["candidate_error"],
            })
    write_csv(output / "observability_forensics.csv", observability)
    write_csv(output / "recovery_forensics.csv", recovery)
    summary = {
        "schema": "l2rar2_r25_r24_zero_physics_forensics_v1",
        "r24_source_read_only": True,
        "observability_episodes": len(observability),
        "signal_not_observed_episodes": sum(row["official_outcome"] == "SIGNAL_NOT_OBSERVED" for row in observability),
        "signal_not_observed_candidate_errors": sum(bool(row["candidate_error"]) for row in observability),
        "fusion_diagnostic_proposals": sum(bool(row["fusion_proposal"]) for row in observability),
        "task_recovery_error_episodes": sum(row["official_failure_stage"] == "TASK_RECOVERY_ERROR" for row in recovery),
        "relocation_error_comparators": sum(row["official_failure_stage"] == "RELOCATION_ERROR" for row in recovery),
        "observation_and_execution_separated": True,
    }
    write_json(output / "summary.json", summary)
    return summary
