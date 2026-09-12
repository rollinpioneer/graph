from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from upgrade_v2.l2r_logical_clock_confirmation.io_utils import read_json, write_csv, write_json

from .registry import ARMS, CASES


def evaluate(root: Path, output: Path) -> dict:
    rows = [read_json(path) for path in sorted(root.glob("*/outcome.json"))]
    metrics, cases = [], []
    for arm in ARMS:
        selected = [row for row in rows if row["arm_id"] == arm.arm_id]
        required = [row for row in selected if row["recovery_required"]]
        metrics.append({
            "arm_id": arm.arm_id, "fusion_enabled": arm.fusion_enabled,
            "supervisor_v2_enabled": arm.supervisor_v2_enabled,
            "rollouts": len(selected),
            "signal_exposure": sum(row["signal_exposed"] for row in required),
            "signal_exposure_denominator": len(required),
            "actual_recovery_success": sum(row["actual_recovery_success"] for row in required),
            "final_task_success": sum(row["final_task_success"] for row in selected),
            "false_executed_actions": sum(row["false_executed_action"] for row in selected),
            "outer_recovery_cycles": sum(row["outer_recovery_cycles"] for row in selected),
            "inner_recovery_loops": sum(row["inner_recovery_loops"] for row in selected),
            "task_duration_median_s": statistics.median(row["task_duration_s"] for row in selected) if selected else None,
            "failure_stages": json.dumps(dict(Counter(row["failure_stage"] for row in selected if row["failure_stage"])), sort_keys=True),
        })
        for case in CASES:
            subset = [row for row in selected if row["case_id"] == case.case_id]
            cases.append({
                "arm_id": arm.arm_id, "case_id": case.case_id, "rollouts": len(subset),
                "signal_exposure": sum(row["signal_exposed"] for row in subset),
                "actual_recovery_success": sum(row["actual_recovery_success"] for row in subset),
                "final_task_success": sum(row["final_task_success"] for row in subset),
                "false_executed_actions": sum(row["false_executed_action"] for row in subset),
                "outer_recovery_cycles": sum(row["outer_recovery_cycles"] for row in subset),
                "failure_stages": json.dumps(dict(Counter(row["failure_stage"] for row in subset if row["failure_stage"])), sort_keys=True),
            })

    def group(case_prefix: str, *, fusion=None, supervisor=None):
        return [row for row in rows if row["case_id"].startswith(case_prefix)
                and (fusion is None or row["fusion_enabled"] is fusion)
                and (supervisor is None or row["supervisor_v2_enabled"] is supervisor)]

    gates = {
        "rollouts_112": len(rows) == 112,
        "signal_not_observed_never_candidate_error": all(not row["candidate_error"] for row in rows if row["signal_not_observed"]),
        "fusion_observability_signal_16_of_16": sum(row["signal_exposed"] for prefix in ("R25C1_", "R25C2_") for row in group(prefix, fusion=True)) == 16,
        "baseline_observability_signal_0_of_16": sum(row["signal_exposed"] for prefix in ("R25C1_", "R25C2_") for row in group(prefix, fusion=False)) == 0,
        "supervisor_relocation_success_8_of_8": sum(row["final_task_success"] for row in group("R25C3_", supervisor=True)) == 8,
        "baseline_relocation_success_0_of_8": sum(row["final_task_success"] for row in group("R25C3_", supervisor=False)) == 0,
        "supervisor_goal_success_8_of_8": sum(row["final_task_success"] for row in group("R25C4_", supervisor=True)) == 8,
        "baseline_goal_success_0_of_8": sum(row["final_task_success"] for row in group("R25C4_", supervisor=False)) == 0,
        "supervisor_secondary_success_8_of_8": sum(row["final_task_success"] for row in group("R25C5_", supervisor=True)) == 8,
        "baseline_secondary_success_0_of_8": sum(row["final_task_success"] for row in group("R25C5_", supervisor=False)) == 0,
        "control_false_actions_0": sum(row["false_executed_action"] for prefix in ("R25C6_", "R25C7_") for row in group(prefix)) == 0,
        "full_factor_cell_success_28_of_28": sum(row["final_task_success"] for row in rows if row["arm_id"] == "F1_S1_FUSION_SUPERVISOR") == 28,
    }
    result = {
        "schema": "l2rar2_r25_factorial_evaluation_v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates, "method_metrics": metrics,
        "r24_development_only": True, "confirmation_parameter_search": False,
        "clp3_modified": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "outcomes.csv", rows)
    write_csv(output / "arm_metrics.csv", metrics)
    write_csv(output / "case_metrics.csv", cases)
    write_json(output / "summary.json", result)
    return result
