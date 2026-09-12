from __future__ import annotations

import csv
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from upgrade_v2.l2r_logical_clock_confirmation.io_utils import read_json, write_csv, write_json

from . import CLP3_ID, METHODS


def _p90(values: list[float]) -> float | None:
    return sorted(values)[max(0, math.ceil(0.9 * len(values)) - 1)] if values else None


def evaluate(root: Path, output: Path) -> dict[str, Any]:
    rows = [read_json(path) for path in sorted(root.glob("*/outcome.json"))]
    if len(rows) != 60: raise RuntimeError(f"EXPECTED_60_OUTCOMES_GOT_{len(rows)}")
    metrics = []
    for method in METHODS:
        items = [row for row in rows if row["method"] == method]
        positives = [row for row in items if row["expected_action"] in {"retry_grasp", "recover_object"}]
        durations = [float(row["recovery"]["duration_s"]) for row in positives
                     if row["recovery"].get("duration_s") is not None]
        metric = {
            "method": method, "rollouts": len(items),
            "final_task_success": sum(bool(row["final_task_success"]) for row in items),
            "overall_success": sum(bool(row["overall_success"]) for row in items),
            "retry_recover_expected": len(positives),
            "retry_recover_actual_success": sum(bool(row["recovery"]["success"]) for row in positives),
            "recovery_duration_median_s": statistics.median(durations) if durations else None,
            "recovery_duration_p90_s": _p90(durations),
            "recovery_loops_total": sum(int(row["recovery"]["loop_count"]) for row in positives),
            "false_executed_actions": sum(bool(row["false_executed_action"]) for row in items),
            "failure_stages": dict(Counter(row["failure_stage"] for row in items if row["failure_stage"])),
        }
        metrics.append(metric)
    paired = []
    keys = sorted({(row["family_id"], row["case_id"], row["rollout_seed"]) for row in rows})
    for family, case, seed in keys:
        group = {(row["method"]): row for row in rows if (row["family_id"], row["case_id"], row["rollout_seed"]) == (family, case, seed)}
        paired.append({"family_id": family, "case_id": case, "rollout_seed": seed,
                       **{f"{method}_overall_success": bool(group[method]["overall_success"]) for method in METHODS},
                       **{f"{method}_final_task_success": bool(group[method]["final_task_success"]) for method in METHODS}})
    clp3 = next(row for row in metrics if row["method"] == CLP3_ID)
    gates = {
        "outcomes_60": len(rows) == 60, "paired_groups_20": len(paired) == 20,
        "clp3_final_task_success_20": clp3["final_task_success"] == 20,
        "clp3_overall_success_20": clp3["overall_success"] == 20,
        "clp3_retry_recover_success_8": clp3["retry_recover_actual_success"] == 8,
        "clp3_false_actions_0": clp3["false_executed_actions"] == 0,
        "clp3_no_failure_stage": not clp3["failure_stages"],
    }
    result = {"schema": "l2rar2_r23_l3_closed_loop_evaluation_v1",
              "status": "PASS" if all(gates.values()) else "FAIL", "gates": gates,
              "metrics": metrics, "candidate_frozen": True, "parameter_search": False,
              "r22_read_only": True}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "outcomes.csv", rows)
    write_csv(output / "method_metrics.csv", metrics)
    write_csv(output / "paired_comparison.csv", paired)
    write_json(output / "summary.json", result)
    return result

