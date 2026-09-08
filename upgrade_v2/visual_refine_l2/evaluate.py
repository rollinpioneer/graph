"""Metric and family-bootstrap helpers for executable graph comparisons."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import numpy as np


METRIC_NAMES = (
    "branch_accuracy", "state_readiness_accuracy", "transition_precondition_precision", "transition_effect_precision",
    "goal_precision", "goal_recall", "false_ready_rate", "unnecessary_manipulation_rate", "failure_recall",
    "recovery_recall", "unknown_rate", "ambiguous_edge_rate", "graph_completion_coverage", "second_view_query_rate",
)


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def summarize_executions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    goal_tp = sum(row["pred_goal"] and row["true_goal"] for row in rows)
    goal_fp = sum(row["pred_goal"] and not row["true_goal"] for row in rows)
    goal_fn = sum(not row["pred_goal"] and row["true_goal"] for row in rows)
    failure_rows = [row for row in rows if row["failure_expected"]]
    recovery_rows = [row for row in rows if row["recovery_expected"]]
    already_rows = [row for row in rows if row["already_satisfied"]]
    return {
        "rollouts": len(rows), "families": len({row["root_family_id"] for row in rows}),
        "branch_accuracy": ratio(sum(row["branch_correct"] for row in rows), len(rows)),
        "state_readiness_accuracy": ratio(sum(row["state_ready_correct"] for row in rows), len(rows)),
        "transition_precondition_precision": ratio(sum(row["precondition_correct"] for row in rows), len(rows)),
        "transition_effect_precision": ratio(sum(row["effect_correct"] for row in rows), len(rows)),
        "goal_precision": ratio(goal_tp, goal_tp + goal_fp), "goal_recall": ratio(goal_tp, goal_tp + goal_fn),
        "false_ready_rate": ratio(goal_fp, goal_tp + goal_fp),
        "unnecessary_manipulation_rate": ratio(sum(row["predicted_branch"] != "stop_no_action" for row in already_rows), len(already_rows)),
        "failure_denominator": len(failure_rows), "failure_recall": ratio(sum(row["failure_handled"] for row in failure_rows), len(failure_rows)),
        "recovery_denominator": len(recovery_rows), "recovery_recall": ratio(sum(row["recovery_handled"] for row in recovery_rows), len(recovery_rows)),
        "unknown_rate": sum(row["unknown_rate"] for row in rows) / max(1, len(rows)),
        "ambiguous_edge_rate": ratio(sum(row["ambiguous"] for row in rows), len(rows)),
        "graph_completion_coverage": sum(row["coverage"] for row in rows) / max(1, len(rows)),
        "second_view_query_rate": ratio(sum(row["second_view_queried"] for row in rows), len(rows)),
    }


def selection_score(metrics: dict[str, Any]) -> float:
    terms = [
        (0.20, "branch_accuracy", 1), (0.15, "goal_precision", 1), (0.15, "failure_recall", 1),
        (0.15, "recovery_recall", 1), (0.10, "graph_completion_coverage", 1), (0.10, "false_ready_rate", -1),
        (0.05, "unnecessary_manipulation_rate", -1), (0.05, "unknown_rate", -1), (0.05, "ambiguous_edge_rate", -1),
    ]
    available = [(weight, metrics.get(name), sign) for weight, name, sign in terms if metrics.get(name) is not None]
    normalizer = sum(weight for weight, _, _ in available)
    return sum(weight * float(value) * sign for weight, value, sign in available) / normalizer if normalizer else float("nan")


def paired_family_bootstrap(rows: list[dict[str, Any]], selected: str, baseline: str, metric: str, resamples: int, seed: int) -> dict[str, Any]:
    by_graph_family: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_graph_family[(row["graph_id"], row["root_family_id"])].append(row)
    families = sorted({family for graph, family in by_graph_family if graph == selected} & {family for graph, family in by_graph_family if graph == baseline})
    deltas = []
    for family in families:
        a = summarize_executions(by_graph_family[(selected, family)]).get(metric)
        b = summarize_executions(by_graph_family[(baseline, family)]).get(metric)
        if a is not None and b is not None:
            deltas.append(float(a) - float(b))
    if not deltas:
        return {"comparison": f"{selected}-{baseline}", "metric": metric, "families": 0, "mean_effect": None, "bootstrap_low": None, "bootstrap_high": None}
    rng = np.random.default_rng(seed)
    values = np.asarray(deltas)
    samples = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    return {"comparison": f"{selected}-{baseline}", "metric": metric, "families": len(values), "mean_effect": float(values.mean()), "bootstrap_low": float(np.quantile(samples, .025)), "bootstrap_high": float(np.quantile(samples, .975)), "bootstrap_resamples": resamples, "bootstrap_seed": seed}
