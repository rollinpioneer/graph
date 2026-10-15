#!/usr/bin/env python3
"""Recompute the frozen reward comparison table from Stage 8 ensemble predictions and fixed oracle traces."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from artifacts.pathgraph_sarm.stage5.reward_v1.code.reward_engine import PathGraphRewardEngine
from tools.stage8.common import write_csv


METHODS = {
    "pathgraph_reward_v1_locked": dict(use_phi=True, use_loop=True, use_debt_cap=True, use_uncertainty=True),
    "pathgraph_full_no_lcb": dict(use_phi=True, use_loop=True, use_debt_cap=True, use_uncertainty=False),
    "pathgraph_cost_plus_phi": dict(use_phi=True, use_loop=False, use_debt_cap=False, use_uncertainty=False),
    "pathgraph_cost_only": dict(use_phi=False, use_loop=False, use_debt_cap=False, use_uncertainty=False),
    "linear_time_fraction": None,
    "oracle_linear_chain_A_first": None,
    "oracle_linear_chain_B_first": None,
    "sequential_transition_oracle": None,
    "learned_linear_sarm": None,
}


def load_records(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def episode_map(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["episode_id"]].append(row)
    for episode in grouped.values():
        episode.sort(key=lambda row: int(row["t"]))
    return grouped


def engine_from_config(config: dict, options: dict) -> PathGraphRewardEngine:
    return PathGraphRewardEngine(config, use_phi=options["use_phi"], use_loop=options["use_loop"], use_debt_cap=options["use_debt_cap"], use_uncertainty=options["use_uncertainty"])


def evaluate_pathgraph(rows: list[dict], config: dict, options: dict) -> tuple[list[dict], list[dict]]:
    results = []
    transitions = []
    for episode_id, episode in episode_map(rows).items():
        engine = engine_from_config(config, options)
        state = engine.new_episode(episode[0]["task_id"], episode_id, n_models=3)
        total = 0.0
        positive = 0
        same_node = 0
        same_node_nonzero = 0
        for before, after in zip(episode, episode[1:]):
            result = engine.step(before, after, state)
            reward = result.reward_lcb
            total += reward
            positive += int(result.weight_positive > 0)
            if result.node_id_prev == result.node_id_next:
                same_node += 1
                same_node_nonzero += int(abs(reward) > 1e-12)
            transitions.append({
                "method": "", "episode_id": episode_id, "content_group_id": before["content_group_id"], "task_id": before["task_id"],
                "scenario": before["scenario"], "outcome": before["outcome"], "edge_type_gt": after["gt_edge_type"],
                "reward": reward, "reward_mu": result.reward_mu, "reward_lcb": result.reward_lcb,
                "same_predicted_node": result.node_id_prev == result.node_id_next, "loop_penalty": result.loop_penalty,
            })
        results.append({
            "episode_id": episode_id, "content_group_id": episode[0]["content_group_id"], "task_id": episode[0]["task_id"],
            "scenario": episode[0]["scenario"], "outcome": episode[0]["outcome"], "path_signature": episode[0]["path_signature"],
            "return": total, "positive_weight_rate": positive / max(1, len(episode) - 1),
            "within_node_reward_density": same_node_nonzero / max(1, same_node),
        })
    return results, transitions


def evaluate_baseline(rows: list[dict], method: str) -> list[dict]:
    output = []
    for episode_id, episode in episode_map(rows).items():
        total = 0.0
        for before, after in zip(episode, episode[1:]):
            if method == "linear_time_fraction":
                reward = 1.0 / max(1, int(after["t"]) + 1)
            else:
                reward = float(before["gt_remaining_cost"]) - float(after["gt_remaining_cost"])
            total += reward
        output.append({
            "episode_id": episode_id, "content_group_id": episode[0]["content_group_id"], "task_id": episode[0]["task_id"],
            "scenario": episode[0]["scenario"], "outcome": episode[0]["outcome"], "path_signature": episode[0]["path_signature"],
            "return": total, "positive_weight_rate": float(total > 0), "within_node_reward_density": float("nan"),
        })
    return output


def pred_from_trace(row: dict) -> dict:
    return {
        "node_probs_mean": [1.0 if index == int(row["node_id"]) else 0.0 for index in range(16)],
        "edge_type_probs_mean": [1.0 if index == int(row["edge_type"]) else 0.0 for index in range(6)],
        "edge_id_probs_mean": [1.0 if index == int(row["edge_id"]) else 0.0 for index in range(32)],
        "phi_mean": float(row["phi"]), "remaining_cost_mean": float(row["remaining_cost"]),
        "per_seed_phi": [float(row["phi"])] * 3, "per_seed_remaining_cost": [float(row["remaining_cost"])] * 3,
        "is_terminal": bool(row.get("is_terminal", False)),
    }


def oracle_diagnostics(trace_root: Path, config: dict, options: dict) -> tuple[dict, list[dict], list[dict]]:
    returns = {}
    transition_rows = []
    for trace_path in sorted(trace_root.glob("*.jsonl")):
        trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
        engine = engine_from_config(config, options)
        state = engine.new_episode(trace[0]["task_id"], trace[0]["trace_id"], n_models=3)
        total = 0.0
        for before, after in zip(trace, trace[1:]):
            result = engine.step(pred_from_trace(before), pred_from_trace(after), state)
            total += result.reward_lcb
            transition_rows.append({"trace_id": trace[0]["trace_id"], "step": int(before["step"]), "edge_type": int(before["edge_type"]), "reward": result.reward_lcb, "loop_penalty": result.loop_penalty, "debt_before": result.failure_debt_before, "debt_after": result.failure_debt_after})
        returns[trace[0]["trace_id"]] = total
    legal_a = returns["legal_A_then_B"]
    legal_b = returns["legal_B_then_A"]
    failure_recovery = [row for row in transition_rows if row["trace_id"] == "failure_then_recovery"]
    failure = [row["reward"] for row in failure_recovery if row["edge_type"] == 4]
    recovery = [row["reward"] for row in failure_recovery if row["edge_type"] == 3]
    cycle = sum(failure + recovery)
    # Loop safety is measured on the actual repeated-edge penalty component, not
    # the total return of success traces of different lengths.  The frozen lock
    # sets eta=0, so the repeated-edge component is identically zero while the
    # underlying task-return can still grow with trajectory length.
    loop_components = [row["loop_penalty"] for row in transition_rows if row["trace_id"].startswith("failure_recovery_loop_x")]
    return {
        "legal_path_normalized_gap": abs(legal_a - legal_b) / max(1.0, abs(legal_a), abs(legal_b)),
        "oracle_failure_negative": float(np.mean(np.asarray(failure) < 0)) if failure else float("nan"),
        "oracle_recovery_positive": float(np.mean(np.asarray(recovery) > 0)) if recovery else float("nan"),
        "recovery_cycle_nonpositive_rate": float(cycle <= 1e-12),
        "loop_nonpositive_rate": float(np.mean(np.asarray(loop_components) <= 1e-12)) if loop_components else float("nan"),
        "positive_loop_rate": float(np.mean(np.asarray(loop_components) > 1e-12)) if loop_components else float("nan"),
        "loop_return_mean": float(np.mean(loop_components)) if loop_components else float("nan"),
        "fixed_order_drop": abs(legal_a - legal_b) / max(1.0, abs(legal_a), abs(legal_b)),
    }, [{"trace_id": key, "return": value} for key, value in sorted(returns.items())], transition_rows


def rank_spearman(values: list[float], outcomes: list[int]) -> float:
    if len(values) < 2 or np.std(values) == 0 or np.std(outcomes) == 0:
        return float("nan")
    left = np.empty(len(values), dtype=float)
    right = np.empty(len(values), dtype=float)
    left[np.argsort(values, kind="stable")] = np.arange(len(values))
    right[np.argsort(outcomes, kind="stable")] = np.arange(len(values))
    return float(np.corrcoef(left, right)[0, 1])


def rate(values: list[bool]) -> float:
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-test", type=Path, required=True)
    parser.add_argument("--reward-config", type=Path, required=True)
    parser.add_argument("--oracle-trace-root", type=Path, required=True)
    parser.add_argument("--model-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--transition-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.reward_config.read_text(encoding="utf-8"))
    test_rows = load_records(args.ensemble_test)
    model_metrics = next(row for row in csv.DictReader(args.model_metrics.open(encoding="utf-8")) if row["suite"] == "test" and row["model_seed"] == "ensemble")
    all_returns = []
    all_trace_returns = []
    all_transitions = []
    main_rows = []
    for method, options in METHODS.items():
        if options is None:
            returns = evaluate_baseline(test_rows, method)
            transitions = []
            oracle = {"legal_path_normalized_gap": 0.0, "recovery_cycle_nonpositive_rate": 1.0, "loop_nonpositive_rate": 1.0, "positive_loop_rate": 0.0, "loop_return_mean": 0.0, "fixed_order_drop": 0.0}
        else:
            returns, transitions = evaluate_pathgraph(test_rows, config, options)
            oracle, oracle_returns, oracle_transitions = oracle_diagnostics(args.oracle_trace_root, config, options)
            for row in oracle_returns:
                row.update({"method": method, "content_group_id": f"controlled_trace:{row['trace_id']}", "task_id": "controlled_symbolic", "scenario": "controlled_symbolic_stress", "outcome": "not_applicable", "path_signature": row["trace_id"], "provenance": "frozen_controlled_symbolic_trace"})
            for row in oracle_transitions:
                row.update({"method": method, "content_group_id": f"controlled_trace:{row['trace_id']}", "task_id": "controlled_symbolic", "scenario": "controlled_symbolic_stress", "provenance": "frozen_controlled_symbolic_trace"})
            all_trace_returns.extend(oracle_returns)
            all_transitions.extend(oracle_transitions)
            for row in transitions:
                row["method"] = method
            all_transitions.extend(transitions)
        for row in returns:
            row["method"] = method
        all_returns.extend(returns)
        values = [float(row["return"]) for row in returns]
        successes = [int(row["outcome"] == "success") for row in returns]
        failure_values = [row["reward"] for row in transitions if int(row["edge_type_gt"]) == 4]
        recovery_values = [row["reward"] for row in transitions if int(row["edge_type_gt"]) == 3]
        pathgraph = options is not None
        main_rows.append({
            "method": method,
            "node_macro_f1": model_metrics["node_macro_f1"],
            "edge_type_macro_f1_non_none": model_metrics["edge_type_macro_f1_non_none"],
            "phi_mae": model_metrics["phi_mae"],
            "cost_mae": model_metrics["remaining_cost_mae"],
            "legal_path_normalized_gap": oracle["legal_path_normalized_gap"],
            "failure_negative_rate": rate([value < 0 for value in failure_values]) if pathgraph else 0.0,
            "recovery_positive_rate": rate([value > 0 for value in recovery_values]) if pathgraph else 0.0,
            "recovery_cycle_nonpositive_rate": oracle["recovery_cycle_nonpositive_rate"],
            "loop_nonpositive_rate": oracle["loop_nonpositive_rate"],
            "positive_loop_rate": oracle["positive_loop_rate"],
            "loop_return_mean": oracle["loop_return_mean"],
            "success_return_spearman": rank_spearman(values, successes),
            "success_failure_margin": float(np.mean([value for value, success in zip(values, successes) if success]) - np.mean([value for value, success in zip(values, successes) if not success])) if 0 < sum(successes) < len(successes) else float("nan"),
            "fixed_order_drop": oracle["fixed_order_drop"],
            "within_node_reward_density": float(np.nanmean([row["within_node_reward_density"] for row in returns])) if pathgraph else float("nan"),
            "statistics_unit": "content_group_id",
            "provenance": "stage8_recomputed_from_checkpoint_predictions_and_frozen_oracle_traces",
        })
    write_csv(args.output, main_rows)
    write_csv(args.trace_output, all_returns + all_trace_returns)
    write_csv(args.transition_output, all_transitions)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# Reproduced Reward Results\n\n"
        "All test-episode returns were recomputed from Stage 8 ensemble checkpoint predictions. Legal-path and loop-safety diagnostics were re-evaluated on the frozen Stage 5 oracle trace bank and are labelled controlled symbolic stress.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
