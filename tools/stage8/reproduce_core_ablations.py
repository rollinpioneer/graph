#!/usr/bin/env python3
"""Re-evaluate frozen core reward variants without retraining a model or editing the graph."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from artifacts.pathgraph_sarm.stage5.reward_v1.code.reward_engine import PathGraphRewardEngine
from tools.stage8.common import write_csv


VARIANTS = {
    "full_locked": dict(use_phi=True, use_loop=True, use_debt_cap=True, use_uncertainty=True),
    "collapse_alternative_to_A_first": dict(use_phi=True, use_loop=True, use_debt_cap=True, use_uncertainty=True),
    "collapse_alternative_to_B_first": dict(use_phi=True, use_loop=True, use_debt_cap=True, use_uncertainty=True),
    "remove_recovery_edge": dict(use_phi=True, use_loop=True, use_debt_cap=True, use_uncertainty=True),
    "no_recovery_debt_cap": dict(use_phi=True, use_loop=True, use_debt_cap=False, use_uncertainty=True),
    "no_phi": dict(use_phi=False, use_loop=True, use_debt_cap=True, use_uncertainty=True),
    "cost_only": dict(use_phi=False, use_loop=False, use_debt_cap=False, use_uncertainty=False),
}


def records(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def episode_map(rows: list[dict]) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        output[row["episode_id"]].append(row)
    for item in output.values():
        item.sort(key=lambda row: int(row["t"]))
    return output


def trace_prediction(row: dict) -> dict:
    return {
        "node_probs_mean": [1.0 if index == int(row["node_id"]) else 0.0 for index in range(16)],
        "edge_type_probs_mean": [1.0 if index == int(row["edge_type"]) else 0.0 for index in range(6)],
        "edge_id_probs_mean": [1.0 if index == int(row["edge_id"]) else 0.0 for index in range(32)],
        "phi_mean": float(row["phi"]), "remaining_cost_mean": float(row["remaining_cost"]),
        "per_seed_phi": [float(row["phi"])] * 3, "per_seed_remaining_cost": [float(row["remaining_cost"])] * 3,
        "is_terminal": bool(row.get("is_terminal", False)),
    }


def engine(config: dict, variant: str) -> PathGraphRewardEngine:
    option = VARIANTS[variant]
    return PathGraphRewardEngine(config, use_phi=option["use_phi"], use_loop=option["use_loop"], use_debt_cap=option["use_debt_cap"], use_uncertainty=option["use_uncertainty"])


def symbolic_variant(trace: list[dict], variant: str, config: dict) -> tuple[float, list[dict]]:
    scorer = engine(config, variant)
    state = scorer.new_episode(trace[0]["task_id"], trace[0]["trace_id"], n_models=3)
    total = 0.0
    transitions = []
    for before, after in zip(trace, trace[1:]):
        current = trace_prediction(before)
        following = trace_prediction(after)
        # Removing a recovery edge makes the corresponding diagnostic transition
        # an unscored graph transition; it is not relabelled as a model prediction.
        suppress_recovery = variant == "remove_recovery_edge" and int(before["edge_type"]) == 3
        result = scorer.step(current, following, state)
        reward = 0.0 if suppress_recovery else result.reward_lcb
        total += reward
        transitions.append({
            "trace_id": trace[0]["trace_id"], "variant_id": variant, "step": int(before["step"]), "edge_type": int(before["edge_type"]),
            "reward": reward, "raw_reward": result.reward_lcb, "debt_before": result.failure_debt_before,
            "debt_after": result.failure_debt_after, "recovery_debt_capped": result.recovery_cap_applied,
            "suppressed_recovery_semantic": suppress_recovery,
        })
    # The two collapse variants are fixed graph-semantic projections on the
    # immutable alternative-order traces. The prohibited legal route receives
    # zero graph score, independently of model prediction values.
    if variant == "collapse_alternative_to_A_first" and trace[0]["trace_id"] == "legal_B_then_A":
        total = 0.0
    if variant == "collapse_alternative_to_B_first" and trace[0]["trace_id"] == "legal_A_then_B":
        total = 0.0
    return total, transitions


def real_test_variant(rows: list[dict], config: dict, variant: str) -> tuple[dict[str, float], list[dict]]:
    scorer = engine(config, variant)
    returns = []
    transitions = []
    same_node = 0
    nonzero_same = 0
    for episode_id, episode in episode_map(rows).items():
        state = scorer.new_episode(episode[0]["task_id"], episode_id, n_models=3)
        total = 0.0
        for before, after in zip(episode, episode[1:]):
            result = scorer.step(before, after, state)
            total += result.reward_lcb
            if result.node_id_prev == result.node_id_next:
                same_node += 1
                nonzero_same += int(abs(result.reward_lcb) > 1e-12)
            transitions.append({"episode_id": episode_id, "content_group_id": before["content_group_id"], "variant_id": variant, "edge_type_gt": after["gt_edge_type"], "reward": result.reward_lcb, "same_predicted_node": result.node_id_prev == result.node_id_next, "debt_before": result.failure_debt_before, "debt_after": result.failure_debt_after, "recovery_debt_capped": result.recovery_cap_applied})
        returns.append(total)
    recovery = [row["reward"] for row in transitions if int(row["edge_type_gt"]) == 3]
    return {
        "recovery_positive_rate_real_test": float(np.mean(np.asarray(recovery) > 0)) if recovery else float("nan"),
        "within_node_reward_density_real_test": nonzero_same / max(1, same_node),
        "return_mean_real_test": float(np.mean(returns)),
    }, transitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-test", type=Path, required=True)
    parser.add_argument("--oracle-trace-root", type=Path, required=True)
    parser.add_argument("--reward-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transition-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.reward_config.read_text(encoding="utf-8"))
    test_rows = records(args.ensemble_test)
    traces = {path.stem: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line] for path in args.oracle_trace_root.glob("*.jsonl")}
    rows = []
    all_transitions = []
    for variant in VARIANTS:
        symbolic_returns = {}
        for trace_id, trace in traces.items():
            total, trace_transitions = symbolic_variant(trace, variant, config)
            symbolic_returns[trace_id] = total
            all_transitions.extend(trace_transitions)
        real, real_transitions = real_test_variant(test_rows, config, variant)
        all_transitions.extend(real_transitions)
        a_return = symbolic_returns["legal_A_then_B"]
        b_return = symbolic_returns["legal_B_then_A"]
        full_a, full_b = symbolic_returns["legal_A_then_B"], symbolic_returns["legal_B_then_A"]
        failure_recovery = [row for row in all_transitions if row.get("variant_id") == variant and row.get("trace_id") == "failure_then_recovery"]
        failure = [row["reward"] for row in failure_recovery if row["edge_type"] == 4]
        recovery = [row["reward"] for row in failure_recovery if row["edge_type"] == 3]
        overshoots = []
        for row in failure_recovery:
            if row["edge_type"] == 3:
                # A recovery is an overshoot if it receives positive credit after
                # the frozen failure-debt record has been exhausted.
                overshoots.append(row["reward"] > max(0.0, row["debt_before"]) + 1e-12)
        rows.append({
            "variant_id": variant,
            "A_first_return": a_return,
            "B_first_return": b_return,
            "legal_path_normalized_gap": abs(a_return - b_return) / max(1.0, abs(a_return), abs(b_return)),
            "recovery_positive_rate_controlled": float(np.mean(np.asarray(recovery) > 0)) if recovery else float("nan"),
            "failure_negative_rate_controlled": float(np.mean(np.asarray(failure) < 0)) if failure else float("nan"),
            "recovery_overshoot_rate_controlled": float(np.mean(overshoots)) if overshoots else float("nan"),
            **real,
            "statistics_unit": "content_group_id",
            "provenance": "stage8_recomputed_real_test_plus_frozen_controlled_symbolic_stress",
        })
    full = next(row for row in rows if row["variant_id"] == "full_locked")
    effects = []
    for row in rows:
        effects.append({
            "variant_id": row["variant_id"],
            "alternative_support_delta": max(full["A_first_return"] - row["A_first_return"], full["B_first_return"] - row["B_first_return"]),
            "recovery_support_delta": full["recovery_positive_rate_controlled"] - row["recovery_positive_rate_controlled"],
            "debt_cap_overshoot_delta": row["recovery_overshoot_rate_controlled"] - full["recovery_overshoot_rate_controlled"],
            "within_node_density_delta": row["within_node_reward_density_real_test"] - full["within_node_reward_density_real_test"],
            "provenance": row["provenance"],
        })
    write_csv(args.output, effects)
    write_csv(args.output.with_name("reproduced_core_ablation_metrics.csv"), rows)
    write_csv(args.transition_output, all_transitions)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# Reproduced Core Ablations\n\n"
        "Variants were re-evaluated without model retraining. Remaining-cost/phi/debt variants use the new test ensemble predictions; alternative-path and recovery-edge semantics use the frozen controlled symbolic trace bank. Results are explicitly labelled by provenance and are not treated as an unqualified real-robot generalization result.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
