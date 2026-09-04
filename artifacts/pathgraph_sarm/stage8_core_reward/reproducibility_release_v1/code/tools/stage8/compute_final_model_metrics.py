#!/usr/bin/env python3
"""Compute model metrics from Stage 8 prediction records, never from a prior result table."""
from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.stage8.common import write_csv


def records(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def f1(true: np.ndarray, predicted: np.ndarray, label: int) -> float:
    tp = np.sum((true == label) & (predicted == label))
    fp = np.sum((true != label) & (predicted == label))
    fn = np.sum((true == label) & (predicted != label))
    return float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0


def macro_f1(true: np.ndarray, predicted: np.ndarray, labels: list[int]) -> float:
    return float(np.mean([f1(true, predicted, label) for label in labels])) if labels else float("nan")


def ordinal_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx = np.empty(len(x), dtype=float)
    ry = np.empty(len(y), dtype=float)
    rx[np.argsort(x, kind="stable")] = np.arange(len(x))
    ry[np.argsort(y, kind="stable")] = np.arange(len(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def metric_row(rows: list[dict], suite: str, seed: str, aggregation: str) -> dict:
    node_true = np.asarray([row["gt_node_id"] for row in rows], dtype=int)
    node_pred = np.asarray([row["node_pred"] if "node_pred" in row else int(np.argmax(row["pred_node_probs"])) for row in rows], dtype=int)
    edge_true = np.asarray([row["gt_edge_type"] for row in rows], dtype=int)
    edge_pred = np.asarray([row["edge_type_pred"] if "edge_type_pred" in row else int(np.argmax(row["pred_edge_type_probs"])) for row in rows], dtype=int)
    edge_id_true = np.asarray([row["gt_edge_id"] for row in rows], dtype=int)
    edge_id_pred = np.asarray([row["edge_id_pred"] if "edge_id_pred" in row else int(np.argmax(row["pred_edge_id_probs"])) for row in rows], dtype=int)
    phi_true = np.asarray([row["gt_phi"] for row in rows], dtype=float)
    phi_pred = np.asarray([row.get("phi_mean", row.get("pred_phi")) for row in rows], dtype=float)
    cost_true = np.asarray([row["gt_remaining_cost"] for row in rows], dtype=float)
    cost_pred = np.asarray([row.get("remaining_cost_mean", row.get("pred_remaining_cost")) for row in rows], dtype=float)
    pair_accuracy = []
    phi_violations = []
    by_episode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_episode[row["episode_id"]].append(row)
    for episode in by_episode.values():
        episode.sort(key=lambda row: int(row["t"]))
        for before, after in zip(episode, episode[1:]):
            if float(before["gt_remaining_cost"]) != float(after["gt_remaining_cost"]):
                actual_direction = float(before["gt_remaining_cost"]) < float(after["gt_remaining_cost"])
                predicted_direction = float(before.get("remaining_cost_mean", before.get("pred_remaining_cost"))) < float(after.get("remaining_cost_mean", after.get("pred_remaining_cost")))
                pair_accuracy.append(actual_direction == predicted_direction)
            if before["gt_node_id"] == after["gt_node_id"] and float(after["gt_phi"]) > float(before["gt_phi"]):
                phi_violations.append(float(after.get("phi_mean", after.get("pred_phi"))) + 1e-9 < float(before.get("phi_mean", before.get("pred_phi"))))
    positive_edge_mask = edge_true > 0
    return {
        "suite": suite,
        "model_seed": seed,
        "aggregation": aggregation,
        "statistics_unit": "content_group_id",
        "content_group_count": len({row["content_group_id"] for row in rows}),
        "prediction_count": len(rows),
        "node_macro_f1": macro_f1(node_true, node_pred, sorted(set(node_true.tolist()))),
        "edge_type_macro_f1_non_none": macro_f1(edge_true[positive_edge_mask], edge_pred[positive_edge_mask], sorted(set(edge_true[positive_edge_mask].tolist()))) if positive_edge_mask.any() else float("nan"),
        "edge_id_macro_f1": macro_f1(edge_id_true[edge_id_true > 0], edge_id_pred[edge_id_true > 0], sorted(set(edge_id_true[edge_id_true > 0].tolist()))) if np.any(edge_id_true > 0) else float("nan"),
        "alternative_edge_f1": f1(edge_true, edge_pred, 2) if np.any(edge_true == 2) else float("nan"),
        "recovery_edge_f1": f1(edge_true, edge_pred, 3) if np.any(edge_true == 3) else float("nan"),
        "phi_mae": float(np.mean(np.abs(phi_pred - phi_true))),
        "phi_spearman": ordinal_spearman(phi_pred, phi_true),
        "phi_monotonic_violation_rate": float(np.mean(phi_violations)) if phi_violations else float("nan"),
        "remaining_cost_mae": float(np.mean(np.abs(cost_pred - cost_true))),
        "remaining_cost_spearman": ordinal_spearman(cost_pred, cost_true),
        "cost_pair_accuracy": float(np.mean(pair_accuracy)) if pair_accuracy else float("nan"),
        "provenance": "stage8_real_frozen_checkpoint_inference",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-seed-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    ensemble_rows = []
    per_seed_rows = []
    for suite in ("val", "test", "stage3_diagnostic"):
        ensemble = records(args.ensemble_root / f"ensemble_{suite}_predictions.jsonl.gz")
        ensemble_rows.append(metric_row(ensemble, suite, "ensemble", "frame_metrics_from_content_group_preserving_records"))
        for seed in ("20260906", "20260907", "20260908"):
            per_seed_rows.append(metric_row(records(args.prediction_root / f"s{seed}__{suite}.jsonl.gz"), suite, seed, "frame_metrics_from_content_group_preserving_records"))
    write_csv(args.output, ensemble_rows)
    write_csv(args.per_seed_output, per_seed_rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    test = next(row for row in ensemble_rows if row["suite"] == "test")
    args.report.write_text(
        "# Reproduced Model Metrics\n\n"
        f"Test ensemble node macro-F1: `{test['node_macro_f1']:.12g}`; edge-type macro-F1: `{test['edge_type_macro_f1_non_none']:.12g}`. "
        "Metrics were recomputed from Stage 8 checkpoint-forward prediction records. Group-level inference will be used for final uncertainty intervals.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
