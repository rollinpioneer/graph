#!/usr/bin/env python3
"""Describe the fixed three-checkpoint ensemble without treating seeds as a population."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-seed-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.per_seed_metrics)
    data = data[(data["suite"] == "test") & (data["model_seed"] != "ensemble")]
    identifiers = ["node_macro_f1", "edge_type_macro_f1_non_none", "phi_mae", "remaining_cost_mae"]
    output = []
    for metric in identifiers:
        values = pd.to_numeric(data[metric], errors="coerce").dropna().to_numpy(dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        direction = -1 if metric.endswith("mae") else 1
        agreement = bool(np.all(direction * (values - (0.5 if "f1" in metric else 0.0)) >= 0))
        output.append({"metric": metric, "seeds": len(values), "mean": mean, "std": std, "min": float(values.min()), "max": float(values.max()), "coefficient_of_variation": std / abs(mean) if mean else float("nan"), "three_of_three_direction_agreement": agreement, "interpretation": "descriptive_fixed_three_checkpoint_summary"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).to_csv(args.output, index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("# Reward-model Seed Stability\n\nThis is a descriptive summary of exactly three frozen checkpoints, not a large-sample significance calculation.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
