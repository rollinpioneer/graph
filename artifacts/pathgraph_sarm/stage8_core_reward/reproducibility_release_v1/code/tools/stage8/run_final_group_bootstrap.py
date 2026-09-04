#!/usr/bin/env python3
"""Deterministic stratified bootstrap over content groups only."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


def strata(groups: pd.DataFrame) -> list[np.ndarray]:
    return [part["content_group_id"].drop_duplicates().to_numpy() for _, part in groups.groupby(["task_id", "provenance"], sort=True)]


def resample_mean(frame: pd.DataFrame, rng: np.random.Generator, resamples: int) -> tuple[float, np.ndarray, int]:
    grouped = frame.groupby(["task_id", "provenance", "content_group_id"], sort=True)["metric_value"].mean().reset_index()
    maps = {group: value for group, value in zip(grouped["content_group_id"], grouped["metric_value"])}
    chunks = strata(grouped)
    out = np.empty(resamples, dtype=float)
    for index in range(resamples):
        selected = np.concatenate([rng.choice(chunk, size=len(chunk), replace=True) for chunk in chunks])
        out[index] = float(np.mean([maps[group] for group in selected]))
    return float(grouped["metric_value"].mean()), out, len(grouped)


def resample_success_margin(frame: pd.DataFrame, rng: np.random.Generator, resamples: int) -> tuple[float, np.ndarray, int]:
    grouped = frame.groupby(["task_id", "provenance", "content_group_id", "outcome"], sort=True)["metric_value"].mean().reset_index()
    if set(grouped["outcome"]) < {"success", "failure"}:
        raise ValueError("success/failure support absent")
    # A success-vs-failure contrast is resampled within outcome inside each
    # task/provenance stratum. This preserves both arms in every resample while
    # retaining content_group_id as the independent observation.
    chunks = [part["content_group_id"].drop_duplicates().to_numpy() for _, part in grouped.groupby(["task_id", "provenance", "outcome"], sort=True)]
    lookup = {group: (float(value), outcome) for group, value, outcome in zip(grouped["content_group_id"], grouped["metric_value"], grouped["outcome"])}
    out = np.empty(resamples, dtype=float)
    for index in range(resamples):
        selected = np.concatenate([rng.choice(chunk, size=len(chunk), replace=True) for chunk in chunks])
        values = [lookup[group] for group in selected]
        success = [value for value, outcome in values if outcome == "success"]
        failure = [value for value, outcome in values if outcome == "failure"]
        out[index] = float(np.mean(success) - np.mean(failure)) if success and failure else np.nan
    out = out[np.isfinite(out)]
    success = grouped[grouped["outcome"] == "success"]["metric_value"]
    failure = grouped[grouped["outcome"] == "failure"]["metric_value"]
    return float(success.mean() - failure.mean()), out, len(grouped)


def resample_contrast(frame: pd.DataFrame, treatment: str, comparator: str, metric: str, rng: np.random.Generator, resamples: int) -> tuple[float, np.ndarray, int]:
    selected = frame[frame["metric_name"] == metric].copy()
    label = selected["variant"].where(selected["variant"].astype(bool), selected["method"])
    selected["_label"] = label
    pivot = selected[selected["_label"].isin([treatment, comparator])].pivot_table(index=["task_id", "provenance", "content_group_id"], columns="_label", values="metric_value", aggfunc="mean")
    pivot = pivot.dropna(subset=[treatment, comparator]).reset_index()
    if pivot.empty:
        raise ValueError(f"paired support absent for {treatment}/{comparator} {metric}")
    pivot["delta"] = pivot[treatment] - pivot[comparator]
    maps = {group: float(value) for group, value in zip(pivot["content_group_id"], pivot["delta"])}
    chunks = strata(pivot)
    out = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sampled = np.concatenate([rng.choice(chunk, size=len(chunk), replace=True) for chunk in chunks])
        out[index] = float(np.mean([maps[group] for group in sampled]))
    return float(pivot["delta"].mean()), out, len(pivot)


def result_row(estimand_id: str, kind: str, point: float, distribution: np.ndarray, groups: int, direction: str) -> dict:
    lo, hi = np.percentile(distribution, [2.5, 97.5])
    if direction == "lower_is_better":
        tail = float((distribution >= 0).mean())
    else:
        tail = float((distribution <= 0).mean())
    # A one-sided bootstrap tail probability is descriptive; this explicit
    # +1 correction avoids an impossible literal zero with finite resamples.
    tail = max(tail, 1.0 / (len(distribution) + 1))
    return {"estimand_id": estimand_id, "estimand_kind": kind, "point_estimate": point, "ci95_low": float(lo), "ci95_high": float(hi), "tail_probability": tail, "bootstrap_resamples_completed": int(len(distribution)), "content_groups": groups, "statistics_unit": "content_group_id", "support_note": "limited_controlled_support" if groups < 4 else "group_bootstrap"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--estimands", type=Path, required=True)
    parser.add_argument("--resamples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--parallel", default="1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distribution-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.resamples != 10000:
        raise SystemExit("frozen Stage 8 protocol requires exactly 10000 resamples")
    data = pq.read_table(args.observations).to_pandas()
    config = yaml.safe_load(args.estimands.read_text(encoding="utf-8"))
    if config["statistics_unit"] != "content_group_id":
        raise SystemExit("estimand lock does not use content_group_id")
    rng = np.random.default_rng(args.seed)
    output: list[dict] = []
    args.distribution_dir.mkdir(parents=True, exist_ok=True)
    main_method = "pathgraph_reward_v1_locked"
    for spec in config["primary_estimands"]:
        identifier, metric, direction = spec["id"], spec["metric"], spec["direction"]
        if identifier == "success_separation":
            frame = data[(data["metric_name"] == "return") & (data["method"] == main_method) & (data["provenance"] == "real_test_checkpoint_ensemble")]
            point, dist, groups = resample_success_margin(frame, rng, args.resamples)
        elif identifier == "path_consistency":
            frame = data[(data["metric_name"] == metric) & (data["method"] == main_method) & (data["provenance"] == "frozen_controlled_symbolic_trace")]
            point, dist, groups = resample_mean(frame, rng, args.resamples)
        elif identifier == "loop_safety":
            frame = data[(data["metric_name"] == metric) & (data["method"] == main_method) & (data["content_group_id"] == "controlled_trace:failure_then_recovery")]
            point, dist, groups = resample_mean(frame, rng, args.resamples)
        else:
            frame = data[(data["metric_name"] == metric) & (data["method"] == main_method) & (data["provenance"] == "real_test_checkpoint_ensemble")]
            point, dist, groups = resample_mean(frame, rng, args.resamples)
        np.save(args.distribution_dir / f"{identifier}_bootstrap.npy", dist)
        output.append(result_row(identifier, "primary", point, dist, groups, direction))
    for spec in config["primary_structural_contrasts"]:
        identifier, metric = spec["id"], spec["metric"]
        point, dist, groups = resample_contrast(data, spec["treatment"], spec["comparator"], metric, rng, args.resamples)
        np.save(args.distribution_dir / f"{identifier}_bootstrap.npy", dist)
        output.append(result_row(identifier, "structural_contrast", point, dist, groups, "lower_is_better" if identifier == "remove_debt_cap" else "higher_is_better"))
    frame = pd.DataFrame(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("# Final Group Bootstrap\n\nAll resampling used `content_group_id` within fixed `task_id, provenance` strata. Controlled symbolic estimands retain their small immutable support count rather than being represented as frame-level samples.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
