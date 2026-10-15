"""Uniform, causal hysteresis, and explicitly offline change-point baselines."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import load_episode, read_csv, write_csv, write_json
from .evaluate import evaluate_predictions


def _enforce_minimum(boundary: np.ndarray, minimum: int) -> np.ndarray:
    out = np.zeros_like(boundary, dtype=np.int8); last = -minimum
    for index in np.where(boundary)[0]:
        if index - last >= minimum: out[index] = 1; last = int(index)
    return out


def _uniform(T: int, length: int) -> np.ndarray:
    out = np.zeros(T, dtype=np.int8); out[max(0, length - 1)::length] = 1; return out


def _hysteresis(probability: np.ndarray, threshold: float, minimum: int = 3) -> np.ndarray:
    above = probability >= threshold; out = np.zeros(len(probability), dtype=np.int8)
    for t in range(1, len(probability)):
        if above[t - 1] and above[t]: out[t] = 1
    return _enforce_minimum(out, minimum)


def _offline_change(observations: np.ndarray, window: int, quantile: float, minimum: int, train_mean: np.ndarray, train_std: np.ndarray, cutoff: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    x = (observations - train_mean) / train_std; T = len(x); score = np.zeros(T, dtype=np.float32)
    for t in range(window, T - window):
        past = x[t-window:t]; future = x[t:t+window]
        mean_gap = np.linalg.norm(past.mean(0) - future.mean(0))
        cov_gap = np.linalg.norm(np.cov(past.T) - np.cov(future.T), ord="fro")
        score[t] = mean_gap + 0.10 * cov_gap
    valid = score[window:T-window]
    if valid.size == 0:
        # Short terminal trajectories are valid episodes; use their available
        # score range rather than excluding them from the offline baseline.
        valid = score
    threshold = float(np.quantile(valid, quantile)) if cutoff is None else cutoff
    return _enforce_minimum(score >= threshold, minimum), score, threshold


def run_baselines(dataset: Path, weak_posteriors: Path, methods: list[str], selection_split: str, output_root: Path, selection_path: Path, grid_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_csv(dataset / "episode_manifest.csv"); train_rows = [row for row in rows if row["split"] == "train"]
    all_train_obs = np.concatenate([load_episode(row)["observations"] for row in train_rows])
    mean = all_train_obs.mean(0); std = np.maximum(all_train_obs.std(0), 1e-6)
    gaps: list[int] = []
    for row in train_rows:
        indexes = np.where(load_episode(row)["gold_boundary"])[0]
        gaps.extend(np.diff(np.r_[0, indexes]).tolist())
    uniform_length = max(3, int(np.median(gaps))) if gaps else 8
    configs: list[dict[str, Any]] = []
    if "uniform" in methods: configs.append({"method": "uniform", "segment_length_train_median": uniform_length, "causal": True})
    if "sensor_hysteresis" in methods:
        configs += [{"method": "sensor_hysteresis", "threshold": x, "minimum_segment_length": 3, "causal": True} for x in (0.35, 0.45, 0.55, 0.65)]
    if "multivariate_change_point" in methods:
        configs += [{"method": "multivariate_change_point", "window": w, "quantile": q, "minimum_segment_length": m, "causal": False, "offline_noncausal": True} for w in (4, 8) for q in (0.85, 0.90, 0.95) for m in (3, 5)]
    grid: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    for ci, config in enumerate(configs):
        name = f"{config['method']}_{ci:02d}"; target = output_root / name; target.mkdir(parents=True, exist_ok=True)
        threshold_cache: float | None = None
        if config["method"] == "multivariate_change_point":
            train_scores = []
            for row in train_rows:
                _, score, _ = _offline_change(load_episode(row)["observations"], config["window"], config["quantile"], config["minimum_segment_length"], mean, std)
                train_scores.append(score)
            threshold_cache = float(np.quantile(np.concatenate(train_scores), config["quantile"]))
        elapsed = 0.0
        for row in rows:
            ep = load_episode(row); start = time.perf_counter()
            with np.load(weak_posteriors / "weak_zero_gold" / f"{row['episode_id']}.npz") as posterior:
                weak_event = posterior["event_argmax"]; weak_prob = posterior["boundary_probability"]; unknown = posterior["unknown"]
            if config["method"] == "uniform": boundary = _uniform(len(ep["observations"]), config["segment_length_train_median"]); score = boundary.astype(np.float32)
            elif config["method"] == "sensor_hysteresis": boundary = _hysteresis(weak_prob, config["threshold"], config["minimum_segment_length"]); score = weak_prob
            else: boundary, score, _ = _offline_change(ep["observations"], config["window"], config["quantile"], config["minimum_segment_length"], mean, std, threshold_cache)
            event = weak_event.copy(); event[~boundary.astype(bool)] = 0
            np.savez_compressed(target / f"{row['episode_id']}.npz", boundary_prediction=boundary, boundary_probability=score.astype(np.float32), event_prediction=event.astype(np.int8), unknown=unknown)
            elapsed += time.perf_counter() - start
        summary, _ = evaluate_predictions(dataset, target, name, selection_split, 2)
        summary.update(config); summary["config_name"] = name; summary["runtime_per_1000_steps"] = elapsed / max(sum(int(r["n_steps"]) for r in rows), 1) * 1000
        # Selection explicitly operates only on validation.
        gold_segments = max(1.0, sum(load_episode(r)["gold_boundary"].sum() for r in rows if r["split"] == selection_split))
        summary["selection_score"] = summary["boundary_f1_tol2"] + .25 * summary["recovery_start_recall"] - .10 * abs(summary["events_per_episode"] * len([r for r in rows if r['split']==selection_split]) / gold_segments - 1)
        grid.append(summary)
    selections: list[dict[str, Any]] = []
    for method in sorted({row["method"] for row in grid}):
        candidates = [row for row in grid if row["method"] == method]; best = max(candidates, key=lambda row: row["selection_score"])
        selections.append({"method": method, "config_name": best["config_name"], "selection_split": selection_split, "test_used_for_selection": False, "selection_score": best["selection_score"], "causal": best["causal"], "offline_noncausal": best.get("offline_noncausal", False)})
    write_csv(grid_path, grid, sorted({key for row in grid for key in row})); write_csv(selection_path, selections, list(selections[0]))
    write_json(output_root.parent / "configs" / "baseline_train_normalization.json", {"fit_split": "train", "mean": mean.tolist(), "std": std.tolist(), "uniform_segment_length_train_median": uniform_length, "test_used_for_selection": False})
    return grid, selections


def evaluate_selected_baselines(dataset: Path, prediction_root: Path, selection: Path, split: str, output: Path, per_event: Path, report: Path, tolerance: int) -> list[dict[str, Any]]:
    selection_rows = read_csv(selection); summaries: list[dict[str, Any]] = []; events: list[dict[str, Any]] = []
    for item in selection_rows:
        summary, details = evaluate_predictions(dataset, prediction_root / item["config_name"], item["method"], split, tolerance)
        summary.update({"config_name": item["config_name"], "causal": item["causal"], "offline_noncausal": item.get("offline_noncausal", "False")})
        summaries.append(summary); events.extend(details)
    write_csv(output, summaries, sorted({x for row in summaries for x in row})); write_csv(per_event, events, list(events[0]))
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text("# U2 segmentation baseline summary\n\n" + "\n".join(f"- {x['method']}: F1±2={x['boundary_f1_tol2']:.4f}" for x in summaries) + "\n", encoding="utf-8")
    return summaries
