#!/usr/bin/env python3
"""
Analyze how the number of saved policy rollouts affects CUPID demonstration-ranking stability.

Run this script from the root of the official CUPID repository after:
1) training one base policy,
2) saving 100 rollout episodes,
3) running train_trak_diffusion.py with finalized scores.

The script does not train a policy and does not modify the dataset.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict, dataclass

REPO_DIR = pathlib.Path(__file__).resolve().parents[1] / "repo"
sys.path.insert(0, str(REPO_DIR))

import hydra
import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr

from diffusion_policy.common import error_util, results_util, trak_util
from diffusion_policy.dataset.episode_dataset import BatchEpisodeDataset


@dataclass(frozen=True)
class StabilityRow:
    budget: int
    repeat: int
    num_success: int
    num_failure: int
    spearman: float
    kendall: float
    top_jaccard: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=pathlib.Path, required=True)
    parser.add_argument("--eval-dir", type=pathlib.Path, required=True)
    parser.add_argument("--train-ckpt", type=str, default="latest")
    parser.add_argument("--result-date", type=str, default="default")
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 25, 50, 100])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--top-fraction", type=float, default=0.20)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--sampling",
        choices=["random", "stratified"],
        default="random",
        help="random: draw from all rollouts; stratified: roughly preserve success/failure ratio.",
    )
    return parser.parse_args()


def resolve_checkpoint(train_dir: pathlib.Path, train_ckpt: str) -> pathlib.Path:
    checkpoint_dir = train_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoints = list(checkpoint_dir.iterdir())
    if train_ckpt.isdigit():
        checkpoint = trak_util.get_index_checkpoint(checkpoints, int(train_ckpt))
    elif train_ckpt == "best":
        checkpoint = trak_util.get_best_checkpoint(checkpoints)
    else:
        checkpoint = checkpoint_dir / f"{train_ckpt}.ckpt"

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint


def finite_correlation(value: float | np.floating | None) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    return float(value)


def scipy_correlation(result: object) -> float:
    """Read correlation from both old and new SciPy result objects."""
    value = getattr(result, "statistic", None)
    if value is None:
        value = getattr(result, "correlation")
    return finite_correlation(value)


def top_set(scores: np.ndarray, fraction: float) -> set[int]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("--top-fraction must be in (0, 1].")
    k = max(1, int(round(len(scores) * fraction)))
    return set(np.argsort(scores)[-k:].tolist())


def jaccard(a: set[int], b: set[int]) -> float:
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def random_indices(rng: np.random.Generator, n: int, budget: int) -> np.ndarray:
    return np.sort(rng.choice(n, size=budget, replace=False))


def stratified_indices(
    rng: np.random.Generator,
    success_mask: np.ndarray,
    budget: int,
) -> np.ndarray:
    success_idx = np.flatnonzero(success_mask)
    failure_idx = np.flatnonzero(~success_mask)
    n_total = len(success_mask)

    desired_success = int(round(budget * len(success_idx) / n_total))
    desired_success = min(desired_success, len(success_idx))
    desired_failure = budget - desired_success

    if desired_failure > len(failure_idx):
        desired_failure = len(failure_idx)
        desired_success = budget - desired_failure
    if desired_success > len(success_idx):
        desired_success = len(success_idx)
        desired_failure = budget - desired_success

    selected = []
    if desired_success:
        selected.extend(rng.choice(success_idx, desired_success, replace=False).tolist())
    if desired_failure:
        selected.extend(rng.choice(failure_idx, desired_failure, replace=False).tolist())

    if len(selected) != budget:
        remaining = np.setdiff1d(np.arange(n_total), np.asarray(selected, dtype=int))
        selected.extend(rng.choice(remaining, budget - len(selected), replace=False).tolist())

    return np.sort(np.asarray(selected, dtype=int))


def quality_scores(
    trajectory_influence: np.ndarray,
    success_mask: np.ndarray,
    rollout_indices: np.ndarray,
) -> np.ndarray:
    result = error_util.compute_demo_quality_scores(
        traj_scores=trajectory_influence[rollout_indices],
        success_mask=success_mask[rollout_indices],
        metric="net",
    )
    if result is None:
        raise RuntimeError("CUPID returned no demonstration-quality scores.")
    result = np.asarray(result, dtype=np.float64)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise RuntimeError("Demonstration-quality scores are malformed or non-finite.")
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = resolve_checkpoint(args.train_dir, args.train_ckpt)
    _, cfg = trak_util.get_policy_from_checkpoint(
        checkpoint=checkpoint,
        return_cfg=True,
        device=torch.device("cpu"),
    )

    train_set = hydra.utils.instantiate(cfg.task.dataset)
    holdout_set = train_set.get_holdout_dataset()
    train_metadata = trak_util.get_dataset_metadata(cfg, train_set)
    holdout_metadata = trak_util.get_dataset_metadata(cfg, holdout_set)

    episode_dir = args.eval_dir / "episodes"
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"Saved rollout directory not found: {episode_dir}")

    test_set = BatchEpisodeDataset(
        batch_size=1,
        dataset_path=episode_dir,
        exec_horizon=1,
        sample_history=0,
    )
    test_metadata = trak_util.get_dataset_metadata(cfg, test_set)

    n_rollouts = int(test_metadata["num_eps"])
    budgets = sorted(set(args.budgets))
    if any(b <= 0 or b > n_rollouts for b in budgets):
        raise ValueError(f"Every budget must be between 1 and {n_rollouts}.")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")

    pairwise_sample_scores = results_util.get_trak_scores(
        eval_dir=args.eval_dir,
        train_set_size=train_metadata["num_samples"] + holdout_metadata["num_samples"],
        test_set_size=test_metadata["num_samples"],
        exp_date=args.result_date,
        return_dtype=np.float32,
        debug=False,
        **results_util.LOAD_TRAK_KWARGS["default_diffusion"],
    )

    train_pairwise_scores = pairwise_sample_scores[:, : train_metadata["num_samples"]]
    trajectory_influence = error_util.pairwise_sample_to_trajectory_scores(
        pairwise_sample_scores=train_pairwise_scores,
        num_test_eps=test_metadata["num_eps"],
        num_train_eps=train_metadata["num_eps"],
        test_ep_idxs=test_metadata["ep_idxs"],
        train_ep_idxs=train_metadata["ep_idxs"],
        test_ep_lens=test_metadata["ep_lens"],
        train_ep_lens=train_metadata["ep_lens"],
        success_mask=test_metadata["success_mask"],
        aggr_fn=error_util.sum_of_sum_influence,
        return_dtype=np.float32,
    )

    all_indices = np.arange(n_rollouts)
    reference_scores = quality_scores(
        trajectory_influence,
        test_metadata["success_mask"],
        all_indices,
    )
    reference_top = top_set(reference_scores, args.top_fraction)

    rows: list[StabilityRow] = []
    base_rng = np.random.default_rng(args.seed)

    for budget in budgets:
        for repeat in range(args.repeats):
            if budget == n_rollouts:
                selected = all_indices
            else:
                repeat_seed = int(base_rng.integers(0, np.iinfo(np.int32).max))
                rng = np.random.default_rng(repeat_seed)
                if args.sampling == "stratified":
                    selected = stratified_indices(
                        rng,
                        test_metadata["success_mask"],
                        budget,
                    )
                else:
                    selected = random_indices(rng, n_rollouts, budget)

            scores = quality_scores(
                trajectory_influence,
                test_metadata["success_mask"],
                selected,
            )
            score_top = top_set(scores, args.top_fraction)
            success_count = int(test_metadata["success_mask"][selected].sum())

            rows.append(
                StabilityRow(
                    budget=budget,
                    repeat=repeat,
                    num_success=success_count,
                    num_failure=int(budget - success_count),
                    spearman=scipy_correlation(spearmanr(scores, reference_scores)),
                    kendall=scipy_correlation(kendalltau(scores, reference_scores)),
                    top_jaccard=jaccard(score_top, reference_top),
                )
            )

    frame = pd.DataFrame([asdict(row) for row in rows])
    summary = (
        frame.groupby("budget", as_index=False)
        .agg(
            repeats=("repeat", "count"),
            success_mean=("num_success", "mean"),
            failure_mean=("num_failure", "mean"),
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
            kendall_mean=("kendall", "mean"),
            kendall_std=("kendall", "std"),
            top_jaccard_mean=("top_jaccard", "mean"),
            top_jaccard_std=("top_jaccard", "std"),
        )
        .fillna(0.0)
    )

    frame.to_csv(args.output_dir / "rollout_budget_repeats.csv", index=False)
    summary.to_csv(args.output_dir / "rollout_budget_summary.csv", index=False)

    metadata = {
        "checkpoint": str(checkpoint),
        "train_demo_count": int(train_metadata["num_eps"]),
        "holdout_demo_count": int(holdout_metadata["num_eps"]),
        "rollout_count": n_rollouts,
        "rollout_success_count": int(test_metadata["success_mask"].sum()),
        "rollout_failure_count": int((~test_metadata["success_mask"]).sum()),
        "budgets": budgets,
        "repeats": args.repeats,
        "sampling": args.sampling,
        "top_fraction": args.top_fraction,
        "seed": args.seed,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nCUPID rollout-budget stability summary")
    print(summary.to_string(index=False))
    print(f"\nSaved results to: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
