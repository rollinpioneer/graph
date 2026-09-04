#!/usr/bin/env python3
"""Freeze the Transport-MH 50% filtering pilot inputs before training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


COMPONENT_KEYS = (
    "sum_of_sum-net-all",
    "min_of_max-net-all",
    "max_of_min-net-all",
)
QUALITY_WEIGHTS = np.asarray([0.50, 0.25, 0.25], dtype=np.float64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def minmax(values: np.ndarray) -> np.ndarray:
    low = float(values.min())
    high = float(values.max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(f"cannot normalize score range [{low}, {high}]")
    return (values - low) / (high - low)


def stable_lowest_ids(scores: np.ndarray, dataset_ids: np.ndarray, count: int) -> np.ndarray:
    order = np.lexsort((dataset_ids, scores))
    return dataset_ids[order[:count]]


def write_ids(path: Path, values: np.ndarray) -> None:
    path.write_text(
        "\n".join(str(int(value)) for value in values) + "\n",
        encoding="ascii",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--score-pickle", type=Path, required=True)
    parser.add_argument("--official-score-csv", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=20260725)
    parser.add_argument("--filter-count", type=int, default=96)
    args = parser.parse_args()

    for path in (
        args.split_json,
        args.score_pickle,
        args.official_score_csv,
        args.dataset,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    dataset_ids = np.asarray(split["train_demo_indices"], dtype=np.int64)
    if dataset_ids.ndim != 1 or len(dataset_ids) != 192:
        raise ValueError(f"expected 192 training IDs, got {dataset_ids.shape}")
    if len(np.unique(dataset_ids)) != len(dataset_ids):
        raise ValueError("training IDs contain duplicates")
    if not 0 < args.filter_count < len(dataset_ids):
        raise ValueError("filter count must be between zero and the training-set size")

    with args.score_pickle.open("rb") as handle:
        payload = pickle.load(handle)
    train_scores = payload["train"]
    components = np.vstack(
        [np.asarray(train_scores[key], dtype=np.float64) for key in COMPONENT_KEYS]
    )
    if components.shape != (3, len(dataset_ids)) or not np.isfinite(components).all():
        raise ValueError(f"invalid component score matrix: {components.shape}")

    cupid_scores = components[0]
    quality_scores = np.sum(
        np.vstack([minmax(row) for row in components]) * QUALITY_WEIGHTS[:, None],
        axis=0,
    )

    official_by_column = np.full(len(dataset_ids), np.nan, dtype=np.float64)
    official_dataset_ids = np.full(len(dataset_ids), -1, dtype=np.int64)
    with args.official_score_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            column = int(row["train_demo_column"])
            official_by_column[column] = float(row["score"])
            official_dataset_ids[column] = int(row["dataset_demo_index"])
    if not np.array_equal(official_dataset_ids, dataset_ids):
        raise ValueError("official score CSV and frozen training split disagree")
    max_official_error = float(np.max(np.abs(official_by_column - cupid_scores)))
    if max_official_error > 2e-3:
        raise ValueError(f"official score reconstruction error {max_official_error}")

    rng = np.random.default_rng(args.random_seed)
    random_ids = np.sort(
        rng.choice(dataset_ids, size=args.filter_count, replace=False).astype(np.int64)
    )
    arm_ids = {
        "cupid50": stable_lowest_ids(cupid_scores, dataset_ids, args.filter_count),
        "quality50": stable_lowest_ids(quality_scores, dataset_ids, args.filter_count),
        "random50": random_ids,
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    ids_dir = args.output_dir / "filter_ids"
    ids_dir.mkdir()
    for arm, values in arm_ids.items():
        if len(values) != args.filter_count or len(np.unique(values)) != len(values):
            raise ValueError(f"invalid frozen IDs for {arm}")
        if not set(values.tolist()).issubset(set(dataset_ids.tolist())):
            raise ValueError(f"{arm} includes a non-training ID")
        write_ids(ids_dir / f"{arm}.txt", values)

    score_csv = args.output_dir / "frozen_scores.csv"
    with score_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "train_demo_column",
                "dataset_demo_index",
                "cupid_score",
                "min_of_max_score",
                "max_of_min_score",
                "cupid_quality_score",
                "filter_cupid50",
                "filter_quality50",
                "filter_random50",
            ]
        )
        membership = {
            arm: set(values.tolist())
            for arm, values in arm_ids.items()
        }
        for column, dataset_id in enumerate(dataset_ids):
            writer.writerow(
                [
                    column,
                    int(dataset_id),
                    f"{cupid_scores[column]:.10g}",
                    f"{components[1, column]:.10g}",
                    f"{components[2, column]:.10g}",
                    f"{quality_scores[column]:.10g}",
                    int(dataset_id in membership["cupid50"]),
                    int(dataset_id in membership["quality50"]),
                    int(dataset_id in membership["random50"]),
                ]
            )

    overlap = {}
    for left in arm_ids:
        for right in arm_ids:
            if left >= right:
                continue
            left_set = set(arm_ids[left].tolist())
            right_set = set(arm_ids[right].tolist())
            overlap[f"{left}__{right}"] = {
                "intersection": len(left_set & right_set),
                "union": len(left_set | right_set),
                "jaccard": len(left_set & right_set) / len(left_set | right_set),
            }

    manifest = {
        "experiment_id": "transport_filter50_pilot_20260725",
        "task": "transport_mh",
        "phase": "seed0_pilot",
        "filter_count": args.filter_count,
        "original_training_count": len(dataset_ids),
        "remaining_training_count": len(dataset_ids) - args.filter_count,
        "training_seed": 0,
        "dataset_seed": 0,
        "random_filter_seed": args.random_seed,
        "filtered_num_epochs": 2301,
        "filtered_expected_last_epoch": 2300,
        "evaluation_test_start_seed": 200000,
        "evaluation_num_episodes": 100,
        "quality_component_keys": list(COMPONENT_KEYS),
        "quality_weights": QUALITY_WEIGHTS.tolist(),
        "official_score_max_abs_error": max_official_error,
        "score_pearson": float(np.corrcoef(cupid_scores, quality_scores)[0, 1]),
        "arms": {
            "baseline": {
                "mode": "eval_only",
                "physical_gpu": 2,
                "filter_ids_file": None,
            },
            "cupid50": {
                "mode": "train_then_eval",
                "physical_gpu": 1,
                "filter_ids_file": str((ids_dir / "cupid50.txt").resolve()),
            },
            "quality50": {
                "mode": "train_then_eval",
                "physical_gpu": 4,
                "filter_ids_file": str((ids_dir / "quality50.txt").resolve()),
            },
            "random50": {
                "mode": "train_then_eval",
                "physical_gpu": 5,
                "filter_ids_file": str((ids_dir / "random50.txt").resolve()),
            },
        },
        "overlap": overlap,
        "inputs": {
            "dataset": str(args.dataset.resolve()),
            "dataset_sha256": sha256(args.dataset),
            "training_split": str(args.split_json.resolve()),
            "training_split_sha256": sha256(args.split_json),
            "score_pickle": str(args.score_pickle.resolve()),
            "score_pickle_sha256": sha256(args.score_pickle),
            "official_score_csv": str(args.official_score_csv.resolve()),
            "official_score_csv_sha256": sha256(args.official_score_csv),
        },
    }
    manifest_path = args.output_dir / "experiment_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )

    hashes = [
        (path.relative_to(args.output_dir), sha256(path))
        for path in sorted(args.output_dir.rglob("*"))
        if path.is_file()
    ]
    with (args.output_dir / "artifact_sha256.txt").open("w", encoding="ascii") as handle:
        for relative, digest in hashes:
            handle.write(f"{digest}  {relative}\n")

    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    print("TRANSPORT FILTER50 PILOT INPUT FREEZE PASS")


if __name__ == "__main__":
    main()
