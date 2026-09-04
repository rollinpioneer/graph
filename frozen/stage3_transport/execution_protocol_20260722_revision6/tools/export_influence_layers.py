#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1] / "repo"
sys.path.insert(0, str(REPO_DIR))

import dill
import hydra
import numpy as np
import pandas as pd
import torch

from diffusion_policy.common import error_util, trak_util
from diffusion_policy.dataset.episode_dataset import BatchEpisodeDataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_finite_in_chunks(array: np.ndarray, chunk_rows: int = 2048) -> None:
    for start in range(0, array.shape[0], chunk_rows):
        chunk = array[start:start + chunk_rows]
        assert np.all(np.isfinite(chunk)), f"non-finite raw scores at row {start}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--train-ckpt", default="latest")
    parser.add_argument("--trak-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--rollout-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.train_dir / "checkpoints" / f"{args.train_ckpt}.ckpt"
    payload = torch.load(checkpoint.open("rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]

    train_set = hydra.utils.instantiate(cfg.task.dataset)
    holdout_set = train_set.get_holdout_dataset()
    test_set = BatchEpisodeDataset(
        batch_size=1,
        dataset_path=args.eval_dir / "episodes",
        exec_horizon=1,
        sample_history=0,
    )
    train_meta = trak_util.get_dataset_metadata(cfg, train_set)
    holdout_meta = trak_util.get_dataset_metadata(cfg, holdout_set)
    test_meta = trak_util.get_dataset_metadata(cfg, test_set)

    total_demo_samples = train_meta["num_samples"] + holdout_meta["num_samples"]
    test_samples = test_meta["num_samples"]
    raw_path = args.trak_dir / "scores" / "all_episodes.mmap"
    assert raw_path.is_file(), raw_path
    # TRAK uses numpy.lib.format.open_memmap, so this file has a .npy header
    # even though its suffix is .mmap.  np.load(..., mmap_mode="r") parses that
    # header while retaining memory-mapped access to the 1.2 GB score matrix.
    raw_stored = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    assert raw_stored.dtype == np.dtype(np.float32), raw_stored.dtype
    assert raw_stored.shape == (total_demo_samples, test_samples), (
        raw_stored.shape,
        (total_demo_samples, test_samples),
    )
    assert_finite_in_chunks(raw_stored)
    pairwise = raw_stored.T
    train_pairwise = pairwise[:, :train_meta["num_samples"]]
    rollout_demo = error_util.pairwise_sample_to_trajectory_scores(
        pairwise_sample_scores=train_pairwise,
        num_test_eps=test_meta["num_eps"],
        num_train_eps=train_meta["num_eps"],
        test_ep_idxs=test_meta["ep_idxs"],
        train_ep_idxs=train_meta["ep_idxs"],
        test_ep_lens=test_meta["ep_lens"],
        train_ep_lens=train_meta["ep_lens"],
        success_mask=test_meta["success_mask"],
        aggr_fn=error_util.sum_of_sum_influence,
        return_dtype=np.float32,
    )
    assert rollout_demo.shape == (100, 192)
    assert np.all(np.isfinite(rollout_demo))

    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    demo_indices = split["train_demo_indices"]
    rollout_frame = pd.read_csv(args.rollout_manifest)
    assert len(demo_indices) == rollout_demo.shape[1]
    assert len(rollout_frame) == rollout_demo.shape[0]
    success_mask = rollout_frame["success"].astype(bool).to_numpy()
    assert np.array_equal(success_mask, test_meta["success_mask"])

    influence_path = args.output_dir / "rollout_demo_influence.npy"
    np.save(influence_path, rollout_demo)
    final_scores = error_util.compute_demo_quality_scores(
        traj_scores=rollout_demo,
        success_mask=success_mask,
        metric="net",
    )
    assert final_scores is not None and np.all(np.isfinite(final_scores))
    final_scores = np.asarray(final_scores, dtype=np.float32)
    np.save(args.output_dir / "final_demo_scores.npy", final_scores)

    order = np.argsort(-final_scores, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(1, len(order) + 1)
    score_frame = pd.DataFrame({
        "train_demo_column": np.arange(len(demo_indices)),
        "dataset_demo_index": demo_indices,
        "score": final_scores,
        "rank_descending": rank,
        "top_20_percent": rank <= max(1, int(round(0.20 * len(rank)))),
    })
    score_frame.sort_values("rank_descending").to_csv(
        args.output_dir / "final_demo_scores.csv", index=False
    )

    raw_hash = sha256(raw_path)
    influence_hash = sha256(influence_path)
    metadata = {
        "checkpoint": str(checkpoint.resolve()),
        "raw_sample_influence": {
            "path": str(raw_path.resolve()),
            "sha256": raw_hash,
            "dtype": "float32",
            "stored_shape": [total_demo_samples, test_samples],
            "logical_shape_test_by_demo": [test_samples, total_demo_samples],
            "train_sample_count": train_meta["num_samples"],
            "holdout_sample_count": holdout_meta["num_samples"],
            "rollout_decision_point_count": test_samples,
        },
        "rollout_demo_influence": {
            "path": str(influence_path.resolve()),
            "sha256": influence_hash,
            "dtype": str(rollout_demo.dtype),
            "shape": list(rollout_demo.shape),
            "aggregation": "sum_of_sum_influence",
            "row_episode_indices": rollout_frame["episode"].astype(int).tolist(),
            "row_seeds": rollout_frame["seed"].astype(int).tolist(),
            "row_success": success_mask.tolist(),
            "column_dataset_demo_indices": demo_indices,
        },
        "final_demo_scores": {
            "metric": "net",
            "success_count": int(success_mask.sum()),
            "failure_count": int((~success_mask).sum()),
            "demo_count": len(demo_indices),
        },
    }
    (args.output_dir / "rollout_demo_influence_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "raw_shape": metadata["raw_sample_influence"]["stored_shape"],
        "rollout_demo_shape": list(rollout_demo.shape),
        "success_count": int(success_mask.sum()),
        "failure_count": int((~success_mask).sum()),
        "output_dir": str(args.output_dir),
    }, indent=2))
    print("INFLUENCE LAYERS EXPORT PASS")


if __name__ == "__main__":
    main()
