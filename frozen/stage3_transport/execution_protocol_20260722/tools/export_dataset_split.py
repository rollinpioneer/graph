#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1] / "repo"
sys.path.insert(0, str(REPO_DIR))

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(
        args.checkpoint.open("rb"),
        pickle_module=dill,
        map_location="cpu",
    )
    cfg = payload["cfg"]
    dataset = hydra.utils.instantiate(cfg.task.dataset)

    required = ["train_mask", "val_mask", "holdout_mask"]
    for name in required:
        if not hasattr(dataset, name):
            raise AttributeError(f"dataset has no {name}")

    masks = {
        "train": np.asarray(dataset.train_mask, dtype=bool),
        "validation": np.asarray(dataset.val_mask, dtype=bool),
        "holdout": np.asarray(dataset.holdout_mask, dtype=bool),
    }

    n = len(masks["train"])
    assert all(len(mask) == n for mask in masks.values())

    total_membership = sum(mask.astype(np.int64) for mask in masks.values())
    assert np.all(total_membership == 1), (
        "Every demonstration must belong to exactly one split."
    )

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "num_total_demos": int(n),
        "train_demo_indices": np.flatnonzero(masks["train"]).astype(int).tolist(),
        "validation_demo_indices": np.flatnonzero(
            masks["validation"]
        ).astype(int).tolist(),
        "holdout_demo_indices": np.flatnonzero(
            masks["holdout"]
        ).astype(int).tolist(),
    }

    (args.output_dir / "dataset_split.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    np.savez_compressed(
        args.output_dir / "dataset_split_masks.npz",
        train_mask=masks["train"],
        validation_mask=masks["validation"],
        holdout_mask=masks["holdout"],
    )

    (args.output_dir / "checkpoint_cfg.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=False),
        encoding="utf-8",
    )

    print(json.dumps({
        "total": n,
        "train": int(masks["train"].sum()),
        "validation": int(masks["validation"].sum()),
        "holdout": int(masks["holdout"].sum()),
    }, indent=2))
    print("saved to:", args.output_dir)


if __name__ == "__main__":
    main()
