"""Deterministic D0 train64/dev10 split. Frozen before any VLM or training result."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np


def _xy(rng, x_lo, x_hi, y_lo, y_hi):
    return [float(rng.uniform(x_lo, x_hi)), float(rng.uniform(y_lo, y_hi))]


def build_split(path: Path):
    train, dev = [], []
    # Layout: robot at -y, table 0.8x0.8. Keep objects in reachable front half.
    container = [0.18, 0.12]
    buffer = [-0.18, 0.12]
    for i in range(64):
        rng = np.random.RandomState(1000 + i)
        t = _xy(rng, -0.22, 0.00, -0.18, -0.02)
        s = _xy(rng, 0.04, 0.22, -0.18, -0.02)
        while float(np.linalg.norm(np.array(t) - np.array(s))) < 0.09:
            s = _xy(rng, 0.04, 0.22, -0.18, -0.02)
        row = {
            "case_id": f"D0_train_{i:02d}",
            "split": "train",
            "seed": int(1000 + i),
            "target_xy": t,
            "second_xy": s,
            "container_xy": container,
            "buffer_xy": buffer,
            "lid_closed": True,
        }
        train.append(row)
    for i in range(10):
        rng = np.random.RandomState(5000 + i)
        t = _xy(rng, -0.22, 0.00, -0.18, -0.02)
        s = _xy(rng, 0.04, 0.22, -0.18, -0.02)
        while float(np.linalg.norm(np.array(t) - np.array(s))) < 0.09:
            s = _xy(rng, 0.04, 0.22, -0.18, -0.02)
        row = {
            "case_id": f"D0_dev_{i:02d}",
            "split": "dev",
            "seed": int(5000 + i),
            "target_xy": t,
            "second_xy": s,
            "container_xy": container,
            "buffer_xy": buffer,
            "lid_closed": True,
        }
        dev.append(row)
    # enforce non-overlap of object seeds / positions by construction of disjoint seed namespaces
    payload = {
        "task_id": "D0",
        "version": "d0-runtime-v2.1-p0",
        "policy": "frozen_before_vlm_or_training",
        "train_count": 64,
        "dev_count": 10,
        "test_isolated": True,
        "reuse_stage_0c_d0": False,
        "reuse_reason": "Action registry dropped MOVE; runtime adds Panda+lid so RGB hash cannot match Stage 0C robot-free scenes. Stage 0C D0 caches archived, not deleted.",
        "train": train,
        "dev": dev,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    build_split(root / "configs/splits/D0_stage_1a.json")
    print("wrote configs/splits/D0_stage_1a.json")
