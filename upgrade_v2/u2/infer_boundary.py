"""Thin inference wrapper kept separate from training for reproducible job manifests."""
from __future__ import annotations
from pathlib import Path
from .train_boundary import load_and_predict

def infer(checkpoint: Path, dataset: Path, weak_posteriors: Path, split: str, output: Path) -> None:
    load_and_predict(checkpoint, dataset, weak_posteriors, split, output)
