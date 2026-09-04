#!/usr/bin/env python3
"""List required but intentionally omitted Stage 8 inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.stage8.common import sha256, write_csv


def artifact(path: Path, artifact_type: str, required_for: str) -> dict:
    return {
        "path": str(path.resolve()),
        "artifact_type": artifact_type,
        "size_bytes": path.stat().st_size if path.is_file() else "",
        "sha256": sha256(path) if path.is_file() else "",
        "required_for": required_for,
        "packaged": False,
        "reason_omitted": "checkpoint_or_raw_prediction_not_in_lightweight_delivery",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--supervision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for line in args.checkpoint_manifest.read_text(encoding="utf-8").splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) > 1:
            rows.append(artifact(Path(fields[1]), "reward_model_checkpoint", "checkpoint_reproduction"))
    for name in ("ensemble_val_predictions.jsonl.gz", "ensemble_test_predictions.jsonl.gz", "ensemble_stage3_diagnostic_predictions.jsonl.gz"):
        rows.append(artifact(args.reference_predictions / "tables" / name, "frozen_reference_prediction", "reference_only_comparison"))
    rows.extend([
        artifact(args.diagnostic / "tables" / "diagnostic_episodes.csv", "stage3_diagnostic_raw_suite", "diagnostic_selection"),
        artifact(args.supervision / "tables" / "sample_index.csv.gz", "supervision_index", "checkpoint_reproduction"),
    ])
    write_csv(args.output, rows, delimiter="\t")


if __name__ == "__main__":
    main()
