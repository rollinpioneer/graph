#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from tools.stage8.common import sha256, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--ensemble-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(list(args.prediction_root.glob("*.jsonl.gz")) + list(args.ensemble_root.glob("*.jsonl.gz"))):
        rows.append({"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path), "artifact_type": "stage8_prediction", "reason_omitted": "per_sample_prediction_omitted_from_lightweight_round_delivery", "required_for_full_recompute": True})
    for line in args.checkpoint_manifest.read_text(encoding="utf-8").splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) > 3:
            rows.append({"path": fields[1], "size_bytes": fields[2], "sha256": fields[3], "artifact_type": "checkpoint", "reason_omitted": "checkpoint_omitted_from_lightweight_round_delivery", "required_for_full_recompute": True})
    write_csv(args.output, rows, delimiter="\t")


if __name__ == "__main__":
    main()
