#!/usr/bin/env python3
"""Build immutable 3 checkpoints x 3 suite inference jobs."""
from __future__ import annotations

import argparse
import csv
import shlex
from pathlib import Path

from tools.stage8.common import read_csv, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--suites", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--supervision", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--job-table", type=Path, required=True)
    parser.add_argument("--commands-dir", type=Path, required=True)
    args = parser.parse_args()
    suites = args.suites.split(",")
    if set(suites) != {"val", "test", "stage3_diagnostic"}:
        raise ValueError("Stage 8 requires exactly val,test,stage3_diagnostic")
    checkpoints = read_csv(args.checkpoint_manifest, delimiter="\t")
    rows = []
    args.commands_dir.mkdir(parents=True, exist_ok=True)
    for checkpoint in checkpoints:
        for suite in suites:
            job_id = f"s{checkpoint['model_seed']}__{suite}"
            job_dir = args.output_root / job_id
            prediction = args.prediction_root / f"{job_id}.jsonl.gz"
            metrics = job_dir / "metrics.json"
            command_parts = [
                args.python_bin, str(args.runner.resolve()), "--checkpoint", checkpoint["checkpoint_path"],
                "--checkpoint-sha256", checkpoint["sha256"], "--seed", checkpoint["model_seed"],
                "--suite", suite, "--supervision-root", str(args.supervision.resolve()),
                "--diagnostic-root", str(args.diagnostic.resolve()), "--inference-protocol", str(args.protocol.resolve()),
                "--output", str(prediction.resolve()), "--metrics-output", str(metrics.resolve()), "--device", "cuda:0",
            ]
            command = " ".join(shlex.quote(part) for part in command_parts)
            (args.commands_dir / f"{job_id}.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n", encoding="utf-8")
            rows.append({
                "job_id": job_id, "model_seed": checkpoint["model_seed"], "suite": suite,
                "checkpoint_path": checkpoint["checkpoint_path"], "checkpoint_sha256": checkpoint["sha256"],
                "output_dir": str(job_dir.resolve()), "prediction_path": str(prediction.resolve()),
                "metrics_path": str(metrics.resolve()), "command": command,
            })
    write_csv(args.job_table, rows, delimiter="\t")
    if len(rows) != 9:
        raise SystemExit("expected exactly nine core reproduction jobs")


if __name__ == "__main__":
    main()
