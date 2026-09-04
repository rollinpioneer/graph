#!/usr/bin/env python3
"""Verify process, CUDA, checkpoint, prediction, and finite-value evidence for all jobs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.stage8.common import dump_json, read_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-table", type=Path, required=True)
    parser.add_argument("--status-table", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    jobs = read_csv(args.job_table, delimiter="\t")
    statuses = {row["job_id"]: row for row in read_csv(args.status_table, delimiter="\t")}
    manifest = {row["sha256"] for row in read_csv(args.checkpoint_manifest, delimiter="\t")}
    checks = []
    for job in jobs:
        status = statuses.get(job["job_id"], {})
        metrics_path = Path(job["metrics_path"])
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        prediction = Path(job["prediction_path"])
        ok = (
            status.get("exit_code") == "0" and status.get("status") == "PASS"
            and metrics.get("status") == "PASS" and metrics.get("cuda_used") is True
            and metrics.get("loaded_checkpoint_sha256") == job["checkpoint_sha256"]
            and metrics.get("loaded_checkpoint_sha256") in manifest
            and int(metrics.get("prediction_count", 0)) > 0 and prediction.is_file()
            and metrics.get("all_prediction_values_finite") is True
        )
        checks.append({"job_id": job["job_id"], "suite": job["suite"], "cuda_used": metrics.get("cuda_used"), "prediction_count": metrics.get("prediction_count", 0), "checkpoint_sha_match": metrics.get("loaded_checkpoint_sha256") == job["checkpoint_sha256"], "status": "PASS" if ok else "FAIL"})
    passed = sum(row["status"] == "PASS" for row in checks)
    decision = "CORE_INFERENCE_JOBS_VERIFIED" if len(checks) == args.expected_count and passed == args.expected_count else "REPAIR_INFERENCE_JOB"
    dump_json(args.output, {"decision": decision, "expected_count": args.expected_count, "jobs_verified": passed, "checks": checks})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(f"# Core Inference Summary\n\nDecision: `{decision}`. Jobs verified: `{passed}/{args.expected_count}`.\n", encoding="utf-8")
    if decision != "CORE_INFERENCE_JOBS_VERIFIED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
