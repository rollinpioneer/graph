#!/usr/bin/env python3
"""Launch independent Stage 8 inference jobs only on explicitly selected idle GPUs."""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from pathlib import Path

from tools.stage8.common import write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-table", type=Path, required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--logs-dir", type=Path, required=True)
    args = parser.parse_args()
    gpus = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if not gpus:
        raise ValueError("at least one privileged-visible idle GPU is required")
    with args.job_table.open(newline="", encoding="utf-8") as handle:
        jobs = list(csv.DictReader(handle, delimiter="\t"))
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    pending = jobs[:]
    active: list[tuple[subprocess.Popen, dict, str, object, float]] = []
    completed = []
    while pending or active:
        while pending and len(active) < len(gpus):
            job = pending.pop(0)
            gpu = gpus[len(active)]
            log = (args.logs_dir / f"{job['job_id']}.log").open("w", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(job["command"], shell=True, cwd="/home/__compress_data/xushijie/CUPID", env=environment, stdout=log, stderr=subprocess.STDOUT)
            active.append((process, job, gpu, log, time.time()))
        remaining = []
        for process, job, gpu, log, started in active:
            code = process.poll()
            if code is None:
                remaining.append((process, job, gpu, log, started))
                continue
            log.close()
            completed.append({**job, "gpu_id": gpu, "exit_code": code, "wall_seconds": round(time.time() - started, 3), "status": "PASS" if code == 0 else "FAIL"})
        active = remaining
        if active:
            time.sleep(0.25)
    write_csv(args.status_output, sorted(completed, key=lambda row: row["job_id"]), delimiter="\t")
    if any(row["status"] != "PASS" for row in completed):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
