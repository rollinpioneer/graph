#!/usr/bin/env python3
"""Project formal TRAK runtime/storage from the frozen one-batch smoke."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--rollout-summary", type=Path, required=True)
    parser.add_argument("--disk-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    smoke = json.loads(args.smoke.read_text(encoding="utf-8"))
    rollout = json.loads(args.rollout_summary.read_text(encoding="utf-8"))

    source_samples = 125_324 + 60_618
    targets = int(rollout["decision_point_count"])
    batch = int(smoke["batch_size"])
    proj_dim = int(smoke["proj_dim"])
    source_batches = (source_samples + batch - 1) // batch
    target_batches = (targets + batch - 1) // batch

    featurize_seconds = float(smoke["featurize_one_batch_seconds"]) * source_batches
    score_seconds = float(smoke["score_one_batch_seconds"]) * target_batches
    fixed_seconds = float(smoke["finalize_features_seconds"])
    # Final score multiplication scales approximately with matrix elements.
    smoke_elements = int(smoke["source_samples"]) * int(smoke["target_samples"])
    formal_elements = source_samples * targets
    finalize_score_seconds = float(smoke["finalize_scores_seconds"]) * (
        formal_elements / smoke_elements
    )
    estimated_seconds = featurize_seconds + score_seconds + fixed_seconds + finalize_score_seconds

    float_bytes = 4
    feature_and_grad_bytes = 2 * source_samples * proj_dim * float_bytes
    target_grad_bytes = targets * proj_dim * float_bytes
    raw_score_bytes = formal_elements * float_bytes
    projected_bytes = int(1.5 * (
        feature_and_grad_bytes + target_grad_bytes + raw_score_bytes
    ))
    stat = os.statvfs(args.disk_root)
    available_bytes = stat.f_bavail * stat.f_frsize
    peak_mib = float(smoke["peak_reserved_memory_mib"])
    passed = (
        smoke.get("status") == "PASS"
        and peak_mib < 24_000
        and projected_bytes < 0.5 * available_bytes
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "time_limit_policy": "record_only_user_authorized_no_wall_time_cap",
        "source_samples_train_plus_holdout": source_samples,
        "target_decision_points": targets,
        "source_batches": source_batches,
        "target_batches": target_batches,
        "estimated_wall_time_seconds": round(estimated_seconds),
        "estimated_wall_time_hours": round(estimated_seconds / 3600, 2),
        "smoke_peak_reserved_memory_mib": peak_mib,
        "projected_storage_bytes_with_1_5x_margin": projected_bytes,
        "projected_storage_gib_with_1_5x_margin": round(projected_bytes / 1024**3, 2),
        "available_disk_bytes": available_bytes,
        "available_disk_gib": round(available_bytes / 1024**3, 2),
        "disk_fraction_projected": projected_bytes / available_bytes,
        "gates": {
            "smoke_pass": smoke.get("status") == "PASS",
            "peak_reserved_below_24000_mib": peak_mib < 24_000,
            "projected_storage_below_half_available": projected_bytes < 0.5 * available_bytes,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
