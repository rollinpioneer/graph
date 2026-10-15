#!/usr/bin/env python3
"""Supplement group-bootstrap results with clearly labelled Wilson intervals."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def wilson(successes: float, total: float, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return centre - radius, centre + radius


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.method != "wilson":
        raise SystemExit("only wilson is supported")
    data = pq.read_table(args.observations).to_pandas()
    methods = ["pathgraph_reward_v1_locked"]
    output = []
    for metric in args.metrics.split(","):
        subset = data[(data["metric_name"] == metric) & (data["method"].isin(methods))].copy()
        for provenance, part in subset.groupby("provenance", sort=True):
            part = part[pd.to_numeric(part["event_denominator"], errors="coerce").notna()]
            successes = float(pd.to_numeric(part["event_numerator"], errors="coerce").sum())
            total = float(pd.to_numeric(part["event_denominator"], errors="coerce").sum())
            if total <= 0:
                continue
            low, high = wilson(successes, total)
            output.append({"metric_name": metric, "method": "pathgraph_reward_v1_locked", "provenance": provenance, "point_estimate_event_rate": successes / total, "wilson95_low": low, "wilson95_high": high, "event_successes": successes, "event_total": total, "content_groups": int(part["content_group_id"].nunique()), "statistics_unit": "content_group_id", "interpretation": "event-level descriptive Wilson interval; clustered content-group bootstrap is the primary inferential result"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
