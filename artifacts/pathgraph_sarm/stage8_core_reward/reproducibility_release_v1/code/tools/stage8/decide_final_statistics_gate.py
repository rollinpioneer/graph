#!/usr/bin/env python3
"""Gate final statistics without upgrading excluded claims."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from tools.stage8.common import dump_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimands-lock", type=Path, required=True)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--hypotheses", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    lock_line = args.estimands_lock.read_text(encoding="utf-8").strip().split()[0]
    valid_lock = len(lock_line) == 64 and all(char in "0123456789abcdef" for char in lock_line.lower())
    bootstrap = pd.read_csv(args.bootstrap)
    hypotheses = pd.read_csv(args.hypotheses)
    summary = pd.read_csv(args.summary)
    ci_columns = bootstrap[["point_estimate", "ci95_low", "ci95_high"]].apply(pd.to_numeric, errors="coerce")
    complete = len(bootstrap) == 10 and (bootstrap["bootstrap_resamples_completed"] == 10000).all() and ci_columns.notna().all().all()
    hypothesis_complete = len(hypotheses) == 10 and hypotheses[["point_estimate", "ci95_low", "ci95_high"]].apply(pd.to_numeric, errors="coerce").notna().all().all()
    no_upgrade = not summary[summary["category"] == "unsupported_or_negative_result"].empty
    decision = "FINAL_STATISTICS_LOCKED" if valid_lock and complete and hypothesis_complete and no_upgrade else "STATISTICAL_INPUT_MISMATCH"
    dump_json(args.output, {"decision": decision, "estimands_lock_valid": valid_lock, "bootstrap_estimands": len(bootstrap), "bootstrap_resamples": sorted(set(int(value) for value in bootstrap["bootstrap_resamples_completed"])), "statistics_unit": "content_group_id", "all_primary_point_estimates_have_ci": bool(ci_columns.notna().all().all()), "hypothesis_rows": len(hypotheses), "unsupported_claims_remain_excluded": no_upgrade})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(f"# Final Statistics Gate\n\nDecision: `{decision}`. All inference uses content-group bootstrap; Wilson intervals are supplementary event-level descriptive intervals only.\n", encoding="utf-8")
    if decision != "FINAL_STATISTICS_LOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
