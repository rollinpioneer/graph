#!/usr/bin/env python3
"""Turn fixed bootstrap estimands into a transparent Holm-adjusted hypothesis table."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--estimands", type=Path, required=True)
    parser.add_argument("--correction", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.correction != "holm":
        raise SystemExit("frozen correction is holm")
    bootstrap = pd.read_csv(args.bootstrap)
    config = yaml.safe_load(args.estimands.read_text(encoding="utf-8"))
    direction = {item["id"]: item.get("direction", "higher_is_better") for item in config["primary_estimands"]}
    direction.update({item["id"]: "higher_is_better" for item in config["primary_structural_contrasts"]})
    direction["remove_debt_cap"] = "lower_is_better"
    rows = []
    eligible = []
    for _, row in bootstrap.iterrows():
        ident = row["estimand_id"]
        limited = int(row["content_groups"]) < 4
        tail = float(row["tail_probability"])
        raw = None if limited else min(1.0, 2.0 * tail)
        eligible.append((ident, raw)) if raw is not None else None
        low, high, point = float(row["ci95_low"]), float(row["ci95_high"]), float(row["point_estimate"])
        d = direction[ident]
        if ident == "path_consistency":
            supported = high <= 0.05
        else:
            supported = low > 0 if d == "higher_is_better" else high < 0
        rows.append({"estimand_id": ident, "point_estimate": point, "ci95_low": low, "ci95_high": high, "raw_p_or_tail_probability": "NA_limited_support" if raw is None else raw, "adjusted_p": "NA_limited_support" if raw is None else "pending_holm", "effect_direction": d, "support_status": "supported" if supported else "not_supported_or_inconclusive", "content_groups": int(row["content_groups"]), "statistics_unit": "content_group_id", "support_note": row["support_note"]})
    pvalues = sorted(eligible, key=lambda pair: pair[1])
    running = 0.0
    adjusted: dict[str, float] = {}
    total = len(pvalues)
    for rank, (ident, value) in enumerate(pvalues, start=1):
        running = max(running, min(1.0, value * (total - rank + 1)))
        adjusted[ident] = running
    for row in rows:
        if row["estimand_id"] in adjusted:
            row["adjusted_p"] = adjusted[row["estimand_id"]]
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("# Primary Hypothesis Results\n\nHolm adjustment is applied only where at least four independent content groups support a stable tail calculation. Controlled immutable traces keep their CI and are explicitly marked `NA_limited_support` for adjusted p-values.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
