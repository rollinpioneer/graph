#!/usr/bin/env python3
"""Assemble a claim-bounded final statistical summary from final CSV inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--rates", type=Path, required=True)
    parser.add_argument("--hypotheses", type=Path, required=True)
    parser.add_argument("--seed-stability", type=Path, required=True)
    parser.add_argument("--claim-scope", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    claims = json.loads(args.claim_scope.read_text(encoding="utf-8"))
    hypotheses = pd.read_csv(args.hypotheses)
    bootstrap = pd.read_csv(args.bootstrap)
    main = pd.read_csv(args.main)
    locked = main[main["method"] == "pathgraph_reward_v1_locked"].iloc[0]
    rows = []
    for _, item in hypotheses.iterrows():
        category = "supported_primary_result" if item["support_status"] == "supported" else "qualified_or_inconclusive_primary_result"
        rows.append({"category": category, "result_id": item["estimand_id"], "point_estimate": item["point_estimate"], "ci95_low": item["ci95_low"], "ci95_high": item["ci95_high"], "support_status": item["support_status"], "provenance": item["support_note"], "statistics_unit": "content_group_id"})
    for metric in ("node_macro_f1", "edge_type_macro_f1_non_none", "phi_mae", "cost_mae"):
        rows.append({"category": "qualified_auxiliary_result", "result_id": metric, "point_estimate": locked[metric], "ci95_low": "", "ci95_high": "", "support_status": "descriptive_checkpoint_reproduction", "provenance": "stage8_reproduced_reward_main_table", "statistics_unit": "content_group_id"})
    for item in claims.get("removed_or_unsupported", claims.get("removed_or_downgraded_claims", claims.get("unsupported_claims", []))):
        label = item.get("id", item) if isinstance(item, dict) else item
        rows.append({"category": "unsupported_or_negative_result", "result_id": label, "point_estimate": "", "ci95_low": "", "ci95_high": "", "support_status": "excluded_by_final_claim_scope", "provenance": "final_claim_scope_lock", "statistics_unit": "content_group_id"})
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("# Final Statistical Summary\n\n## Supported primary results\n\n" + "\n".join(f"- {row['result_id']}" for _, row in output[output.category == "supported_primary_result"].iterrows()) + "\n\n## Qualified auxiliary results\n\nCheckpoint metrics and controlled symbolic evidence retain their provenance and limited-support labels.\n\n## Unsupported / negative results\n\nCoverage scaling, unseen-order extension, stable policy improvement, and automatic graph as a main contribution remain excluded by the frozen claim scope.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
