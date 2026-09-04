#!/usr/bin/env python3
"""Create the final claim registry with explicit claim-boundary safeguards."""
from __future__ import annotations

import argparse
from pathlib import Path

from tools.stage8.common import read_csv, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim-matrix", type=Path, required=True)
    parser.add_argument("--input-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    claims = read_csv(args.claim_matrix)
    source_by_id = {
        "C1": ("core_ablation_effects.csv", "alternative_structural_support", "primary"),
        "C2": ("core_ablation_effects.csv", "recovery_structural_support", "primary"),
        "C3": ("history_granularity_summary.csv", "history32_vs_history1_node_f1_gain", "qualified"),
        "C4": ("policy_secondary_evidence.csv", "strict_seed_consistency", "negative"),
        "C5": ("auto_graph_test_metrics.csv", "cross_seed_node_ari", "negative"),
        "C6": ("uncertainty_error_detection.csv", "reward_error_AUROC", "qualified"),
        "C7": ("coverage_scaling_metrics.csv", "real_coverage_node_macro_f1", "negative"),
        "C8": ("ood_reward_metrics.csv", "unseen_order_alternative_edge_f1/path_gap", "negative"),
    }
    rows = []
    for claim in claims:
        source_table, metric, category = source_by_id.get(claim["claim_id"], ("", "", "negative"))
        status = claim.get("support_status", "not_supported")
        allowed = status == "supported" and category == "primary"
        rows.append({
            "claim_id": claim["claim_id"],
            "claim_text": claim["claim_text"],
            "final_status": status,
            "priority": claim.get("priority", "secondary"),
            "source_table": source_table,
            "source_metric": metric,
            "source_artifact": str(args.input_index.resolve()),
            "provenance": claim.get("evidence_source", ""),
            "allowed_in_abstract": allowed,
            "allowed_in_main_results": allowed or (status == "partially_supported" and category == "qualified"),
            "required_qualifier": "" if allowed else ("current benchmark only" if status == "partially_supported" else "limitations or negative result only"),
        })
    write_csv(args.output, rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# Final Evidence Scope\n\n"
        "Only supported primary claims may appear as abstract conclusions. History and uncertainty are qualified. "
        "Policy stability, automatic graph as the main method, coverage scaling, and unseen-order generalization remain negative or unsupported results.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
