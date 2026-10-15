#!/usr/bin/env python3
"""Render publication tables directly from locked final CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def fmt(value: object) -> str:
    if pd.isna(value) or value == "":
        return "NA"
    if isinstance(value, str):
        try: value = float(value)
        except ValueError: return value.replace("_", " ")
    return f"{float(value):.3f}"


def write_table(name: str, frame: pd.DataFrame, out: Path) -> None:
    frame.to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}.md").write_text(frame.to_markdown(index=False) + "\n", encoding="utf-8")
    (out / f"{name}.tex").write_text(frame.to_latex(index=False, escape=True) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    for arg in ("main_results", "model_results", "ablations", "statistics", "bootstrap", "history", "uncertainty", "policy", "coverage", "ood", "claim_matrix"):
        p.add_argument("--" + arg.replace("_", "-"), type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--source-map", type=Path, required=True)
    a = p.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    main = pd.read_csv(a.main_results); model = pd.read_csv(a.model_results); effects = pd.read_csv(a.ablations); stats = pd.read_csv(a.statistics); boot = pd.read_csv(a.bootstrap)
    ci = boot.set_index("estimand_id")
    m = main[["method", "node_macro_f1", "edge_type_macro_f1_non_none", "failure_negative_rate", "recovery_positive_rate", "success_failure_margin", "legal_path_normalized_gap"]].copy()
    m.columns = ["Method", "Node macro-F1", "Edge-type macro-F1", "Failure negative rate", "Recovery positive rate", "Success–failure margin", "Legal-path gap"]
    for col in m.columns[1:]: m[col] = m[col].map(fmt)
    write_table("table_1_main_reward_results", m, a.output_dir)
    contrast_id = {"collapse_alternative_to_A_first": "alternative_A_collapse", "collapse_alternative_to_B_first": "alternative_B_collapse", "remove_recovery_edge": "remove_recovery", "no_recovery_debt_cap": "remove_debt_cap", "no_phi": "remove_phi"}
    e = effects.copy(); e["estimand_id"] = e["variant_id"].map(contrast_id); e = e.merge(boot[["estimand_id", "point_estimate", "ci95_low", "ci95_high"]], how="left", on="estimand_id")
    e["Bootstrap estimate [95% CI]"] = e.apply(lambda r: "NA" if pd.isna(r.point_estimate) else f"{r.point_estimate:.3f} [{r.ci95_low:.3f}, {r.ci95_high:.3f}]", axis=1)
    e = e[["variant_id", "alternative_support_delta", "recovery_support_delta", "debt_cap_overshoot_delta", "within_node_density_delta", "Bootstrap estimate [95% CI]", "provenance"]]
    e.columns = ["Variant", "Alternative support Δ", "Recovery support Δ", "Debt overshoot Δ", "Within-node density Δ", "Locked contrast", "Provenance"]
    for col in e.columns[1:5]: e[col] = e[col].map(fmt)
    write_table("table_2_structural_ablations", e, a.output_dir)
    test = model[(model.suite == "test") & (model.model_seed == "ensemble")].copy()
    comp = test[["node_macro_f1", "edge_type_macro_f1_non_none", "edge_id_macro_f1", "phi_mae", "remaining_cost_mae", "cost_pair_accuracy"]].T.reset_index(); comp.columns = ["Component metric", "Value"]; comp["Value"] = comp["Value"].map(fmt)
    write_table("table_3_model_components", comp, a.output_dir)
    scope = stats[["category", "result_id", "support_status", "provenance"]].copy(); scope.columns = ["Evidence role", "Result / claim", "Status", "Provenance"]
    write_table("table_4_final_claim_scope", scope, a.output_dir)
    seed = model[(model.suite == "test") & (model.model_seed != "ensemble")][["model_seed", "node_macro_f1", "edge_type_macro_f1_non_none", "phi_mae", "remaining_cost_mae"]].copy(); seed.columns = ["Seed", "Node macro-F1", "Edge-type macro-F1", "Phi MAE", "Cost MAE"]
    for col in seed.columns[1:]: seed[col] = seed[col].map(fmt)
    write_table("table_A1_model_seed_results", seed, a.output_dir)
    for name, path in (("table_A2_history_granularity", a.history), ("table_A3_uncertainty", a.uncertainty), ("table_A4_policy_secondary_mixed", a.policy)):
        write_table(name, pd.read_csv(path), a.output_dir)
    negative = pd.concat([pd.read_csv(a.coverage).assign(evidence_group="coverage_negative_extension"), pd.read_csv(a.ood).assign(evidence_group="unseen_order_negative_extension")], ignore_index=True, sort=False).fillna("Not estimable")
    write_table("table_A5_negative_extensions", negative, a.output_dir)
    sources = []
    input_map = {"table_1_main_reward_results": [a.main_results, a.bootstrap], "table_2_structural_ablations": [a.ablations, a.bootstrap], "table_3_model_components": [a.model_results], "table_4_final_claim_scope": [a.statistics], "table_A1_model_seed_results": [a.model_results], "table_A2_history_granularity": [a.history], "table_A3_uncertainty": [a.uncertainty], "table_A4_policy_secondary_mixed": [a.policy], "table_A5_negative_extensions": [a.coverage, a.ood]}
    for artifact, inputs in input_map.items():
        for source in inputs: sources.append({"artifact_id": artifact, "artifact_type": "table", "source_path": str(source.resolve()), "source_kind": "final_csv", "statistics_unit_or_scope": "content_group_id_or_explicit_auxiliary_provenance", "unsupported_claim_in_main": False})
    pd.DataFrame(sources).to_csv(a.source_map, index=False)


if __name__ == "__main__": main()
