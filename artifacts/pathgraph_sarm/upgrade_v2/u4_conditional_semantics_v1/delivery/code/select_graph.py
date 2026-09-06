"""Select a conditional graph using frozen development evidence only."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .evaluator_v2 import graph_metrics
from .io import read_json, read_jsonl, write_csv, write_json


def _macro(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(x[field]) for x in rows if x.get(field) not in {None, ""}]
    return sum(values) / len(values) if values else None


def evaluate_graphs(graph_paths: list[Path], occurrences: Path, output: Path, table: Path) -> dict[str, Any]:
    rows = read_jsonl(occurrences)
    all_metrics = []
    for path in graph_paths:
        metric = graph_metrics(read_json(path), rows)
        family = metric["family_rows"]
        all_metrics.append({"graph_id": read_json(path).get("graph_id", path.stem), "path": str(path), "transition_coverage_macro": _macro(family, "transition_coverage"), "typed_occurrence_coverage_macro": _macro(family, "typed_occurrence_coverage"), "failure_event_recall_macro": _macro(family, "failure_event_recall"), "false_terminal_claim_rate_macro": _macro(family, "false_terminal_claim_rate"), "eligible_family_count": len(family), "horizon_excluded_from_failure": True})
    write_csv(table, all_metrics)
    write_json(output, {"schema": "u4r1_graph_metrics_v1", "split": "dev_fit", "graphs": all_metrics, "label_source": "evaluator_semantics"})
    return {"status": "PASS", "graphs": all_metrics}


def select(metrics_path: Path, output: Path, report: Path | None = None) -> dict[str, Any]:
    data = read_json(metrics_path)
    rows = data.get("graphs", [])
    by_id = {str(x["graph_id"]): x for x in rows}
    chosen = "G2_conditional_multigraph" if "G2_conditional_multigraph" in by_id else "G1_semantic_only"
    if chosen in by_id and by_id[chosen].get("typed_occurrence_coverage_macro") is not None:
        g1 = by_id.get("G1_semantic_only", {})
        g2 = by_id[chosen]
        if g1.get("typed_occurrence_coverage_macro") is not None and g2["typed_occurrence_coverage_macro"] < g1["typed_occurrence_coverage_macro"]:
            chosen = "G1_semantic_only"
    result = {"schema": "u4r1_graph_selection_v1", "selected_graph": chosen, "selection_split": "dev_fit", "confirmation_locked_after_selection": True, "metric_version": "u4r1_evaluator_v2", "reason": "conditional guarded edges retained only when observable evaluator condition is satisfiable"}
    write_json(output, result)
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(f"# Conditional graph selection\n\n- selected: `{chosen}`\n- selection split: `dev_fit`\n- evaluator semantics: `evaluator_semantics`\n", encoding="utf-8")
    return result
