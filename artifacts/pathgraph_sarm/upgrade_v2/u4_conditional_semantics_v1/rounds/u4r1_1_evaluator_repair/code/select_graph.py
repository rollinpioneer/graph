"""Validation-only conditional graph metrics and selection."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .evaluator_v2 import graph_metrics
from .io import read_json, read_jsonl, write_csv, write_json


def _macro(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(x[field]) for x in rows if x.get(field) not in {None, ""}]
    return sum(values) / len(values) if values else None


def _f1(metric: dict[str, Any], label: str) -> float | None:
    return metric.get("aggregate", {}).get(f"{label}_f1_macro")


def _score(item: dict[str, Any]) -> tuple[float | None, list[str]]:
    weights = (("typed_occurrence_coverage_macro", .20), ("failure_event_f1_macro", .15), ("recovery_attempt_f1_macro", .15), ("recovery_achieved_f1_macro", .15), ("failure_terminal_f1_macro", .15), ("transition_coverage_macro", .10), ("unknown_rate_macro", -.05), ("ambiguous_guard_rate_macro", -.05))
    total = 0.0; weight_sum = 0.0; missing = []
    for field, weight in weights:
        value = item.get(field)
        if value in {None, ""}:
            missing.append(field); continue
        total += weight * float(value); weight_sum += abs(weight)
    return (total / weight_sum if weight_sum else None), missing


def _graph_paths(graphs: Path | list[Path]) -> list[Path]:
    if isinstance(graphs, list): return graphs
    if graphs.is_dir():
        preferred = ("G0_raw_topology.json", "G1_single_label_v2.json", "G3_guard_rule_multigraph.json", "G4_tree_compiled_multigraph.json")
        paths = [graphs / name for name in preferred if (graphs / name).is_file()]
        return paths or sorted(path for path in graphs.glob("*.json") if path.name not in {"edit_log.json", "selection.json"})
    return [graphs]


def evaluate_graphs(graph_paths: Path | list[Path], occurrences: Path, output: Path, table: Path | None = None, split: str | None = None, *args: Any, **kwargs: Any) -> dict[str, Any]:
    rows = read_jsonl(occurrences)
    if split:
        rows = [row for row in rows if row.get("split") == split]
    all_metrics = []
    for path in _graph_paths(graph_paths):
        graph = read_json(path); metric = graph_metrics(graph, rows); aggregate = metric["aggregate"]
        item = {"graph_id": graph.get("graph_id", path.stem), "path": str(path), "split": split or "dev_route", "eligible_family_count": metric["eligible_family_count"], "occurrence_count": len(rows), "transition_coverage_macro": aggregate.get("transition_coverage_macro"), "typed_occurrence_coverage_macro": aggregate.get("typed_occurrence_coverage_macro"), "unknown_rate_macro": aggregate.get("unknown_rate_macro"), "ambiguous_guard_rate_macro": aggregate.get("ambiguous_guard_rate_macro"), "failure_event_precision_macro": aggregate.get("failure_event_precision_macro"), "failure_event_recall_macro": aggregate.get("failure_event_recall_macro"), "failure_event_f1_macro": aggregate.get("failure_event_f1_macro"), "recovery_attempt_precision_macro": aggregate.get("recovery_attempt_precision_macro"), "recovery_attempt_recall_macro": aggregate.get("recovery_attempt_recall_macro"), "recovery_attempt_f1_macro": aggregate.get("recovery_attempt_f1_macro"), "recovery_achieved_precision_macro": aggregate.get("recovery_achieved_precision_macro"), "recovery_achieved_recall_macro": aggregate.get("recovery_achieved_recall_macro"), "recovery_achieved_f1_macro": aggregate.get("recovery_achieved_f1_macro"), "failure_terminal_precision_macro": aggregate.get("failure_terminal_precision_macro"), "failure_terminal_recall_macro": aggregate.get("failure_terminal_recall_macro"), "failure_terminal_f1_macro": aggregate.get("failure_terminal_f1_macro"), "false_terminal_claim_rate_macro": aggregate.get("false_terminal_claim_rate_macro"), "terminal_claim_coverage_macro": aggregate.get("terminal_claim_coverage_macro"), "censored_occurrence_count": aggregate.get("censored_occurrence_count"), "guard_ambiguous_count": aggregate.get("guard_ambiguous_count"), "value_consistent_edge_direction": None, "graph_nodes": len(graph.get("nodes", [])), "conditional_edges": sum(bool(edge.get("guard")) for edge in graph.get("edges", [])), "label_source": "evaluator_semantics", "horizon_excluded_from_failure": True}
        item["selection_score"], item["not_estimable_terms"] = _score(item); all_metrics.append(item)
    if output.suffix.lower() == ".csv": write_csv(output, all_metrics)
    else: write_json(output, {"schema": "u4r1_graph_metrics_v2", "split": split or "dev_route", "graphs": all_metrics, "selection_only_source": split or "dev_route", "label_source": "evaluator_semantics"})
    if table: write_csv(table, all_metrics)
    return {"status": "PASS" if all_metrics else "BLOCKED", "graphs": all_metrics}


def select(metrics_path: Path, output: Path, report: Path | None = None, *args: Any, **kwargs: Any) -> dict[str, Any]:
    if metrics_path.suffix.lower() == ".csv":
        import csv
        with metrics_path.open(encoding="utf-8", newline="") as handle: rows = list(csv.DictReader(handle))
    else: rows = read_json(metrics_path).get("graphs", [])
    for row in rows:
        row["selection_score"] = float(row["selection_score"]) if row.get("selection_score") not in {None, ""} else _score(row)[0]
    eligible = [row for row in rows if row.get("selection_score") is not None and float(row.get("ambiguous_guard_rate_macro") or 0.0) <= float(kwargs.get("max_ambiguous_guard_rate", .10))]
    chosen_row = max(eligible or rows, key=lambda row: float(row.get("selection_score") or -1.0)) if rows else None
    chosen = chosen_row.get("graph_id") if chosen_row else None
    result = {"schema": "u4r1_graph_selection_v2", "selected_graph": chosen, "selection_split": "dev_route", "confirmation_locked_after_selection": bool(chosen), "metric_version": "u4r1_evaluator_metrics_v2", "selection_score": chosen_row.get("selection_score") if chosen_row else None, "not_estimable_terms": chosen_row.get("not_estimable_terms", []) if chosen_row else [], "reason": "validation-only weighted score; missing metric terms removed from normalization; transition and ambiguity constraints applied"}
    write_json(output, result)
    lock = kwargs.get("lock")
    if lock: write_json(lock, {"schema": "u4r1_selection_lock_v2", "selected_graph": chosen, "metrics_sha256": __import__("hashlib").sha256(metrics_path.read_bytes()).hexdigest(), "selection_split": "dev_route", "confirmation_locked_after_selection": bool(chosen)})
    if report:
        report.parent.mkdir(parents=True, exist_ok=True); report.write_text("# Conditional graph selection\n\n" + f"- selected: `{chosen}`\n- selection split: `dev_route`\n- score: `{result['selection_score']}`\n- metric version: `u4r1_evaluator_metrics_v2`\n", encoding="utf-8")
    return result
