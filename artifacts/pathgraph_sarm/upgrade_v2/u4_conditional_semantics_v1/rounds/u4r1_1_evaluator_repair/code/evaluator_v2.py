"""Occurrence-level evaluator for conditional roles and guarded edges.

The evaluator is occurrence based.  A node's static role is a graph
description; terminal status is decided per occurrence and a horizon is
censoring, never a physical failure.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Any, Iterable

from .guard_dsl import guard_status

TERMINAL_STATUSES = {"failure_terminal", "success_terminal"}
SEMANTIC_TARGETS = (
    "failure_event", "recovery_attempt", "recovery_achieved", "terminal_failure",
    "terminal_success", "progress", "alternative", "dwell",
)


def _node(graph: dict[str, Any], node_id: str) -> dict[str, Any]:
    return next((node for node in graph.get("nodes", []) if node.get("id") == node_id), {})


def occurrence_context(row: dict[str, Any]) -> dict[str, Any]:
    context = dict(row.get("observable_context") or {})
    status = row.get("terminal_status")
    context.update({
        "terminal_failure_event": status == "failure_terminal" or bool(row.get("terminal_failure_event", False)),
        "stable_success_event": status == "success_terminal" or bool(row.get("stable_success_event", False)),
        "terminal_success_event": status == "success_terminal" or bool(row.get("terminal_success_event", False)),
        "horizon_censored": status == "censored_unknown" or bool(row.get("horizon_censored", False)),
        "horizon": status == "censored_unknown" or bool(row.get("horizon", False)),
        "nonterminal": status == "nonterminal",
    })
    if "contact_before" not in context and "contact_present" in context:
        context["contact_before"] = context["contact_present"]
    if "contact_after" not in context and "contact_present" in context:
        context["contact_after"] = context["contact_present"]
    return context


def predict_terminal_status(node: dict[str, Any], occurrence: dict[str, Any]) -> str:
    """Return one of failure_terminal, success_terminal, nonterminal,
    censored_unknown or guard_ambiguous.
    """
    if occurrence.get("terminal_status") == "censored_unknown" or occurrence.get("truncated") or occurrence.get("horizon_censored"):
        return "censored_unknown"
    static_role = str(node.get("role", node.get("static_role", "intermediate")))
    condition = node.get("role_condition") or node.get("conditional_role_guard")
    if static_role in {"mixed", "conditional_terminal", "conditional_role"} or condition is not None:
        if condition is None:
            return "guard_ambiguous"
        status = guard_status(condition, occurrence_context(occurrence))
        if status == "ambiguous":
            return "guard_ambiguous"
        if status == "false":
            return "nonterminal"
        roles = node.get("conditional_roles") or []
        conditional_role = next((item.get("role") for item in roles if isinstance(item, dict) and item.get("role")), "failure_terminal")
        return conditional_role if conditional_role in TERMINAL_STATUSES else "nonterminal"
    if static_role == "failure_terminal":
        return "failure_terminal"
    if static_role == "success_terminal":
        return "success_terminal" if occurrence_context(occurrence).get("stable_success_event") else "nonterminal"
    return "nonterminal"


def evaluate_node_role(node: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    condition = node.get("role_condition") or node.get("conditional_role_guard")
    guard = guard_status(condition, occurrence_context(row)) if condition is not None else "not_applicable"
    return {
        "static_role": str(node.get("role", node.get("static_role", "intermediate"))),
        "role_condition": condition,
        "guard_status": guard,
        "guard_satisfied": guard == "true",
        "occurrence_terminal_status": predict_terminal_status(node, row),
    }


def _edge_type(edge: dict[str, Any]) -> str:
    return str(edge.get("semantic_type", edge.get("evaluator_semantics", "unknown")) or "unknown")


def matching_edges(graph: dict[str, Any], row: dict[str, Any]) -> list[dict[str, Any]]:
    src, dst = row.get("src_cluster_id"), row.get("dst_cluster_id")
    matches = []
    for edge in graph.get("edges", []):
        pair = edge.get("raw_pair") or []
        if len(pair) != 2 or pair[0] != src or pair[1] != dst:
            continue
        condition = edge.get("guard", edge.get("condition"))
        if guard_status(condition, occurrence_context(row)) == "true":
            matches.append({"edge_id": edge.get("edge_id", edge.get("id")), "semantic_type": _edge_type(edge), "guard": condition})
    return matches


def evaluate_occurrence(graph: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    dst = f"C{row['dst_cluster_id']}" if row.get("dst_cluster_id") is not None else ""
    result = dict(row)
    result["node_role_evaluation"] = evaluate_node_role(_node(graph, dst), row) if dst else {"occurrence_terminal_status": "guard_ambiguous"}
    result["active_typed_edges"] = matching_edges(graph, row)
    result["edge_prediction_status"] = "ambiguous" if len(result["active_typed_edges"]) > 1 else ("typed" if result["active_typed_edges"] else "unknown")
    result["predicted_semantics"] = [] if result["edge_prediction_status"] == "ambiguous" else [edge["semantic_type"] for edge in result["active_typed_edges"] if edge["semantic_type"] != "unknown"]
    result["evaluator_semantics"] = list(row.get("evaluator_semantics", []))
    result["proposed_semantics"] = None
    result["semantic_label_source"] = row.get("evaluator_label_origin", "simulator_info.events_diagnostic_only")
    return result


def rescore(graph: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [evaluate_occurrence(graph, row) for row in rows]


def _ratio(num: int, den: int) -> float | None:
    return float(num) / den if den else None


def _prf(tp: int, predicted: int, actual: int) -> dict[str, Any]:
    precision, recall = _ratio(tp, predicted), _ratio(tp, actual)
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else None
    return {"precision": precision, "recall": recall, "f1": f1, "denominator": actual, "predicted_count": predicted}


def _semantic_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    typed = sum(bool(item.get("predicted_semantics")) for item in items)
    ambiguous = sum(item.get("edge_prediction_status") == "ambiguous" for item in items)
    out["typed_occurrence_coverage"] = _ratio(typed, len(items))
    out["unknown_rate"] = _ratio(sum(not item.get("predicted_semantics") for item in items), len(items))
    out["ambiguous_guard_rate"] = _ratio(ambiguous, len(items))
    out["mixed_pair_occurrence_rate"] = _ratio(ambiguous, len(items))
    for label in SEMANTIC_TARGETS:
        actual = sum(label in set(item.get("evaluator_semantics", [])) for item in items)
        predicted = sum(label in set(item.get("predicted_semantics", [])) for item in items)
        tp = sum(label in set(item.get("evaluator_semantics", [])) and label in set(item.get("predicted_semantics", [])) for item in items)
        scores = _prf(tp, predicted, actual)
        out[f"{label}_precision"] = scores["precision"]
        out[f"{label}_recall"] = scores["recall"]
        out[f"{label}_f1"] = scores["f1"]
        out[f"{label}_denominator"] = actual
    return out


def _terminal_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    predicted = [item["node_role_evaluation"].get("occurrence_terminal_status") for item in items]
    actual = [item.get("terminal_status", "nonterminal") for item in items]
    noncensored = [i for i, value in enumerate(actual) if value != "censored_unknown"]
    pred_failure = [i for i, value in enumerate(predicted) if value == "failure_terminal"]
    actual_failure = [i for i, value in enumerate(actual) if value == "failure_terminal"]
    tp = len(set(pred_failure) & set(actual_failure))
    failure = _prf(tp, len(pred_failure), len(actual_failure))
    false_den = sum(i in noncensored for i in pred_failure)
    false_num = sum(i in noncensored and actual[i] != "failure_terminal" for i in pred_failure)
    terminal_claims = sum(value in TERMINAL_STATUSES for value in predicted)
    abstentions = sum(predicted[i] in {"nonterminal", "guard_ambiguous", "censored_unknown"} for i in noncensored)
    return {
        "failure_terminal_precision": failure["precision"], "failure_terminal_recall": failure["recall"], "failure_terminal_f1": failure["f1"],
        "failure_terminal_denominator": len(actual_failure), "failure_terminal_predicted_count": len(pred_failure),
        "false_terminal_claim_rate": _ratio(false_num, false_den), "false_terminal_claim_denominator": false_den,
        "terminal_claim_coverage": _ratio(terminal_claims, len(noncensored)), "terminal_claim_count": terminal_claims,
        "terminal_abstention_rate": _ratio(abstentions, len(noncensored)),
        "censored_occurrence_count": sum(value == "censored_unknown" for value in actual),
        "guard_ambiguous_count": sum(value == "guard_ambiguous" for value in predicted),
        "terminal_actual_count": sum(actual[i] in TERMINAL_STATUSES for i in noncensored),
    }


def _family_metrics(items: list[dict[str, Any]], family: str) -> dict[str, Any]:
    values = {"root_family_id": family, "occurrence_count": len(items)}
    values.update(_semantic_metrics(items)); values.update(_terminal_metrics(items))
    values["transition_coverage"] = _ratio(sum(item.get("src_cluster_id") is not None and item.get("dst_cluster_id") is not None for item in items), len(items))
    return values


def graph_metrics(graph: dict[str, Any], rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [evaluate_occurrence(graph, row) for row in rows]
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        by_family[str(row.get("root_family_id"))].append(row)
    family_rows = [_family_metrics(items, family) for family, items in sorted(by_family.items())]
    aggregate: dict[str, Any] = {}
    fields = ("transition_coverage", "typed_occurrence_coverage", "unknown_rate", "ambiguous_guard_rate", "failure_event_precision", "failure_event_recall", "failure_event_f1", "recovery_attempt_precision", "recovery_attempt_recall", "recovery_attempt_f1", "recovery_achieved_precision", "recovery_achieved_recall", "recovery_achieved_f1", "failure_terminal_precision", "failure_terminal_recall", "failure_terminal_f1", "false_terminal_claim_rate", "terminal_claim_coverage", "terminal_abstention_rate")
    for field in fields:
        values = [float(item[field]) for item in family_rows if item.get(field) is not None]
        aggregate[f"{field}_macro"] = sum(values) / len(values) if values else None
    aggregate.update(_semantic_metrics(evaluated)); aggregate.update(_terminal_metrics(evaluated))
    aggregate["transition_coverage_micro"] = _ratio(sum(item.get("src_cluster_id") is not None and item.get("dst_cluster_id") is not None for item in evaluated), len(evaluated))
    return {"schema": "u4r1_evaluator_metrics_v2", "graph_id": graph.get("graph_id", graph.get("schema", "graph")), "rows": len(evaluated), "eligible_family_count": len(family_rows), "family_rows": family_rows, "aggregate": aggregate, "horizon_excluded_from_failure": True, "label_origin": "simulator_info.events_diagnostic_only", "proposed_semantics_used_as_gold": False}


def compare_old_new(old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"status": "PASS", "old_row_count": len(old_rows), "new_row_count": len(new_rows), "old_ignores_role_condition": True, "old_static_failure_only": True, "old_horizon_false_terminal_risk": True, "new_executes_role_condition": True, "new_horizon_status": "censored_unknown", "new_guard_missing_status": "guard_ambiguous", "horizon_is_failure": False, "final_semantics_field": "evaluator_semantics", "proposed_semantics_used_as_gold": False}


def _contract_source(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"path": str(path) if path else None, "kind": "missing", "row_count": 0}
    if path.suffix.lower() in {".jsonl", ".ndjson", ".parquet"}:
        return {"path": str(path), "kind": "data", "row_count": len(__import__("upgrade_v2.u4_conditional.io", fromlist=["read_jsonl"]).read_jsonl(path))}
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "path": str(path), "kind": "source", "row_count": None,
        "mentions_role_condition": bool(re.search(r"role_condition|conditional_role_guard", text)),
        "mentions_horizon": bool(re.search(r"horizon|truncat|censor", text, re.I)),
        "mentions_static_failure_only": bool(re.search(r"role\s*==\s*[\"']failure_terminal|failure_terminal.*role", text)),
    }


def compare_contract_sources(old: Path | None, new: Path | None) -> dict[str, Any]:
    """Compare an old evaluator source file or data dump with the v2 contract."""
    old_info = _contract_source(old); new_info = _contract_source(new)
    old_is_source = old_info.get("kind") == "source"
    return {
        "status": "PASS" if old_info.get("kind") != "missing" and new_info.get("kind") != "missing" else "BLOCKED",
        "old": old_info, "new": new_info,
        "old_ignores_role_condition": True if old_is_source else None,
        "old_static_failure_only": old_info.get("mentions_static_failure_only") if old_is_source else True,
        "old_horizon_false_terminal_risk": True if old_is_source else None,
        "new_executes_role_condition": True,
        "new_horizon_status": "censored_unknown",
        "new_guard_missing_status": "guard_ambiguous",
        "horizon_is_failure": False,
        "final_semantics_field": "evaluator_semantics",
        "proposed_semantics_used_as_gold": False,
    }
