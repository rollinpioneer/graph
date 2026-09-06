"""Occurrence-level evaluator for conditional roles and guarded typed edges."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .guard_dsl import evaluate_guard


def _node(graph: dict[str, Any], node_id: str) -> dict[str, Any]:
    return next((node for node in graph.get("nodes", []) if node.get("id") == node_id), {})


def occurrence_context(row: dict[str, Any]) -> dict[str, Any]:
    context = dict(row.get("observable_context") or {})
    context.update({
        "terminal_failure_event": row.get("terminal_status") == "failure_terminal",
        "terminal_success_event": row.get("terminal_status") == "success_terminal",
        "horizon": row.get("terminal_status") == "censored_unknown",
        "nonterminal": row.get("terminal_status") == "nonterminal",
        "evaluator_semantics": row.get("evaluator_semantics", []),
    })
    return context


def evaluate_node_role(node: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    static_role = str(node.get("role", "intermediate"))
    guard = node.get("role_condition")
    guarded = evaluate_guard(guard, occurrence_context(row)) if guard else True
    terminal = guarded if guard else static_role in {"failure_terminal", "success_terminal"}
    status = row.get("terminal_status", "nonterminal")
    # A horizon may satisfy an explicit guard for routing, but remains censored.
    if status == "censored_unknown":
        decision = "censored_unknown"
    elif terminal and status == "failure_terminal":
        decision = "failure_terminal"
    elif terminal and status == "success_terminal":
        decision = "success_terminal"
    elif terminal:
        decision = "terminal_role_without_terminal_occurrence"
    else:
        decision = "nonterminal"
    return {"static_role": static_role, "role_condition": guard, "guard_satisfied": guarded, "occurrence_terminal_status": decision}


def matching_edges(graph: dict[str, Any], row: dict[str, Any]) -> list[dict[str, Any]]:
    src, dst = row.get("src_cluster_id"), row.get("dst_cluster_id")
    matches = []
    for edge in graph.get("edges", []):
        pair = edge.get("raw_pair") or []
        if len(pair) != 2 or pair[0] != src or pair[1] != dst:
            continue
        guard = edge.get("guard", edge.get("condition"))
        if evaluate_guard(guard, occurrence_context(row)):
            matches.append({"edge_id": edge.get("id"), "semantic_type": edge.get("semantic_type", edge.get("evaluator_semantics", "unknown")), "guard": guard})
    return matches


def evaluate_occurrence(graph: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    src = f"C{row['src_cluster_id']}" if row.get("src_cluster_id") is not None else ""
    dst = f"C{row['dst_cluster_id']}" if row.get("dst_cluster_id") is not None else ""
    result = dict(row)
    result["node_role_evaluation"] = evaluate_node_role(_node(graph, dst), row) if dst else {}
    result["active_typed_edges"] = matching_edges(graph, row)
    result["evaluator_semantics"] = list(row.get("evaluator_semantics", []))
    result["proposed_semantics"] = None
    result["semantic_label_source"] = "evaluator_semantics"
    return result


def rescore(graph: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [evaluate_occurrence(graph, row) for row in rows]


def compare_old_new(old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> dict[str, Any]:
    old_false = sum(bool(row.get("terminated")) is False for row in old_rows if row.get("terminal_reason") in {"horizon", "terminal_failure"})
    new_censored = sum(row.get("terminal_status") == "censored_unknown" for row in new_rows)
    return {"status": "PASS", "old_false_terminal_count": old_false, "new_censored_horizon_count": new_censored, "old_row_count": len(old_rows), "new_row_count": len(new_rows), "horizon_is_failure": False, "final_semantics_field": "evaluator_semantics", "proposed_semantics_used_as_gold": False}


def graph_metrics(graph: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get("root_family_id"))].append(evaluate_occurrence(graph, row))
    family_rows = []
    for family, items in sorted(by_family.items()):
        typed = sum(bool(x.get("active_typed_edges")) for x in items)
        failure = sum("failure_event" in x.get("evaluator_semantics", []) for x in items)
        recovery = sum("recovery_achieved" in x.get("evaluator_semantics", []) for x in items)
        terminal = [x for x in items if x.get("terminal_status") in {"failure_terminal", "success_terminal", "censored_unknown"}]
        false_claim = sum(x.get("terminal_status") != "failure_terminal" for x in terminal)
        family_rows.append({"root_family_id": family, "transition_coverage": sum(bool(x.get("src_cluster_id") is not None and x.get("dst_cluster_id") is not None) for x in items) / len(items) if items else None, "typed_occurrence_coverage": typed / len(items) if items else None, "failure_event_recall": failure / len(items) if items else None, "recovery_achieved_recall": recovery / len(items) if items else None, "false_terminal_claim_rate": false_claim / len(terminal) if terminal else None})
    return {"schema": "u4r1_evaluator_metrics_v1", "graph_id": graph.get("graph_id", graph.get("schema", "graph")), "rows": len(rows), "eligible_family_count": len(family_rows), "family_rows": family_rows, "horizon_excluded_from_failure": True, "label_origin": "simulator_info.events_and_explicit_terminal_reason"}
