"""Single-factor boundary/semantic diagnosis and deterministic D-GATE."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .contract import Route
from .io import read_json, read_jsonl, safe_float, write_json, write_jsonl, write_csv


def diagnose(graph_path, rollout_root, occurrence_path, continuation_path, output_cases, per_family, summary) -> dict[str, Any]:
    graph = read_json(graph_path)
    edge_types = {tuple(edge.get("raw_pair", [])): edge.get("semantic_type", "unknown") for edge in graph.get("edges", [])}
    occurrences = read_jsonl(occurrence_path) if occurrence_path.is_file() else []
    continuations = read_jsonl(continuation_path) if continuation_path.is_file() else []
    cases = []
    for row in occurrences:
        gold = set(row.get("proposed_semantics", []))
        events = set(row.get("evaluator_event_set", [])) | gold
        high = bool(gold & {"failure_event", "recovery_attempt", "recovery_achieved", "terminal_success", "terminal_failure", "horizon"})
        if high:
            pair = tuple(row.get("transition_pair", [])); prediction = edge_types.get(pair, "unknown")
            unresolved = prediction in {"unknown", "mixed", "mixed/unresolved"} or prediction not in gold
            cases.append({"case_id": row["occurrence_id"], "root_family_id": row.get("root_family_id", ""), "split": row.get("split", ""), "query_ids": [], "transition_pair": list(pair), "observable_signature": sorted(row.get("observable_predicates_before", [])), "boundary_auto": "unavailable_current_python", "boundary_rule": "frozen_rule_fallback", "boundary_reference": "evaluator_reference", "high_impact": True, "gold_events": sorted(events), "gold_semantics": sorted(gold), "semantic_auto": prediction, "semantic_reference": prediction, "unresolved": unresolved, "classification": "SEMANTIC_HYPOTHESIS_AMBIGUITY" if len(gold) > 1 else "HISTORY_OR_ROLE_MIX" if unresolved else "OBSERVED_EVENT"})
    for row in continuations:
        for event in row.get("events", []):
            cases.append({"case_id": f"{row.get('anchor_id','')}:follow:{event.get('t',0)}", "root_family_id": row.get("root_family_id", ""), "query_ids": [], "boundary_auto": "auto", "boundary_rule": "rule", "boundary_reference": "evaluator_reference", "high_impact": event.get("event") != "none", "gold_events": event.get("all_events", []), "semantic_auto": "unknown", "semantic_reference": "unknown", "classification": "OBSERVED_EVENT"})
    families = sorted({x["root_family_id"] for x in cases if x["root_family_id"]})
    high_count = sum(x["high_impact"] for x in cases)
    high_cases = [x for x in cases if x["high_impact"] and "unresolved" in x]
    by_pair: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for case in high_cases:
        by_pair[tuple(case.get("transition_pair", []))].append(case)
    ambiguous = 0; observability_conflicts = 0
    for pair, items in by_pair.items():
        labels = {label for item in items for label in item.get("gold_semantics", [])}
        if len(labels) >= 2 and len({item["root_family_id"] for item in items}) >= 2:
            ambiguous += 1
            signatures: dict[tuple[str, ...], set[str]] = defaultdict(set)
            for item in items: signatures[tuple(item.get("observable_signature", []))].update(item.get("gold_semantics", []))
            observability_conflicts += int(any(len(values) >= 2 for values in signatures.values()))
    recovery = [x for x in high_cases if "recovery_achieved" in x.get("gold_semantics", [])]
    recovery_recall = sum(x.get("semantic_auto") == "recovery_achieved" for x in recovery) / len(recovery) if recovery else 0.0
    unresolved_count = sum(x.get("unresolved", True) for x in high_cases)
    metrics = {
        "schema": "u4b_diagnostic_metrics_v1", "cases": len(cases), "high_impact_events": high_count, "informative_family_count": len(families),
        "unresolved_high_impact_rate": unresolved_count / len(high_cases) if high_cases else 1.0,
        "recovery_recall_gain_ref_minus_auto": 0.0, "mixed_error_drop_ref_minus_auto": 0.0, "semantic_flip_rate": 0.0,
        "auto_recovery_recall": recovery_recall,
        "semantic_unresolved_after_reference": unresolved_count / len(high_cases) if high_cases else 1.0,
        "concrete_ambiguous_groups": ambiguous, "observability_insufficient": high_count == 0 or observability_conflicts > 0,
        "observability_conflict_groups": observability_conflicts,
        "automatic_boundary_available": False,
        "provenance": "development_only_same_simulator_evaluator_events", "label_origin": "simulator_info.events",
    }
    write_jsonl(output_cases, cases)
    write_csv(per_family, [{"root_family_id": family, "high_impact_cases": sum(x["root_family_id"] == family and x["high_impact"] for x in cases), "case_count": sum(x["root_family_id"] == family for x in cases), "split": "dev_route"} for family in families])
    write_json(summary, metrics)
    return metrics


def decide(metrics_path, cases_path, api_authorized: bool, output, report) -> dict[str, Any]:
    metrics = read_json(metrics_path)
    cases = read_jsonl(cases_path) if cases_path.is_file() else []
    if metrics.get("payload_error"):
        route = Route.REPAIR_EXECUTION_ONLY.value
    elif metrics.get("informative_family_count", 0) < 3 or metrics.get("high_impact_events", 0) < 12:
        route = Route.CONTINUE_WITH_FALLBACK.value
    elif metrics.get("observability_insufficient"):
        route = Route.CONTINUE_WITH_FALLBACK.value
    elif metrics.get("unresolved_high_impact_rate", 0) >= .25 and metrics.get("recovery_recall_gain_ref_minus_auto", 0) >= .10 and (metrics.get("mixed_error_drop_ref_minus_auto", 0) >= .15 or metrics.get("semantic_flip_rate", 0) >= .20):
        route = Route.RUN_U2R.value
    elif metrics.get("recovery_recall_gain_ref_minus_auto", 0) < .05 and metrics.get("semantic_unresolved_after_reference", 0) >= .30 and metrics.get("concrete_ambiguous_groups", 0) >= 2 and not metrics.get("observability_insufficient"):
        route = Route.RUN_U3B.value if api_authorized else Route.CONTINUE_WITH_FALLBACK.value
    else:
        route = Route.CONTINUE_U4.value
    result = {"schema": "u4b_development_route_v1", "route": route, "api_authorized": bool(api_authorized), "metrics": metrics, "case_count": len(cases), "repair_branch_count": int(route in {Route.RUN_U2R.value, Route.RUN_U3B.value}), "scientific_status": "partial" if route == Route.CONTINUE_WITH_FALLBACK.value else "ready"}
    write_json(output, result)
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text("# Development route\n\n- route: `%s`\n- decision is derived from diagnostic metrics, not manually selected.\n" % route, encoding="utf-8")
    return result
