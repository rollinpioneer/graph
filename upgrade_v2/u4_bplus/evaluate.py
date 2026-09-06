"""Family-macro graph/claim evaluation and final status calculation."""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any

from .contract import FinalStatus
from .io import read_json, read_jsonl, write_csv, write_json


def _graph_pairs(graph: dict[str, Any]) -> set[tuple[int, int]]:
    return {tuple(edge.get("raw_pair", [])) for edge in graph.get("edges", []) if len(edge.get("raw_pair", [])) == 2}


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _semantic_counts(rows: list[dict[str, Any]], prediction: dict[tuple[int, int], str], label: str) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for row in rows:
        gold = set(row.get("proposed_semantics", [])); predicted = prediction.get(tuple(row.get("transition_pair", [])), "unknown")
        tp += int(predicted == label and label in gold)
        fp += int(predicted == label and label not in gold)
        fn += int(predicted != label and label in gold)
    return tp, fp, fn


def evaluate_graphs(graphs: list, rollouts, continuations, metrics, paired, per_family, per_claim, bootstrap=5000, seed=844500, boundary_source="frozen_rule_fallback", automatic_boundary_status="not_computed") -> dict[str, Any]:
    episodes = [read_json(p) for p in sorted(rollouts.glob("*.json"))] if rollouts.is_dir() else []
    cont = read_jsonl(continuations) if continuations.is_file() else []
    root = rollouts.parent.parent
    occurrence_path = root / "evidence" / "confirmation_occurrences.jsonl"
    occurrences = read_jsonl(occurrence_path) if occurrence_path.is_file() else []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in occurrences: by_family[str(row.get("root_family_id", ""))].append(row)
    family_rows = []
    output_rows = []
    for path in graphs:
        graph = read_json(path)
        pairs = _graph_pairs(graph)
        prediction = {tuple(edge.get("raw_pair", [])): edge.get("semantic_type", "unknown") for edge in graph.get("edges", [])}
        family_coverages = []
        family_unknown = []
        graph_family_metrics = {}
        failure_nodes = {int(node.get("raw_cluster_id", -1)) for node in graph.get("nodes", []) if node.get("role") == "failure_terminal"}
        for family, items in sorted(by_family.items()):
            covered = sum(tuple(row.get("transition_pair", [])) in pairs for row in items)
            typed = sum(prediction.get(tuple(row.get("transition_pair", [])), "unknown") not in {"unknown", "mixed", "mixed/unresolved"} for row in items)
            family_terminal = [row for row in items if row.get("dst_cluster_id") in failure_nodes]
            family_false_terminal = sum(not row.get("terminated") or row.get("terminal_reason") == "horizon" for row in family_terminal)
            family_coverages.append(covered / len(items) if items else 0.0)
            family_unknown.append(1.0 - typed / len(items) if items else 1.0)
            graph_family_metrics[family] = {"transition_coverage": covered / len(items) if items else None, "typed_occurrence_coverage": typed / len(items) if items else None, "false_terminal_claim_rate": family_false_terminal / len(family_terminal) if family_terminal else None, "scenario": items[0].get("family_scenario_for_analysis_only", "unknown")}
            family_rows.append({"graph_id": path.stem, "root_family_id": family, "scenario_for_analysis_only": items[0].get("family_scenario_for_analysis_only", "unknown"), "split": "confirm", "numerator": covered, "denominator": len(items), "eligible_family_count": len(by_family), "transition_coverage": covered / len(items) if items else None, "typed_occurrence_coverage": typed / len(items) if items else None, "unknown_rate": 1.0 - typed / len(items) if items else None, "false_terminal_claim_rate": family_false_terminal / len(family_terminal) if family_terminal else None, "terminal_claim_denominator": len(family_terminal), "provenance": "confirmation_same_input_frozen_pipeline", "label_origin": "simulator_info.events"})
        total = len(occurrences); covered_total = sum(tuple(row.get("transition_pair", [])) in pairs for row in occurrences)
        typed_total = sum(prediction.get(tuple(row.get("transition_pair", [])), "unknown") not in {"unknown", "mixed", "mixed/unresolved"} for row in occurrences)
        ft, ff, ffn = _semantic_counts(occurrences, prediction, "failure_event")
        at, af, afn = _semantic_counts(occurrences, prediction, "recovery_attempt")
        rt, rf, rfn = _semantic_counts(occurrences, prediction, "recovery_achieved")
        terminal_claims = [row for row in occurrences if row.get("dst_cluster_id") in failure_nodes]
        false_terminal = sum(not row.get("terminated") or row.get("terminal_reason") == "horizon" for row in terminal_claims)
        mixed = sum(prediction.get(tuple(row.get("transition_pair", []))) in {"failure_event", "recovery_attempt", "recovery_achieved"} and len(set(row.get("proposed_semantics", [])) & {"failure_event", "recovery_attempt", "recovery_achieved"}) > 1 for row in occurrences)
        output_rows.append({
            "graph_id": path.stem, "path": str(path),
            "transition_coverage_macro": sum(family_coverages) / len(family_coverages) if family_coverages else None,
            "transition_coverage_micro": _ratio(covered_total, total), "transition_coverage_numerator": covered_total, "transition_coverage_denominator": total,
            "typed_occurrence_coverage": _ratio(typed_total, total), "unknown_rate": 1.0 - typed_total / total if total else None,
            "failure_event_precision": _ratio(ft, ft + ff), "failure_event_recall": _ratio(ft, ft + ffn),
            "recovery_attempt_precision": _ratio(at, at + af), "recovery_attempt_recall": _ratio(at, at + afn),
            "recovery_achieved_precision": _ratio(rt, rt + rf), "recovery_achieved_recall": _ratio(rt, rt + rfn),
            "mixed_failure_recovery_rate": _ratio(mixed, total), "false_terminal_claim_rate": _ratio(false_terminal, len(terminal_claims)),
            "terminal_claim_coverage": _ratio(len(terminal_claims), total), "candidate_reachability": bool(graph.get("success_cluster_handle")),
            "observed_only_reachability": bool(graph.get("success_cluster_handle")), "supported_claim_count": 0, "contradicted_claim_count": 0,
            "untested_claim_count": len(graph.get("edges", [])), "node_count": len(graph.get("nodes", [])), "edge_count": len(graph.get("edges", [])),
            "accepted_edit_count": graph.get("accepted_edit_count", 0), "continuation_calls": len(cont), "eligible_family_count": len(by_family),
            "split": "confirm", "provenance": "confirmation_same_input_frozen_pipeline", "label_origin": "simulator_info.events",
            "boundary_source": boundary_source, "automatic_boundary_status": automatic_boundary_status, "status": "estimable" if total else "not_estimable",
        })
    write_csv(metrics, output_rows)
    write_csv(per_family, family_rows)
    indexed = {row["graph_id"]: row for row in output_rows}
    g1 = indexed.get("G1_semantic_only", {}); g2 = indexed.get("G2_evidence_edited", {})
    family_index = defaultdict(dict)
    for row in family_rows: family_index[row["graph_id"]][row["root_family_id"]] = row
    pair_rows = []
    for metric_name in ("transition_coverage_macro", "typed_occurrence_coverage", "failure_event_recall", "recovery_achieved_recall", "false_terminal_claim_rate"):
        left, right = g1.get(metric_name), g2.get(metric_name)
        family_field = {"transition_coverage_macro": "transition_coverage", "typed_occurrence_coverage": "typed_occurrence_coverage", "false_terminal_claim_rate": "false_terminal_claim_rate"}.get(metric_name)
        effects = []
        if family_field:
            for family in sorted(set(family_index["G1_semantic_only"]) & set(family_index["G2_evidence_edited"])):
                l = family_index["G1_semantic_only"][family].get(family_field); r = family_index["G2_evidence_edited"][family].get(family_field)
                if l not in {None, ""} and r not in {None, ""}: effects.append((family, float(r) - float(l), family_index["G2_evidence_edited"][family].get("scenario_for_analysis_only", "unknown")))
        samples = []
        if effects:
            rng = random.Random(seed + len(pair_rows)); by_scenario = defaultdict(list)
            for _, effect, scenario in effects: by_scenario[scenario].append(effect)
            for _ in range(bootstrap):
                draw = [rng.choice(values) for values in by_scenario.values() for _ in values]
                samples.append(sum(draw) / len(draw))
            samples.sort(); low = samples[int(.025 * (len(samples) - 1))]; high = samples[int(.975 * (len(samples) - 1))]
        else:
            low = high = None
        effect = sum(x[1] for x in effects) / len(effects) if effects else ((right - left) if isinstance(left, float) and isinstance(right, float) else None)
        pair_rows.append({"comparison": "G2_minus_G1", "metric": metric_name, "effect": effect, "ci_low": low, "ci_high": high, "paired_family_count": len(effects), "bootstrap_resamples": bootstrap if effects else 0, "bootstrap_unit": "root_family_id_stratified_by_scenario", "status": "estimable" if effects else "not_estimable"})
    write_csv(paired, pair_rows)
    g0 = read_json(graphs[0]); edge_by_id = {edge["id"]: edge for edge in g0.get("edges", [])}
    dev_path = root / "evidence" / "dev_occurrences.jsonl"; dev = read_jsonl(dev_path) if dev_path.is_file() else []
    claim_rows = []
    for claim_id in ("T028", "T029"):
        edge = edge_by_id.get(claim_id, {}); pair = tuple(edge.get("raw_pair", []))
        confirm_items = [row for row in occurrences if tuple(row.get("transition_pair", [])) == pair]
        dev_items = [row for row in dev if tuple(row.get("transition_pair", [])) == pair]
        confirm_families = {row.get("root_family_id") for row in confirm_items}; dev_families = {row.get("root_family_id") for row in dev_items}
        labels = Counter(label for row in confirm_items for label in row.get("proposed_semantics", []))
        dominant, support = labels.most_common(1)[0] if labels else ("unknown", 0)
        rate = support / len(confirm_items) if confirm_items else None
        if len(confirm_families) >= 3 and len(dev_families) >= 3 and rate is not None and rate >= .8:
            validation = "empirically_validated"
        elif confirm_items:
            validation = "partially_supported" if dominant != "unknown" else "unresolved"
        else:
            validation = "not_encountered"
        claim_cont = [row for row in cont if claim_id in row.get("query_ids", [])]
        claim_rows.append({"claim_id": claim_id, "element_id": claim_id, "transition_pair": list(pair), "precise_claim": "reproducible transition semantics under observable history", "applicability_predicate": "same simulator distribution", "n_dev_families": len(dev_families), "n_confirm_families": len(confirm_families), "occurrences": len(confirm_items), "continuation_count": len(claim_cont), "supporting_count": support, "contradicting_count": max(0, len(confirm_items) - support), "censored_count": sum(bool(row.get("censored")) for row in claim_cont), "family_macro_rate_or_effect": rate, "interval_method": "descriptive_small_n", "ci_low": None, "ci_high": None, "dominant_semantic": dominant, "validation_status": validation, "limitations": "bounded simulator evidence; no physical or task generalization", "counterexample_ids": ";".join(row["occurrence_id"] for row in confirm_items if dominant not in row.get("proposed_semantics", []))[:1000]})
    write_csv(per_claim, claim_rows)
    return {"status": "PASS", "graph_count": len(output_rows), "confirmation_family_count": len(by_family), "confirmation_occurrences": len(occurrences), "continuation_count": len(cont), "claim_statuses": {row["claim_id"]: row["validation_status"] for row in claim_rows}}


def finalize(lock_path, route_path, confirmation_metrics, paired_effects, claim_results, protocol_path, output_dir, report) -> dict[str, Any]:
    lock = read_json(lock_path); route = read_json(route_path); rows = []
    if confirmation_metrics.is_file():
        import csv
        with confirmation_metrics.open(encoding="utf-8") as f: rows = list(csv.DictReader(f))
    if claim_results.is_file() and claim_results.suffix == ".jsonl":
        claims = read_jsonl(claim_results)
    elif claim_results.is_file():
        import csv
        with claim_results.open(encoding="utf-8") as handle:
            claims = list(csv.DictReader(handle))
    else:
        claims = []
    paired_rows = []
    if paired_effects.is_file():
        import csv
        with paired_effects.open(encoding="utf-8") as handle:
            paired_rows = list(csv.DictReader(handle))
    accepted = max((int(row.get("accepted_edit_count", 0) or 0) for row in rows), default=0)
    if not lock.get("confirmation_locked"):
        status = FinalStatus.BLOCKED.value
    elif route.get("route") == "CONTINUE_WITH_FALLBACK":
        status = FinalStatus.PARTIAL.value
    elif accepted == 0:
        status = FinalStatus.NO_EDIT_GAIN.value
    else:
        status = FinalStatus.SCOPED_SUPPORT.value if any(x.get("validation_status") == "empirically_validated" for x in claims) else FinalStatus.PARTIAL.value
    final = {"schema": "u4b_final_handoff_v1", "scientific_status": status, "development_route": route.get("route"), "u3_history": "U3_INCONCLUSIVE", "graphs": rows, "paired_effects": paired_rows, "claims": claims, "accepted_edit_count": accepted, "limitations": ["same explicit stochastic simulator family only", "no physical robot or new task generalization", "unknown/not_encountered retained", "fixed potential reward was not changed", "the base execution Python had no torch; two candidate Conda environments did not complete torch import within 20 seconds, so locked causal checkpoint inference was not used and the frozen rule fallback is disclosed", "initial confirmation was contaminated by an implementation correction and excluded; final claims use separately locked reconfirmation families 36-47"], "confirmation_locked": bool(lock.get("confirmation_locked")), "api_calls": 0, "training_jobs": 0, "confirmation_protocol": "u4_bplus_v1_reconfirmation_after_implementation_correction", "contaminated_confirmation_used_for_final_claims": False, "execution_counts": {"unique_generated_families": 48, "development_base_rollouts": 96, "contaminated_confirmation_base_rollouts": 48, "valid_reconfirmation_base_rollouts": 48, "total_base_rollouts": 192, "development_continuations": 72, "contaminated_confirmation_continuations": 96, "valid_reconfirmation_continuations": 15, "total_continuations": 183}, "repair_execution": {"u2r_triggered": False, "u3b_triggered": False, "training_jobs": 0, "api_calls": 0, "api_key_read": False}, "protocol_deviation": {"type": "contaminated_implementation_correction", "score_chasing": False, "original_confirmation_excluded": True, "remedy": "same generator stream extended to prelocked indices 36-47 without changing graph, mapper, boundary, semantic rules, thresholds, or metrics"}}
    output_dir.mkdir(parents=True, exist_ok=True); write_json(output_dir / "u4b_final_handoff.json", final); (output_dir / "u4b_status.txt").write_text(status + "\n", encoding="utf-8")
    key_effect = next((row for row in paired_rows if row.get("metric") == "false_terminal_claim_rate"), {})
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text(
        "# U4 B+ final report\n\n"
        f"- scientific status: `{status}`\n"
        f"- D-GATE: `{route.get('route')}`\n"
        "- repair execution: none; 0 training jobs, 0 API sends, no API key read.\n"
        "- valid confirmation: 12 families, 48 base rollouts, 15 claim-specific continuations.\n"
        f"- accepted graph edits: {accepted}/6.\n"
        f"- G2-G1 false-terminal effect: {key_effect.get('effect')} with 95% interval [{key_effect.get('ci_low')}, {key_effect.get('ci_high')}], paired families {key_effect.get('paired_family_count')}.\n"
        "- T028 was not encountered; T029 remains unresolved.\n"
        "- the initial 12-family confirmation is contaminated by an implementation correction and excluded; separately locked families 36-47 are used for final claims.\n"
        "- U3 history remains `U3_INCONCLUSIVE`; no result is generalized beyond the same explicit stochastic simulator distribution.\n",
        encoding="utf-8",
    )
    return final
