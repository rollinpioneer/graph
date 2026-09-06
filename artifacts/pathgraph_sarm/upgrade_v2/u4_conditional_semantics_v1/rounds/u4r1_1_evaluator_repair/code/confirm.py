"""Frozen and fresh confirmation evaluation with an explicit input gate."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .evaluator_v2 import SEMANTIC_TARGETS, graph_metrics
from .io import read_json, read_jsonl, sha256_file, write_csv, write_json, write_jsonl
from .occurrence_table import build_occurrences


def build_occurrences_confirm(rollouts: Path, output: Path, repo: Path, predictions: Path | None = None) -> dict[str, Any]:
    return build_occurrences(rollouts, output, repo, "confirm", predictions)


def verify_lock(lock: Path, family_lock: Path) -> dict[str, Any]:
    if not lock.is_file() or not family_lock.is_file():
        return {"status": "BLOCKED", "reason": "final pipeline lock or family lock missing"}
    payload = read_json(lock); failures = []
    for path, digest in (payload.get("input_hashes") or {}).items():
        target = Path(path)
        if not target.is_file() or sha256_file(target) != digest:
            failures.append(str(target))
    expected = payload.get("family_lock_sha256")
    if expected and expected != sha256_file(family_lock):
        failures.append("family lock hash mismatch")
    return {"status": "PASS" if not failures and payload.get("confirmation_locked") else "BLOCKED", "failures": failures}


def build_lock(output: Path, graph_paths: list[Path], selection: Path, protocol: Path, family_lock: Path) -> dict[str, Any]:
    inputs = [*graph_paths, selection, protocol, family_lock]
    payload = {"schema": "u4r1_final_pipeline_lock_v2", "confirmation_locked": True, "graph_paths": [str(path.resolve()) for path in graph_paths], "selected_graph": read_json(selection).get("selected_graph"), "family_lock_sha256": sha256_file(family_lock), "input_hashes": {str(path.resolve()): sha256_file(path) for path in inputs}, "metric_version": "u4r1_evaluator_metrics_v2", "guard_dsl_version": "u4r1_guard_dsl_v2", "confirmation_started_after_lock": True}
    write_json(output, payload)
    return {"status": "PASS", "lock": str(output), "input_count": len(inputs), "selected_graph": payload["selected_graph"]}


def _macro(rows: list[dict[str, Any]], field: str) -> float | None:
    vals = [float(row[field]) for row in rows if row.get(field) not in {None, ""}]
    return sum(vals) / len(vals) if vals else None


def _paired_ci(left: list[float], right: list[float], bootstrap: int, seed: int) -> tuple[float | None, float | None, float | None]:
    if not left or not right or len(left) != len(right):
        return None, None, None
    diffs = [r - l for l, r in zip(left, right)]
    point = sum(diffs) / len(diffs)
    rng = random.Random(seed); samples = []
    for _ in range(max(1, bootstrap)):
        samples.append(sum(rng.choice(diffs) for _ in diffs) / len(diffs))
    samples.sort(); lo = samples[max(0, int(0.025 * len(samples)) - 1)]; hi = samples[min(len(samples) - 1, int(0.975 * len(samples)))]
    return point, lo, hi


def evaluate(graph_paths: list[Path], occurrences: Path, output: Path, paired: Path, family_table: Path, pipeline_lock: Path | None = None, family_lock: Path | None = None, *args: Any, **kwargs: Any) -> dict[str, Any]:
    if pipeline_lock is None or family_lock is None:
        result = {"status": "BLOCKED", "reason": "confirmation requires final pipeline lock and family lock"}
        write_json(output.with_suffix(".json"), result); return result
    gate = verify_lock(pipeline_lock, family_lock)
    if gate["status"] != "PASS":
        result = {"status": "BLOCKED", "reason": "confirmation gate failed", "gate": gate}
        write_json(output.with_suffix(".json"), result); return result
    rows = read_jsonl(occurrences); metrics = []; family_rows = []; semantic_rows = []
    for path in graph_paths:
        graph = read_json(path); result = graph_metrics(graph, rows); aggregate = result["aggregate"]
        item = {"graph_id": graph.get("graph_id", path.stem), "path": str(path), "split": "confirm", "occurrence_count": len(rows), "eligible_family_count": result["eligible_family_count"], "label_origin": result["label_origin"], "transition_coverage_macro": aggregate.get("transition_coverage_macro"), "typed_occurrence_coverage": aggregate.get("typed_occurrence_coverage_macro"), "unknown_rate": aggregate.get("unknown_rate_macro"), "ambiguous_guard_rate": aggregate.get("ambiguous_guard_rate_macro"), "failure_event_precision": aggregate.get("failure_event_precision_macro"), "failure_event_recall": aggregate.get("failure_event_recall_macro"), "failure_event_denominator": aggregate.get("failure_event_denominator"), "recovery_attempt_precision": aggregate.get("recovery_attempt_precision_macro"), "recovery_attempt_recall": aggregate.get("recovery_attempt_recall_macro"), "recovery_attempt_denominator": aggregate.get("recovery_attempt_denominator"), "recovery_achieved_precision": aggregate.get("recovery_achieved_precision_macro"), "recovery_achieved_recall": aggregate.get("recovery_achieved_recall_macro"), "recovery_achieved_denominator": aggregate.get("recovery_achieved_denominator"), "failure_terminal_precision": aggregate.get("failure_terminal_precision_macro"), "failure_terminal_recall": aggregate.get("failure_terminal_recall_macro"), "failure_terminal_f1": aggregate.get("failure_terminal_f1_macro"), "terminal_failure_denominator": aggregate.get("failure_terminal_denominator"), "false_terminal_claim_rate": aggregate.get("false_terminal_claim_rate_macro"), "terminal_claim_coverage": aggregate.get("terminal_claim_coverage_macro"), "censored_horizon_count": aggregate.get("censored_occurrence_count"), "guard_ambiguous_count": aggregate.get("guard_ambiguous_count"), "bootstrap_resamples": int(kwargs.get("bootstrap", 5000)), "bootstrap_unit": "root_family_id"}
        metrics.append(item)
        for row in result["family_rows"]:
            family_rows.append({"graph_id": item["graph_id"], **row, "scenario_for_analysis_only": next((x.get("family_scenario_for_analysis_only") for x in rows if x.get("root_family_id") == row["root_family_id"]), "")})
        for label in SEMANTIC_TARGETS:
            semantic_rows.append({"graph_id": item["graph_id"], "semantic": label, "precision": aggregate.get(f"{label}_precision"), "recall": aggregate.get(f"{label}_recall"), "f1": aggregate.get(f"{label}_f1"), "denominator": aggregate.get(f"{label}_denominator")})
    write_csv(output, metrics); write_csv(family_table, family_rows)
    write_csv(paired.with_name("conditional_minus_single_by_semantic.csv"), semantic_rows)
    lookup = {str(x["graph_id"]): x for x in metrics}; baseline = lookup.get("G1_single_label_v2", lookup.get("G1_semantic_only", {})); selected = lookup.get("G3_guard_rule_multigraph", lookup.get("G2_conditional_multigraph", metrics[-1] if metrics else {}))
    fields = ("transition_coverage", "typed_occurrence_coverage", "unknown_rate", "ambiguous_guard_rate", "failure_event_precision", "failure_event_recall", "recovery_attempt_precision", "recovery_attempt_recall", "recovery_achieved_precision", "recovery_achieved_recall", "failure_terminal_precision", "failure_terminal_recall", "false_terminal_claim_rate", "terminal_claim_coverage")
    paired_rows = []
    # Family-level paired values, preserving scenario/family units instead of
    # bootstrapping individual frames.
    left_by = {(x["root_family_id"]): x for x in family_rows if x["graph_id"] == baseline.get("graph_id")}
    right_by = {(x["root_family_id"]): x for x in family_rows if x["graph_id"] == selected.get("graph_id")}
    common = sorted(set(left_by) & set(right_by)); bootstrap = int(kwargs.get("bootstrap", 5000)); seed = int(kwargs.get("seed", kwargs.get("bootstrap_seed", 912001)))
    for field in fields:
        left = [float(left_by[f][field]) for f in common if left_by[f].get(field) is not None and right_by[f].get(field) is not None]; right = [float(right_by[f][field]) for f in common if left_by[f].get(field) is not None and right_by[f].get(field) is not None]
        point, lo, hi = _paired_ci(left, right, bootstrap, seed)
        paired_rows.append({"comparison": f"{selected.get('graph_id')}_minus_{baseline.get('graph_id')}", "metric": field, "effect": point, "ci_low": lo, "ci_high": hi, "paired_family_count": len(left), "bootstrap_resamples": bootstrap if point is not None else 0, "status": "estimable" if point is not None else "not_estimable", "bootstrap_unit": "root_family_id"})
    scenarios = sorted({str(row.get("scenario_for_analysis_only") or row.get("family_scenario_for_analysis_only") or "unknown") for row in family_rows if row.get("graph_id") == baseline.get("graph_id")})
    for scenario in scenarios:
        scenario_families = [family for family in common if str(left_by[family].get("scenario_for_analysis_only") or "unknown") == scenario]
        for field in ("typed_occurrence_coverage", "unknown_rate"):
            left = [float(left_by[f][field]) for f in scenario_families if left_by[f].get(field) is not None and right_by[f].get(field) is not None]
            right = [float(right_by[f][field]) for f in scenario_families if left_by[f].get(field) is not None and right_by[f].get(field) is not None]
            point, lo, hi = _paired_ci(left, right, bootstrap, seed)
            paired_rows.append({"comparison": f"{selected.get('graph_id')}_minus_{baseline.get('graph_id')}", "scope": "scenario", "scenario": scenario, "metric": field, "effect": point, "ci_low": lo, "ci_high": hi, "paired_family_count": len(left), "bootstrap_resamples": bootstrap if point is not None else 0, "status": "estimable" if point is not None else "not_estimable", "bootstrap_unit": "root_family_id"})
    write_csv(paired, paired_rows)
    per_semantic = kwargs.get("per_semantic")
    if per_semantic:
        write_csv(Path(per_semantic), semantic_rows)
    report = kwargs.get("report")
    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text("# Fresh confirmation\n\n- confirmation gate: PASS\n- statistics unit: root_family_id\n- horizon is censored and excluded from failure-terminal denominators\n- no post-confirmation tuning was performed\n", encoding="utf-8")
    payload = {"schema": "u4r1_confirmation_metrics_v2", "graphs": metrics, "paired_effects": paired_rows, "confirmation_frozen": True, "horizon_is_censored": True, "confirmation_gate": gate, "denominators": {"failure_occurrence": sum("failure_event" in x.get("evaluator_semantics", []) for x in rows), "recovery_attempt": sum("recovery_attempt" in x.get("evaluator_semantics", []) for x in rows), "recovery_achieved": sum("recovery_achieved" in x.get("evaluator_semantics", []) for x in rows), "terminal_failure": sum(x.get("terminal_status") == "failure_terminal" for x in rows), "censored_horizon": sum(x.get("terminal_status") == "censored_unknown" for x in rows)}}
    write_json(output.with_suffix(".json"), payload)
    return {"status": "PASS", "graphs": metrics, "paired_effects": paired_rows, "confirmation_gate": gate, "denominators": payload["denominators"]}
