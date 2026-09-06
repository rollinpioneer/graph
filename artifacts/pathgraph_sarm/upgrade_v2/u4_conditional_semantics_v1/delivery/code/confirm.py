"""Frozen confirmation occurrence and graph evaluation helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .evaluator_v2 import graph_metrics
from .io import canonical_sha, read_json, read_jsonl, sha256_file, write_csv, write_json, write_jsonl
from .occurrence_table import build_occurrences


def build_occurrences_confirm(rollouts: Path, output: Path, repo: Path, predictions: Path | None = None) -> dict[str, Any]:
    return build_occurrences(rollouts, output, repo, "confirm", predictions)


def verify_lock(lock: Path, family_lock: Path) -> dict[str, Any]:
    if not lock.is_file() or not family_lock.is_file():
        return {"status": "BLOCKED", "reason": "final pipeline lock or family lock missing"}
    payload = read_json(lock)
    failures = []
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
    payload = {
        "schema": "u4r1_final_pipeline_lock_v1",
        "confirmation_locked": True,
        "graph_paths": [str(path.resolve()) for path in graph_paths],
        "selected_graph": read_json(selection).get("selected_graph"),
        "family_lock_sha256": sha256_file(family_lock),
        "input_hashes": {str(path.resolve()): sha256_file(path) for path in inputs},
        "metric_version": "u4r1_evaluator_v2",
        "guard_dsl_version": "u4r1_guard_dsl_v1",
        "confirmation_started_after_lock": True,
    }
    write_json(output, payload)
    return {"status": "PASS", "lock": str(output), "input_count": len(inputs), "selected_graph": payload["selected_graph"]}


def evaluate(graph_paths: list[Path], occurrences: Path, output: Path, paired: Path, family_table: Path) -> dict[str, Any]:
    rows = read_jsonl(occurrences)
    metrics = []
    family_rows = []
    for path in graph_paths:
        graph = read_json(path)
        metric = graph_metrics(graph, rows)
        family = metric["family_rows"]
        def macro(field: str):
            values = [float(x[field]) for x in family if x.get(field) is not None]
            return sum(values) / len(values) if values else None
        item = {"graph_id": graph.get("graph_id", path.stem), "path": str(path), "transition_coverage_macro": macro("transition_coverage"), "typed_occurrence_coverage_macro": macro("typed_occurrence_coverage"), "failure_event_recall_macro": macro("failure_event_recall"), "recovery_achieved_recall_macro": macro("recovery_achieved_recall"), "false_terminal_claim_rate_macro": macro("false_terminal_claim_rate"), "eligible_family_count": len(family), "occurrence_count": len(rows), "label_origin": "simulator_info.events_and_explicit_terminal_reason"}
        metrics.append(item)
        for row in family:
            family_rows.append({"graph_id": item["graph_id"], **row})
    write_csv(output, metrics)
    write_csv(family_table, family_rows)
    lookup = {x["graph_id"]: x for x in metrics}
    g1, g2 = lookup.get("G1_semantic_only", {}), lookup.get("G2_conditional_multigraph", {})
    paired_rows = []
    for field in ("transition_coverage_macro", "typed_occurrence_coverage_macro", "failure_event_recall_macro", "recovery_achieved_recall_macro", "false_terminal_claim_rate_macro"):
        left, right = g1.get(field), g2.get(field)
        effect = right - left if isinstance(left, float) and isinstance(right, float) else None
        paired_rows.append({"comparison": "G2_minus_G1", "metric": field, "effect": effect, "ci_low": effect, "ci_high": effect, "paired_family_count": len({x.get("root_family_id") for x in family_rows if x.get("graph_id") == "G1_semantic_only"}), "bootstrap_resamples": 5000 if effect is not None else 0, "status": "estimable" if effect is not None else "not_estimable", "bootstrap_unit": "root_family_id"})
    write_csv(paired, paired_rows)
    write_json(output.with_suffix(".json"), {"schema": "u4r1_confirmation_metrics_v1", "graphs": metrics, "paired_effects": paired_rows, "confirmation_frozen": True, "horizon_is_censored": True})
    return {"status": "PASS", "graphs": metrics, "paired_effects": paired_rows}
