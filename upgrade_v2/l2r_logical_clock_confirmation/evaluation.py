from __future__ import annotations

import json
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from upgrade_v2.l2r_contact_loss_guard.temporal_evaluation import score_records
from upgrade_v2.l2r_rgb_temporal_confirmation.candidate_inputs import load_candidate_input
from upgrade_v2.l2r_rgb_temporal_confirmation.evaluation import _actions, _run_o, _run_s

from .audits import candidate_source_audit, fault_injection_audit, input_provenance_audit, parameter_audit
from .candidate_adapter import apply_logical_guard
from .io_utils import read_json, read_jsonl, write_csv, write_json

METHODS = ("O_B2_RAW", "O_C3_RAW", "O_C3_CLP2_LOGICAL_CLOCK", "S_G_H")


def _run(method: str, rows: list[dict[str, Any]], physics: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if method == "O_B2_RAW": return _run_o(rows, "O_B2")
    if method == "O_C3_RAW": return _run_o(rows, "O_C3")
    if method == "O_C3_CLP2_LOGICAL_CLOCK": return apply_logical_guard(rows, _run_o(rows, "O_C3"))
    return _run_s(rows, physics or [], "S_G_H")


def _projection(records: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [(row.get("selected_action"), row.get("reason_code"), row.get("hold_state"),
             row.get("time"), row.get("capture_order")) for row in records]


def _percentile(values: list[float], q: float) -> float | None:
    if not values: return None
    ordered = sorted(values); return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def evaluate(repo: Path, confirmation_root: Path, reference_root: Path, output_root: Path) -> dict[str, Any]:
    gate = read_json(reference_root.parent / "generator_gate.json")
    if not gate.get("candidate_evaluation_allowed"): raise RuntimeError("GENERATOR_GATE_REQUIRED")
    output_root.mkdir(parents=True, exist_ok=False)
    online, predictions = {}, {}
    for metadata in sorted(confirmation_root.glob("*/metadata.json")):
        rollout_id = metadata.parent.name
        rows = load_candidate_input(metadata.parent / "candidate_input"); online[rollout_id] = rows
        for method in METHODS[:3]:
            saved = metadata.parent / "candidate_output" / f"{method}.jsonl"
            predictions[(method, rollout_id)] = read_jsonl(saved)
    references = read_json(reference_root / "physical_reference_index.json")["rows"]
    decisions = []
    prefix = {"schema": "l2rar2_r20_prefix_causality_audit_v1", "passed": True,
              "checked": 0, "expected": 1152, "mismatches": []}
    provenance = input_provenance_audit(); source = candidate_source_audit(repo)
    parameters = parameter_audit(); fault = fault_injection_audit(confirmation_root)
    for method in METHODS:
        for reference in references:
            rollout_id = reference["rollout_id"]; rows = online[rollout_id]
            physics = read_jsonl((reference_root / reference["physics_trace_path"]).resolve())
            records = predictions[(method, rollout_id)] if method != "S_G_H" else _run(method, rows, physics)
            actions = _actions(records); scored = score_records(reference, rows, actions)
            onset = reference.get("loss_onset_time_abs") if reference.get("reference_action") == "recover_object" else None
            onset = float(onset) if onset is not None else None
            first_post = next((float(row["time"]) for row in rows if onset is not None and float(row["time"]) >= onset), None)
            observable, previous_contact = None, None
            for row in rows:
                if previous_contact is True and row.get("contact_present") is False:
                    observable = float(row["time"]); break
                previous_contact = row.get("contact_present")
            if observable is None: observable = first_post
            decisions.append({"method": method,
                              "input_tier": provenance["methods"][method]["input_tier"],
                              "eligible_for_selection": method == "O_C3_CLP2_LOGICAL_CLOCK",
                              "rollout_id": rollout_id, "family_id": reference["family_id"],
                              "case_id": reference["case_id"], "truth_action": reference["reference_action"],
                              "reference_state": reference["state"], "reference_resolved": bool(reference["resolvable"]),
                              "physical_loss_onset": onset,
                              **{key: value for key, value in scored.items() if key != "all_actions"},
                              "actions_json": json.dumps(actions, sort_keys=True),
                              "unknown": bool(not actions and records and records[-1].get("selected_action") == "needs_observation"),
                              "requested_phase_offset_ms": reference.get("requested_phase_offset_ms"),
                              "actual_phase_offset_ms": reference.get("actual_phase_offset_ms"),
                              "latency_from_physical_onset": None if onset is None or scored["primary_action_time"] is None else scored["primary_action_time"] - onset,
                              "latency_from_method_observable": None if observable is None or scored["primary_action_time"] is None else scored["primary_action_time"] - observable,
                              "time_provenance": "candidate row.time; frozen physical onset joined only after prediction"})
            for fraction in (0.25, 0.50, 0.75, 1.0):
                cut = max(1, math.ceil(len(rows) * fraction))
                rerun = _run(method, rows[:cut], physics)
                prefix["checked"] += 1
                if _projection(rerun) != _projection(records[:cut]):
                    prefix["passed"] = False; prefix["mismatches"].append({"method": method, "rollout_id": rollout_id, "fraction": fraction})
    prefix["passed"] = bool(prefix["passed"] and prefix["checked"] == prefix["expected"])
    metrics, phase_metrics = [], []
    for method in METHODS:
        items = [row for row in decisions if row["method"] == method]
        resolved = [row for row in items if row["reference_resolved"]]
        counts = defaultdict(int)
        for row in items: counts[row["temporal_outcome"]] += 1
        latency = [float(row["latency_from_physical_onset"]) for row in items
                   if row["temporal_outcome"] == "CORRECT_IN_WINDOW" and row["latency_from_physical_onset"] is not None]
        false_group = [row for row in items if row["case_id"].startswith(("T1_", "T2_", "T3_", "T4_", "T12_"))]
        strong = [row for row in items if row["case_id"].startswith(tuple(f"T{n}_" for n in range(7, 12)))]
        variable = [row for row in items if row["case_id"].startswith(("T5_", "T6_"))]
        correct = sum(bool(row["accurate"]) for row in resolved)
        metric = {"method": method, "input_tier": provenance["methods"][method]["input_tier"],
                  "eligible_for_selection": method == "O_C3_CLP2_LOGICAL_CLOCK", "total": len(items),
                  "resolved": len(resolved), "correct": correct, "overall_temporal_accuracy": correct / len(resolved),
                  "unknown_rate": sum(bool(row["unknown"]) for row in items) / len(items),
                  "early_actions": counts["EARLY_ACTION"],
                  "false_actions_T1_T2_T3_T4_T12": sum(row["primary_action"] is not None for row in false_group),
                  "strong_T7_T11_correct_in_window": sum(row["temporal_outcome"] == "CORRECT_IN_WINDOW" for row in strong),
                  "T5_T6_outcome_correct": sum(bool(row["accurate"]) for row in variable),
                  "T3_no_action": sum(row["primary_action"] is None for row in items if row["case_id"].startswith("T3_")),
                  "T4_no_action": sum(row["primary_action"] is None for row in items if row["case_id"].startswith("T4_")),
                  "T12_no_recovery": sum(row["primary_action"] != "recover_object" for row in items if row["case_id"].startswith("T12_")),
                  "latency_p50_s": statistics.median(latency) if latency else None,
                  "latency_p90_s": _percentile(latency, 0.90), "outcome_counts": dict(counts)}
        metrics.append(metric)
        for offset in (0, 10, 20, 30, 40):
            group = [row for row in items if row.get("requested_phase_offset_ms") == offset]
            phase_metrics.append({"method": method, "phase_offset_ms": offset, "total": len(group),
                                  "correct": sum(bool(row["accurate"]) for row in group),
                                  "early": sum(row["temporal_outcome"] == "EARLY_ACTION" for row in group)})
    candidate = next(row for row in metrics if row["method"] == "O_C3_CLP2_LOGICAL_CLOCK")
    passed = bool(provenance["passed"] and prefix["passed"] and parameters["passed"] and source["passed"] and fault["passed"]
                  and candidate["false_actions_T1_T2_T3_T4_T12"] == 0 and candidate["early_actions"] == 0
                  and candidate["unknown_rate"] <= 0.05 and candidate["strong_T7_T11_correct_in_window"] >= 29
                  and candidate["T5_T6_outcome_correct"] == 12 and candidate["correct"] >= 71
                  and candidate["latency_p90_s"] is not None and candidate["latency_p90_s"] <= 0.20
                  and candidate["T3_no_action"] == 6 and candidate["T4_no_action"] == 6
                  and candidate["T12_no_recovery"] == 6)
    candidate["confirmation_status"] = "CONFIRMATION_PASS" if passed else "CONFIRMATION_FAIL"
    ranking = {"schema": "l2rar2_r20_candidate_ranking_v1",
               "selected_candidate_id": "O_C3_CLP2_LOGICAL_CLOCK" if passed else None,
               "confirmation_run": True, "l3_entry_ready": passed, "l3_entry_allowed": False,
               "raw_methods_selection_eligible": False, "state_assisted_selection_eligible": False,
               "methods": metrics}
    payload = {"schema": "l2rar2_r20_method_metrics_v1", "methods": metrics,
               "selected_candidate_id": ranking["selected_candidate_id"]}
    write_csv(output_root / "per_event_decisions.csv", decisions)
    flat = ("method", "input_tier", "eligible_for_selection", "total", "resolved", "correct",
            "overall_temporal_accuracy", "unknown_rate", "early_actions", "false_actions_T1_T2_T3_T4_T12",
            "strong_T7_T11_correct_in_window", "T5_T6_outcome_correct", "T3_no_action", "T4_no_action",
            "T12_no_recovery", "latency_p50_s", "latency_p90_s", "confirmation_status")
    write_csv(output_root / "method_metrics.csv", metrics, flat); write_json(output_root / "method_metrics.json", payload)
    write_csv(output_root / "latency_summary.csv", metrics, ("method", "latency_p50_s", "latency_p90_s"))
    write_csv(output_root / "phase_offset_metrics.csv", phase_metrics)
    coverage_src = reference_root / "logical_clock_coverage.json"; shutil.copyfile(coverage_src, output_root / "logical_clock_coverage.json")
    transient = [{"method": method, "T3_no_action": next(row for row in metrics if row["method"] == method)["T3_no_action"],
                  "T4_no_action": next(row for row in metrics if row["method"] == method)["T4_no_action"]} for method in METHODS]
    write_csv(output_root / "transient_contact_fault_metrics.csv", transient)
    rgb = [{"method": method, "T11_correct": sum(row["accurate"] for row in decisions if row["method"] == method and row["case_id"].startswith("T11_"))} for method in METHODS]
    write_csv(output_root / "rgb_dropout_metrics.csv", rgb)
    write_json(output_root / "input_provenance_audit.json", provenance); write_json(output_root / "prefix_causality_audit.json", prefix)
    write_json(output_root / "parameter_audit.json", parameters); write_json(output_root / "candidate_source_audit.json", source)
    write_json(output_root / "fault_injection_audit.json", fault); write_json(output_root / "candidate_ranking.json", ranking)
    (output_root / "candidate_recommendation.md").write_text(
        "# Candidate recommendation\n\n" + ("Confirm O_C3_CLP2_LOGICAL_CLOCK; L3 remains closed.\n" if passed else "No online candidate passes R20.\n"), encoding="utf-8")
    return payload
