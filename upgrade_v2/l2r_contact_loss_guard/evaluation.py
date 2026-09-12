from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from upgrade_v2.l2r_rgb_temporal_confirmation.candidate_inputs import ALLOWED, load_candidate_input
from upgrade_v2.l2r_rgb_temporal_confirmation.evaluation import _actions, _projection, _run_o, _run_s

from .audits import input_provenance_audit, parameter_audit
from .guard import apply_guard
from .io_utils import read_json, read_jsonl, write_csv, write_json
from .temporal_evaluation import score_records

METHODS = ("O_B2_RAW", "O_C3_RAW", "O_C3_CLP1", "S_G_H")


def _records(method: str, rows: list[dict[str, Any]], physics: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if method == "O_B2_RAW": return _run_o(rows, "O_B2")
    if method == "O_C3_RAW": return _run_o(rows, "O_C3")
    if method == "O_C3_CLP1": return apply_guard(rows, _run_o(rows, "O_C3"))
    return _run_s(rows, physics or [], "S_G_H")


def _percentile(values: list[float], q: float) -> float | None:
    if not values: return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def evaluate(confirmation_root: Path, reference_root: Path, output_root: Path) -> dict[str, Any]:
    gate = read_json(reference_root / "generator_gate.json")
    if not gate.get("candidate_evaluation_allowed"):
        raise RuntimeError("GENERATOR_GATE_REQUIRED")
    output_root.mkdir(parents=True, exist_ok=False)

    online: dict[str, list[dict[str, Any]]] = {}
    predictions: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for metadata in sorted(confirmation_root.glob("*/metadata.json")):
        rollout_id = metadata.parent.name
        rows = load_candidate_input(metadata.parent / "candidate_input")
        online[rollout_id] = rows
        for method in METHODS[:3]:
            predictions[(method, rollout_id)] = _records(method, rows)

    references = read_json(reference_root / "physical_reference_index.json")["rows"]
    decisions: list[dict[str, Any]] = []
    prefix = {"schema": "l2rar2_r18_prefix_causality_audit_v1", "passed": True,
              "checked": 0, "expected": len(METHODS) * len(references) * 4, "mismatches": []}
    provenance: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        online_method = method != "S_G_H"
        provenance[method] = {
            "input_tier": "ONLINE_TRUE_RGB" if online_method else "STATE_ASSISTED_DIAGNOSTIC",
            "eligible_for_online_selection": method == "O_C3_CLP1",
            "files_read": ["candidate_input/detections_rgb.jsonl", "candidate_input/contact_proxy.jsonl",
                           "candidate_input/attempt_lifecycle.jsonl", "candidate_input/gripper_commands.jsonl",
                           "candidate_input/request_context.jsonl"] + ([] if online_method else ["reference/physics_trace.jsonl"]),
            "fields_read": sorted(ALLOWED) if online_method else ["object_world_position", "gripper_world_position"],
            "forbidden_online_reads": 0,
        }
        for reference in references:
            rollout_id = reference["rollout_id"]
            rows = online[rollout_id]
            physics = read_jsonl((reference_root / reference["physics_trace_path"]).resolve())
            records = predictions[(method, rollout_id)] if online_method else _records(method, rows, physics)
            actions = _actions(records)
            scored = score_records(reference, rows, actions)
            onset = reference.get("loss_onset_time_abs") if reference.get("reference_action") == "recover_object" else None
            onset = float(onset) if onset is not None else None
            first_post = next((float(row["time"]) for row in rows if onset is not None and float(row["time"]) >= onset), None)
            previous_contact = None
            observable = None
            for row in rows:
                if previous_contact is True and row.get("contact_present") is False:
                    observable = float(row["time"]); break
                previous_contact = row.get("contact_present")
            if observable is None: observable = first_post
            decisions.append({
                "method": method, "input_tier": provenance[method]["input_tier"],
                "eligible_for_online_selection": method == "O_C3_CLP1", "rollout_id": rollout_id,
                "family_id": reference["family_id"], "case_id": reference["case_id"],
                "phase_offset_ms": reference.get("phase_offset_ms"), "truth_action": reference["reference_action"],
                "reference_state": reference.get("state"), "reference_resolved": bool(reference.get("resolvable")),
                "physical_loss_onset": onset, **{key: value for key, value in scored.items() if key != "all_actions"},
                "actions_json": json.dumps(actions, sort_keys=True),
                "unknown": bool(not actions and records and records[-1].get("selected_action") == "needs_observation"),
                "t_first_online_capture_after_onset": first_post, "t_method_first_observable": observable,
                "latency_from_physical_onset": None if onset is None or scored["primary_action_time"] is None else scored["primary_action_time"] - onset,
                "latency_from_method_observable": None if observable is None or scored["primary_action_time"] is None else scored["primary_action_time"] - observable,
                "time_provenance": "online capture time drives prediction; frozen physics row.time is joined only for scoring",
            })
            for fraction in (0.25, 0.50, 0.75, 1.0):
                cut = max(1, math.ceil(len(rows) * fraction))
                rerun = _records(method, rows[:cut], physics)
                prefix["checked"] += 1
                if _projection(rerun) != _projection(records[:cut]):
                    prefix["passed"] = False
                    prefix["mismatches"].append({"method": method, "rollout_id": rollout_id, "fraction": fraction})
    prefix["passed"] = bool(prefix["passed"] and prefix["checked"] == prefix["expected"])

    metrics = []
    phase_rows = []
    transient_rows = []
    for method in METHODS:
        items = [row for row in decisions if row["method"] == method]
        resolved = [row for row in items if row["reference_resolved"]]
        counts = defaultdict(int)
        for row in items: counts[row["temporal_outcome"]] += 1
        correct = sum(bool(row["accurate"]) for row in resolved)
        latency = [float(row["latency_from_physical_onset"]) for row in items
                   if row["latency_from_physical_onset"] is not None and row["temporal_outcome"] == "CORRECT_IN_WINDOW"]
        false_group = [row for row in items if row["case_id"].startswith(("T1_", "T2_", "T3_", "T12_"))]
        strong = [row for row in items if row["case_id"].startswith(tuple(f"T{i}_" for i in range(6, 12)))]
        variable = [row for row in items if row["case_id"].startswith(("T4_", "T5_"))]
        metric = {"method": method, "input_tier": provenance[method]["input_tier"], "total": len(items),
                  "resolved": len(resolved), "correct": correct,
                  "overall_temporal_accuracy": correct / len(resolved),
                  "unknown_rate": sum(bool(row["unknown"]) for row in items) / len(items),
                  "early_actions": counts["EARLY_ACTION"], "false_actions_T1_T2_T3_T12": sum(row["primary_action"] is not None for row in false_group),
                  "strong_loss_correct_in_window": sum(row["temporal_outcome"] == "CORRECT_IN_WINDOW" for row in strong),
                  "T4_T5_outcome_conditioned_correct": sum(bool(row["accurate"]) for row in variable),
                  "latency_median_s": statistics.median(latency) if latency else None,
                  "latency_p90_s": _percentile(latency, 0.90), "outcome_counts": dict(counts)}
        metrics.append(metric)
        for offset in (0, 10, 20, 30, 40):
            group = [row for row in items if row.get("phase_offset_ms") == offset]
            phase_rows.append({"method": method, "phase_offset_ms": offset, "total": len(group),
                               "correct": sum(bool(row["accurate"]) for row in group),
                               "early": sum(row["temporal_outcome"] == "EARLY_ACTION" for row in group)})
        for prefix_name in ("T3_", "T11_"):
            group = [row for row in items if row["case_id"].startswith(prefix_name)]
            transient_rows.append({"method": method, "case": prefix_name.rstrip("_"), "total": len(group),
                                   "correct": sum(bool(row["accurate"]) for row in group),
                                   "false_actions": sum(row["primary_action"] is not None and row["truth_action"] == "none" for row in group)})

    input_audit = input_provenance_audit(provenance)
    parameters = parameter_audit()
    candidate = next(row for row in metrics if row["method"] == "O_C3_CLP1")
    passed = bool(input_audit["passed"] and prefix["passed"] and parameters["passed"]
                  and candidate["false_actions_T1_T2_T3_T12"] == 0 and candidate["early_actions"] == 0
                  and candidate["unknown_rate"] <= 0.05 and candidate["strong_loss_correct_in_window"] >= 35
                  and candidate["T4_T5_outcome_conditioned_correct"] == 12
                  and candidate["correct"] >= 71 and candidate["latency_p90_s"] is not None
                  and candidate["latency_p90_s"] <= 0.20)
    candidate["confirmation_status"] = "CONFIRMATION_PASS" if passed else "CONFIRMATION_FAIL"
    ranking = {"schema": "l2rar2_r18_candidate_ranking_v1",
               "selected_candidate_id": "O_C3_CLP1" if passed else None,
               "confirmation_run": True, "l3_entry_ready": passed, "l3_entry_allowed": False,
               "candidates": [row for row in metrics if row["method"] != "S_G_H"]}
    payload = {"schema": "l2rar2_r18_method_metrics_v1", "methods": metrics,
               "selected_candidate_id": ranking["selected_candidate_id"],
               "input_provenance": input_audit, "prefix_causality": prefix, "parameter_audit": parameters}
    write_csv(output_root / "per_event_decisions.csv", decisions)
    write_json(output_root / "method_metrics.json", payload)
    write_csv(output_root / "latency_summary.csv", metrics,
              ("method", "latency_median_s", "latency_p90_s", "strong_loss_correct_in_window"))
    write_csv(output_root / "phase_offset_metrics.csv", phase_rows)
    write_csv(output_root / "transient_dropout_metrics.csv", transient_rows)
    write_json(output_root / "input_provenance_audit.json", input_audit)
    write_json(output_root / "prefix_causality_audit.json", prefix)
    write_json(output_root / "parameter_audit.json", parameters)
    write_json(output_root / "candidate_ranking.json", ranking)
    recommendation = "Select O_C3_CLP1 for L3 preparation." if passed else "No online candidate passes the frozen R18 gate."
    (output_root / "candidate_recommendation.md").write_text("# Candidate recommendation\n\n" + recommendation + "\n", encoding="utf-8")
    return payload
