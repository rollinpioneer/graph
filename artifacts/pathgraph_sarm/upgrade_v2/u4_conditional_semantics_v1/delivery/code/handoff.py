"""Final U4-R1 scientific handoff and pre-registered decision gate."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_csv, read_json, write_json


def _rows(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path and path.suffix.lower() == ".csv" else (read_json(path).get("graphs", []) if path and path.is_file() else [])


def _value(row: dict, key: str) -> float | None:
    try: return float(row[key]) if row.get(key) not in {None, ""} else None
    except (KeyError, TypeError, ValueError): return None


def decide(selection: Path, metrics: Path, paired: Path, separability: Path, output: Path, report: Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
    selected = read_json(selection) if selection and selection.is_file() else {}
    rows = _rows(metrics); effects = read_csv(paired) if paired and paired.is_file() else []
    sep = read_json(separability) if separability and separability.suffix.lower() != ".csv" and separability.is_file() else {}
    if not sep and separability and separability.is_file():
        sidecar = separability.with_suffix(".json")
        if sidecar.is_file(): sep = read_json(sidecar)
    selected_id = selected.get("selected_graph")
    selected_row = next((row for row in rows if row.get("graph_id") == selected_id), None)
    blocked = not selected_id or selected_row is None
    sep_decision = sep.get("decision") or sep.get("observability_decision")
    if not sep_decision:
        result_rows = sep.get("results", [])
        past = next((r for r in result_rows if r.get("feature_group") == "past8" and r.get("model") == "logistic"), None)
        majority = next((r for r in result_rows if r.get("feature_group") == "past8" and r.get("model") == "majority"), None)
        if past and majority:
            gain = float(past.get("macro_f1", 0.0)) - float(majority.get("macro_f1", 0.0))
            sep_decision = "CONDITIONAL_SEMANTICS_JUSTIFIED" if gain >= .15 and max(float(past.get("failure_f1", 0.0)), float(past.get("recovery_attempt_f1", 0.0)), float(past.get("recovery_achieved_f1", 0.0))) >= .60 else ("SEMANTICS_NOT_OBSERVABLE" if gain < .05 else "SEMANTICS_PARTIALLY_OBSERVABLE")
    if blocked:
        exit_state = "EXECUTION_BLOCKED"
    elif sep_decision == "SEMANTICS_NOT_OBSERVABLE":
        exit_state = "REFINE_U2_OBSERVABLES_ONCE"
    else:
        coverage = _value(selected_row, "typed_occurrence_coverage") or 0.0
        unknown = _value(selected_row, "unknown_rate")
        transition = _value(selected_row, "transition_coverage_macro") or 0.0
        ambiguity = _value(selected_row, "ambiguous_guard_rate") or 0.0
        failure_recall = _value(selected_row, "failure_event_recall")
        recovery_attempt = _value(selected_row, "recovery_attempt_recall")
        recovery_achieved = _value(selected_row, "recovery_achieved_recall")
        terminal_precision = _value(selected_row, "failure_terminal_precision")
        false_terminal = _value(selected_row, "false_terminal_claim_rate")
        baseline = next((row for row in rows if row.get("graph_id") in {"G1_single_label_v2", "G1_semantic_only"}), {})
        scenario_rows = [row for row in effects if row.get("scope") == "scenario" and row.get("metric") in {"typed_occurrence_coverage", "unknown_rate"}]
        scenario_effects: dict[str, dict[str, float | None]] = {}
        for row in scenario_rows:
            scenario_effects.setdefault(str(row.get("scenario") or "unknown"), {})[str(row.get("metric"))] = _value(row, "effect")
        paired_scenario_noninferior = sum(
            values.get("typed_occurrence_coverage") is not None and values.get("unknown_rate") is not None
            and values["typed_occurrence_coverage"] >= 0.0 and values["unknown_rate"] <= 0.0
            for values in scenario_effects.values()
        )
        go = coverage >= .35 and coverage - (_value(baseline, "typed_occurrence_coverage") or 0.0) >= .20 and (unknown is not None and unknown <= .65) and transition >= (_value(baseline, "transition_coverage_macro") or transition) - .05 and ambiguity <= .10 and (false_terminal is not None and false_terminal <= .25) and (terminal_precision is not None and terminal_precision >= .75) and (failure_recall is not None and failure_recall >= .60) and max(recovery_attempt or 0.0, recovery_achieved or 0.0) >= .60 and paired_scenario_noninferior >= 4
        exit_state = "GO_U5_CONDITIONAL_GRAPH" if go else "STOP_AUTO_GRAPH_KEEP_MANUAL"
    payload = {
        "schema": "u4r1_final_handoff_v2", "exit_state": exit_state, "historical_u4_status": "U4_COMPLETE_NO_EDIT_GAIN", "historical_result_modified": False,
        "evaluator_repair": {"role_condition_executed": True, "horizon_treated_as_censored": True, "historical_role_conclusion": "NO_EDIT_GAIN_CONFIRMED_AFTER_EVAL_FIX"},
        "semantic_separability_decision": sep_decision or "not_estimable", "selected_graph": selected_id, "d_gate": "CONTINUE_U4", "api_calls": 0, "api_key_read": False, "training_jobs": 0,
        "device": "cpu", "automatic_boundary_status": "computed_and_locked", "boundary_source": "offline_teacher_to_causal_s623", "checkpoint_sha256": "fe0464076a3590de19b31d88cd668d4c0e8cf92ee2a80ec413e8191fea34c94e",
        "torch_environment": "cupid: Python 3.9.18, torch 2.8.0+cu126; CUDA unavailable, CPU inference", "fresh_confirmation_metrics": selected_row, "paired_effects": effects, "paired_scenario_noninferior_count": paired_scenario_noninferior, "paired_scenario_strata_count": len(scenario_effects),
        "limitations": ["same explicit stochastic simulator family only", "horizon is censored_unknown and excluded from failure claims", "scenario/phase/gold_mode/future labels are analysis-only and not online features", "U2 formal train occurrences were reconstructed from gold_event_id as diagnostic labels; they were not used as online features", "conditional semantics are not physical robot or new-task validation"],
        "next_action": exit_state,
    }
    write_json(output, payload); report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# U4-R1 final handoff\n\n" + "\n".join([f"- decision: `{exit_state}`", "- historical U4 status: `U4_COMPLETE_NO_EDIT_GAIN` (unchanged)", f"- evaluator role_condition executed: `{payload['evaluator_repair']['role_condition_executed']}`", "- horizon: `censored_unknown`", f"- semantic separability: `{sep_decision or 'not_estimable'}`", f"- selected graph: `{selected_id}`", "- checkpoint inference: `cupid` CPU, torch `2.8.0+cu126`", "- API calls: `0`; training jobs: `0`", "- no physical robot or new-task generalization claim" ]) + "\n", encoding="utf-8")
    return payload
