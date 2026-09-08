"""One-shot new-family confirmation with a verified selection lock."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from upgrade_v2.visual_refine_l2.dataset import collect_dataset
from upgrade_v2.visual_refine_l2.predicates import infer_dataset
from upgrade_v2.visual_refine_l2.execute_graph import execute_graphs
from upgrade_v2.visual_refine_l2.io import read_csv, read_json, read_jsonl

from .evaluate import _event_metrics, candidate_sequence
from .probes import collect_challenge_probes
from .reference_events import extract_reference_events


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else ["status", "reason"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_probe_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _validate_lock(selection_lock: Path, protocol_path: Path, lock: dict[str, Any]) -> None:
    if lock.get("protocol_sha256") != _sha(protocol_path):
        raise RuntimeError("confirmation refused: protocol hash differs from selection lock")
    for item in lock.get("locked_files", []):
        path = Path(item["path"])
        if not path.is_file() or _sha(path) != item["sha256"]:
            raise RuntimeError(f"confirmation refused: locked input changed: {item['logical_id']}")
    if lock.get("status") not in {"LOCKED_BEFORE_NEW_CONFIRMATION", "DEVELOPMENT_NOT_READY"}:
        raise RuntimeError(f"confirmation refused: invalid selection lock status {lock.get('status')}")


def _standard_items(manifest: Path, prediction_manifest: Path) -> list[dict[str, Any]]:
    data = {row["rollout_id"]: row for row in read_csv(manifest)}
    items = []
    for row in read_csv(prediction_manifest):
        truth = data[row["rollout_id"]]
        predictions = read_jsonl(Path(row["prediction_path"]))
        rollout = {
            "rollout_id": truth["rollout_id"],
            "root_family_id": truth["root_family_id"],
            "rollout_path": truth["path"],
        }
        event_records = extract_reference_events(rollout)
        primary = event_records[0] if event_records else None
        event_type = primary["reference_event_type"] if primary else "none"
        action = primary["reference_action_class"] if primary else "none"
        decision_frame = primary["decision_frame_index"] if primary else None
        items.append({"probe_id": row["rollout_id"], "root_family_id": row["root_family_id"], "stratum": truth["scenario"],
                      "observations": predictions, "reference": {"event_type": event_type, "action_class": action,
                      "label_status": primary["reference_label_status"] if primary else "no_decidable_event",
                      "observable_at_decision": bool(primary and primary["reference_observable_at_decision"]),
                      "history_complete": bool(primary and primary["reference_history_complete"]) if primary else True,
                      "decision_frame_index": decision_frame, "reference_events": event_records}})
    return items


def _candidate_rows(g2_rows: list[dict[str, Any]], items: list[dict[str, Any]], candidate_id: str, hold: int, loss: int) -> list[dict[str, Any]]:
    item_by_id = {item["probe_id"]: item for item in items}
    result = []
    for base in g2_rows:
        item = item_by_id[base["rollout_id"]]
        sequence = candidate_sequence(item, candidate_id, hold, loss)
        predicted = sequence["selected_action"] if sequence["selected_action"] in {"retry_grasp", "recover_object", "needs_observation"} else base["predicted_branch"]
        row = dict(base)
        row["graph_id"] = candidate_id
        row["predicted_branch"] = predicted
        expected = base["expected_branch"]
        row["branch_correct"] = predicted == expected if expected != "observe_or_request_view" else predicted in {"request_second_view", "request_clarification"}
        row["precondition_correct"] = row["branch_correct"]
        row["failure_handled"] = (item["reference"]["event_type"] == "missed_grasp_retry_required" and predicted == "retry_grasp") or (item["reference"]["event_type"] == "held_object_loss_recovery_required" and predicted == "recover_object")
        row["recovery_handled"] = row["failure_handled"]
        row["ambiguous"] = bool(sequence["any_conflict"])
        result.append(row)
    return result


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    goal_tp = sum(bool(r["pred_goal"]) and bool(r["true_goal"]) for r in rows)
    goal_fp = sum(bool(r["pred_goal"]) and not bool(r["true_goal"]) for r in rows)
    goal_fn = sum(not bool(r["pred_goal"]) and bool(r["true_goal"]) for r in rows)
    failures = [r for r in rows if r["failure_expected"]]
    recoveries = [r for r in rows if r["recovery_expected"]]
    already = [r for r in rows if r["already_satisfied"]]
    return {"rollouts": n, "families": len({r["root_family_id"] for r in rows},),
            "branch_accuracy": sum(bool(r["branch_correct"]) for r in rows) / n if n else None,
            "goal_precision": goal_tp / (goal_tp + goal_fp) if goal_tp + goal_fp else None,
            "goal_recall": goal_tp / (goal_tp + goal_fn) if goal_tp + goal_fn else None,
            "false_ready_rate": goal_fp / (goal_tp + goal_fp) if goal_tp + goal_fp else None,
            "unnecessary_manipulation_rate": sum(r["predicted_branch"] != "stop_no_action" for r in already) / len(already) if already else None,
            "failure_denominator": len(failures), "failure_recall": sum(bool(r["failure_handled"]) for r in failures) / len(failures) if failures else None,
            "recovery_denominator": len(recoveries), "recovery_recall": sum(bool(r["recovery_handled"]) for r in recoveries) / len(recoveries) if recoveries else None,
            "unknown_rate": sum(float(r["unknown_rate"]) for r in rows) / n if n else None,
            "ambiguous_edge_rate": sum(bool(r["ambiguous"]) for r in rows) / n if n else None,
            "graph_completion_coverage": sum(float(r["coverage"]) for r in rows) / n if n else None}


def _bootstrap(rows_by_graph_family: dict[tuple[str, str], list[dict[str, Any]]], selected: str, baseline: str, metric: str) -> dict[str, Any]:
    import numpy as np
    families = sorted({family for graph, family in rows_by_graph_family if graph == selected} & {family for graph, family in rows_by_graph_family if graph == baseline})
    deltas = []
    for family in families:
        a = _summary(rows_by_graph_family[(selected, family)]).get(metric)
        b = _summary(rows_by_graph_family[(baseline, family)]).get(metric)
        if a is not None and b is not None:
            deltas.append(float(a) - float(b))
    if not deltas:
        return {"comparison": f"{selected}-{baseline}", "metric": metric, "families": 0, "mean_effect": None, "bootstrap_low": None, "bootstrap_high": None}
    rng = np.random.default_rng(460008)
    values = np.asarray(deltas)
    sample = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
    return {"comparison": f"{selected}-{baseline}", "metric": metric, "families": len(values), "mean_effect": float(values.mean()), "bootstrap_low": float(np.quantile(sample, .025)), "bootstrap_high": float(np.quantile(sample, .975)), "bootstrap_resamples": 5000, "bootstrap_seed": 460008}


def _scenario_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("scenario", "unknown"))].append(row)
    return {key: sum(bool(row["branch_correct"]) for row in value) / len(value) for key, value in grouped.items() if value}


def _confirmation_gate(protocol: dict[str, Any], metric_rows: list[dict[str, Any]], all_rows: dict[str, list[dict[str, Any]]], event_metrics: dict[str, Any], challenge_metrics: dict[str, Any], challenge_false: list[dict[str, Any]]) -> dict[str, Any]:
    thresholds = protocol["old_go_thresholds_on_standard"]
    new = protocol["new_diagnostic_thresholds"]
    selected = next(row for row in metric_rows if row["graph_id"] not in {"G0_coarse_direct", "G1_predicate_bound", "G2_evidence_refined"})
    g0 = next(row for row in metric_rows if row["graph_id"] == "G0_coarse_direct")
    g1 = next(row for row in metric_rows if row["graph_id"] == "G1_predicate_bound")
    g2 = next(row for row in metric_rows if row["graph_id"] == "G2_evidence_refined")
    def at_least(value: Any, minimum: float) -> bool: return value is not None and float(value) >= minimum
    def at_most(value: Any, maximum: float) -> bool: return value is not None and float(value) <= maximum
    scenario_selected = _scenario_accuracy(all_rows[selected["graph_id"]])
    scenario_g1 = _scenario_accuracy(all_rows["G1_predicate_bound"])
    noninferior = sum(name in scenario_g1 and scenario_selected.get(name, -1) + 1e-12 >= scenario_g1[name] - .02 for name in scenario_g1)
    false_by = {row["stratum"]: row for row in challenge_false}
    old_checks = {
        "branch_accuracy": at_least(selected["branch_accuracy"], thresholds["branch_accuracy_min"]),
        "goal_precision": at_least(selected["goal_precision"], thresholds["goal_precision_min"]),
        "false_ready_rate": at_most(selected["false_ready_rate"], thresholds["false_ready_rate_max"]),
        "unnecessary_manipulation_rate": at_most(selected["unnecessary_manipulation_rate"], thresholds["unnecessary_manipulation_rate_max"]),
        "failure_denominator": selected["failure_denominator"] >= thresholds["failure_denominator_min"],
        "failure_recall": at_least(selected["failure_recall"], thresholds["failure_recall_min"]),
        "recovery_denominator": selected["recovery_denominator"] >= thresholds["recovery_denominator_min"],
        "recovery_recall": at_least(selected["recovery_recall"], thresholds["recovery_recall_min"]),
        "unknown_rate": at_most(selected["unknown_rate"], thresholds["unknown_rate_max"]),
        "ambiguous_edge_rate": at_most(selected["ambiguous_edge_rate"], thresholds["ambiguous_edge_rate_max"]),
        "coverage": at_least(selected["graph_completion_coverage"], thresholds["graph_completion_coverage_min"]),
        "selected_minus_g0_branch": selected["branch_accuracy"] - g0["branch_accuracy"] >= thresholds["selected_minus_g0_branch_min"],
        "selected_minus_g1_branch": selected["branch_accuracy"] - g1["branch_accuracy"] >= thresholds["selected_minus_g1_branch_min"],
        "scenario_noninferior": noninferior >= thresholds["noninferior_scenarios_min"],
    }
    compat_checks = {
        "g2_branch_noninferior": selected["branch_accuracy"] >= g2["branch_accuracy"] - .02,
        "g2_goal_precision_noninferior": selected["goal_precision"] is not None and g2["goal_precision"] is not None and selected["goal_precision"] >= g2["goal_precision"] - .02,
        "g2_goal_recall_noninferior": selected["goal_recall"] is not None and g2["goal_recall"] is not None and selected["goal_recall"] >= g2["goal_recall"] - .02,
        "g2_coverage_noninferior": selected["graph_completion_coverage"] >= g2["graph_completion_coverage"] - .02,
        "g2_failure_noninferior": selected["failure_recall"] is not None and g2["failure_recall"] is not None and selected["failure_recall"] >= g2["failure_recall"] - .05,
        "g2_recovery_noninferior": selected["recovery_recall"] is not None and g2["recovery_recall"] is not None and selected["recovery_recall"] >= g2["recovery_recall"] - .05,
    }
    standard_checks = {
        "effective_conflict_rollout": at_most(event_metrics.get("effective_conflict_rollout_rate"), new["standard_any_conflict_rollout_rate_max"]),
        "event_window_conflict": at_most(event_metrics.get("event_window_conflict_rate"), new["event_window_conflict_rate_max"]),
        "miss_event_support": event_metrics.get("miss_events", 0) >= 12 and event_metrics.get("miss_families", 0) >= 3,
        "loss_event_support": event_metrics.get("loss_events", 0) >= 12 and event_metrics.get("loss_families", 0) >= 3,
        "miss_type_recall": at_least(event_metrics.get("miss_type_recall"), new["event_type_recall_min"]),
        "loss_type_recall": at_least(event_metrics.get("loss_type_recall"), new["event_type_recall_min"]),
        "wrong_or_unknown": at_most(event_metrics.get("wrong_or_unknown_rate"), new["wrong_or_unknown_fraction_max"]),
    }
    challenge_checks = {
        "touch_false_emergency": at_most(false_by.get("touch_without_hold_then_loss", {}).get("false_emergency_rate"), new["false_emergency_rate_max"]),
        "release_false_emergency": at_most(false_by.get("commanded_release", {}).get("false_emergency_rate"), new["false_emergency_rate_max"]),
        "long_gap_recovery": at_least(challenge_metrics.get("loss_type_recall") if challenge_metrics.get("loss_type_recall") is not None else None, new["long_gap_recovery_recall_min"]),
        "history_unjustified_definite": at_most(challenge_metrics.get("unjustified_definite_rate"), new["unjustified_definite_rate_max"]),
        "delay_p95": at_most(challenge_metrics.get("delay_observation_steps_p95"), new["detection_delay_observation_steps_p95_max"]),
    }
    checks = {**{f"old_{key}": value for key, value in old_checks.items()}, **compat_checks, **{f"standard_{key}": value for key, value in standard_checks.items()}, **{f"challenge_{key}": value for key, value in challenge_checks.items()}}
    return {"schema": "pathgraph_l2ra_confirmation_gate_v1", "status": "CONFIRMATION_PASS" if all(checks.values()) else "CONFIRMATION_FAIL", "selected_candidate_id": selected["graph_id"], "checks": checks, "old_metrics": selected, "event_metrics": event_metrics, "challenge_metrics": challenge_metrics, "challenge_false_emergency": challenge_false, "scenario_noninferior_count": noninferior, "scenario_count": len(scenario_g1), "all_pass": all(checks.values())}


def _make_consumption(path: Path, lock: dict[str, Any], status: str, reason: str | None = None, confirmation_status: str | None = None) -> dict[str, Any]:
    payload = {"schema": "pathgraph_l2ra_confirmation_consumption_v2", "status": status, "started_once": status in {"STARTED", "COMPLETE"},
               "selection_lock_sha256": lock["_selection_lock_sha256"], "selection_lock_schema": lock.get("schema"), "selected_candidate_id": lock.get("selected_candidate_id"),
               "standard_family_count": 24, "standard_rollouts": 96, "challenge_family_count": 12, "challenge_rollouts": 48,
               "reason": reason, "confirmation_status": confirmation_status, "api_calls": 0, "training_jobs": 0}
    _write_json(path, payload)
    return payload


def confirm(selection_lock: Path, protocol: dict[str, Any], resolved: dict[str, Any], run_root: Path, frozen_root: Path, workers: int = 4) -> dict[str, Any]:
    lock = read_json(selection_lock)
    lock["_selection_lock_sha256"] = _sha(selection_lock)
    consumption_path = run_root / "rounds/l2ra_4_fresh_confirmation/confirmation_consumption.json"
    existing = read_json(consumption_path) if consumption_path.is_file() else None
    if existing and existing.get("selection_lock_sha256") != lock["_selection_lock_sha256"] and existing.get("started_once"):
        raise RuntimeError("confirmation refused: consumption lock hash differs")
    if not lock.get("selected_candidate_id"):
        if lock.get("protocol_path"):
            _validate_lock(selection_lock, Path(lock["protocol_path"]), lock)
        _make_consumption(consumption_path, lock, "BLOCKED", "development route did not select a candidate")
        return {"status": "BLOCKED", "reason": "development_not_ready", "standard_confirmation": "NOT_RUN", "challenge_confirmation": "NOT_RUN"}
    protocol_path = Path(lock["protocol_path"])
    _validate_lock(selection_lock, protocol_path, lock)
    if existing and existing.get("status") == "COMPLETE":
        gate = read_json(run_root / "rounds/l2ra_4_fresh_confirmation/confirmation_gate.json", {})
        return {"status": "COMPLETE", "resumed": True, "confirmation_status": gate.get("status")}
    _make_consumption(consumption_path, lock, "STARTED")
    round_root = run_root / "rounds/l2ra_4_fresh_confirmation"
    data_root = run_root / "data/standard_confirmation"
    manifest = data_root / "rollout_manifest.csv"
    family_split = data_root / "family_split.csv"
    if not manifest.exists():
        collect_dataset("fresh_confirmation", 24, 4, ["normal_pick_place", "already_satisfied_stable", "already_on_target_offcenter", "missed_grasp_then_retry", "slip_then_recover", "target_occupied", "distractor_object_ambiguity", "primary_view_occlusion"], 440000, 44100000, data_root / "rollouts", manifest, family_split, workers=workers, render_side=True)
    if len(read_csv(manifest)) != 96:
        raise RuntimeError("confirmation refused: standard confirmation manifest is not 96 rollouts")
    pred_root = data_root / "predicates"
    pred_manifest = pred_root / "prediction_manifest.csv"
    if not pred_manifest.exists():
        infer_dataset(data_root / "rollouts", frozen_root / "locks/predicate_thresholds.json", pred_root, pred_manifest, camera="front")
    graph_paths = [frozen_root / f"graphs/{name}.json" for name in ("G0_coarse_direct", "G1_predicate_bound", "G2_evidence_refined")]
    execution_root = round_root / "standard_execution"
    execute_graphs(graph_paths, manifest, pred_manifest, None, None, execution_root, execution_root / "metrics.csv", None, execution_root / "per_family.csv")
    g2_rows = read_jsonl(execution_root / "G2_evidence_refined.jsonl")
    items = _standard_items(manifest, pred_manifest)
    selected = lock["selected_candidate_id"]
    config = lock.get("selected_candidate_config") or {}
    hold, loss = int(config.get("hold_confirm_observations", 1)), int(config.get("loss_confirm_observations", 1))
    candidate_rows = _candidate_rows(g2_rows, items, selected, hold, loss)
    all_rows = {name: read_jsonl(execution_root / f"{name}.jsonl") for name in ("G0_coarse_direct", "G1_predicate_bound", "G2_evidence_refined")}
    all_rows[selected] = candidate_rows
    metric_rows = [{"graph_id": name, **_summary(rows)} for name, rows in sorted(all_rows.items())]
    _write_csv(round_root / "standard_confirmation.csv", metric_rows)
    by_graph_family = defaultdict(list)
    for graph, rows in all_rows.items():
        for row in rows:
            by_graph_family[(graph, row["root_family_id"])].append(row)
    effects = [_bootstrap(by_graph_family, selected, baseline, metric) for baseline in ("G0_coarse_direct", "G1_predicate_bound", "G2_evidence_refined") for metric in ("branch_accuracy", "goal_precision", "goal_recall", "graph_completion_coverage", "failure_recall", "recovery_recall")]
    _write_csv(round_root / "paired_family_effects.csv", effects)
    event_metrics, false_rows, delays = _event_metrics(items, selected, hold, loss)
    _write_csv(round_root / "event_type_metrics.csv", [event_metrics])
    _write_csv(round_root / "false_emergency_and_unknown.csv", false_rows)
    _write_csv(round_root / "decision_delay.csv", delays)
    challenge_root = run_root / "data/challenge_confirmation"
    challenge_result = collect_challenge_probes(resolved, protocol, challenge_root, workers)
    challenge_items = _load_probe_records(challenge_root / "probe_records.jsonl")
    challenge_metrics, challenge_false, challenge_delay = _event_metrics(challenge_items, selected, hold, loss)
    challenge_metrics["data_status"] = "NEW_DYNAMIC_MUJOCO_CHALLENGE"
    _write_csv(round_root / "challenge_confirmation.csv", [challenge_metrics])
    _write_csv(round_root / "challenge_false_emergency_and_unknown.csv", challenge_false)
    _write_csv(round_root / "challenge_decision_delay.csv", challenge_delay)
    gate = _confirmation_gate(protocol, metric_rows, all_rows, event_metrics, challenge_metrics, challenge_false)
    _write_json(round_root / "confirmation_gate.json", gate)
    _make_consumption(consumption_path, lock, "COMPLETE", confirmation_status=gate["status"])
    return {"status": "COMPLETE", "confirmation_status": gate["status"], "standard_metrics": metric_rows, "event_metrics": event_metrics, "challenge_metrics": challenge_metrics, "challenge_collection": challenge_result, "selected_candidate": selected}
