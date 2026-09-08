"""One-shot new-family confirmation with a consumption lock."""
from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from upgrade_v2.visual_refine_l2.dataset import collect_dataset
from upgrade_v2.visual_refine_l2.predicates import infer_dataset
from upgrade_v2.visual_refine_l2.execute_graph import execute_graphs
from upgrade_v2.visual_refine_l2.io import read_csv, read_json, read_jsonl

from .evaluate import _event_metrics, candidate_sequence


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def _load_probe_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _standard_items(manifest: Path, prediction_manifest: Path) -> list[dict[str, Any]]:
    data = {row["rollout_id"]: row for row in read_csv(manifest)}
    items = []
    for row in read_csv(prediction_manifest):
        truth = data[row["rollout_id"]]
        predictions = read_jsonl(Path(row["prediction_path"]))
        event_type = "held_object_loss_recovery_required" if truth["contact_loss"] == "True" else "missed_grasp_retry_required" if truth["missed_grasp"] == "True" else "none"
        action = "recover_object" if event_type.startswith("held") else "retry_grasp" if event_type.startswith("missed") else "none"
        decision_frame = next((i for i, p in enumerate(predictions) if p["predicates"].get("gripper_command_closed") == "true" and p["predicates"].get("contact_present") == "false"), None)
        items.append({"probe_id": row["rollout_id"], "root_family_id": row["root_family_id"], "stratum": truth["scenario"],
                      "observations": predictions, "reference": {"event_type": event_type, "action_class": action, "label_status": "reference_labeled", "history_complete": True, "decision_frame_index": decision_frame}})
    return items


def _candidate_rows(g2_rows: list[dict[str, Any]], items: list[dict[str, Any]], candidate_id: str, hold: int, loss: int) -> list[dict[str, Any]]:
    item_by_id = {item["probe_id"]: item for item in items}
    result = []
    for base in g2_rows:
        item = item_by_id[base["rollout_id"]]
        seq = candidate_sequence(item, candidate_id, hold, loss)
        predicted = seq["selected_action"] if seq["selected_action"] in {"retry_grasp", "recover_object", "needs_observation"} else base["predicted_branch"]
        row = dict(base)
        row["graph_id"] = candidate_id
        row["predicted_branch"] = predicted
        expected = base["expected_branch"]
        row["branch_correct"] = predicted == expected if expected != "observe_or_request_view" else predicted in {"request_second_view", "request_clarification"}
        row["precondition_correct"] = row["branch_correct"]
        row["failure_handled"] = (item["reference"]["event_type"] == "missed_grasp_retry_required" and predicted == "retry_grasp") or (item["reference"]["event_type"] == "held_object_loss_recovery_required" and predicted == "recover_object")
        row["recovery_handled"] = row["failure_handled"]
        row["ambiguous"] = bool(seq["any_conflict"])
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
    return {"rollouts": n, "families": len({r["root_family_id"] for r in rows}),
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
        a, b = _summary(rows_by_graph_family[(selected, family)]).get(metric), _summary(rows_by_graph_family[(baseline, family)]).get(metric)
        if a is not None and b is not None: deltas.append(float(a) - float(b))
    if not deltas: return {"comparison": f"{selected}-{baseline}", "metric": metric, "families": 0, "mean_effect": None, "bootstrap_low": None, "bootstrap_high": None}
    rng = np.random.default_rng(460008)
    values = np.asarray(deltas); sample = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
    return {"comparison": f"{selected}-{baseline}", "metric": metric, "families": len(values), "mean_effect": float(values.mean()), "bootstrap_low": float(np.quantile(sample, .025)), "bootstrap_high": float(np.quantile(sample, .975)), "bootstrap_resamples": 5000, "bootstrap_seed": 460008}


def _make_consumption(path: Path, lock: dict[str, Any], status: str, reason: str | None = None) -> dict[str, Any]:
    payload = {"schema": "pathgraph_l2ra_confirmation_consumption_v1", "status": status, "started_once": status in {"STARTED", "COMPLETE"},
               "selection_lock_sha256": lock["_selection_lock_sha256"], "selected_candidate_id": lock.get("selected_candidate_id"),
               "standard_family_count": 24, "standard_rollouts": 96, "challenge_family_count": 12, "challenge_rollouts": 48,
               "reason": reason, "api_calls": 0, "training_jobs": 0}
    _write_json(path, payload); return payload


def confirm(selection_lock: Path, protocol: dict[str, Any], resolved: dict[str, Any], run_root: Path, frozen_root: Path, workers: int = 4) -> dict[str, Any]:
    lock = read_json(selection_lock)
    lock["_selection_lock_sha256"] = _sha(selection_lock)
    consumption_path = run_root / "rounds/l2ra_4_fresh_confirmation/confirmation_consumption.json"
    existing = read_json(consumption_path) if consumption_path.is_file() else None
    if not lock.get("selected_candidate_id"):
        _make_consumption(consumption_path, lock, "BLOCKED", "development route did not select a candidate")
        return {"status": "BLOCKED", "reason": "development_not_ready", "standard_confirmation": "NOT_RUN", "challenge_confirmation": "NOT_RUN"}
    if existing and existing.get("status") == "COMPLETE":
        return {"status": "COMPLETE", "resumed": True}
    _make_consumption(consumption_path, lock, "STARTED")
    round_root = run_root / "rounds/l2ra_4_fresh_confirmation"
    data_root = run_root / "data/standard_confirmation"
    manifest = data_root / "rollout_manifest.csv"; family_split = data_root / "family_split.csv"
    if not manifest.exists():
        collect_dataset("fresh_confirmation", 24, 4, ["normal_pick_place", "already_satisfied_stable", "already_on_target_offcenter", "missed_grasp_then_retry", "slip_then_recover", "target_occupied", "distractor_object_ambiguity", "primary_view_occlusion"], 440000, 44100000, data_root / "rollouts", manifest, family_split, workers=workers, render_side=True)
    pred_root = data_root / "predicates"; pred_manifest = pred_root / "prediction_manifest.csv"
    if not pred_manifest.exists():
        infer_dataset(data_root / "rollouts", frozen_root / "locks/predicate_thresholds.json", pred_root, pred_manifest, camera="front")
    graph_paths = [frozen_root / f"graphs/{name}.json" for name in ("G0_coarse_direct", "G1_predicate_bound", "G2_evidence_refined")]
    execution_root = round_root / "standard_execution"
    execute_graphs(graph_paths, manifest, pred_manifest, None, None, execution_root, execution_root / "metrics.csv", None, execution_root / "per_family.csv")
    g2_rows = read_jsonl(execution_root / "G2_evidence_refined.jsonl")
    items = _standard_items(manifest, pred_manifest)
    selected = lock["selected_candidate_id"]; config = lock.get("selected_candidate_config") or {}
    hold, loss = int(config.get("hold_confirm_observations", 1)), int(config.get("loss_confirm_observations", 1))
    candidate_rows = _candidate_rows(g2_rows, items, selected, hold, loss)
    all_rows = {name: read_jsonl(execution_root / f"{name}.jsonl") for name in ("G0_coarse_direct", "G1_predicate_bound", "G2_evidence_refined")}
    all_rows[selected] = candidate_rows
    metric_rows = [{"graph_id": name, **_summary(rows)} for name, rows in sorted(all_rows.items())]
    _write_csv(round_root / "standard_confirmation.csv", metric_rows)
    by_graph_family = defaultdict(list)
    for graph, rows in all_rows.items():
        for row in rows: by_graph_family[(graph, row["root_family_id"])].append(row)
    effects = [_bootstrap(by_graph_family, selected, baseline, metric) for baseline in ("G0_coarse_direct", "G1_predicate_bound", "G2_evidence_refined") for metric in ("branch_accuracy", "goal_precision", "goal_recall", "graph_completion_coverage", "failure_recall", "recovery_recall")]
    _write_csv(round_root / "paired_family_effects.csv", effects)
    event_metrics, false_rows, delays = _event_metrics(items, selected, hold, loss)
    _write_csv(round_root / "event_type_metrics.csv", [event_metrics])
    _write_csv(round_root / "false_emergency_and_unknown.csv", false_rows)
    _write_csv(round_root / "decision_delay.csv", delays)
    # The challenge pool is explicitly an observation intervention until an
    # independent dynamic adapter exists; it is not silently treated as new physics.
    probe_path = run_root / "data/new_development/probe_records.jsonl"
    challenge_items = [item for item in _load_probe_records(probe_path) if item["stratum"] in {"touch_without_hold_then_loss", "commanded_release", "long_gap_after_loss", "history_or_visual_unavailable"}]
    challenge_metrics, challenge_false, challenge_delay = _event_metrics(challenge_items, selected, hold, loss)
    challenge_metrics["data_status"] = "OBSERVATION_INTERVENTION_NOT_INDEPENDENT_PHYSICAL_CONFIRMATION"
    _write_csv(round_root / "challenge_confirmation.csv", [challenge_metrics])
    _write_csv(round_root / "challenge_false_emergency_and_unknown.csv", challenge_false)
    _write_csv(round_root / "challenge_decision_delay.csv", challenge_delay)
    _make_consumption(consumption_path, lock, "COMPLETE")
    return {"status": "COMPLETE", "standard_metrics": metric_rows, "event_metrics": event_metrics, "challenge_metrics": challenge_metrics, "selected_candidate": selected}
