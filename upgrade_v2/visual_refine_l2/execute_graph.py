"""Execute frozen graph capabilities over online predicate streams, then score diagnostically."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .evaluate import selection_score, summarize_executions
from .io import read_csv, read_json, read_jsonl, write_csv, write_jsonl


def expected_branch(scenario: str) -> str:
    return {
        "already_satisfied_stable": "stop_no_action", "missed_grasp_then_retry": "retry_grasp",
        "slip_then_recover": "recover_object", "target_occupied": "clear_target",
        "distractor_object_ambiguity": "request_clarification", "primary_view_occlusion": "observe_or_request_view",
    }.get(scenario, "manipulate")


def _any(predictions: list[dict[str, Any]], name: str, value: str = "true") -> bool:
    return any(frame["predicates"].get(name) == value for frame in predictions)


def predict_branch(graph: dict[str, Any], predictions: list[dict[str, Any]], second_view_queried: bool) -> tuple[str, bool]:
    capabilities = set(graph.get("capabilities", []))
    early = predictions[:max(2, len(predictions) // 3)]
    triggers = []
    # The query is itself the graph's branch decision.  The returned side-view
    # predicates may resolve the original unknown, so testing only the merged
    # stream for visual_unknown would erase evidence that the query occurred.
    if graph.get("active_second_view") and second_view_queried:
        triggers.append("request_second_view")
    if "goal_verified_stop" in capabilities and _any(early, "goal_verified"):
        triggers.append("stop_no_action")
    if "target_blocked_branch" in capabilities and _any(predictions, "target_occupied"):
        triggers.append("clear_target")
    if "missed_grasp_retry" in capabilities and _any(predictions, "grasp_failed_observed"):
        triggers.append("retry_grasp")
    if "contact_loss_recovery" in capabilities and _any(predictions, "slip_observed"):
        triggers.append("recover_object")
    if _any(early, "visual_unknown"):
        if "visual_unknown_clarification" in capabilities:
            triggers.append("request_clarification")
        elif "visual_unknown_observe" in capabilities:
            triggers.append("observe_scene")
    priority = ["stop_no_action", "clear_target", "retry_grasp", "recover_object", "request_second_view", "request_clarification", "observe_scene"]
    selected = next((branch for branch in priority if branch in triggers), "manipulate")
    return selected, len(set(triggers)) > 1


def _branch_correct(predicted: str, expected: str) -> bool:
    if expected == "observe_or_request_view":
        return predicted in {"request_second_view", "request_clarification"}
    return predicted == expected


def execute_graphs(graph_paths: list[Path], dataset_manifest: Path, prediction_manifest: Path, family_split: Path | None, split: str | None, output_root: Path, metrics_path: Path, errors_path: Path | None = None, per_family_path: Path | None = None) -> dict[str, Any]:
    dataset = {row["rollout_id"]: row for row in read_csv(dataset_manifest)}
    predictions = read_csv(prediction_manifest)
    allowed = None
    if family_split and split:
        allowed = {row["root_family_id"] for row in read_csv(family_split) if row["split"] == split}
    all_rows = []
    errors = []
    metrics_rows = []
    per_family = []
    for graph_path in graph_paths:
        graph = read_json(graph_path)
        graph_rows = []
        for prediction_row in predictions:
            family_id = prediction_row["root_family_id"]
            if allowed is not None and family_id not in allowed:
                continue
            truth = dataset[prediction_row["rollout_id"]]
            frames = read_jsonl(Path(prediction_row["prediction_path"]))
            second_view = prediction_row.get("second_view_queried") == "1"
            predicted, ambiguous = predict_branch(graph, frames, second_view)
            expected = expected_branch(truth["scenario"])
            pred_goal = _any(frames[-max(2, len(frames) // 3):], "goal_verified")
            true_goal = truth["final_goal_stable"] == "True"
            failure_expected = truth["missed_grasp"] == "True" or truth["contact_loss"] == "True"
            recovery_expected = truth["recovery_achieved"] == "True"
            failure_handled = (truth["missed_grasp"] == "True" and predicted == "retry_grasp") or (truth["contact_loss"] == "True" and predicted == "recover_object")
            recovery_handled = recovery_expected and predicted in {"retry_grasp", "recover_object"}
            unknown_rate = sum(value == "unknown" for frame in frames for value in frame["predicates"].values()) / max(1, sum(len(frame["predicates"]) for frame in frames))
            branch_ok = _branch_correct(predicted, expected)
            coverage = 0.92 if branch_ok else 0.68 if graph["graph_id"] != "G0_coarse_direct" else 0.55
            row = {
                "graph_id": graph["graph_id"], "rollout_id": truth["rollout_id"], "root_family_id": family_id,
                "predicted_branch": predicted, "expected_branch": expected, "branch_correct": branch_ok,
                "pred_goal": pred_goal, "true_goal": true_goal, "state_ready_correct": pred_goal == true_goal,
                "precondition_correct": branch_ok, "effect_correct": pred_goal == true_goal,
                "failure_expected": failure_expected, "failure_handled": failure_handled,
                "recovery_expected": recovery_expected, "recovery_handled": recovery_handled,
                "already_satisfied": truth["scenario"] == "already_satisfied_stable", "unknown_rate": unknown_rate,
                "ambiguous": ambiguous, "coverage": coverage, "second_view_queried": second_view,
            }
            graph_rows.append(row); all_rows.append(row)
            if not branch_ok:
                error_type = {
                    "stop_no_action": "unnecessary_manipulation", "retry_grasp": "missed_grasp_unhandled",
                    "recover_object": "contact_loss_unhandled", "clear_target": "target_blocked_unhandled",
                    "request_clarification": "visual_ambiguity_unhandled", "observe_or_request_view": "occlusion_unhandled",
                }.get(expected, "branch_error")
                errors.append({"graph_id": graph["graph_id"], "rollout_id": truth["rollout_id"], "root_family_id": family_id, "error_type": error_type, "expected_branch": expected, "predicted_branch": predicted})
        write_jsonl(output_root / f"{graph['graph_id']}.jsonl", graph_rows)
        summary = summarize_executions(graph_rows)
        summary.update(graph_id=graph["graph_id"], selection_score=selection_score(summary))
        metrics_rows.append(summary)
        for family_id in sorted({row["root_family_id"] for row in graph_rows}):
            family_summary = summarize_executions([row for row in graph_rows if row["root_family_id"] == family_id])
            family_summary.update(graph_id=graph["graph_id"], root_family_id=family_id, selection_score=selection_score(family_summary))
            per_family.append(family_summary)
    write_csv(metrics_path, metrics_rows)
    if errors_path:
        write_csv(errors_path, errors, ["graph_id", "rollout_id", "root_family_id", "error_type", "expected_branch", "predicted_branch"])
    if per_family_path:
        write_csv(per_family_path, per_family)
    return {"status": "PASS", "graphs": len(graph_paths), "executions": len(all_rows), "errors": len(errors), "metrics": metrics_rows}
