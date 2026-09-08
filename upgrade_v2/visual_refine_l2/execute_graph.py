"""Execute frozen graph capabilities over online predicate streams, then score diagnostically."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .evaluate import selection_score, summarize_executions
from .io import read_csv, read_json, read_jsonl, write_csv, write_jsonl
from .predicate_dsl import TRUE, evaluate


def expected_branch(scenario: str) -> str:
    return {
        "already_satisfied_stable": "stop_no_action", "missed_grasp_then_retry": "retry_grasp",
        "slip_then_recover": "recover_object", "target_occupied": "clear_target",
        "distractor_object_ambiguity": "request_clarification", "primary_view_occlusion": "observe_or_request_view",
    }.get(scenario, "manipulate")


def _any(predictions: list[dict[str, Any]], name: str, value: str = "true") -> bool:
    return any(frame["predicates"].get(name) == value for frame in predictions)


def _graph_edges(graph: dict[str, Any]) -> list[dict[str, Any]]:
    return graph.get("edges", graph.get("topology", {}).get("edges", []))


def _edge_truth(graph: dict[str, Any], predictions: list[dict[str, Any]]) -> list[dict[str, str]]:
    history = [frame["predicates"] for frame in predictions]
    return [
        {edge["id"]: evaluate(edge.get("condition", "UNKNOWN"), history, index) for edge in _graph_edges(graph)}
        for index in range(len(history))
    ]


def predict_branch(graph: dict[str, Any], predictions: list[dict[str, Any]], second_view_queried: bool) -> tuple[str, bool]:
    edges = _graph_edges(graph)
    truths = _edge_truth(graph, predictions)
    early_limit = max(2, len(predictions) // 3)
    by_action: dict[str, list[int]] = {}
    for index, values in enumerate(truths):
        for edge in edges:
            if values[edge["id"]] == TRUE:
                by_action.setdefault(edge["action"], []).append(index)
    if by_action.get("stop_no_action") and min(by_action["stop_no_action"]) < early_limit:
        return "stop_no_action", False
    if by_action.get("clear_target") and min(by_action["clear_target"]) < early_limit:
        return "clear_target", False
    event_actions = [action for action in ("retry_grasp", "recover_object") if by_action.get(action)]
    if event_actions:
        latest = max(max(by_action[action]) for action in event_actions)
        active = [action for action in event_actions if latest in by_action[action]]
        selected = "recover_object" if "recover_object" in active else active[0]
        return selected, len(active) > 1
    if graph.get("active_second_view") and second_view_queried:
        return "request_second_view", False
    if by_action.get("request_clarification"):
        return "request_clarification", False
    if by_action.get("observe_scene") and min(by_action["observe_scene"]) < early_limit:
        return "observe_scene", False
    return "manipulate", False


ACTION_TO_GRAPH = {
    "observe_scene": "observe_scene",
    "approach_object": "grasp_object",
    "close_gripper": "grasp_object",
    "lift": "grasp_object",
    "transport_to_target": "transport_object",
    "align": "align_with_target",
    "lower": "place_object",
    "open_gripper": "release_object",
    "verify": "verify_goal_relation",
    "retry": "retry_grasp",
    "recover": "recover_object",
    "clear_target": "clear_target",
    "stop": "stop_no_action",
}


def graph_coverage(graph: dict[str, Any], predictions: list[dict[str, Any]], actions: list[str]) -> float:
    edges = _graph_edges(graph)
    edge_actions = {edge["action"] for edge in edges}
    if "fixed_manipulation_path" in graph.get("capabilities", []):
        edge_actions.update({"observe_scene", "grasp_object", "transport_object", "align_with_target", "place_object", "release_object", "verify_goal_relation"})
    truths = _edge_truth(graph, predictions)
    explained = 0
    for index, action in enumerate(actions):
        canonical = ACTION_TO_GRAPH.get(action)
        condition_true = any(
            edge["action"] == canonical and truths[index].get(edge["id"]) == TRUE
            for edge in edges
        )
        fixed_path = canonical in edge_actions and canonical not in {"retry_grasp", "recover_object", "clear_target", "stop_no_action"}
        explained += int(condition_true or fixed_path)
    return explained / len(actions) if actions else 0.0


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
            action_rows = read_csv(Path(truth["path"]) / "actions.csv")
            actions = [row["action"] for row in action_rows]
            if len(actions) != len(frames):
                raise ValueError(f"action/predicate frame mismatch: {truth['rollout_id']}")
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
            coverage = graph_coverage(graph, frames, actions)
            row = {
                "graph_id": graph["graph_id"], "rollout_id": truth["rollout_id"], "root_family_id": family_id,
                "predicted_branch": predicted, "expected_branch": expected, "branch_correct": branch_ok,
                "pred_goal": pred_goal, "true_goal": true_goal, "state_ready_correct": pred_goal == true_goal,
                "precondition_correct": branch_ok, "effect_correct": pred_goal == true_goal,
                "failure_expected": failure_expected, "failure_handled": failure_handled,
                "recovery_expected": recovery_expected, "recovery_handled": recovery_handled,
                "already_satisfied": truth["scenario"] == "already_satisfied_stable", "unknown_rate": unknown_rate,
                "ambiguous": ambiguous, "coverage": coverage, "second_view_queried": second_view,
                "front_unknown_frames": int(prediction_row.get("front_unknown_frames") or 0),
                "resolved_unknown_frames": int(prediction_row.get("resolved_unknown_frames") or 0),
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
