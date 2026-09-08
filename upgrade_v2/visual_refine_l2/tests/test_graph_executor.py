import json

import pytest

from upgrade_v2.visual_refine_l2.confirm import _unique_locked_graphs
from upgrade_v2.visual_refine_l2.execute_graph import predict_branch
from upgrade_v2.visual_refine_l2.io import sha256_file


def frames(**values: str):
    defaults = {"goal_verified": "false", "target_occupied": "false", "grasp_failed_observed": "false", "slip_observed": "false", "visual_unknown": "false"}
    defaults.update(values)
    return [{"predicates": defaults} for _ in range(3)]


def test_refined_graph_selects_failure_and_goal_branches() -> None:
    graph = {"capabilities": ["goal_verified_stop", "missed_grasp_retry", "contact_loss_recovery"], "active_second_view": False}
    assert predict_branch(graph, frames(goal_verified="true"), False)[0] == "stop_no_action"
    assert predict_branch(graph, frames(grasp_failed_observed="true"), False)[0] == "retry_grasp"
    assert predict_branch(graph, frames(slip_observed="true"), False)[0] == "recover_object"


def test_active_view_query_remains_visible_after_side_view_resolves_unknown() -> None:
    graph = {"capabilities": ["visual_unknown_clarification"], "active_second_view": True}
    resolved_side_view = frames(visual_unknown="false")
    assert predict_branch(graph, resolved_side_view, True)[0] == "request_second_view"


def test_confirmation_deduplicates_same_locked_graph_id(tmp_path) -> None:
    graph = tmp_path / "g1.json"
    graph.write_text(json.dumps({"graph_id": "G1_predicate_bound"}), encoding="utf-8")
    lock = tmp_path / "selection.json"
    lock.write_text(json.dumps({"graph_sha256_by_id": {"G1_predicate_bound": sha256_file(graph)}}), encoding="utf-8")
    assert _unique_locked_graphs([graph, graph], lock) == [graph]

    graph.write_text(json.dumps({"graph_id": "G1_predicate_bound", "changed": True}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="mismatched"):
        _unique_locked_graphs([graph], lock)
