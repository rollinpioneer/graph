import json

import pytest

from upgrade_v2.visual_refine_l2.confirm import _unique_locked_graphs
from upgrade_v2.visual_refine_l2.execute_graph import graph_coverage, predict_branch
from upgrade_v2.visual_refine_l2.io import sha256_file, write_csv, write_json
from upgrade_v2.visual_refine_l2.confirm import decide_status
from upgrade_v2.visual_refine_l2.refine_graph import propose_refinements


def frames(**values: str):
    defaults = {"goal_verified": "false", "target_occupied": "false", "grasp_failed_observed": "false", "slip_observed": "false", "visual_unknown": "false"}
    defaults.update(values)
    return [{"predicates": dict(defaults)} for _ in range(3)]


def test_refined_graph_selects_failure_and_goal_branches() -> None:
    graph = {
        "active_second_view": False,
        "edges": [
            {"id": "goal", "action": "stop_no_action", "condition": "goal_verified"},
            {"id": "retry", "action": "retry_grasp", "condition": "grasp_failed_observed"},
            {"id": "recover", "action": "recover_object", "condition": "slip_observed"},
        ],
    }
    assert predict_branch(graph, frames(goal_verified="true"), False)[0] == "stop_no_action"
    assert predict_branch(graph, frames(grasp_failed_observed="true"), False)[0] == "retry_grasp"
    assert predict_branch(graph, frames(slip_observed="true"), False)[0] == "recover_object"


def test_active_view_query_remains_visible_after_side_view_resolves_unknown() -> None:
    graph = {"capabilities": ["visual_unknown_clarification"], "active_second_view": True}
    resolved_side_view = frames(visual_unknown="false")
    assert predict_branch(graph, resolved_side_view, True)[0] == "request_second_view"


def test_g1_persistent_unknown_remains_observation_branch() -> None:
    graph = {
        "edges": [
            {"id": "observe", "action": "observe_scene", "condition": "visual_unknown"},
            {
                "id": "manipulate",
                "action": "grasp_object",
                "condition": {"op": "AND", "args": ["object_visible", "target_visible"]},
            },
        ]
    }
    history = frames(visual_unknown="true", object_visible="true", target_visible="true")
    assert predict_branch(graph, history, False)[0] == "observe_scene"


def test_e7_alone_adds_unknown_node_and_splits_persistent_unknown(tmp_path) -> None:
    base = tmp_path / "g1.json"
    errors = tmp_path / "errors.csv"
    output = tmp_path / "g2.json"
    edit_log = tmp_path / "edits.csv"
    report = tmp_path / "report.md"
    write_json(base, {
        "graph_id": "G1_predicate_bound",
        "capabilities": ["fixed_manipulation_path"],
        "nodes": [{"id": "scene_unverified"}],
        "edges": [
            {"id": "observe_unknown", "src": "scene_unverified", "dst": "scene_unverified", "action": "observe_scene", "condition": "visual_unknown"},
            {"id": "manipulate", "src": "scene_unverified", "dst": "grasp_candidate", "action": "grasp_object", "condition": "object_visible"},
        ],
    })
    write_csv(errors, [
        {"graph_id": "G1_predicate_bound", "rollout_id": f"r{index}", "root_family_id": f"f{index}", "error_type": "visual_ambiguity_unhandled"}
        for index in range(3)
    ])
    propose_refinements(base, errors, {"E7"}, 1, 3, output, edit_log, report)
    graph = json.loads(output.read_text(encoding="utf-8"))
    assert any(node["id"] == "visual_unknown" for node in graph["nodes"])
    edges = {edge["id"]: edge for edge in graph["edges"]}
    assert edges["observe_unknown"]["dst"] == "visual_unknown"
    assert edges["observe_unknown"]["condition"]["op"] == "AND"
    assert edges["clarify_persistent_unknown"]["condition"]["op"] == "CONSECUTIVE"


def test_latest_failure_evidence_selects_recovery_over_retry() -> None:
    graph = {
        "edges": [
            {"id": "retry", "action": "retry_grasp", "condition": "grasp_failed_observed"},
            {"id": "recover", "action": "recover_object", "condition": "slip_observed"},
        ]
    }
    history = frames()
    history[1]["predicates"]["grasp_failed_observed"] = "true"
    history[2]["predicates"]["slip_observed"] = "true"
    assert predict_branch(graph, history, False) == ("recover_object", False)


def test_graph_coverage_is_observed_action_fraction() -> None:
    graph = {"capabilities": ["fixed_manipulation_path"], "edges": []}
    history = frames()
    assert graph_coverage(graph, history, ["observe_scene", "approach_object", "retry"]) == pytest.approx(2 / 3)


def test_active_multiview_go_requires_half_unknown_resolution(tmp_path) -> None:
    lock = tmp_path / "selection.json"
    confirmation = tmp_path / "confirmation.csv"
    effects = tmp_path / "effects.csv"
    scenarios = tmp_path / "scenarios.csv"
    write_json(lock, {"selected_graph_id": "G3_active_second_view"})
    metric = {
        "graph_id": "G3_active_second_view", "branch_accuracy": .9, "goal_precision": .9,
        "false_ready_rate": .0, "unnecessary_manipulation_rate": .0,
        "failure_denominator": 24, "failure_recall": .8, "recovery_denominator": 24,
        "recovery_recall": .8, "unknown_rate": .1, "ambiguous_edge_rate": .0,
        "graph_completion_coverage": .9, "second_view_query_rate": .2,
        "second_view_unknown_resolution_rate": .49,
    }
    write_csv(confirmation, [metric])
    write_csv(effects, [
        {"comparison": "G3_active_second_view-G0_coarse_direct", "mean_effect": .2},
        {"comparison": "G3_active_second_view-G1_predicate_bound", "mean_effect": .1},
    ])
    scenario_rows = []
    for index in range(8):
        scenario_rows.extend([
            {"graph_id": "G3_active_second_view", "scenario": str(index), "branch_accuracy": .9},
            {"graph_id": "G1_predicate_bound", "scenario": str(index), "branch_accuracy": .8},
        ])
    write_csv(scenarios, scenario_rows)
    status, reasons = decide_status(lock, confirmation, effects, scenarios)
    assert status == "L2R_PARTIAL_KEEP_COARSE_GRAPH"
    assert "second_view_unknown_resolution_rate>=0.50" in reasons[0]
    metric["second_view_unknown_resolution_rate"] = .5
    write_csv(confirmation, [metric])
    assert decide_status(lock, confirmation, effects, scenarios)[0] == "GO_L3_REWARD_GROUNDING_ACTIVE_MULTIVIEW"


def test_confirmation_deduplicates_same_locked_graph_id(tmp_path) -> None:
    graph = tmp_path / "g1.json"
    graph.write_text(json.dumps({"graph_id": "G1_predicate_bound"}), encoding="utf-8")
    lock = tmp_path / "selection.json"
    lock.write_text(json.dumps({"graph_sha256_by_id": {"G1_predicate_bound": sha256_file(graph)}}), encoding="utf-8")
    assert _unique_locked_graphs([graph, graph], lock) == [graph]

    graph.write_text(json.dumps({"graph_id": "G1_predicate_bound", "changed": True}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="mismatched"):
        _unique_locked_graphs([graph], lock)
