from upgrade_v2.visual_refine_l2.execute_graph import predict_branch


def frames(**values: str):
    defaults = {"goal_verified": "false", "target_occupied": "false", "grasp_failed_observed": "false", "slip_observed": "false", "visual_unknown": "false"}
    defaults.update(values)
    return [{"predicates": defaults} for _ in range(3)]


def test_refined_graph_selects_failure_and_goal_branches() -> None:
    graph = {"capabilities": ["goal_verified_stop", "missed_grasp_retry", "contact_loss_recovery"], "active_second_view": False}
    assert predict_branch(graph, frames(goal_verified="true"), False)[0] == "stop_no_action"
    assert predict_branch(graph, frames(grasp_failed_observed="true"), False)[0] == "retry_grasp"
    assert predict_branch(graph, frames(slip_observed="true"), False)[0] == "recover_object"
