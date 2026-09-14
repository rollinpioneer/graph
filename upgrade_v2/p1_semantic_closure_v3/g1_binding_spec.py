"""Explicit G1 reward-binding matrix. New views are proposals, not original dispatch."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from .util import sha256_file


def load_g1(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_spec(repo: Path) -> dict[str, Any]:
    g1_path = repo / "artifacts/pathgraph_sarm/upgrade_v2/visual_refine_l2_v1/final_v1/graphs/G1_predicate_bound.json"
    g1 = load_g1(g1_path)
    consumer = repo / "artifacts/pathgraph_sarm/upgrade_v2/p1_state_conditioned_reward_v1/continuation_v2/consumer_trace.json"
    controller = repo / "upgrade_v2/visual_refine_l2/scripted_controller.py"
    executor = repo / "upgrade_v2/visual_refine_l2/dynamic_simulator.py"
    engine = repo / "tools/stage5/lib/reward_engine.py"
    rows = [
        dict(source_graph_id="G1_predicate_bound", source_node_or_edge="already_done",
             source_origin="ORIGINAL_G1_EDGE", runtime_function="scripted_controller.action_program:stop / already_satisfied verify",
             current_state_guard="goal_verified", completion_evidence="oracle_timeline.goal_stable at current frame, not metadata.final_goal_stable",
             available_at_rule="current-frame predicate/oracle_timeline only", cost_rule="NONE_IN_G1", phi_rule="NONE_IN_G1",
             binding_status="ORIGINAL_GRAPH_NO_NUMERIC_REWARD", new_structural_extension="no",
             source_commit="de3e54310270eeb47e00a99745b3ea6987aa8795", source_path=str(g1_path.relative_to(repo)), source_lines="already_done"),
        dict(source_graph_id="G1_predicate_bound", source_node_or_edge="observe_unknown",
             source_origin="ORIGINAL_G1_EDGE", runtime_function="scripted_controller.action_program:observe_scene",
             current_state_guard="visual_unknown", completion_evidence="action observe_scene is command, not scene-verified state",
             available_at_rule="current frame", cost_rule="NONE_IN_G1", phi_rule="NONE_IN_G1",
             binding_status="ORIGINAL_GRAPH_NO_NUMERIC_REWARD", new_structural_extension="no",
             source_commit="de3e54310270eeb47e00a99745b3ea6987aa8795", source_path=str(g1_path.relative_to(repo)), source_lines="observe_unknown"),
        dict(source_graph_id="G1_predicate_bound", source_node_or_edge="manipulate",
             source_origin="ORIGINAL_G1_EDGE", runtime_function="scripted_controller.action_program:grasp_object via close_gripper",
             current_state_guard="object_visible AND target_visible AND NOT goal_verified",
             completion_evidence="contact_present and weld_state at current frame; action name is not completion",
             available_at_rule="current frame contact/weld", cost_rule="NONE_IN_G1", phi_rule="NONE_IN_G1",
             binding_status="ORIGINAL_GRAPH_NO_NUMERIC_REWARD", new_structural_extension="no",
             source_commit="de3e54310270eeb47e00a99745b3ea6987aa8795", source_path=str(g1_path.relative_to(repo)), source_lines="manipulate"),
        dict(source_graph_id="G1_predicate_bound", source_node_or_edge="transport_child",
             source_origin="TEMPLATE_SLOT_NOT_IN_G1", runtime_function="scripted_controller.action_program:lift/transport_to_target/align/lower",
             current_state_guard="contact/weld currently true", completion_evidence="in_transit node on legacy graph is a NEW reward view",
             available_at_rule="current frame", cost_rule="legacy D_G(in_transit)=2/6 under normalized contract",
             phi_rule="clip(1-d_current/max(d_entry,eps),0,1) only after in_transit entry; proposal",
             binding_status="RUNTIME_REWARD_VIEW_PROPOSAL", new_structural_extension="yes: G1 has no transport edge",
             source_commit="de3e54310270eeb47e00a99745b3ea6987aa8795", source_path=str(controller.relative_to(repo)), source_lines="BASE"),
        dict(source_graph_id="G1_predicate_bound", source_node_or_edge="release_child",
             source_origin="TEMPLATE_SLOT_NOT_IN_G1", runtime_function="scripted_controller.action_program:open_gripper",
             current_state_guard="gripper open and object near target", completion_evidence="placed/success require goal_stable, not the open action itself",
             available_at_rule="current frame goal_stable", cost_rule="legacy placed/success costs", phi_rule="phi not used on node change",
             binding_status="RUNTIME_REWARD_VIEW_PROPOSAL", new_structural_extension="yes: G1 has no release edge",
             source_commit="de3e54310270eeb47e00a99745b3ea6987aa8795", source_path=str(controller.relative_to(repo)), source_lines="open_gripper"),
        dict(source_graph_id="G1_predicate_bound", source_node_or_edge="recovery_child",
             source_origin="TEMPLATE_SLOT_NOT_IN_G1", runtime_function="scripted_controller.action_program:retry/recover",
             current_state_guard="missed_grasp or contact_loss observed on current prefix",
             completion_evidence="recovery_achieved event or contact re-established; G1 has no failure/recovery edge",
             available_at_rule="events.jsonl observable_online at event time", cost_rule="legacy dropped/recovery costs",
             phi_rule="not defined on G1",
             binding_status="STRUCTURAL_EXTENSION_REQUIRED", new_structural_extension="yes: failure/recovery absent from G1",
             source_commit="de3e54310270eeb47e00a99745b3ea6987aa8795", source_path=str(controller.relative_to(repo)), source_lines="missed_grasp_then_retry/slip_then_recover"),
    ]
    unresolved = [r for r in rows if r["binding_status"] != "ORIGINAL_GRAPH_NO_NUMERIC_REWARD"]
    return dict(
        schema="p1_v3_g1_binding_spec_v1",
        graph_id=g1["graph_id"],
        n_nodes=len(g1["nodes"]),
        n_edges=len(g1["edges"]),
        numeric_reward_or_cost=g1.get("numeric_reward_or_cost"),
        source_sha256=sha256_file(g1_path),
        consumer_trace=str(consumer.relative_to(repo)),
        executor=str(executor.relative_to(repo)),
        reward_engine=str(engine.relative_to(repo)),
        actual_graph_to_controller_dispatch=False,
        existing_mapping="Three G1 edges only: already_done, observe_unknown, manipulate. Used for branch/coverage diagnostics.",
        new_binding_proposal="Legacy transport_recovery graph can host a reward view of scripted pick-place, but that view is new.",
        structural_extension_needed=["failure/recovery edges", "transport/release children", "numeric cost/phi on G1"],
        rows=rows,
        unresolved=unresolved,
        confirmation_passed=False,
    )