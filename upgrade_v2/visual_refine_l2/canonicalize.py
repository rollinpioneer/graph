"""Deterministic canonicalization of frozen natural-language L1V graphs."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from .io import read_json, write_csv, write_json, write_jsonl, write_report


ACTION_VOCABULARY = [
    "observe_scene", "approach_object", "grasp_object", "verify_grasp", "lift_object", "transport_object",
    "align_with_target", "place_object", "release_object", "verify_goal_relation", "retry_grasp", "recover_object",
    "clear_target", "request_second_view", "request_clarification", "stop_no_action",
]
STATE_VOCABULARY = [
    "scene_unverified", "task_already_satisfied_candidate", "object_localized", "target_localized",
    "object_target_relation_verified", "grasp_candidate", "contact_established", "stable_hold_candidate", "object_lifted",
    "object_in_transport", "object_above_target", "object_on_target_candidate", "object_released",
    "goal_stability_candidate", "goal_verified", "grasp_failed", "contact_lost", "recovery_required", "target_blocked",
    "visual_unknown", "failure_terminal_candidate",
]


def _text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(map(str, value)).lower()
    return str(value or "").lower()


def map_action(value: str) -> tuple[str | None, str]:
    text = _text(value)
    ordered = [
        ("request_second_view", ("second view", "side view", "另一视角", "第二视角")),
        ("request_clarification", ("clarif", "澄清", "确认对象")),
        ("retry_grasp", ("retry", "regrasp", "重新抓", "再次抓")),
        ("recover_object", ("recover", "recovery", "恢复", "滑落")),
        ("clear_target", ("clear", "remove obstacle", "移除", "清理")),
        ("verify_goal_relation", ("verify_goal", "verify placement", "验证放置", "确认任务", "check_goal", "final_verify")),
        ("release_object", ("release", "open_gripper", "松开", "释放")),
        ("place_object", ("place", "lower", "放置", "放入", "下降")),
        ("align_with_target", ("align", "center", "对齐", "居中")),
        ("transport_object", ("transport", "move_to", "carry", "移动至", "移动到", "搬运")),
        ("lift_object", ("lift", "raise", "提起", "抬起")),
        ("verify_grasp", ("verify_hold", "check_grasp", "确认抓", "验证抓", "hold_check")),
        ("grasp_object", ("grasp", "pick", "close_gripper", "抓取", "夹持")),
        ("approach_object", ("approach", "reach", "靠近", "接近")),
        ("observe_scene", ("observe", "inspect", "check", "scan", "观察", "检查", "评估")),
        ("stop_no_action", ("stop", "no_action", "无需移动", "不移动")),
    ]
    matches = [name for name, keys in ordered if any(key in text for key in keys)]
    if not matches:
        return None, "No finite action synonym matched; retained as unmapped language."
    return matches[0], f"Matched the explicit finite-vocabulary synonym set for {matches[0]}."


def map_state(node: dict[str, Any]) -> tuple[str, str]:
    text = _text([node.get("id"), node.get("description"), node.get("role"), node.get("observable_conditions"), node.get("unknown_conditions")])
    ordered = [
        ("task_already_satisfied_candidate", ("already", "已在", "已经位于", "already_satisfied")),
        ("goal_verified", ("goal_verified", "任务完成并验证", "verified goal")),
        ("goal_stability_candidate", ("stable", "稳定放置", "稳定性", "goal_achieved", "verify_goal")),
        ("object_released", ("release", "释放", "松开")),
        ("object_on_target_candidate", ("on_target", "in_target", "放入", "盘子中央", "inside tray")),
        ("object_above_target", ("above", "上方", "悬于")),
        ("object_in_transport", ("transport", "移动至", "运输", "搬运")),
        ("object_lifted", ("lift", "提起", "离开桌面")),
        ("stable_hold_candidate", ("stable hold", "牢固", "稳固", "grasped", "被抓取")),
        ("grasp_candidate", ("grasp", "抓取", "夹取")),
        ("target_blocked", ("blocked", "occupied", "障碍", "占据")),
        ("recovery_required", ("recover", "recovery", "恢复", "重新抓")),
        ("contact_lost", ("contact lost", "slip", "滑落", "掉落")),
        ("grasp_failed", ("grasp failed", "missed grasp", "抓取失败")),
        ("visual_unknown", ("occlud", "unknown", "不可见", "遮挡", "待确认")),
        ("object_target_relation_verified", ("relation", "位置关系", "relative")),
        ("target_localized", ("target localized", "目标位置", "盘位置", "托盘位置", "盒子位置")),
        ("object_localized", ("object localized", "物体位置")),
    ]
    for state, keys in ordered:
        if any(key in text for key in keys):
            return state, f"Mapped explicit node language to the frozen semantic state {state}."
    return "scene_unverified", "No stronger observable state was explicit, so the node remains scene_unverified."


def canonicalize(candidate_paths: list[Path], rules: Path, action_vocabulary: Path, state_vocabulary: Path, output_dir: Path, canonical_graph: Path, case_mapping: Path, unmapped: Path, evidence: Path, report: Path) -> dict[str, Any]:
    if len(candidate_paths) != 18:
        raise ValueError(f"expected 18 source candidates, got {len(candidate_paths)}")
    allowed_actions = set(read_json(action_vocabulary)["values"])
    allowed_states = set(read_json(state_vocabulary)["values"])
    if allowed_actions != set(ACTION_VOCABULARY) or allowed_states != set(STATE_VOCABULARY):
        raise ValueError("vocabulary lock does not match implementation constants")
    if read_json(rules).get("dynamic_results_used") is not False:
        raise ValueError("canonicalization rules must exclude dynamic results")
    evidence_rows = []
    mapping_rows = []
    unmapped_rows = []
    state_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    normalized_ids = []
    for candidate_path in sorted(candidate_paths):
        source = read_json(candidate_path)
        request_id = candidate_path.stem
        normalized = {**source, "schema_version": "l2r_normalized_source_graph_v1", "source_request_id": request_id}
        node_map = {}
        for node in normalized.get("nodes", []):
            state, reason = map_state(node)
            if state not in allowed_states:
                raise AssertionError(state)
            node_map[node["id"]] = state
            node["canonical_state"] = state
            node["status"] = "hypothesized"
            state_counts[state] += 1
            evidence_rows.append({
                "source_candidate_id": request_id, "element_type": "node", "source_element_id": node["id"],
                "source_text": node.get("description", ""), "canonical_value": state, "mapping_reason": reason,
                "dynamic_validation_required": state not in {"scene_unverified", "object_localized", "target_localized"},
            })
        for edge in normalized.get("edges", []):
            action, reason = map_action(edge.get("action", ""))
            edge["canonical_action"] = action or "unmapped"
            edge["status"] = "hypothesized"
            if action:
                action_counts[action] += 1
            else:
                unmapped_rows.append({"source_candidate_id": request_id, "source_element_id": edge["id"], "source_text": edge.get("action", ""), "retained_as": "unknown_action"})
            evidence_rows.append({
                "source_candidate_id": request_id, "element_type": "edge", "source_element_id": edge["id"],
                "source_text": edge.get("action", ""), "canonical_value": action or "unmapped", "mapping_reason": reason,
                "dynamic_validation_required": True,
            })
        output_path = output_dir / candidate_path.name
        write_json(output_path, normalized)
        normalized_ids.append(request_id)
        mapping_rows.append({
            "source_candidate_id": request_id, "source_nodes": len(source.get("nodes", [])), "source_edges": len(source.get("edges", [])),
            "canonical_states": ";".join(sorted(set(node_map.values()))),
            "mapped_edges": sum(1 for edge in normalized.get("edges", []) if edge["canonical_action"] != "unmapped"),
            "unmapped_edges": sum(1 for edge in normalized.get("edges", []) if edge["canonical_action"] == "unmapped"),
        })
    canonical = {
        "schema": "pathgraph_l2r_canonical_coarse_graph_v1",
        "status": "hypothesized",
        "source": "18 frozen L1V V1 candidates",
        "source_candidate_ids": normalized_ids,
        "objects": [
            {"id": "manipulated_object", "role": "manipulated_object", "status": "hypothesized"},
            {"id": "target_object", "role": "target", "status": "hypothesized"},
        ],
        "nodes": [
            {"id": "scene_unverified", "state": "scene_unverified", "status": "hypothesized"},
            {"id": "grasp_candidate", "state": "grasp_candidate", "status": "hypothesized"},
            {"id": "stable_hold_candidate", "state": "stable_hold_candidate", "status": "hypothesized"},
            {"id": "object_in_transport", "state": "object_in_transport", "status": "hypothesized"},
            {"id": "object_above_target", "state": "object_above_target", "status": "hypothesized"},
            {"id": "object_on_target_candidate", "state": "object_on_target_candidate", "status": "hypothesized"},
            {"id": "object_released", "state": "object_released", "status": "hypothesized"},
            {"id": "goal_stability_candidate", "state": "goal_stability_candidate", "status": "hypothesized"},
        ],
        "edges": [
            {"id": "observe", "src": "scene_unverified", "dst": "grasp_candidate", "action": "observe_scene", "condition": "unknown", "status": "hypothesized"},
            {"id": "grasp", "src": "grasp_candidate", "dst": "stable_hold_candidate", "action": "grasp_object", "condition": "unknown", "status": "hypothesized"},
            {"id": "transport", "src": "stable_hold_candidate", "dst": "object_in_transport", "action": "transport_object", "condition": "unknown", "status": "hypothesized"},
            {"id": "align", "src": "object_in_transport", "dst": "object_above_target", "action": "align_with_target", "condition": "unknown", "status": "hypothesized"},
            {"id": "place", "src": "object_above_target", "dst": "object_on_target_candidate", "action": "place_object", "condition": "unknown", "status": "hypothesized"},
            {"id": "release", "src": "object_on_target_candidate", "dst": "object_released", "action": "release_object", "condition": "unknown", "status": "hypothesized"},
            {"id": "verify", "src": "object_released", "dst": "goal_stability_candidate", "action": "verify_goal_relation", "condition": "unknown", "status": "hypothesized"},
        ],
        "source_state_counts": dict(sorted(state_counts.items())),
        "source_action_counts": dict(sorted(action_counts.items())),
        "unmapped_language_retained": len(unmapped_rows),
        "numeric_reward_or_cost": None,
    }
    write_json(canonical_graph, canonical)
    write_csv(case_mapping, mapping_rows)
    write_csv(unmapped, unmapped_rows, ["source_candidate_id", "source_element_id", "source_text", "retained_as"])
    write_jsonl(evidence, evidence_rows)
    write_report(report, "V1 Coarse Graph Canonicalization", [("status", "V1_COARSE_GRAPH_CANONICALIZED"), ("candidates", 18), ("evidence records", len(evidence_rows)), ("unmapped edges", len(unmapped_rows)), ("dynamic results used", False)])
    return {"status": "V1_COARSE_GRAPH_CANONICALIZED", "candidates": 18, "evidence_records": len(evidence_rows), "unmapped": len(unmapped_rows)}


def review_canonicalization(canonical_graph: Path, mapping: Path, evidence: Path, output: Path, report: Path) -> dict[str, Any]:
    graph = read_json(canonical_graph)
    mappings = __import__("upgrade_v2.visual_refine_l2.io", fromlist=["read_csv"]).read_csv(mapping)
    evidence_rows = __import__("upgrade_v2.visual_refine_l2.io", fromlist=["read_jsonl"]).read_jsonl(evidence)
    source_ids = set(graph.get("source_candidate_ids", []))
    failures = []
    if len(mappings) != 18 or len(source_ids) != 18:
        failures.append("18/18 V1 candidates are not represented")
    if {row["source_candidate_id"] for row in mappings} != source_ids:
        failures.append("case mapping is not source-complete")
    if any(len(row.get("mapping_reason", "").strip()) < 12 for row in evidence_rows):
        failures.append("mapping evidence has a non-substantive reason")
    if any(item.get("status") != "hypothesized" for key in ("nodes", "edges") for item in graph.get(key, [])) or graph.get("status") != "hypothesized":
        failures.append("canonical graph contains a non-hypothesized status")
    if graph.get("numeric_reward_or_cost") is not None:
        failures.append("numeric reward or cost was introduced")
    result = {
        "schema": "pathgraph_l2r_canonicalization_gate_v1",
        "status": "V1_COARSE_GRAPH_CANONICALIZED" if not failures else "CANONICALIZATION_REVIEW_FAILED",
        "candidate_count": len(source_ids), "mapping_rows": len(mappings), "evidence_rows": len(evidence_rows),
        "unmapped_language_retained": graph.get("unmapped_language_retained", 0), "failures": failures,
    }
    write_json(output, result)
    write_report(report, "Canonicalization Gate", [("status", result["status"]), ("represented", f"{len(source_ids)}/18"), ("evidence records", len(evidence_rows)), ("failures", len(failures))])
    return result
