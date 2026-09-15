"""State-first binding onto the frozen recovery graph. Actions are context only."""
from __future__ import annotations
from typing import Any

NODES = ["start", "grasped", "in_transit", "placed", "dropped_or_misaligned", "recovery", "success", "terminal_failure"]
NODE_INDEX = {n: i for i, n in enumerate(NODES)}
EDGES = {
    ("start", "grasped"): ("start_to_grasped", "forward", 0),
    ("grasped", "in_transit"): ("grasped_to_in_transit", "forward", 1),
    ("in_transit", "placed"): ("in_transit_to_placed", "forward", 2),
    ("placed", "success"): ("placed_to_success", "forward", 3),
    ("in_transit", "dropped_or_misaligned"): ("in_transit_to_dropped_or_misaligned", "failure", 4),
    ("in_transit", "terminal_failure"): ("in_transit_to_terminal_failure", "failure", 5),
    ("dropped_or_misaligned", "recovery"): ("dropped_or_misaligned_to_recovery", "recovery", 6),
    ("recovery", "grasped"): ("recovery_to_grasped", "recovery", 7),
    ("recovery", "recovery"): ("stagnation_loop", "stagnation", 8),
}
COMPRESSED = {("recovery", "in_transit"), ("dropped_or_misaligned", "in_transit"), ("dropped_or_misaligned", "grasped")}
OUTSIDE = {("start", "success"), ("scene_unverified", "goal_verified")}
LINEAR = {"start": 0.0, "grasped": 0.25, "in_transit": 0.5, "placed": 0.75, "success": 1.0,
          "dropped_or_misaligned": 0.0, "recovery": 0.25, "terminal_failure": 0.0}
NONE_EDGE = 9


def step_node(prev: str, facts: dict[str, Any], action: str) -> str:
    hold = facts.get("hold_current") is True
    loss = facts.get("unexpected_loss_eligible") is True
    goal = facts.get("goal_verified") is True
    rec = facts.get("recovery_completed") is True
    earlier = facts.get("hold_established_earlier") is True
    if prev == "start" and goal and action in ("verify", "stop") and not earlier:
        return "success"
    if loss and prev in ("in_transit", "grasped"):
        return "dropped_or_misaligned"
    if prev == "dropped_or_misaligned":
        if rec or action in ("retry", "recover"):
            return "recovery"
        return prev
    if prev == "recovery":
        if hold:
            return "grasped"
        return "recovery"
    if hold and action == "close_gripper":
        return "grasped"
    if hold and action in ("lift", "transport_to_target", "align", "lower"):
        return "in_transit"
    if prev == "in_transit" and hold and action in ("transport_to_target", "align", "lower"):
        return "in_transit"
    if action == "open_gripper" and goal:
        return "placed"
    if action in ("verify", "stop") and goal:
        return "success"
    if hold and prev == "start" and action == "close_gripper":
        return "grasped"
    return prev


def classify(a: str, b: str) -> tuple[str, str | None, str, int]:
    if a == b:
        return "BOUND_NODE_DWELL", None, "none", NONE_EDGE
    if (a, b) in EDGES:
        e = EDGES[(a, b)]
        return "BOUND", e[0], e[1], e[2]
    if (a, b) in COMPRESSED:
        return "SAMPLED_PATH_COMPRESSED_UNRESOLVED", None, "none", NONE_EDGE
    if (a, b) in OUTSIDE or (a == "start" and b == "success"):
        return "TASK_OUTSIDE_GRAPH_DOMAIN", None, "none", NONE_EDGE
    return "GRAPH_FORBIDDEN", None, "none", NONE_EDGE


def unordered(node: str, facts: dict[str, Any]) -> float:
    hold = 1.0 if facts.get("hold_established_earlier") else 0.0
    transit = 1.0 if node in ("in_transit", "placed", "success") else 0.0
    placed = 1.0 if node in ("placed", "success") else 0.0
    success = 1.0 if node == "success" else 0.0
    return (hold + transit + placed + success) / 4.0


def bind_kernel_trace(kernel_rows: list[dict[str, Any]], ingest_events: list[dict[str, Any]],
                      costs: dict[str, float]) -> dict[str, Any]:
    by_id = {e["event_id"]: e for e in ingest_events}
    node = "start"
    frames = []
    for row in kernel_rows:
        src = by_id[row["event_id"]] if "event_id" in row else by_id.get(f"{row['episode_id']}:f{row['capture_order']}")
        # kernel output may not include event_id; recover from capture_order
        if src is None:
            src = ingest_events[row["capture_order"]]
        nxt = step_node(node, row, src.get("action_context", ""))
        frames.append(dict(node=nxt, facts=row, src=src))
        node = nxt
    transitions = []
    for i in range(len(frames) - 1):
        a, b = frames[i]["node"], frames[i + 1]["node"]
        cls, eid, et, eidx = classify(a, b)
        src0, src1 = frames[i]["src"], frames[i + 1]["src"]
        rec_done = frames[i + 1]["facts"].get("recovery_completed") is True
        transitions.append(dict(
            episode_id=frames[i]["facts"]["episode_id"],
            classification=cls,
            node_before=a, node_after=b, edge_id=eid, edge_type=et, edge_index=eidx,
            cost_before=costs.get(a), cost_after=costs.get(b),
            linear_before=LINEAR[a], linear_after=LINEAR[b],
            unordered_before=unordered(a, frames[i]["facts"]),
            unordered_after=unordered(b, frames[i + 1]["facts"]),
            t0_ns=src0["available_at_ns"], t1_ns=src1["available_at_ns"],
            available_at_ns=src1["available_at_ns"],
            action_after=src1.get("action_context"),
            distance_before=src0.get("object_target_distance"),
            distance_after=src1.get("object_target_distance"),
            loss_episode_id=src1.get("loss_episode_id") or src0.get("loss_episode_id"),
            recovery_completed=rec_done,
            recovered_loss_episode_id=(src1.get("loss_episode_id") if rec_done else None),
            hold_current=frames[i + 1]["facts"].get("hold_current"),
            hold_established_earlier=frames[i + 1]["facts"].get("hold_established_earlier"),
            goal_verified=frames[i + 1]["facts"].get("goal_verified"),
            source_reference=src1.get("source_reference"),
            source_row=src1.get("source_row"),
            reference_strength=src1.get("reference_strength"),
        ))
    return dict(frames=frames, transitions=transitions)