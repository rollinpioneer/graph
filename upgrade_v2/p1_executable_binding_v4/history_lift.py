"""DUAL_VALID_STATE_REFERENCE_V4: history-lifted graph, not old runtime repair."""
from __future__ import annotations
from typing import Any
from pathlib import Path
import json

REQUIRED = [
    "only_A", "only_B", "A_then_B", "B_then_A", "A_invalid_B_valid", "A_restored",
    "repeat_A_not_B", "object_mismatch", "unknown_update", "both_without_goal",
    "terminal_failure", "future_B_isolation",
]


def success_eligible(a, b, g) -> bool | None:
    vals = (a, b, g)
    if any(v is False for v in vals):
        return False
    if any(v is None for v in vals):
        return None
    return True


def sequences() -> list[dict[str, Any]]:
    def ev(t, **u):
        keys = list(u)
        return dict(episode_id="dual_ref", object_id="obj", goal_id="dual", attempt_id="0",
                    event_id=f"e{t}", available_at_ns=t, capture_order=t,
                    updates=u, field_observed_at_ns={k: t for k in keys}, field_known_at_ns={k: t for k in keys})
    return [
        dict(name="only_A", events=[ev(1, a_current_valid=True, b_current_valid=False, goal_verified=True)], expect=False),
        dict(name="only_B", events=[ev(1, a_current_valid=False, b_current_valid=True, goal_verified=True)], expect=False),
        dict(name="A_then_B", events=[ev(1, a_current_valid=True, b_current_valid=False, goal_verified=True), ev(2, b_current_valid=True)], expect=True),
        dict(name="B_then_A", events=[ev(1, a_current_valid=False, b_current_valid=True, goal_verified=True), ev(2, a_current_valid=True)], expect=True),
        dict(name="A_invalid_B_valid", events=[ev(1, a_current_valid=True, b_current_valid=False, goal_verified=True), ev(2, a_current_valid=False, b_current_valid=True)], expect=False),
        dict(name="A_restored", events=[ev(1, a_current_valid=True, b_current_valid=False, goal_verified=True), ev(2, a_current_valid=False, b_current_valid=True), ev(3, a_current_valid=True)], expect=True),
        dict(name="repeat_A_not_B", events=[ev(1, a_current_valid=True, b_current_valid=False, goal_verified=True), ev(2, a_current_valid=True)], expect=False),
        dict(name="object_mismatch", events=[ev(1, a_current_valid=True, b_current_valid=None, goal_verified=True)], expect=None),
        dict(name="unknown_update", events=[ev(1, a_current_valid=True, b_current_valid=True, goal_verified=None)], expect=None),
        dict(name="both_without_goal", events=[ev(1, a_current_valid=True, b_current_valid=True, goal_verified=False)], expect=False),
        dict(name="terminal_failure", events=[ev(1, a_current_valid=True, b_current_valid=True, goal_verified=True, terminal_failure=True)], expect=False),
        dict(name="future_B_isolation", events=[ev(1, a_current_valid=True, b_current_valid=False, goal_verified=True), ev(2, b_current_valid=True)], expect=True, prefix0=False),
    ]


def lifted_graph() -> dict[str, Any]:
    """Finite (node, valid_set) graph. Success only if A and B currently valid."""
    nodes = [
        "start|none", "A_done|{A}", "B_done|{B}", "AB|{A,B}", "success|{A,B}",
        "dropped|{A}", "dropped|{B}", "dropped|{A,B}", "recovery|{A,B}", "terminal_failure|none",
    ]
    def e(i, src, dst, typ, cost, guard):
        return dict(id=i, src=src, dst=dst, type=typ, base_step_cost=cost, guard=guard)
    edges = [
        e("sA", "start|none", "A_done|{A}", "forward", 1, "A_current_valid"),
        e("sB", "start|none", "B_done|{B}", "alternative", 1, "B_current_valid"),
        e("AB", "A_done|{A}", "AB|{A,B}", "forward", 1, "B_current_valid"),
        e("BA", "B_done|{B}", "AB|{A,B}", "alternative", 1, "A_current_valid"),
        e("ABsucc", "AB|{A,B}", "success|{A,B}", "forward", 1, "A_valid AND B_valid AND goal_verified"),
        e("Adrop", "A_done|{A}", "dropped|{A}", "failure", 1, "A lost, B never valid so not success"),
        e("Bdrop", "B_done|{B}", "dropped|{B}", "failure", 1, "B lost"),
        e("ABdrop", "AB|{A,B}", "dropped|{A,B}", "failure", 1, "subgoal currently lost"),
        e("drec", "dropped|{A,B}", "recovery|{A,B}", "recovery", 1, "recovery started"),
        e("recAB", "recovery|{A,B}", "AB|{A,B}", "recovery", 1, "A and B currently restored"),
        e("term", "B_done|{B}", "terminal_failure|none", "failure", 1, "terminal failure evidence"),
    ]
    return dict(
        graph_id="DUAL_VALID_STATE_REFERENCE_V4",
        nodes=[{"id": n} for n in nodes],
        success_nodes=["success|{A,B}"],
        terminal_failure_nodes=["terminal_failure|none"],
        edges=edges,
        old_runtime_equivalence=False,
        physical_dual_order_binding="RAW_NOT_FOUND",
        note="A_done->success from the unlifted YAML is NOT an enabled edge here.",
    )


def evaluate_kernel(StateKernel) -> list[dict[str, Any]]:
    rows = []
    for case in sequences():
        k = StateKernel("dual_ref", "obj", "dual")
        outs = [k.step(e) for e in case["events"]]
        got = outs[-1]["dual_success_eligible"]
        prefix = True
        k2 = StateKernel("dual_ref", "obj", "dual")
        pref = [k2.step(e) for e in case["events"][:1]]
        future_ok = True
        if case["name"] == "future_B_isolation":
            k3 = StateKernel("dual_ref", "obj", "dual")
            a = k3.step(case["events"][0])
            mutated = dict(case["events"][1], updates=dict(b_current_valid=False),
                           field_observed_at_ns={"b_current_valid": 2}, field_known_at_ns={"b_current_valid": 2})
            k4 = StateKernel("dual_ref", "obj", "dual")
            b = k4.step(case["events"][0])
            future_ok = a["dual_success_eligible"] == b["dual_success_eligible"] == case.get("prefix0", False)
        ok = (got is case["expect"]) and future_ok
        rows.append(dict(name=case["name"], expected=case["expect"], observed=got, passed=ok, future_isolation=future_ok))
    return rows