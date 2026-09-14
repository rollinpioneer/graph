"""Source-bound dual-order overlay. Package truth tables are not runtime proof."""
from __future__ import annotations
from typing import Any

SOURCE_SUCCESS_FUNCTION = {
    "synthetic_oracle": "tools/stage2/stage2_pipeline.py:synthetic_episode",
    "source_lines": "95-102",
    "a_field": "info.subgoal_A_done",
    "b_field": "info.subgoal_B_done",
    "terminal_field": "info.success",
    "task_config": "configs/stage2/task_state_fields.yaml:transport_dual_order.success=info.success",
    "yaml_history_policy": "completed_subgoal_set is part of state",
    "yaml_success_guard": "state evidence only; A_done_to_success and B_done_to_success",
}


def tri_and(*values: bool | None) -> bool | None:
    if any(v is not None and type(v) is not bool for v in values):
        raise ValueError("three-valued bool expected")
    if any(v is False for v in values):
        return False
    return None if any(v is None for v in values) else True


def success_guard(a_valid: bool | None, b_valid: bool | None, task_goal_verified: bool | None) -> bool | None:
    """Proposed overlay: current-valid A and B plus an explicit terminal requirement."""
    return tri_and(a_valid, b_valid, task_goal_verified)


def oracle_success_from_source(a_done: bool | None, b_done: bool | None, scenario_success: bool | None) -> bool | None:
    """Exact synthetic-oracle terminal used in stage2. Not A∧B."""
    if scenario_success is False:
        return False
    if scenario_success is None:
        return None
    return True


def guard_events(events: list[dict[str, Any]], *, rule: str = "overlay") -> list[dict[str, Any]]:
    a = b = g = None
    out: list[dict[str, Any]] = []
    prev_time = -1
    for event in events:
        t = event["available_at_ns"]
        if type(t) is not int or t < prev_time:
            raise ValueError("invalid event order")
        prev_time = t
        for key in ("a_valid", "b_valid", "goal_verified"):
            if key in event and event[key] is not None and type(event[key]) is not bool:
                raise ValueError("invalid state observation")
        a = event.get("a_valid", a)
        b = event.get("b_valid", b)
        g = event.get("goal_verified", g)
        eligible = success_guard(a, b, g) if rule == "overlay" else oracle_success_from_source(a, b, g)
        out.append(dict(available_at_ns=t, a_valid=a, b_valid=b, goal_verified=g, success_eligible=eligible, rule=rule))
    return out


def prefix_check_events(events: list[dict[str, Any]], *, rule: str = "overlay") -> bool:
    full = guard_events(events, rule=rule)
    return all(guard_events(events[:i], rule=rule) == full[:i] for i in range(len(events) + 1))


def required_sequences() -> list[dict[str, Any]]:
    """MANUAL 8.3 historical sequences. Overlay rule, not the synthetic oracle."""
    cases = [
        dict(name="only_A", events=[dict(available_at_ns=1, a_valid=True, b_valid=False, goal_verified=True)], expect=False),
        dict(name="only_B", events=[dict(available_at_ns=1, a_valid=False, b_valid=True, goal_verified=True)], expect=False),
        dict(name="A_then_B", events=[
            dict(available_at_ns=1, a_valid=True, b_valid=False, goal_verified=True),
            dict(available_at_ns=2, b_valid=True),
        ], expect=True),
        dict(name="B_then_A", events=[
            dict(available_at_ns=1, a_valid=False, b_valid=True, goal_verified=True),
            dict(available_at_ns=2, a_valid=True),
        ], expect=True),
        dict(name="A_invalidated_only_B", events=[
            dict(available_at_ns=1, a_valid=True, b_valid=False, goal_verified=True),
            dict(available_at_ns=2, a_valid=False, b_valid=True),
        ], expect=False),
        dict(name="A_restored", events=[
            dict(available_at_ns=1, a_valid=True, b_valid=False, goal_verified=True),
            dict(available_at_ns=2, a_valid=False, b_valid=True),
            dict(available_at_ns=3, a_valid=True),
        ], expect=True),
        dict(name="repeat_A_is_not_B", events=[
            dict(available_at_ns=1, a_valid=True, b_valid=False, goal_verified=True),
            dict(available_at_ns=2, a_valid=True),
        ], expect=False),
        dict(name="object_mismatch_unknown", events=[
            dict(available_at_ns=1, a_valid=True, b_valid=None, goal_verified=True),
        ], expect=None),
        dict(name="future_B_does_not_rewrite_past", events=[
            dict(available_at_ns=1, a_valid=True, b_valid=False, goal_verified=True),
            dict(available_at_ns=2, b_valid=True),
        ], expect_prefix0=False, expect=True),
        dict(name="terminal_label_without_current_evidence", events=[
            dict(available_at_ns=1, a_valid=True, b_valid=False, goal_verified=True),
        ], expect=False),
    ]
    return cases


def evaluate_required_sequences() -> list[dict[str, Any]]:
    rows = []
    for case in required_sequences():
        out = guard_events(case["events"])
        got = out[-1]["success_eligible"]
        prefix_ok = prefix_check_events(case["events"])
        future_ok = True
        if case["name"] == "future_B_does_not_rewrite_past":
            mutated = [dict(case["events"][0]), dict(case["events"][1], b_valid=False)]
            future_ok = guard_events(case["events"])[:1] == guard_events(mutated)[:1]
        expect = case.get("expect")
        passed = (got is expect) and prefix_ok and future_ok
        rows.append(dict(
            name=case["name"], expected=expect, observed=got, prefix_stable=prefix_ok,
            future_isolation=future_ok, passed=passed, evidence_tier="PROPOSED_GUARD_UNIT_TEST_NOT_RUNTIME_BINDING",
        ))
    return rows


def overlay_document(source_graph_sha256: str | None = None) -> dict[str, Any]:
    seq = evaluate_required_sequences()
    return {
        "schema": "p1_v3_dual_order_guard_overlay_v1",
        "status": "SOURCE_FOUND_RUNTIME_NOT_EQUIVALENT",
        "source_graph_sha256": source_graph_sha256,
        "immutable_old_yaml": True,
        "source_success_function": SOURCE_SUCCESS_FUNCTION["synthetic_oracle"],
        "source_code_lines": SOURCE_SUCCESS_FUNCTION["source_lines"],
        "a_currently_valid_field": SOURCE_SUCCESS_FUNCTION["a_field"],
        "b_currently_valid_field": SOURCE_SUCCESS_FUNCTION["b_field"],
        "terminal_requirement_source": SOURCE_SUCCESS_FUNCTION["terminal_field"] + " ; " + SOURCE_SUCCESS_FUNCTION["task_config"],
        "proposed_rule": "A_currently_valid AND B_currently_valid AND task_specific_terminal_requirement",
        "task_specific_terminal_requirement": (
            "Synthetic oracle sets info.success = (scenario != 'terminal_failure'), independent of A∧B. "
            "A/B flags are monotonic time thresholds except terminal_failure which clears the first subgoal. "
            "drop_and_regrasp does not clear A/B, so they behave as ever-completed rather than currently-valid. "
            "YAML success edges only say 'state evidence'. Overlay is therefore proposed, not runtime-closed."
        ),
        "missing_evidence_policy": "UNKNOWN_NO_SUCCESS_CREDIT",
        "historical_ever_completed_is_not_current_valid": True,
        "guard_units_passed": all(r["passed"] for r in seq),
        "runtime_binding_closed": False,
        "sequence_tests": seq,
        "comparison_note": (
            "A simple unordered count of {A_done,B_done} matches the overlay when both flags are current-valid booleans. "
            "It does not match the synthetic oracle, which can mark success from scenario while A/B are incomplete, "
            "and does not invalidate A during drop_and_regrasp."
        ),
    }