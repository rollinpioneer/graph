from __future__ import annotations

from typing import Any

from upgrade_v2.l2r_ambiguity.event_memory import AttemptScopedMemory, tri, tri_and, tri_not, tri_or


def infer_history_complete(observations: list[dict[str, Any]]) -> bool:
    """Infer prefix completeness from the stream, never from answer metadata."""
    if not observations:
        return False
    first_visible = None
    for row in observations:
        if row.get("observation_masked") or row.get("observation_missing"):
            if first_visible is None:
                return False
            continue
        first_visible = row
        break
    if first_visible is None:
        return False
    required = ("contact_present", "gripper_command")
    if any(first_visible.get(key) is None for key in required):
        return False
    # A complete attempt starts with the gripper open.  Starting closed is
    # compatible with a held object, but its prior history is unobserved.
    return first_visible.get("gripper_command") == "open"


def run_event_interface(observations: list[dict[str, Any]], evidence_rows: list[dict[str, Any]], history_complete: bool = True) -> list[dict[str, Any]]:
    memory = AttemptScopedMemory(1, 1, history_complete)
    output = []
    for obs, evidence in zip(observations, evidence_rows):
        predicates = dict(obs.get("predicates", obs))
        # Lifecycle markers are controller-owned online context.  They are
        # copied from the post-update observation and are never reconstructed
        # from event labels, action names, or future rows.
        for key in ("attempt_id", "attempt_phase", "attempt_active", "attempt_end", "attempt_end_reason"):
            if key in obs:
                predicates[key] = obs[key]
        predicates["stable_hold_observed"] = evidence["hold_evidence"]
        row = memory.observe(predicates, obs.get("time"), obs.get("frame_index"))
        row["hold_evidence"] = evidence["hold_evidence"]
        row["hold_memory"] = evidence["hold_memory"]
        row["effective_guards"] = dict(row["effective_guards"])
        output.append(row)
    return output


def first_action(rows: list[dict[str, Any]]) -> tuple[str, int | None, bool]:
    actions = []
    for index, row in enumerate(rows):
        guards = row.get("effective_guards", {})
        retry, recover = guards.get("retry_grasp"), guards.get("recover_object")
        if retry == "true":
            actions.append((index, "retry_grasp"))
        if recover == "true":
            actions.append((index, "recover_object"))
    if not actions:
        return "none", None, False
    first_index = min(index for index, _ in actions)
    at_first = [action for index, action in actions if index == first_index]
    return ("conflict" if len(set(at_first)) > 1 else at_first[0]), first_index, len(set(at_first)) > 1
