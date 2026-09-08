from __future__ import annotations

from typing import Any

from upgrade_v2.l2r_ambiguity.event_memory import AttemptScopedMemory, tri, tri_and, tri_not, tri_or


def run_event_interface(observations: list[dict[str, Any]], evidence_rows: list[dict[str, Any]], history_complete: bool = True) -> list[dict[str, Any]]:
    memory = AttemptScopedMemory(1, 1, history_complete)
    output = []
    for obs, evidence in zip(observations, evidence_rows):
        predicates = dict(obs.get("predicates", obs))
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
