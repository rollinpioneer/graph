"""Canonical causal episode/transition records for U0/U1.

The adapter intentionally refuses to manufacture an after-observation.  A
sequence with N actions and fewer than N+1 observations consequently yields
only the transitions whose before/action/after triplet actually exists.
"""
from __future__ import annotations

from typing import Any, Iterable


def transitions(episode_uid: str, observations: list[Any], actions: list[Any],
                incoming_labels: list[Any] | None = None) -> list[dict[str, Any]]:
    usable = min(len(actions), max(0, len(observations) - 1))
    output: list[dict[str, Any]] = []
    for step in range(usable):
        label = incoming_labels[step + 1] if incoming_labels and step + 1 < len(incoming_labels) else None
        output.append({"transition_uid": f"{episode_uid}:{step}", "episode_uid": episode_uid,
                       "source_step": step, "obs_before": observations[step],
                       "action_applied": actions[step], "obs_after": observations[step + 1],
                       "edge_event": label, "edge_label_source": "incoming[k+1]" if label is not None else "unknown"})
    return output


def event_instances(event_labels: Iterable[Any]) -> list[int | None]:
    """Assign an ID at entry/re-entry; repeated frame labels share an ID."""
    ids: list[int | None] = []
    previous: Any = object()
    count = 0
    active: int | None = None
    for label in event_labels:
        if label is None:
            active = None
        elif label != previous or active is None:
            count += 1
            active = count
        ids.append(active)
        previous = label
    return ids
