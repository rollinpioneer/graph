"""Frozen transition-aligned gold-event schema for U2."""
from __future__ import annotations

from typing import Final

EVENT_NAMES: Final[dict[int, str]] = {
    0: "none",
    1: "contact_on",
    2: "transport_on",
    3: "contact_off_failure",
    4: "recovery_start",
    5: "contact_reestablished",
    6: "detour_start",
    7: "goal_enter",
    8: "stable_success",
    9: "terminal_failure",
    10: "stagnation_onset",
}
EVENT_IDS: Final[dict[str, int]] = {value: key for key, value in EVENT_NAMES.items()}

MODE_NAMES: Final[dict[int, str]] = {
    0: "approach", 1: "contact_established", 2: "transport", 3: "contact_lost",
    4: "recovery_approach", 5: "contact_reestablished", 6: "detour",
    7: "goal_entered", 8: "stable_success", 9: "terminal_failure", 10: "stagnation",
}
MODE_IDS: Final[dict[str, int]] = {value: key for key, value in MODE_NAMES.items()}


def event_id(name: str) -> int:
    return EVENT_IDS[name]


def schema_payload() -> dict[str, object]:
    return {
        "schema": "u2_event_schema_v1",
        "events": EVENT_NAMES,
        "modes": MODE_NAMES,
        "boundary_timing": "gold_boundary[t]=1 denotes an event on transition (x[t-1], a[t-1], x[t])",
        "model_input_forbidden": ["gold_event", "gold_mode", "scenario", "future_outcome", "t_over_T", "root_family_id", "episode_id"],
    }
