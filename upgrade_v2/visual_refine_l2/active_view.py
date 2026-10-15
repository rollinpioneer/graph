"""Active-view policy definition; side RGB is queried only on visual unknown."""

from __future__ import annotations

from typing import Any


def should_query_second_view(predicate_frames: list[dict[str, Any]], maximum_rate: float = 0.35) -> bool:
    del maximum_rate
    return any(frame["predicates"].get("visual_unknown") == "true" for frame in predicate_frames)


def active_view_contract() -> dict[str, Any]:
    return {
        "schema": "pathgraph_l2r_active_view_contract_v1",
        "trigger": "visual_unknown == true under frozen front-view thresholds",
        "side_view_default": False,
        "maximum_rollout_query_rate": 0.35,
        "topology_change_allowed": False,
        "new_api_calls": 0,
    }
