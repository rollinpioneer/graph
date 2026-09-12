"""Explicit hash domains for reproducible checkpoint comparisons.

Record identity is intentionally separate from physical and semantic state.
This prevents the callback/return sampling labels from masquerading as a
simulation-state change while keeping cross-run record identity strict.
"""

from __future__ import annotations

from typing import Any

from .protocol import canonical_hash

RECORD_IDENTITY_FIELDS = ("sequence", "sampling_point")
PHYSICAL_FIELDS = (
    "time_hex", "arrays", "object_xyz", "gripper_xyz", "object_qpos",
    "object_qvel", "rng_state",
)
SEMANTIC_FIELDS = (
    "action", "action_index", "control_index", "events",
    "attempt_lifecycle", "simulator_flags",
)


def record_identity(state: dict[str, Any]) -> dict[str, Any]:
    return {field: state.get(field) for field in RECORD_IDENTITY_FIELDS}


def physical_payload(state: dict[str, Any]) -> dict[str, Any]:
    return {field: state.get(field) for field in PHYSICAL_FIELDS}


def semantic_payload(state: dict[str, Any]) -> dict[str, Any]:
    return {field: state.get(field) for field in SEMANTIC_FIELDS}


def physical_state_sha256(state: dict[str, Any]) -> str:
    return canonical_hash(physical_payload(state))


def semantic_state_sha256(state: dict[str, Any]) -> str:
    return canonical_hash(semantic_payload(state))


def record_sha256(state: dict[str, Any]) -> str:
    physical = state.get("physical_state_sha256") or physical_state_sha256(state)
    semantic = state.get("semantic_state_sha256") or semantic_state_sha256(state)
    return canonical_hash({
        **record_identity(state),
        "physical_state_sha256": physical,
        "semantic_state_sha256": semantic,
    })


def pair_callback_return_states(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pair each action-end callback with exactly one adjacent return row."""
    callbacks = [row for row in rows if row.get("sampling_point") == "action_end_callback"]
    returns = [row for row in rows if row.get("sampling_point") == "after_perform_return"]
    return_map: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in returns:
        return_map.setdefault((int(row.get("action_index", -1)), str(row.get("action"))), []).append(row)

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    errors: list[dict[str, Any]] = []
    for callback in sorted(callbacks, key=lambda row: int(row.get("sequence", -1))):
        key = (int(callback.get("action_index", -1)), str(callback.get("action")))
        candidates = return_map.get(key, [])
        if len(candidates) != 1:
            errors.append({"pair_key": key, "return_candidates": len(candidates)})
            continue
        returned = candidates[0]
        if int(returned.get("sequence", -999)) != int(callback.get("sequence", -999)) + 1:
            errors.append({
                "pair_key": key,
                "callback_sequence": callback.get("sequence"),
                "return_sequence": returned.get("sequence"),
                "error": "non_adjacent_sequence",
            })
            continue
        pairs.append((callback, returned))
    if len(callbacks) != len(returns):
        errors.append({"callback_count": len(callbacks), "return_count": len(returns)})
    return {
        "callbacks": callbacks,
        "returns": returns,
        "pairs": pairs,
        "pairing_complete": len(pairs) == len(callbacks) == len(returns) and not errors,
        "errors": errors,
    }


def compare_callback_return_pair(callback: dict[str, Any], returned: dict[str, Any]) -> dict[str, Any]:
    adjacent = int(returned.get("sequence", -999)) == int(callback.get("sequence", -999)) + 1
    identity_expected = (
        callback.get("sampling_point") == "action_end_callback"
        and returned.get("sampling_point") == "after_perform_return"
        and callback.get("action") == returned.get("action")
        and callback.get("action_index") == returned.get("action_index")
        and adjacent
    )
    physical_left = physical_state_sha256(callback)
    physical_right = physical_state_sha256(returned)
    semantic_left = semantic_state_sha256(callback)
    semantic_right = semantic_state_sha256(returned)
    return {
        "identity_expected": identity_expected,
        "adjacent_sequence": adjacent,
        "physical_exact": physical_left == physical_right,
        "semantic_exact": semantic_left == semantic_right,
        "record_exact": record_sha256(callback) == record_sha256(returned),
        "physical_state_sha256_callback": physical_left,
        "physical_state_sha256_return": physical_right,
        "semantic_state_sha256_callback": semantic_left,
        "semantic_state_sha256_return": semantic_right,
        "record_sha256_callback": record_sha256(callback),
        "record_sha256_return": record_sha256(returned),
    }
