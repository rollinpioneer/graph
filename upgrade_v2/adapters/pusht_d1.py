"""Minimal state-snapshot adapter for the repository's Pymunk PushT environment.

D1 needs an environment whose state evolves through a physics engine and can be
captured at an anchor, restored in a fresh process, and continued with the same
controls.  The adapter deliberately stores the complete dynamic state required
by PushT rather than inferring a state from its five-dimensional observation.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def _pair(value: Any) -> list[float]:
    return [float(value[0]), float(value[1])]


def capture_state(env: Any) -> dict[str, Any]:
    """Return a JSON-serializable dynamic snapshot of a live PushT environment."""
    return {
        "schema": "pusht_pymunk_snapshot_v1",
        "agent_position": _pair(env.agent.position),
        "agent_velocity": _pair(env.agent.velocity),
        "block_position": _pair(env.block.position),
        "block_velocity": _pair(env.block.velocity),
        "block_angle": float(env.block.angle),
        "block_angular_velocity": float(env.block.angular_velocity),
        "latest_action": None if env.latest_action is None else np.asarray(env.latest_action, dtype=float).tolist(),
        "n_contact_points": int(env.n_contact_points),
        "sim_hz": int(env.sim_hz),
        "control_hz": int(env.control_hz),
    }


def restore_state(env: Any, snapshot: dict[str, Any]) -> None:
    """Restore ``capture_state`` output into an already-reset PushT instance."""
    if snapshot.get("schema") != "pusht_pymunk_snapshot_v1":
        raise ValueError("unsupported snapshot schema")
    if int(snapshot["sim_hz"]) != int(env.sim_hz) or int(snapshot["control_hz"]) != int(env.control_hz):
        raise ValueError("snapshot simulator timing does not match the target environment")
    env.agent.position = snapshot["agent_position"]
    env.agent.velocity = snapshot["agent_velocity"]
    env.block.position = snapshot["block_position"]
    env.block.velocity = snapshot["block_velocity"]
    env.block.angle = float(snapshot["block_angle"])
    env.block.angular_velocity = float(snapshot["block_angular_velocity"])
    action = snapshot.get("latest_action")
    env.latest_action = None if action is None else np.asarray(action, dtype=np.float64)
    env.n_contact_points = int(snapshot["n_contact_points"])
    # Pymunk must refresh broad-phase indices after direct body assignment.
    env.space.reindex_shapes_for_body(env.agent)
    env.space.reindex_shapes_for_body(env.block)


def state_vector(env: Any) -> np.ndarray:
    """Numeric dynamic state used only for restoration equality checks."""
    return np.asarray([
        *env.agent.position, *env.agent.velocity,
        *env.block.position, *env.block.velocity,
        env.block.angle, env.block.angular_velocity,
    ], dtype=np.float64)


def state_digest(snapshot: dict[str, Any]) -> str:
    """Stable digest for anchor provenance without converting it to a label."""
    payload = repr(snapshot).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
