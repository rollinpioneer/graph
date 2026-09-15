"""Constructed holdout states. No reward, potential, or scorer imports."""
from __future__ import annotations

import copy
from typing import Any

EVIDENCE = "STATE_CONDITIONED_HOLDOUT"
DT_NS = 10_000_000
Z = 0.03
QUAT = [1.0, 0.0, 0.0, 0.0]
ZERO3 = [0.0, 0.0, 0.0]


def _phase_flags(phase: str) -> tuple[bool, bool]:
    if phase == "VALID":
        return False, True
    if phase in ("HELD", "TRANSPORT"):
        return True, False
    if phase in ("WAIT", "PLACED", "LOST", "RECOVERING"):
        return False, False
    raise ValueError(f"unsupported constructed phase {phase}")


def object_state(phase: str, pos, target_xy, vel=None) -> dict[str, Any]:
    held, valid = _phase_flags(phase)
    return {
        "held": held,
        "valid": valid,
        "phase": phase,
        "pos": [float(pos[0]), float(pos[1]), float(pos[2] if len(pos) > 2 else Z)],
        "vel": list(vel) if vel is not None else list(ZERO3),
        "target_xy": [float(target_xy[0]), float(target_xy[1])],
        "quat": list(QUAT),
        "angular_vel": list(ZERO3),
    }


def lerp_xy(start, target, frac: float) -> list[float]:
    return [
        float(start[0] + frac * (target[0] - start[0])),
        float(start[1] + frac * (target[1] - start[1])),
        float(start[2] if len(start) > 2 else Z),
    ]


def reward_relevant(state: dict) -> dict:
    objs = {}
    for key, obj in state["objects"].items():
        objs[key] = {
            "held": obj["held"],
            "valid": obj["valid"],
            "phase": obj["phase"],
            "pos": list(obj["pos"]),
            "vel": list(obj["vel"]),
            "target_xy": list(obj["target_xy"]),
            "quat": list(obj["quat"]),
            "angular_vel": list(obj["angular_vel"]),
        }
    return {
        "task": state["task"],
        "success": state["success"],
        "terminal_failure": state["terminal_failure"],
        "gripper_closed": state["gripper_closed"],
        "eef": list(state["eef"]),
        "objects": objs,
        "events": copy.deepcopy(state.get("events") or []),
    }


class EpisodeBuilder:
    def __init__(self, episode_id: str, task: str, family_id: str, case_id: str, seed: int):
        self.episode_id = episode_id
        self.task = task
        self.family_id = family_id
        self.case_id = case_id
        self.seed = seed
        self.states: list[dict] = []
        self.segments: dict[str, Any] = {}

    def add(self, objects: dict, *, success: bool = False, terminal_failure: bool = False,
            gripper_closed: bool | None = None, eef=None, events=None) -> int:
        idx = len(self.states)
        t_ns = idx * DT_NS
        if gripper_closed is None:
            gripper_closed = any(o["held"] for o in objects.values())
        if eef is None:
            held = [o for o in objects.values() if o["held"]]
            eef = list(held[0]["pos"]) if held else [0.0, 0.0, 0.18]
        state = {
            "episode_id": self.episode_id,
            "state_index": idx,
            "task": self.task,
            "available_at_ns": int(t_ns),
            "physical_time_ns": int(t_ns),
            "capture_order": int(idx),
            "success": bool(success),
            "terminal_failure": bool(terminal_failure),
            "gripper_closed": bool(gripper_closed),
            "eef": [float(eef[0]), float(eef[1]), float(eef[2] if len(eef) > 2 else 0.18)],
            "events": copy.deepcopy(events or []),
            "objects": copy.deepcopy(objects),
            "legacy": {"node": "constructed_holdout", "cost": 0.0, "phi": 0.0},
            "legacy_edge_type": "none",
            "legacy_binding_status": "CONSTRUCTED_HOLDOUT_NO_OLD_ENGINE",
            "evidence_tier": EVIDENCE,
            "provenance": {
                "family_id": self.family_id,
                "case_id": self.case_id,
                "seed": self.seed,
                "source": "holdout_generator",
                "note": "constructed STATE_CONDITIONED_HOLDOUT; generator does not import or read reward",
            },
        }
        for event in state["events"]:
            event.setdefault("known_at_ns", int(t_ns))
            event.setdefault("object_id", event.get("object_id"))
        self.states.append(state)
        return idx

    def mark(self, name: str, value) -> None:
        self.segments[name] = value