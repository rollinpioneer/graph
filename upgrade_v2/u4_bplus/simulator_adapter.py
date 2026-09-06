"""Adapter around the real U2 simulator; no dynamics are reimplemented here."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from upgrade_v2.u2.simulator import FamilySpec, StochasticBoundarySimulator, make_family_specs
from upgrade_v2.u2.event_schema import EVENT_NAMES

from .io import write_json, write_jsonl, read_csv


def family_specs(count: int, seed: int) -> list[FamilySpec]:
    raw = make_family_specs(count, seed)
    result = []
    for index, spec in enumerate(raw):
        result.append(FamilySpec(
            root_family_id=f"u4b_v1_{index:02d}", scenario=spec.scenario, side=spec.side,
            obstacle_margin=spec.obstacle_margin, process_noise=spec.process_noise,
            slip_step=spec.slip_step, stagnation_steps=spec.stagnation_steps,
        ))
    return result


def family_to_dict(spec: FamilySpec) -> dict[str, Any]:
    return dict(spec.__dict__)


def collect_episode(spec: FamilySpec, rollout_seed: int, capture_snapshots: bool = True) -> dict[str, Any]:
    sim = StochasticBoundarySimulator(spec, rollout_seed)
    states = [sim.state_vector().tolist()]
    observations = [sim.observable().tolist()]
    actions: list[list[float]] = []
    events: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    boundaries: list[bool] = []
    while not sim.done:
        snapshot = sim.snapshot() if capture_snapshots else None
        action = sim.policy_action()
        if snapshot is not None:
            snapshots.append(snapshot)
        obs_after, event_id, boundary, info = sim.step(action)
        actions.append(action.astype(float).tolist())
        observations.append(obs_after.tolist())
        states.append(sim.state_vector().tolist())
        boundaries.append(bool(boundary))
        events.append({"t": len(actions) - 1, "event_id": int(event_id), "event": EVENT_NAMES[int(event_id)], "all_events": list(info.get("events", [])), "terminal_reason": info.get("terminal_reason", ""), "success": bool(info.get("success", False)), "contact_before": bool(info.get("contact_before", False)), "contact_after": bool(info.get("contact_after", False))})
    return {
        "schema": "u4b_episode_v1", "episode_id": f"u4b_{spec.root_family_id}_{rollout_seed}",
        "root_family_id": spec.root_family_id, "family": family_to_dict(spec), "rollout_seed": int(rollout_seed),
        "states": states, "observations": observations, "actions": actions, "events": events,
        "boundaries": boundaries, "snapshots": snapshots, "success": bool(sim.success),
        "terminal_reason": sim.terminal_reason or "stable_success", "n_steps": len(actions),
    }


def save_episode(path: Path, episode: dict[str, Any]) -> None:
    write_json(path, episode)


def continuation(anchor: dict[str, Any], seed: int, horizon: int = 32, exact: bool = False) -> dict[str, Any]:
    family = FamilySpec(**anchor["family"])
    snapshot = copy.deepcopy(anchor["snapshot"])
    sim = StochasticBoundarySimulator(family, int(anchor.get("rollout_seed", seed)))
    sim.restore(snapshot)
    if not exact:
        sim.rng = np.random.default_rng(int(seed))
    states = []
    events = []
    for _ in range(min(horizon, max(0, sim.horizon - sim.step_index))):
        if sim.done:
            break
        action = sim.policy_action()
        before = sim.state_vector().tolist()
        _, event_id, _, info = sim.step(action)
        states.append({"before": before, "after": sim.state_vector().tolist(), "action": action.tolist()})
        events.append({"event_id": int(event_id), "event": EVENT_NAMES[int(event_id)], "all_events": list(info.get("events", [])), "t": sim.step_index - 1})
    return {"anchor_id": anchor.get("anchor_id", ""), "root_family_id": family.root_family_id, "split": anchor.get("split", ""), "snapshot_sha": hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest(), "controller_sha": "policy_action@u2/simulator.py", "rng_seed": int(seed), "followup_steps": len(states), "remaining_budget": max(0, sim.horizon - sim.step_index), "events": events, "states": states, "success": bool(sim.success), "terminal_reason": sim.terminal_reason, "censored": not sim.done, "exact_replay": bool(exact)}


def validate_snapshot_replay(spec: FamilySpec, rollout_seed: int = 840999) -> dict[str, Any]:
    episode = collect_episode(spec, rollout_seed, True)
    anchors = [i for i, snap in enumerate(episode["snapshots"]) if i < episode["n_steps"]][:6]
    checks = []
    for index in anchors:
        anchor = {"family": episode["family"], "rollout_seed": rollout_seed, "snapshot": episode["snapshots"][index], "anchor_id": f"{episode['episode_id']}:t{index}"}
        left = continuation(anchor, rollout_seed, 6, exact=True)
        right = continuation(anchor, rollout_seed, 6, exact=True)
        checks.append({"anchor_id": anchor["anchor_id"], "state_equal": left["states"] == right["states"], "events_equal": left["events"] == right["events"]})
    return {"status": "PASS" if all(x["state_equal"] and x["events_equal"] for x in checks) else "FAIL", "checks": checks}
