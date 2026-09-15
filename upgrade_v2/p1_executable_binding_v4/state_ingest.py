"""Raw file -> DELTA events. Missing current facts become explicit None, not sticky True."""
from __future__ import annotations
import csv, json
from pathlib import Path
from typing import Any
import numpy as np
from .util import sha256_file

TRI = ("hold_current", "hold_loss_confirmed", "release_intent", "a_current_valid",
       "b_current_valid", "goal_verified", "terminal_failure", "recovery_completed")


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _b(v: str) -> bool:
    return str(v) in ("1", "True", "true")


def ingest_rollout(path: Path, *, goal_id: str = "place_on_target") -> dict[str, Any]:
    path = Path(path)
    actions = _csv(path / "actions.csv")
    contact = _csv(path / "contact_sensor.csv")
    gripper = _csv(path / "gripper_command.csv")
    oracle = _csv(path / "oracle_timeline.csv")
    online = np.load(path / "online_observation.npz")
    if any(k in online.files for k in ("future_outcome", "scenario")):
        raise ValueError("online observation must not carry future labels")
    ts = [float(x) for x in online["timestamps"]]
    n = len(ts)
    if not (len(actions) == len(contact) == len(gripper) == len(oracle) == n):
        raise ValueError(f"clock/schema mismatch in {path}")
    events_raw = []
    ep = path / "events.jsonl"
    if ep.is_file() and ep.stat().st_size:
        for line in ep.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events_raw.append(json.loads(line))
    files = {name: sha256_file(path / name) for name in
             ("actions.csv", "contact_sensor.csv", "gripper_command.csv", "oracle_timeline.csv",
              "online_observation.npz", "events.jsonl") if (path / name).is_file()}
    episode_id = path.parent.name + "_" + path.name.replace("rollout_", "r")
    events = []
    established = False
    prev_hold = None
    loss_id = None
    loss_i = 0
    attempt = "0"
    prev_facts = {k: None for k in TRI}
    for i in range(n):
        contact_now = _b(contact[i]["contact_present"])
        weld_now = _b(oracle[i]["weld_state"])
        hold = bool(contact_now and weld_now)
        release = gripper[i]["gripper_command"] == "open"
        goal = _b(oracle[i]["goal_stable"])
        action = actions[i]["action"]
        if hold:
            established = True
        hold_loss = False
        unexpected_onset_release = None
        if established and prev_hold is True and hold is False:
            hold_loss = True
            unexpected_onset_release = release
            if not release:
                loss_i += 1
                loss_id = f"{episode_id}:loss:{loss_i}"
                attempt = str(int(attempt) + 1)
        recovery_done = bool(action in ("retry", "recover") and hold and loss_id)
        facts = dict(hold_current=hold, hold_loss_confirmed=hold_loss if hold_loss else False,
                     release_intent=release, a_current_valid=None, b_current_valid=None,
                     goal_verified=goal, terminal_failure=False, recovery_completed=recovery_done)
        updates = {k: facts[k] for k in TRI if facts[k] != prev_facts[k]}
        # Always emit hold_current so a missing frame cannot keep stale True.
        if "hold_current" not in updates:
            updates["hold_current"] = hold
        ns = int(np.int64(np.round(np.float64(ts[i]) * 1_000_000_000.0)))
        ev = dict(
            episode_id=episode_id, object_id="active_object", goal_id=goal_id,
            attempt_id=attempt, event_id=f"{episode_id}:f{i}",
            available_at_ns=ns, capture_order=i,
            updates=updates,
            field_observed_at_ns={k: ns for k in updates},
            field_known_at_ns={k: ns for k in updates},
            source_reference=str(path), source_row=i, source_sha256=files.get("oracle_timeline.csv"),
            reference_strength="LEGACY_CONTACT_WELD_PROXY",
            action_context=action,
            contact_present=contact_now, weld_state=weld_now,
            object_target_distance=float(oracle[i]["object_target_distance"]),
            loss_episode_id=loss_id,
            release_intent_at_loss_onset=unexpected_onset_release,
        )
        events.append(ev)
        prev_facts = facts
        prev_hold = hold
    return dict(episode_id=episode_id, path=str(path), files=files, n=n, events=events,
                raw_events=events_raw, reference_strength="LEGACY_CONTACT_WELD_PROXY")