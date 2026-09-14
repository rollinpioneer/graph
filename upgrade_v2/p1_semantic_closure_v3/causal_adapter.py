"""Prefix-only causal adapter. Action names are commands, not success evidence."""
from __future__ import annotations
import csv
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .util import sha256_file

NODES = ["start", "grasped", "in_transit", "placed", "dropped_or_misaligned", "recovery", "success", "terminal_failure"]
NODE_INDEX = {name: i for i, name in enumerate(NODES)}
EDGES = {
    ("start", "grasped"): ("start_to_grasped", "forward", 0),
    ("grasped", "in_transit"): ("grasped_to_in_transit", "forward", 1),
    ("in_transit", "placed"): ("in_transit_to_placed", "forward", 2),
    ("placed", "success"): ("placed_to_success", "forward", 3),
    ("in_transit", "dropped_or_misaligned"): ("in_transit_to_dropped_or_misaligned", "failure", 4),
    ("in_transit", "terminal_failure"): ("in_transit_to_terminal_failure", "failure", 5),
    ("dropped_or_misaligned", "recovery"): ("dropped_or_misaligned_to_recovery", "recovery", 6),
    ("recovery", "grasped"): ("recovery_to_grasped", "recovery", 7),
    ("recovery", "recovery"): ("stagnation_loop", "stagnation", 8),
}
NONE_EDGE_ID = 9
COSTS = {
    "start": 4 / 6,
    "grasped": 3 / 6,
    "in_transit": 2 / 6,
    "placed": 1 / 6,
    "dropped_or_misaligned": 5 / 6,
    "recovery": 4 / 6,
    "success": 0.0,
    "terminal_failure": 1.0,
}
FUTURE_FORBIDDEN = ("future_outcome", "scenario")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _ns(value: float) -> int:
    return int(np.int64(np.round(np.float64(value) * 1_000_000_000.0)))


def phi_distance(current: float, entry: float) -> float:
    if any(isinstance(x, bool) or not math.isfinite(x) or x < 0 for x in (current, entry)):
        raise ValueError("invalid geometry")
    return max(0.0, min(1.0, 1.0 - current / max(entry, 1e-9)))


def step_node(prev: str, frame: dict[str, Any]) -> str:
    action = frame["action"]
    held = bool(frame["contact_present"] or frame["weld_state"])
    goal = bool(frame["goal_stable"])
    if action in ("observe_scene", "approach_object", "clear_target"):
        return prev if prev != "success" else prev
    if action == "close_gripper":
        return "grasped" if held else prev
    if action == "retry":
        if held:
            return "grasped"
        return "recovery"
    if action == "recover":
        return "recovery" if not (held and prev == "in_transit") else "recovery"
    if action == "lift":
        if held:
            return "in_transit"
        return "recovery" if prev in ("recovery", "dropped_or_misaligned") else prev
    if action in ("transport_to_target", "align", "lower"):
        if held:
            return "in_transit"
        if prev == "in_transit":
            return "dropped_or_misaligned"
        return prev
    if action == "open_gripper":
        return "placed" if goal else prev
    if action in ("verify", "stop"):
        return "success" if goal else prev
    return prev


def load_rollout(path: Path) -> dict[str, Any]:
    path = Path(path)
    actions = _read_csv(path / "actions.csv")
    contact = _read_csv(path / "contact_sensor.csv")
    gripper = _read_csv(path / "gripper_command.csv")
    oracle = _read_csv(path / "oracle_timeline.csv")
    events = []
    events_path = path / "events.jsonl"
    if events_path.is_file() and events_path.stat().st_size:
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    online = np.load(path / "online_observation.npz")
    if any(k in online.files for k in FUTURE_FORBIDDEN):
        raise ValueError("online observation unexpectedly contains future fields")
    timestamps = [float(x) for x in online["timestamps"]]
    n = len(timestamps)
    if not (len(actions) == len(contact) == len(gripper) == len(oracle) == n):
        raise ValueError(f"frame count mismatch in {path}")
    frames = []
    for i in range(n):
        frames.append(dict(
            frame_index=i,
            time_s=timestamps[i],
            time_ns=_ns(timestamps[i]),
            action=actions[i]["action"],
            contact_present=str(contact[i]["contact_present"]) in ("1", "True", "true"),
            gripper_closed=gripper[i]["gripper_command"] == "closed",
            weld_state=str(oracle[i]["weld_state"]) in ("1", "True", "true"),
            goal_stable=str(oracle[i]["goal_stable"]) in ("1", "True", "true"),
            object_target_distance=float(oracle[i]["object_target_distance"]),
        ))
    files = {
        name: sha256_file(path / name)
        for name in ("actions.csv", "contact_sensor.csv", "gripper_command.csv",
                     "online_observation.npz", "oracle_timeline.csv", "events.jsonl")
        if (path / name).is_file()
    }
    return dict(path=str(path), frames=frames, events=events, files=files, n=n)


def _bind(prev: str, nxt: str) -> tuple[str | None, str, int, str]:
    if prev == nxt:
        return None, "none", NONE_EDGE_ID, "BOUND_NODE_DWELL"
    spec = EDGES.get((prev, nxt))
    if spec is None:
        return None, "none", NONE_EDGE_ID, "ILLEGAL_TRANSITION"
    return spec[0], spec[1], spec[2], "BOUND"


def adapt_prefix(raw: dict[str, Any], n_frames: int | None = None) -> dict[str, Any]:
    frames = raw["frames"] if n_frames is None else raw["frames"][:n_frames]
    if not frames:
        raise ValueError("empty prefix")
    nodes = []
    prev = "start"
    entry_anchor: dict[str, float] = {}
    for frame in frames:
        nxt = step_node(prev, frame)
        if nxt == "in_transit" and "in_transit" not in entry_anchor:
            entry_anchor["in_transit"] = float(frame["object_target_distance"])
        nodes.append(nxt)
        prev = nxt
    state_sequence = []
    for frame, node in zip(frames, nodes):
        state_sequence.append(dict(
            frame_index=frame["frame_index"],
            time_ns=frame["time_ns"],
            action=frame["action"],
            state=node,
            contact_present=frame["contact_present"],
            weld_state=frame["weld_state"],
            gripper_closed=frame["gripper_closed"],
            goal_stable_current=frame["goal_stable"],
            object_target_distance=frame["object_target_distance"],
        ))
    transitions = []
    attempt_id = 0
    for i in range(len(frames) - 1):
        a, b = nodes[i], nodes[i + 1]
        if a in ("dropped_or_misaligned", "recovery") and b in ("grasped", "in_transit", "recovery"):
            attempt_id = max(attempt_id, 1)
        edge_id, edge_type, edge_index, binding = _bind(a, b)
        phi_before = 0.0
        phi_after = 0.0
        phi_status = "PHI_UNAVAILABLE"
        if "in_transit" in entry_anchor:
            if a == "in_transit":
                phi_before = phi_distance(frames[i]["object_target_distance"], entry_anchor["in_transit"])
            if b == "in_transit":
                phi_after = phi_distance(frames[i + 1]["object_target_distance"], entry_anchor["in_transit"])
            if a == "in_transit" and b == "in_transit":
                phi_status = "WITHIN_NODE_ENTRY_ANCHORED"
            elif a == "in_transit" or b == "in_transit":
                phi_status = "NODE_CHANGE_PHI_ENDPOINT"
        row = dict(
            episode_id=Path(raw["path"]).parent.name + "_" + Path(raw["path"]).name.replace("rollout_", "r"),
            task_id="transport_recovery",
            root_group_id=Path(raw["path"]).parent.name,
            evidence_tier="CAUSAL_STATE_REPLAY",
            cost_contract_id="P1_NORMALIZED_RUNTIME_GRAPH_V1",
            step=i,
            t0_ns=frames[i]["time_ns"],
            t1_ns=frames[i + 1]["time_ns"],
            available_at_ns=frames[i + 1]["time_ns"],
            reference_known_at_ns=frames[i + 1]["time_ns"],
            physical_time_ns=frames[i + 1]["time_ns"],
            object_id="active_object",
            active_goal="place_on_target",
            current_valid_subgoals=[],
            historical_hold=bool(frames[i + 1]["contact_present"] or frames[i + 1]["weld_state"]),
            release_intent=frames[i + 1]["action"] == "open_gripper",
            attempt_id=attempt_id,
            node_before=a,
            node_after=b,
            edge_id=edge_id,
            edge_type=edge_type,
            cost_before=COSTS[a],
            cost_after=COSTS[b],
            phi_before=phi_before,
            phi_after=phi_after,
            phi_entry_distance=(entry_anchor.get("in_transit") if (a == "in_transit" or b == "in_transit") else None),
            phi_entry_time=None,
            phi_source="oracle_timeline.object_target_distance vs first in_transit distance" if phi_status != "PHI_UNAVAILABLE" else None,
            phi_status=phi_status,
            binding_status=binding,
            evidence_tier_row="CAUSAL_STATE_REPLAY",
            raw_state_reference=raw["path"],
            source_file="oracle_timeline.csv+actions.csv+contact_sensor.csv+gripper_command.csv+online_observation.npz",
            source_row=i + 1,
            source_sha256=raw["files"].get("oracle_timeline.csv"),
            future_outcome_used=False,
            scenario_used=False,
        )
        transitions.append(row)
    illegal = sum(1 for t in transitions if t["binding_status"] == "ILLEGAL_TRANSITION")
    unknown = sum(1 for t in transitions if t["binding_status"] == "UNRESOLVED_STATE")
    return dict(
        status="PASS" if not illegal else "ILLEGAL_TRANSITION_PRESENT",
        raw_rollout=raw["path"],
        raw_files=raw["files"],
        frames=len(frames),
        transitions=transitions,
        state_sequence=state_sequence,
        entry_distance=entry_anchor,
        illegal_transition_count=illegal,
        unresolved_state_count=unknown,
        phi_available_count=sum(1 for t in transitions if t["phi_status"] != "PHI_UNAVAILABLE"),
        future_fields_not_read=["oracle_diagnostic.npz:future_outcome", "oracle_diagnostic.npz:scenario", "metadata.json:scenario/final_goal_stable"],
        time_conversion="int(round(float64(seconds)*1e9)); timestamps from online_observation.npz",
    )


def scoring_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for t in result["transitions"]:
        a = t["node_before"]
        b = t["node_after"]
        rows.append(dict(
            episode_id=t["episode_id"],
            task_id=t["task_id"],
            root_group_id=t["root_group_id"],
            evidence_tier="CAUSAL_STATE_REPLAY",
            cost_contract_id=t["cost_contract_id"],
            step=t["step"],
            t0_ns=t["t0_ns"],
            t1_ns=t["t1_ns"],
            available_at_ns=t["available_at_ns"],
            reference_known_at_ns=t["reference_known_at_ns"],
            node_before=NODE_INDEX[a],
            node_after=NODE_INDEX[b],
            edge_id=EDGES[(a, b)][2] if (a, b) in EDGES else NONE_EDGE_ID,
            edge_type=t["edge_type"],
            cost_before=t["cost_before"],
            cost_after=t["cost_after"],
            phi_before=t["phi_before"],
            phi_after=t["phi_after"],
            linear_before=0.0 if a != "success" else 1.0,
            linear_after=0.0 if b != "success" else 1.0,
            unordered_before=0.0 if a != "success" else 1.0,
            unordered_after=0.0 if b != "success" else 1.0,
            success_before=a == "success",
            success_after=b == "success",
            terminal_before=a in ("success", "terminal_failure"),
            terminal_after=b in ("success", "terminal_failure"),
            raw_state_reference=t["raw_state_reference"],
            node_before_label=a,
            node_after_label=b,
            edge_id_label=t["edge_id"],
            binding_status=t["binding_status"],
        ))
    return rows


def prefix_checks(raw: dict[str, Any], cuts: list[int]) -> list[dict[str, Any]]:
    full = adapt_prefix(raw)
    rows = []
    n = len(raw["frames"])
    for cut in cuts:
        if cut < 2 or cut > n:
            rows.append(dict(prefix_frames=cut, status="SKIP", reason="need at least one transition"))
            continue
        part = adapt_prefix(raw, cut)
        equal = part["transitions"] == full["transitions"][: cut - 1]
        rows.append(dict(prefix_frames=cut, status="PASS" if equal else "FAIL", rows_equal_to_full_prefix=equal))
    mutated = deepcopy(raw)
    if mutated["frames"]:
        mutated["frames"][-1]["object_target_distance"] = mutated["frames"][-1]["object_target_distance"] + 1.0
        mutated["frames"][-1]["goal_stable"] = not mutated["frames"][-1]["goal_stable"]
        mid = max(2, n // 2)
        left = adapt_prefix(raw, mid)
        right = adapt_prefix(mutated, mid)
        rows.append(dict(prefix_frames=mid, status="PASS" if left["transitions"] == right["transitions"] else "FAIL",
                         future_row_mutation_test=left["transitions"] == right["transitions"]))
    return rows