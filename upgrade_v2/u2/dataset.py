"""U2 dataset collection, loading, split validation, and state-restore tests."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .event_schema import EVENT_NAMES, MODE_IDS, schema_payload
from .simulator import FamilySpec, SCENARIOS, StochasticBoundarySimulator, make_family_specs


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def split_specs(specs: list[FamilySpec], ratios: tuple[float, float, float]) -> dict[str, str]:
    """Stratify exact family counts per scenario; no rollout can cross a split."""
    out: dict[str, str] = {}
    for scenario in SCENARIOS:
        members = [spec for spec in specs if spec.scenario == scenario]
        n_train = round(len(members) * ratios[0])
        n_val = round(len(members) * ratios[1])
        # The final remainder preserves the requested total even under rounding.
        for index, spec in enumerate(members):
            out[spec.root_family_id] = "train" if index < n_train else ("val" if index < n_train + n_val else "test")
    return out


def rollout_episode(sim: StochasticBoundarySimulator) -> dict[str, Any]:
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    contact: list[int] = []
    collision: list[int] = []
    in_goal: list[int] = []
    event_ids: list[int] = []
    boundary: list[int] = []
    mode_ids: list[int] = []
    while not sim.done:
        action = sim.policy_action()
        observation, event, is_boundary, info = sim.step(action)
        observations.append(observation); actions.append(action.astype(np.float32))
        contact.append(int(observation[12] > 0.5)); collision.append(int(observation[13] > 0.5)); in_goal.append(int(observation[14] > 0.5))
        event_ids.append(event); boundary.append(int(is_boundary)); mode_ids.append(MODE_IDS[info["gold_mode"]])
    return {
        "observations": np.asarray(observations, dtype=np.float32), "actions": np.asarray(actions, dtype=np.float32),
        "contact_sensor": np.asarray(contact, dtype=np.int8), "collision_sensor": np.asarray(collision, dtype=np.int8),
        "object_in_goal": np.asarray(in_goal, dtype=np.int8), "gold_event_id": np.asarray(event_ids, dtype=np.int8),
        "gold_boundary": np.asarray(boundary, dtype=np.int8), "gold_mode_id": np.asarray(mode_ids, dtype=np.int8),
        "success": sim.success, "terminal_reason": sim.terminal_reason, "event_log": sim.gold_events(),
    }


def collect_dataset(mode: str, root_families: int, rollouts_per_family: int, seed: int, output_root: Path, ratios: tuple[float, float, float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs = make_family_specs(root_families, seed)
    splits = split_specs(specs, ratios)
    output_root.mkdir(parents=True, exist_ok=True)
    episodes_dir = output_root / "episodes"; episodes_dir.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, Any]] = []
    split_rows = [{"root_family_id": spec.root_family_id, "scenario_for_analysis_only": spec.scenario, "split": splits[spec.root_family_id]} for spec in specs]
    for family_index, spec in enumerate(specs):
        for rollout_index in range(rollouts_per_family):
            episode_id = f"{mode}_{spec.root_family_id}_r{rollout_index:02d}"
            rollout_seed = seed + family_index * 1000 + rollout_index
            episode = rollout_episode(StochasticBoundarySimulator(spec, rollout_seed))
            path = episodes_dir / f"{episode_id}.npz"
            np.savez_compressed(path, **{key: value for key, value in episode.items() if isinstance(value, np.ndarray)})
            counts = Counter(int(value) for value in episode["gold_event_id"])
            manifests.append({
                "episode_id": episode_id, "root_family_id": spec.root_family_id, "split": splits[spec.root_family_id],
                "scenario_for_analysis_only": spec.scenario, "rollout_seed": rollout_seed, "n_steps": len(episode["actions"]),
                "success": bool(episode["success"]), "terminal_reason": episode["terminal_reason"] or "stable_success",
                "event_instance_count": int(sum(value != 0 for value in episode["gold_event_id"])),
                "recovery_event_count": int(counts.get(4, 0) + counts.get(5, 0)), "source_provenance": "explicit_state_stochastic_simulator",
                "npz_path": str(path.resolve()),
            })
    write_csv(output_root / "episode_manifest.csv", manifests, list(manifests[0]))
    write_csv(output_root / "root_family_split.csv", split_rows, list(split_rows[0]))
    write_json(output_root / "configs" / "event_schema.json", schema_payload())
    write_json(output_root / "configs" / "observable_schema.json", {
        "dimension": 17, "features": ["agent_to_object_position_x", "agent_to_object_position_y", "agent_velocity_x", "agent_velocity_y", "object_to_goal_position_x", "object_to_goal_position_y", "object_velocity_x", "object_velocity_y", "agent_to_obstacle_position_x", "agent_to_obstacle_position_y", "object_to_obstacle_position_x", "object_to_obstacle_position_y", "contact_sensor", "collision_sensor", "object_in_goal_sensor", "previous_action_x", "previous_action_y"],
        "forbidden": ["gold_event", "gold_mode", "scenario", "future outcome", "time fraction", "root_family_id", "episode_id"],
    })
    return manifests, split_rows


def load_episode(row: dict[str, str]) -> dict[str, np.ndarray]:
    with np.load(row["npz_path"], allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def group_by_split(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows: result[row["split"]].append(row)
    return result


def verify_state_restore(families: int, anchors_per_family: int, continuation_steps: int, seed: int, tolerance: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    specs = make_family_specs(families, seed)
    details: list[dict[str, Any]] = []
    for family_index, spec in enumerate(specs):
        for anchor_index in range(anchors_per_family):
            sim = StochasticBoundarySimulator(spec, seed + family_index * 100 + anchor_index)
            for _ in range(5 + anchor_index * 4):
                if sim.done: sim.reset()
                sim.step(sim.policy_action())
            snapshot = sim.snapshot()
            original = []
            for _ in range(continuation_steps):
                if sim.done: break
                _, event, _, _ = sim.step(sim.policy_action())
                original.append((sim.state_vector().copy(), event))
            restored = StochasticBoundarySimulator(spec, seed + 999999 + family_index * 100 + anchor_index)
            restored.restore(snapshot)
            replay = []
            for _ in range(continuation_steps):
                if restored.done: break
                _, event, _, _ = restored.step(restored.policy_action())
                replay.append((restored.state_vector().copy(), event))
            error = max([float(np.max(np.abs(a - b))) for (a, _), (b, _) in zip(original, replay)] or [0.0])
            event_match = [event for _, event in original] == [event for _, event in replay]
            details.append({"root_family_id": spec.root_family_id, "anchor_index": anchor_index, "continuation_steps": len(original), "max_abs_state_error": error, "gold_event_sequence_identical": event_match, "pass": error <= tolerance and event_match})
    summary = {"families": families, "anchors_per_family": anchors_per_family, "anchors_total": len(details), "anchors_pass": sum(row["pass"] for row in details), "max_continuation_error": max(row["max_abs_state_error"] for row in details), "tolerance": tolerance, "status": "U2_STATE_RESTORE_PASS" if all(row["pass"] for row in details) else "U2_STATE_RESTORE_FAIL"}
    return summary, details


def dataset_gate(dataset_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], str]:
    rows = read_csv(dataset_root / "episode_manifest.csv")
    splits = group_by_split(rows)
    families_by_split = {split: {row["root_family_id"] for row in values} for split, values in splits.items()}
    leakage = bool(families_by_split.get("train", set()) & families_by_split.get("val", set()) or families_by_split.get("train", set()) & families_by_split.get("test", set()) or families_by_split.get("val", set()) & families_by_split.get("test", set()))
    counts: Counter[tuple[str, int]] = Counter()
    scenarios: Counter[tuple[str, str]] = Counter()
    finite = True
    family_outcomes: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        episode = load_episode(row)
        finite = finite and bool(np.isfinite(episode["observations"]).all()) and len(episode["observations"]) > 0 and len(episode["actions"]) > 0
        for event in episode["gold_event_id"]:
            counts[(row["split"], int(event))] += 1
        scenarios[(row["split"], row["scenario_for_analysis_only"])] += 1
        family_outcomes[row["root_family_id"]].add("success" if row["success"].lower() == "true" else "failure")
    event_rows = [{"split": split, "event_id": event, "event": EVENT_NAMES[event], "count": count} for (split, event), count in sorted(counts.items())]
    scenario_rows = [{"split": split, "scenario_for_analysis_only": scenario, "episodes": count} for (split, scenario), count in sorted(scenarios.items())]
    total = len(rows); unique_families = len({row["root_family_id"] for row in rows})
    total_event = Counter(event for (_, event), count in counts.items() for _ in range(count))
    grazing = [row for row in rows if row["scenario_for_analysis_only"] == "grazing_contact"]
    grazing_rate = sum(row["success"].lower() == "true" for row in grazing) / len(grazing)
    mixed = sum(len(values) > 1 for values in family_outcomes.values())
    requirements = {
        "episodes": total == 720, "root_families": unique_families == 120, "leakage": not leakage,
        "finite": finite, "contact_on": total_event[1] >= 100, "contact_off_failure": total_event[3] >= 60,
        "recovery_start": total_event[4] >= 60, "contact_reestablished": total_event[5] >= 60,
        "goal_enter": total_event[7] >= 100, "terminal_failure": total_event[9] >= 60,
        "stagnation_onset": total_event[10] >= 60, "grazing_mixed": 0.0 < grazing_rate < 1.0,
        "mixed_outcome_families": mixed >= 10,
    }
    status = "U2_EVENTFUL_DATASET_READY" if all(requirements.values()) else "U2_DATASET_GATE_FAIL"
    summary = {"status": status, "episodes": total, "unique_root_families": unique_families, "root_family_leakage": leakage, "finite_observations_actions": finite, "grazing_success_rate": grazing_rate, "mixed_outcome_families": mixed, "requirements": requirements}
    report = "# U2 eventful dataset gate\n\n" + "\n".join(f"- {key}: {value}" for key, value in summary.items() if key != "requirements") + "\n\n" + "\n".join(f"- gate_{key}: {value}" for key, value in requirements.items()) + "\n"
    return summary, event_rows, scenario_rows, report
