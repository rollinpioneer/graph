"""Causal sensor-change candidates and reproducible weak event posteriors."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .dataset import load_episode, read_csv, write_csv, write_json
from .event_schema import EVENT_NAMES


DEFAULT_RULES: dict[str, Any] = {
    "window": {"causal": 8, "offline_future": 4},
    "contact": {"on_threshold": 0.5, "off_threshold": 0.5, "hysteresis_steps": 2},
    "motion": {"object_speed_on": 0.015, "object_speed_off": 0.006, "velocity_correlation_on": 0.55},
    "goal": {"enter_sensor_threshold": 0.5, "progress_derivative_abs": 0.015},
    "action": {"change_norm": 0.30}, "dynamics": {"residual_quantile_train": 0.90},
    "stagnation": {"object_speed_max": 0.004, "goal_progress_abs_max": 0.003, "consecutive_steps": 6},
    "posterior": {"unknown_max_probability": 0.55, "unknown_margin": 0.12},
}


def save_rules(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(DEFAULT_RULES, sort_keys=False), encoding="utf-8")


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    out = np.exp(shifted); return out / out.sum(axis=1, keepdims=True)


def candidate_scores(observations: np.ndarray, actions: np.ndarray, rules: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Produce only causal sensor/history votes; no gold/scenario/outcome is read."""
    T = len(observations); scores = np.zeros((T, 11), dtype=np.float32)
    contact = observations[:, 12] > rules["contact"]["on_threshold"]
    collision = observations[:, 13] > 0.5; in_goal = observations[:, 14] > rules["goal"]["enter_sensor_threshold"]
    object_vel = observations[:, 6:8]; agent_vel = observations[:, 2:4]
    object_speed = np.linalg.norm(object_vel, axis=1)
    agent_speed = np.linalg.norm(agent_vel, axis=1)
    goal_distance = np.linalg.norm(observations[:, 4:6], axis=1)
    action_delta = np.r_[0.0, np.linalg.norm(actions[1:] - actions[:-1], axis=1)]
    contact_on = np.zeros(T, dtype=np.float32); contact_off = np.zeros(T, dtype=np.float32)
    goal_enter = np.zeros(T, dtype=np.float32); collision_on = np.zeros(T, dtype=np.float32)
    contact_on[1:] = (~contact[:-1] & contact[1:]); contact_on[0] = contact[0]
    contact_off[1:] = (contact[:-1] & ~contact[1:])
    goal_enter[1:] = (~in_goal[:-1] & in_goal[1:]); goal_enter[0] = in_goal[0]
    collision_on[1:] = (~collision[:-1] & collision[1:]); collision_on[0] = collision[0]
    motion_on = np.zeros(T, dtype=np.float32); motion_on[1:] = (object_speed[1:] >= rules["motion"]["object_speed_on"]) & (object_speed[:-1] < rules["motion"]["object_speed_on"])
    motion_off = np.zeros(T, dtype=np.float32); motion_off[1:] = (object_speed[1:] <= rules["motion"]["object_speed_off"]) & (object_speed[:-1] > rules["motion"]["object_speed_off"])
    correlation = (agent_vel * object_vel).sum(axis=1) / np.maximum(agent_speed * object_speed, 1e-8)
    velocity_corr = (correlation > rules["motion"]["velocity_correlation_on"]).astype(np.float32)
    progress = np.r_[0.0, goal_distance[:-1] - goal_distance[1:]]
    progress_change = np.r_[0.0, np.abs(progress[1:] - progress[:-1])]
    loss_history = np.cumsum(contact_off) > 0
    recovery_start = np.r_[0.0, contact_off[:-1]]
    reestablished = contact_on * loss_history
    detour = collision_on.copy()
    # A sharp steering change near the obstacle is an observable detour cue.
    near_obstacle = np.linalg.norm(observations[:, 10:12], axis=1) < 0.31
    detour = np.maximum(detour, ((action_delta > rules["action"]["change_norm"]) & near_obstacle).astype(np.float32))
    stagnant = np.zeros(T, dtype=np.float32); w = int(rules["stagnation"]["consecutive_steps"])
    for t in range(w - 1, T):
        speed_ok = object_speed[t - w + 1:t + 1].max() <= rules["stagnation"]["object_speed_max"]
        delta = abs(goal_distance[t - w + 1] - goal_distance[t])
        if speed_ok and delta <= rules["stagnation"]["goal_progress_abs_max"]: stagnant[t] = 1.0
    # Candidate-to-event mapping, deliberately kept independent of simulator gold.
    scores[:, 1] = contact_on * 2.6
    scores[:, 2] = np.maximum(motion_on, velocity_corr * contact.astype(np.float32)) * 1.2
    scores[:, 3] = contact_off * 2.7
    scores[:, 4] = recovery_start * 2.4
    scores[:, 5] = reestablished * 2.6
    scores[:, 6] = detour * 1.4
    scores[:, 7] = goal_enter * 2.8
    scores[:, 8] = (in_goal & contact & (object_speed < 0.045)).astype(np.float32) * 0.8
    scores[:, 9] = collision_on * 2.3
    scores[:, 10] = stagnant * 2.0
    signals = {"contact_on_vote": contact_on, "contact_off_vote": contact_off, "object_motion_on_vote": motion_on,
               "object_motion_off_vote": motion_off, "agent_object_velocity_correlation": velocity_corr,
               "goal_distance_drop": np.maximum(progress, 0.0), "goal_distance_derivative_change": progress_change,
               "collision_on_vote": collision_on, "object_goal_enter_vote": goal_enter,
               "action_change_vote": action_delta, "dynamics_residual_vote": np.abs(object_speed - agent_speed),
               "stagnation_vote": stagnant}
    return scores, signals


def extract_candidates(dataset: Path, rules_path: Path, output_root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    rules = yaml.safe_load(rules_path.read_text(encoding="utf-8")); rows = read_csv(dataset / "episode_manifest.csv")
    output_root.mkdir(parents=True, exist_ok=True); manifests: list[dict[str, Any]] = []
    for row in rows:
        episode = load_episode(row); scores, signals = candidate_scores(episode["observations"], episode["actions"], rules)
        out = output_root / f"{row['episode_id']}.npz"; np.savez_compressed(out, event_scores=scores, **signals)
        # Raw candidate records are excluded from packages, while the compact NPZ
        # powers downstream aggregation.
        record_file = output_root / f"{row['episode_id']}.jsonl"
        with record_file.open("w", encoding="utf-8") as handle:
            for name, values in signals.items():
                causal = not name.startswith("offline_vote_")
                for t, value in enumerate(values):
                    if float(value) != 0.0:
                        handle.write(json.dumps({"episode_id": row["episode_id"], "root_family_id": row["root_family_id"], "split": row["split"], "t": t, "candidate_name": name, "vote": bool(value > 0), "strength": float(value), "causal": causal, "source_signal": name}) + "\n")
        manifests.append({"episode_id": row["episode_id"], "root_family_id": row["root_family_id"], "split": row["split"], "candidate_path": str(out.resolve()), "record_path": str(record_file.resolve()), "n_steps": len(scores)})
    write_csv(manifest_path, manifests, list(manifests[0])); return manifests


def _calibration_families(rows: list[dict[str, str]], fraction: float, seed: int) -> set[str]:
    train = sorted({row["root_family_id"] for row in rows if row["split"] == "train"})
    rng = np.random.default_rng(seed); rng.shuffle(train)
    return set(train[:max(1, round(len(train) * fraction))])


def aggregate_posteriors(candidate_root: Path, dataset: Path, modes: list[str], fraction: float, seed: int, output_root: Path, weights_path: Path, families_path: Path, rules_path: Path) -> None:
    rows = read_csv(dataset / "episode_manifest.csv"); rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    calibration = _calibration_families(rows, fraction, seed)
    write_csv(families_path, [{"root_family_id": x, "split": "train", "purpose": "simulator_gold_calibration_10pct"} for x in sorted(calibration)], ["root_family_id", "split", "purpose"])
    weight_rows: list[dict[str, Any]] = []
    for mode in modes:
        output = output_root / mode; output.mkdir(parents=True, exist_ok=True)
        # A strong none prior prevents a weak motion correlation from becoming
        # a boundary, while a discrete sensor transition remains decisive.
        event_bias = np.zeros(11, dtype=np.float32); event_bias[0] = 3.0
        scales = np.ones(11, dtype=np.float32)
        if mode == "weak_plus_small_gold_calibration":
            # Calibration gold is restricted to the frozen 10% train families.
            observed = np.zeros(11); expected = np.zeros(11)
            for row in rows:
                if row["root_family_id"] not in calibration: continue
                with np.load(candidate_root / f"{row['episode_id']}.npz") as payload: score = payload["event_scores"]
                gold = load_episode(row)["gold_event_id"]
                for k in range(11):
                    observed[k] += float((gold == k).sum()); expected[k] += float(score[:, k].sum())
            ratio = (observed + 2.0) / (expected + 2.0)
            scales = np.clip(1.0 + 0.10 * np.log(ratio), 0.70, 1.30).astype(np.float32)
        for k, value in enumerate(scales): weight_rows.append({"mode": mode, "event_id": k, "event": EVENT_NAMES[k], "weight": float(value), "bias": float(event_bias[k]), "source": "candidate_consensus" if mode == "weak_zero_gold" else "candidate_consensus_plus_10pct_train_simulator_gold"})
        for row in rows:
            with np.load(candidate_root / f"{row['episode_id']}.npz") as payload: score = payload["event_scores"].astype(np.float32)
            logits = score * (3.0 * scales[None, :]) + event_bias[None, :]
            probs = _softmax(logits).astype(np.float32); maxp = probs.max(axis=1); second = np.partition(probs, -2, axis=1)[:, -2]
            unknown = ((maxp < rules["posterior"]["unknown_max_probability"]) | ((maxp - second) < rules["posterior"]["unknown_margin"])).astype(np.int8)
            np.savez_compressed(output / f"{row['episode_id']}.npz", boundary_probability=(1.0 - probs[:, 0]).astype(np.float32), event_probability=probs, event_argmax=probs.argmax(axis=1).astype(np.int8), unknown=unknown, posterior_entropy=(-np.sum(probs * np.log(np.maximum(probs, 1e-8)), axis=1)).astype(np.float32))
    write_csv(weights_path, weight_rows, list(weight_rows[0]))
    write_json(output_root / "aggregation_manifest.json", {"modes": modes, "episodes": len(rows), "calibration_fraction": fraction, "calibration_family_count": len(calibration), "test_gold_used": False})
