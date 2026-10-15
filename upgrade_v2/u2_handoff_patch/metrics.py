"""Corrected episode-local boundary and reward metrics for U2 handoff."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .primitives import (
    incoming_segment_return,
    match_events,
    metric_from_counts,
    read_csv_rows,
    sha256_inventory,
    write_csv_rows,
    write_json,
)


EVENT_NAMES = {
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


def _load_episode(row: dict[str, str], repo_root: Path) -> dict[str, np.ndarray]:
    raw = Path(row["npz_path"])
    candidates = [raw, repo_root / raw]
    if not raw.is_absolute():
        candidates.append(repo_root / "artifacts" / "pathgraph_sarm" / "upgrade_v2" / "u2_stochastic_boundary" / "data_v1" / Path(row["split"]) / "episodes" / f"{row['episode_id']}.npz")
    for path in candidates:
        if path.is_file():
            with np.load(path) as payload:
                return {key: np.asarray(payload[key]) for key in payload.files}
    raise FileNotFoundError(f"episode cache missing for {row['episode_id']}: {candidates}")


def _prediction_candidates(root: Path, episode_id: str, split: str) -> list[Path]:
    # Prediction roots in the existing archive use either ``root/<episode>``
    # or ``root/<split>/<episode>``.  Keep this resolver explicit and bounded.
    return [root / f"{episode_id}.npz", root / split / f"{episode_id}.npz"]


def _load_prediction(root: Path, episode_id: str, split: str) -> tuple[dict[str, np.ndarray] | None, str | None]:
    for path in _prediction_candidates(root, episode_id, split):
        if path.is_file():
            with np.load(path) as payload:
                return {key: np.asarray(payload[key]) for key in payload.files}, str(path)
    return None, None


def _payload_array(payload: dict[str, np.ndarray], primary: str, secondary: str, length: int, threshold: float = 0.5) -> np.ndarray:
    if primary in payload:
        return np.asarray(payload[primary])
    if secondary in payload:
        return np.asarray(payload[secondary]) >= threshold
    return np.zeros(length, dtype=bool)


def _metric_row(prefix: str, metric: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metric.items()}


def _boundary_metric(pred: Sequence[int], gold: Sequence[int], tolerance: int) -> dict[str, Any]:
    matched = match_events(np.where(np.asarray(pred, dtype=bool))[0], np.where(np.asarray(gold, dtype=bool))[0], tolerance)
    return metric_from_counts(matched["tp"], matched["fp"], matched["fn"], matched["errors"])


def _event_metric(pred: Sequence[int], gold: Sequence[int], event_id: int, tolerance: int) -> dict[str, Any]:
    p = np.where(np.asarray(pred) == event_id)[0]
    g = np.where(np.asarray(gold) == event_id)[0]
    matched = match_events(p, g, tolerance)
    return metric_from_counts(matched["tp"], matched["fp"], matched["fn"], matched["errors"])


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.mean(clean)) if clean else None


def _family_macro_interval(rows: Sequence[dict[str, Any]], value_key: str, seed: int) -> dict[str, Any]:
    """Family-macro mean and deterministic nonparametric 95% interval."""

    values = [float(row[value_key]) for row in rows if row.get(value_key) is not None and math.isfinite(float(row[value_key]))]
    if not values:
        return {
            "root_family_macro": None,
            "root_family_macro_ci_low": None,
            "root_family_macro_ci_high": None,
            "root_family_macro_estimability": "not_estimable",
        }
    sample = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(sample, size=(5000, len(sample)), replace=True).mean(axis=1)
    return {
        "root_family_macro": float(sample.mean()),
        "root_family_macro_ci_low": float(np.quantile(draws, 0.025)),
        "root_family_macro_ci_high": float(np.quantile(draws, 0.975)),
        "root_family_macro_estimability": "estimable",
    }


def _unknown_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(row.get("frame_count", 0)) for row in rows)
    predicted = sum(int(row.get("unknown_predicted_frame_count", 0)) for row in rows)
    gold = sum(int(row.get("unknown_gold_frame_count", 0)) for row in rows)
    return {
        "unknown_frame_count": predicted,
        "unknown_gold_frame_count": gold,
        "unknown_frame_rate": (predicted / total) if total else None,
        "unknown_gold_frame_rate": (gold / total) if total else None,
        "unknown_retained": True,
    }


def recompute_boundaries(
    u2_root: Path,
    output: Path,
    splits: Sequence[str] = ("val", "test"),
    tolerances: Sequence[int] = (1, 2),
    group_key: str = "root_family_id",
) -> dict[str, Any]:
    """Recompute all available boundary predictions without cross-episode matching."""

    repo_root = _find_repo_root(u2_root)
    dataset = u2_root / "data_v1" / "formal" / "episode_manifest.csv"
    manifest = read_csv_rows(dataset)
    rows = [row for row in manifest if row["split"] in set(splits)]
    output.mkdir(parents=True, exist_ok=True)
    prediction_roots: dict[str, Path] = {
        "weak_zero_gold": u2_root / "weak_events_v1" / "posteriors" / "weak_zero_gold",
        "weak_plus_small_gold_calibration": u2_root / "weak_events_v1" / "posteriors" / "weak_plus_small_gold_calibration",
        "sensor_hysteresis_01": u2_root / "segmentation_baselines_v1" / "predictions" / "sensor_hysteresis_01",
        "uniform_00": u2_root / "segmentation_baselines_v1" / "predictions" / "uniform_00",
    }
    for parent, label in ((u2_root / "boundary_models_v1" / "predictions", "boundary_model"), (u2_root / "budgeted_correction_v1" / "predictions", "budget_model")):
        if parent.is_dir():
            for child in sorted(parent.iterdir()):
                if child.is_dir():
                    prediction_roots[f"{label}:{child.name}"] = child

    by_episode: list[dict[str, Any]] = []
    by_model: list[dict[str, Any]] = []
    by_family: list[dict[str, Any]] = []
    event_episode_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    available_models: list[str] = []
    input_paths: list[Path] = [dataset]
    for model_name, root in prediction_roots.items():
        model_episode_rows: list[dict[str, Any]] = []
        if not root.exists():
            continue
        for row in rows:
            payload, path = _load_prediction(root, row["episode_id"], row["split"])
            if payload is None:
                missing.append({"model": model_name, "episode_id": row["episode_id"], "split": row["split"], "expected_root": str(root)})
                continue
            episode = _load_episode(row, repo_root)
            input_paths.extend([repo_root / row["npz_path"], Path(path)])
            predicted_boundary = _payload_array(payload, "boundary_prediction", "boundary_probability", len(episode["gold_boundary"]))
            predicted_event = np.asarray(payload.get("event_prediction", payload.get("event_argmax", np.zeros(len(episode["gold_event_id"]), dtype=np.int8))))
            unknown = np.asarray(payload.get("unknown", np.zeros(len(predicted_boundary), dtype=np.int8)), dtype=bool)
            common = {
                "model": model_name,
                "episode_id": row["episode_id"],
                "root_family_id": row[group_key],
                "split": row["split"],
                "prediction_path": path,
                "frame_count": len(predicted_boundary),
                "unknown_predicted_frame_count": int(unknown.sum()),
                "unknown_gold_frame_count": 0,
            }
            for tolerance in tolerances:
                metric = _boundary_metric(predicted_boundary, episode["gold_boundary"], tolerance)
                model_episode_rows.append({**common, "tolerance": tolerance, **_metric_row("boundary", metric)})
                for event_id, event_name in EVENT_NAMES.items():
                    if event_id == 0:
                        continue
                    event_metric = _event_metric(predicted_event, episode["gold_event_id"], event_id, tolerance)
                    event_episode_rows.append({**common, "tolerance": tolerance, "event_id": event_id, "event": event_name, **event_metric})
        if not model_episode_rows:
            continue
        available_models.append(model_name)
        by_episode.extend(model_episode_rows)
        for (split, tolerance), group in _group_rows(model_episode_rows, ("split", "tolerance")):
            family_rows: list[dict[str, Any]] = []
            for (_, _, family), family_group in _group_rows(group, ("split", "tolerance", "root_family_id")):
                family_rows.append({"root_family_id": family, **_aggregate_metric_rows(family_group, "boundary")})
            macro = _family_macro_interval(family_rows, "boundary_f1", seed=20260957 + int(tolerance))
            by_model.append({
                "model": model_name,
                "split": split,
                "tolerance": tolerance,
                "episodes": len(group),
                "root_family_count": len(family_rows),
                **_aggregate_metric_rows(group, "boundary"),
                **{f"boundary_{key}": value for key, value in macro.items()},
                **_unknown_summary(group),
            })
        for (split, tolerance, family), group in _group_rows(model_episode_rows, ("split", "tolerance", "root_family_id")):
            by_family.append({"model": model_name, "split": split, "tolerance": tolerance, "root_family_id": family, "episodes": len(group), **_aggregate_metric_rows(group, "boundary"), **_unknown_summary(group)})

    by_event: list[dict[str, Any]] = []
    for (model, split, tolerance, event_id, event), group in _group_rows(event_episode_rows, ("model", "split", "tolerance", "event_id", "event")):
        tp = sum(int(row["tp"]) for row in group)
        fp = sum(int(row["fp"]) for row in group)
        fn = sum(int(row["fn"]) for row in group)
        error_sum = sum(float(row.get("error_sum", 0.0) or 0.0) for row in group)
        matched_count = sum(int(row.get("matched_count", 0) or 0) for row in group)
        metric = metric_from_counts(tp, fp, fn)
        metric["error_sum"] = error_sum
        metric["matched_count"] = matched_count
        metric["mae"] = error_sum / matched_count if matched_count else None
        by_event.append({"model": model, "split": split, "tolerance": tolerance, "event_id": event_id, "event": event, "episodes": len(group), **metric})

    old_vs_corrected: list[dict[str, Any]] = []
    # The old table is retained for traceability; these are not recomputed with
    # the old buggy semantics, only linked to the corrected estimand.
    old_candidates = [
        u2_root / "rounds" / "u2_3_causal_boundary_models" / "tables" / "u2_boundary_model_metrics.csv",
        u2_root / "rounds" / "u2_2_segmentation_baselines" / "tables" / "baseline_test_metrics.csv",
    ]
    for old_path in old_candidates:
        if not old_path.is_file():
            continue
        for old in read_csv_rows(old_path):
            model = old.get("job_id") or old.get("config_name") or old.get("variant") or old.get("method") or "unknown"
            old_split = old.get("split", old.get("evaluation_split", "unknown"))
            corrected = next((item for item in by_model if item["tolerance"] == 2 and item["split"] == old_split and (item["model"] == model or item["model"].endswith(f":{model}"))), None)
            old_f1 = old.get("boundary_f1_tol2", "")
            corrected_f1 = corrected.get("boundary_f1") if corrected else None
            try:
                delta = float(corrected_f1) - float(old_f1) if corrected_f1 is not None and old_f1 != "" else None
            except ValueError:
                delta = None
            old_vs_corrected.append({
                "model": model,
                "old_source": _relative_path(old_path, repo_root),
                "old_split": old_split,
                "old_boundary_f1_tol2": old_f1,
                "corrected_boundary_f1_tol2": corrected_f1 if corrected else "not_estimable",
                "corrected_minus_old_f1_tol2": delta if delta is not None else "not_estimable",
                "corrected_estimability": corrected.get("boundary_estimability", "not_estimable") if corrected else "not_estimable",
                "correction_note": "episode-local dynamic-program matching; historical checkpoint selection unchanged",
            })

    write_csv_rows(output / "boundary_metrics_by_episode.csv", by_episode)
    write_csv_rows(output / "boundary_metrics_by_family.csv", by_family)
    write_csv_rows(output / "boundary_metrics_by_model.csv", by_model)
    write_csv_rows(output / "event_metrics_by_type.csv", by_event)
    write_csv_rows(output / "old_vs_corrected_metrics.csv", old_vs_corrected)
    inventory = sha256_inventory(input_paths, repo_root)
    status = {
        "status": "BOUNDARY_CACHE_RECOMPUTED" if by_episode else "BOUNDARY_CACHE_INCOMPLETE",
        "source_commit": _source_commit(repo_root),
        "dataset_manifest": str(dataset.relative_to(repo_root)) if dataset.is_relative_to(repo_root) else str(dataset),
        "splits": list(splits),
        "tolerances": list(tolerances),
        "group_key": group_key,
        "models_evaluated": available_models,
        "episodes_evaluated": len({(x["model"], x["episode_id"]) for x in by_episode}),
        "missing_prediction_count": len(missing),
        "missing_predictions": missing,
        "input_file_inventory": {key: inventory[key] for key in ("file_count", "sha256", "missing_paths")},
        "algorithm": "episode-local one-to-one monotone dynamic programming; maximize TP then minimize total absolute error",
        "unknown_is_retained": True,
    }
    write_json(output / "recompute_status.json", status)
    return status


def _source_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _find_repo_root(path: Path) -> Path:
    current = path.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return path.resolve()


def _relative_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _group_rows(rows: Sequence[dict[str, Any]], keys: Sequence[str]):
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups.items()


def _aggregate_metric_rows(rows: Sequence[dict[str, Any]], prefix: str) -> dict[str, Any]:
    tp = sum(int(row[f"{prefix}_tp"]) for row in rows)
    fp = sum(int(row[f"{prefix}_fp"]) for row in rows)
    fn = sum(int(row[f"{prefix}_fn"]) for row in rows)
    error_sum = sum(float(row.get(f"{prefix}_error_sum", 0.0) or 0.0) for row in rows)
    matched_count = sum(int(row.get(f"{prefix}_matched_count", 0) or 0) for row in rows)
    metric = metric_from_counts(tp, fp, fn, ())
    metric["error_sum"] = error_sum
    metric["matched_count"] = matched_count
    metric["mae"] = error_sum / matched_count if matched_count else None
    return _metric_row(prefix, metric)


def _segment_rows(path: Path) -> list[dict[str, str]]:
    return read_csv_rows(path)


def _select_boundary_path(u2_root: Path, source: str, row: dict[str, str]) -> tuple[Path | None, str]:
    episode, split = row["episode_id"], row["split"]
    if source == "gold":
        return None, "gold"
    if source == "best_causal":
        lock = json.loads((u2_root / "segment_representation_v1" / "configs" / "boundary_source_lock.json").read_text())
        if lock.get("source_type") == "causal_model":
            return u2_root / "boundary_models_v1" / "predictions" / lock["source_method"] / split / f"{episode}.npz", lock["source_method"]
        source = "best_rule"
    if source == "best_rule":
        return u2_root / "segmentation_baselines_v1" / "predictions" / "sensor_hysteresis_01" / f"{episode}.npz", "sensor_hysteresis_01"
    if source == "uniform":
        return u2_root / "segmentation_baselines_v1" / "predictions" / "uniform_00" / f"{episode}.npz", "uniform_00"
    if source == "best_budget":
        lock = u2_root / "budgeted_correction_v1" / "models" / "selection.csv"
        selected = read_csv_rows(lock)
        best = max(selected, key=lambda x: float(x.get("boundary_f1_tol2", 0)))
        return u2_root / "budgeted_correction_v1" / "predictions" / best["job_id"] / split / f"{episode}.npz", best["job_id"]
    raise ValueError(f"unknown boundary source: {source}")


def _boundary_indices(source: str, u2_root: Path, row: dict[str, str], episode: dict[str, np.ndarray]) -> tuple[np.ndarray, str, Path | None]:
    path, resolved = _select_boundary_path(u2_root, source, row)
    if source == "gold":
        return np.asarray(episode["gold_boundary"], dtype=bool), resolved, None
    if path is None or not path.is_file():
        raise FileNotFoundError(f"missing boundary prediction for {source}: {path}")
    with np.load(path) as payload:
        return _payload_array(payload, "boundary_prediction", "boundary_probability", len(episode["gold_boundary"])), resolved, path


def _segments_from_boundary(boundary: Sequence[int]) -> list[tuple[int, int]]:
    ends = [int(x) for x in np.where(np.asarray(boundary, dtype=bool))[0]]
    result: list[tuple[int, int]] = []
    start = 0
    for end in ends + [len(boundary) - 1]:
        if end < start:
            continue
        result.append((start, end))
        start = end + 1
    return result


def recompute_reward(
    u2_root: Path,
    output: Path,
    split: str = "test",
    bootstrap: int = 5000,
    seed: int = 20260957,
    potential_lock: Path | None = None,
    boundary_lock: Path | None = None,
) -> dict[str, Any]:
    """Recompute incoming segment returns and test-only family bootstrap tables."""

    repo_root = _find_repo_root(u2_root)
    manifest = read_csv_rows(u2_root / "data_v1" / "formal" / "episode_manifest.csv")
    rows = [x for x in manifest if x["split"] == split]
    potential_root = u2_root / "reward_impact_v1" / "predictions"
    potential_lock = potential_lock or (u2_root / "reward_impact_v1" / "configs" / "value_potential_lock.json")
    boundary_lock = boundary_lock or (u2_root / "segment_representation_v1" / "configs" / "boundary_source_lock.json")
    output.mkdir(parents=True, exist_ok=True)
    sources = ["gold", "best_causal", "best_rule", "uniform", "best_budget"]
    segment_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    conservation: list[dict[str, Any]] = []
    missing: list[str] = []
    input_paths: list[Path] = [u2_root / "data_v1" / "formal" / "episode_manifest.csv", potential_lock, boundary_lock]
    for row in rows:
        episode = _load_episode(row, repo_root)
        potential_path = potential_root / f"{row['episode_id']}.npz"
        if not potential_path.is_file():
            missing.append(str(potential_path))
            continue
        input_paths.extend([repo_root / row["npz_path"], potential_path])
        with np.load(potential_path) as payload:
            phi = np.asarray(payload["phi"], dtype=float)
        whole_return = float(phi[-1] - phi[0]) if len(phi) > 1 else 0.0
        for source in sources:
            try:
                boundary, resolved, boundary_path = _boundary_indices(source, u2_root, row, episode)
            except FileNotFoundError as exc:
                missing.append(str(exc))
                continue
            if boundary_path is not None:
                input_paths.append(boundary_path)
            current: list[dict[str, Any]] = []
            for index, (start, end) in enumerate(_segments_from_boundary(boundary)):
                events = np.asarray(episode["gold_event_id"][start : end + 1], dtype=int)
                nonzero = events[events != 0]
                purity: float | str = "not_applicable"
                if len(nonzero):
                    counts = Counter(int(x) for x in nonzero)
                    purity = float(max(counts.values()) / len(nonzero))
                value = incoming_segment_return(phi, start, end)
                row_out = {"boundary_source": source, "resolved_source": resolved, "segment_id": f"{row['episode_id']}_{source}_{index}", "episode_id": row["episode_id"], "root_family_id": row["root_family_id"], "split": split, "start_t": start, "end_t": end, "segment_return": value, "dominant_gold_event": int(Counter(events).most_common(1)[0][0]) if len(events) else 0, "event_purity": purity, "contains_failure": bool(np.isin(events, [3, 9]).any()), "contains_recovery": bool(np.isin(events, [4, 5]).any()), "contains_success": bool(np.isin(events, [7, 8]).any()), "mixed_failure_recovery": bool(np.isin(events, [3, 9]).any() and np.isin(events, [4, 5]).any()), "direction_evaluable": True}
                current.append(row_out)
                segment_rows.append(row_out)
                for local_index, event_value in enumerate(events):
                    event_id = int(event_value)
                    if event_id in (3, 4, 5, 7, 8, 9):
                        event_t = start + local_index
                        direction_evaluable = event_t > 0
                        incoming_return = float(phi[event_t] - phi[event_t - 1]) if direction_evaluable else None
                        direction_correct: bool | str
                        if not direction_evaluable:
                            direction_correct = "not_estimable"
                        elif event_id in (3, 9):
                            direction_correct = incoming_return < 0
                        else:
                            direction_correct = incoming_return > 0
                        event_rows.append({
                            "boundary_source": source,
                            "resolved_source": resolved,
                            "segment_id": row_out["segment_id"],
                            "episode_id": row["episode_id"],
                            "root_family_id": row["root_family_id"],
                            "event_t": event_t,
                            "event_id": event_id,
                            "event": EVENT_NAMES[event_id],
                            "event_incoming_return": incoming_return,
                            "segment_return": value,
                            "direction_evaluable": direction_evaluable,
                            "direction_correct": direction_correct,
                            "event_covered_by_segment": True,
                            "event_at_segment_boundary": event_t in {start, end},
                            "direction_metric": "incoming_transition_phi_t_minus_phi_t_minus_1",
                        })
            partition_sum = float(sum(item["segment_return"] for item in current))
            conservation.append({"boundary_source": source, "episode_id": row["episode_id"], "root_family_id": row["root_family_id"], "split": split, "whole_return_phi_last_minus_phi_first": whole_return, "segment_return_sum": partition_sum, "partition_residual": abs(partition_sum - whole_return), "closed_full_input_cycle_residual": "not_measured"})
    write_csv_rows(output / "segment_returns_test_v2.csv", segment_rows)
    write_csv_rows(output / "event_returns_test_v2.csv", event_rows)
    write_csv_rows(output / "partition_conservation_by_episode.csv", conservation)

    family_metrics: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for source in sources:
        data = [row for row in segment_rows if row["boundary_source"] == source]
        source_events = [row for row in event_rows if row["boundary_source"] == source]
        families = sorted({row["root_family_id"] for row in data})
        per_family: list[dict[str, Any]] = []
        for family in families:
            subset = [row for row in data if row["root_family_id"] == family]
            family_events = [row for row in source_events if row["root_family_id"] == family]
            failures = [bool(row["direction_correct"]) for row in family_events if row["event_id"] in (3, 9) and row["direction_evaluable"]]
            recoveries = [bool(row["direction_correct"]) for row in family_events if row["event_id"] in (4, 5) and row["direction_evaluable"]]
            successes = [bool(row["direction_correct"]) for row in family_events if row["event_id"] in (7, 8) and row["direction_evaluable"]]
            per_family.append({
                "boundary_source": source,
                "root_family_id": family,
                "failure_negative_rate": float(np.mean(failures)) if failures else None,
                "recovery_positive_rate": float(np.mean(recoveries)) if recoveries else None,
                "success_positive_rate": float(np.mean(successes)) if successes else None,
                "segments": len(subset),
                "events": len(family_events),
                "direction_evaluable_events": sum(bool(row["direction_evaluable"]) for row in family_events),
                "direction_not_estimable_events": sum(not bool(row["direction_evaluable"]) for row in family_events),
            })
        family_metrics.extend(per_family)
        fail_values = [float(x["failure_negative_rate"]) for x in per_family if x["failure_negative_rate"] is not None]
        rec_values = [float(x["recovery_positive_rate"]) for x in per_family if x["recovery_positive_rate"] is not None]
        success_values = [float(x["success_positive_rate"]) for x in per_family if x["success_positive_rate"] is not None]
        fail_boot = rng.choice(np.asarray(fail_values), size=(bootstrap, len(fail_values)), replace=True).mean(axis=1) if fail_values else np.asarray([])
        rec_boot = rng.choice(np.asarray(rec_values), size=(bootstrap, len(rec_values)), replace=True).mean(axis=1) if rec_values else np.asarray([])
        def ci(values: np.ndarray) -> tuple[float | None, float | None]:
            return (float(np.quantile(values, .025)), float(np.quantile(values, .975))) if len(values) else (None, None)
        fl, fh = ci(fail_boot); rl, rh = ci(rec_boot)
        summary.append({
            "boundary_source": source,
            "split": split,
            "n_root_families": len(families),
            "n_segments": len(data),
            "n_events": len(source_events),
            "n_direction_evaluable_events": sum(bool(row["direction_evaluable"]) for row in source_events),
            "n_direction_not_estimable_events": sum(not bool(row["direction_evaluable"]) for row in source_events),
            "direction_metric": "incoming_transition_phi_t_minus_phi_t_minus_1",
            "failure_negative_rate": float(np.mean(fail_values)) if fail_values else None,
            "failure_negative_rate_ci_low": fl,
            "failure_negative_rate_ci_high": fh,
            "recovery_positive_rate": float(np.mean(rec_values)) if rec_values else None,
            "recovery_positive_rate_ci_low": rl,
            "recovery_positive_rate_ci_high": rh,
            "success_positive_rate": float(np.mean(success_values)) if success_values else None,
            "failure_recovery_mixed_segment_rate": float(np.mean([bool(x["mixed_failure_recovery"]) for x in data])) if data else None,
            "empty_event_segment_fraction": float(np.mean([x["event_purity"] == "not_applicable" for x in data])) if data else None,
            "event_segment_purity": float(np.mean([float(x["event_purity"]) for x in data if x["event_purity"] != "not_applicable"])) if any(x["event_purity"] != "not_applicable" for x in data) else None,
            "max_partition_residual": max((float(x["partition_residual"]) for x in conservation if x["boundary_source"] == source), default=None),
            "closed_full_input_cycle_residual": "not_measured",
            "bootstrap_resamples": bootstrap,
        })
    gold = next((x for x in summary if x["boundary_source"] == "gold"), None)
    for row in summary:
        if gold:
            row["failure_negative_rate_delta_vs_gold"] = row["failure_negative_rate"] - gold["failure_negative_rate"] if row["failure_negative_rate"] is not None and gold["failure_negative_rate"] is not None else None
            row["recovery_positive_rate_delta_vs_gold"] = row["recovery_positive_rate"] - gold["recovery_positive_rate"] if row["recovery_positive_rate"] is not None and gold["recovery_positive_rate"] is not None else None
    write_csv_rows(output / "reward_metrics_by_family_v2.csv", family_metrics)
    write_csv_rows(output / "reward_summary_test_v2.csv", summary)
    inventory = sha256_inventory(input_paths, repo_root)
    source_map = {
        "status": "REWARD_CACHE_RECOMPUTED" if summary else "REWARD_CACHE_INCOMPLETE",
        "source_commit": _source_commit(repo_root),
        "split": split,
        "potential_lock": _relative_path(potential_lock, repo_root),
        "boundary_lock": _relative_path(boundary_lock, repo_root),
        "sources": sources,
        "missing": missing,
        "missing_item_count": len(missing) + len(inventory["missing_paths"]),
        "input_file_inventory": {key: inventory[key] for key in ("file_count", "sha256", "missing_paths")},
        "direction_metric_is_separate_from_boundary_localization": True,
        "closed_full_input_cycle_residual": "not_measured",
    }
    write_json(output / "reward_source_map.json", source_map)
    return source_map
