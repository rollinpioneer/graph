"""Event and boundary metrics shared by weak labels, baselines, and models."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import load_episode, read_csv, write_csv
from .event_schema import EVENT_NAMES


def _match(pred: np.ndarray, gold: np.ndarray, tolerance: int) -> tuple[int, int, int, list[int]]:
    used: set[int] = set(); matched: list[int] = []; gold_index = np.where(gold)[0]
    for p in np.where(pred)[0]:
        left = int(np.searchsorted(gold_index, p - tolerance, side="left"))
        right = int(np.searchsorted(gold_index, p + tolerance, side="right"))
        choices = [int(g) for g in gold_index[left:right] if int(g) not in used]
        if choices:
            g = min(choices, key=lambda x: abs(int(x) - int(p))); used.add(int(g)); matched.append(abs(int(g) - int(p)))
    return len(matched), int(pred.sum()) - len(matched), int(gold.sum()) - len(matched), matched


def prf(pred: np.ndarray, gold: np.ndarray, tolerance: int = 0) -> dict[str, float]:
    tp, fp, fn, distances = _match(pred.astype(bool), gold.astype(bool), tolerance)
    precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "mae": float(np.mean(distances)) if distances else float("nan"), "tp": tp, "fp": fp, "fn": fn}


def evaluate_predictions(dataset: Path, prediction_root: Path, method: str, split: str | None, tolerance: int = 2, event_source: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [r for r in read_csv(dataset / "episode_manifest.csv") if split is None or r["split"] == split]
    aggregate: dict[str, list[np.ndarray]] = defaultdict(list); counts: dict[str, int] = defaultdict(int)
    per_event: list[dict[str, Any]] = []
    total_pred: list[np.ndarray] = []; total_gold: list[np.ndarray] = []; unknowns: list[np.ndarray] = []
    for row in rows:
        ep = load_episode(row)
        with np.load(prediction_root / f"{row['episode_id']}.npz") as payload:
            if "event_argmax" in payload: predicted_event = payload["event_argmax"]
            elif "event_prediction" in payload: predicted_event = payload["event_prediction"]
            else: predicted_event = np.zeros(len(ep["gold_event_id"]), dtype=np.int8)
            boundary = payload["boundary_prediction"] if "boundary_prediction" in payload else payload["boundary_probability"] >= 0.5
            unknown = payload["unknown"] if "unknown" in payload else np.zeros(len(boundary), dtype=np.int8)
        gold = ep["gold_event_id"]; total_pred.append(np.asarray(boundary, bool)); total_gold.append(gold != 0); unknowns.append(unknown)
        for event in range(1, 11):
            aggregate[f"pred_{event}"].append(predicted_event == event); aggregate[f"gold_{event}"].append(gold == event)
    all_pred = np.concatenate(total_pred); all_gold = np.concatenate(total_gold); b1 = prf(all_pred, all_gold, 1); b2 = prf(all_pred, all_gold, tolerance)
    for event in range(1, 11):
        metric = prf(np.concatenate(aggregate[f"pred_{event}"]), np.concatenate(aggregate[f"gold_{event}"]), tolerance)
        per_event.append({"method": method, "split": split or "all", "event_id": event, "event": EVENT_NAMES[event], **metric})
    macro = float(np.mean([row["f1"] for row in per_event]))
    summary = {"method": method, "split": split or "all", "episodes": len(rows), "boundary_precision_tol1": b1["precision"], "boundary_recall_tol1": b1["recall"], "boundary_f1_tol1": b1["f1"], "boundary_precision_tol2": b2["precision"], "boundary_recall_tol2": b2["recall"], "boundary_f1_tol2": b2["f1"], "boundary_mae": b2["mae"], "event_macro_f1": macro, "recovery_start_recall": next(x["recall"] for x in per_event if x["event_id"] == 4), "contact_reestablished_recall": next(x["recall"] for x in per_event if x["event_id"] == 5), "unknown_rate": float(np.concatenate(unknowns).mean()), "events_per_episode": float(all_pred.sum() / max(len(rows), 1))}
    return summary, per_event


def evaluate_weak(dataset: Path, posterior_root: Path, output: Path, per_event_out: Path, report: Path, tolerance: int) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []; events_out: list[dict[str, Any]] = []
    for mode_dir in sorted(x for x in posterior_root.iterdir() if x.is_dir()):
        summary, events = evaluate_predictions(dataset, mode_dir, mode_dir.name, None, tolerance)
        rows_out.append(summary); events_out.extend(events)
    write_csv(output, rows_out, list(rows_out[0])); write_csv(per_event_out, events_out, list(events_out[0]))
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text("# U2 weak-event evaluation\n\n" + "\n".join(f"- {row['method']}: boundary F1±2={row['boundary_f1_tol2']:.4f}; recovery recall={row['recovery_start_recall']:.4f}" for row in rows_out) + "\n", encoding="utf-8")
    return rows_out


def write_evaluation(dataset: Path, prediction_root: Path, method: str, split: str, output: Path, per_event: Path, report: Path, tolerance: int = 2) -> dict[str, Any]:
    summary, details = evaluate_predictions(dataset, prediction_root, method, split, tolerance)
    write_csv(output, [summary], list(summary)); write_csv(per_event, details, list(details[0]))
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text("# Segmentation evaluation\n\n" + "\n".join(f"- {k}: {v}" for k, v in summary.items()) + "\n", encoding="utf-8")
    return summary
