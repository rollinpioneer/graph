from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any

from .replay import replay
from .temporal_scoring import score_event, truth_for

# The protocol's locked operating point plus immediate sensitivity checks.
# This keeps grouped validation reproducible without an unbounded search.
WINDOWS = (250_000_000, 500_000_000, 750_000_000)
THETA_MOTION = (0.20, 0.35, 0.50)
THETA_SEP = (0.05, 0.10, 0.20)


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _select(train: list[dict[str, Any]], method: str, replay_cache: dict[tuple[str, str, int, float, float], dict[str, Any]]) -> tuple[int, float, float]:
    combos = [(window, theta, sep) for window, theta, sep in itertools.product(WINDOWS, THETA_MOTION, THETA_SEP)]
    if method == "B1":
        combos = [(500_000_000, 0.35, sep) for sep in THETA_SEP]
    elif method == "B2":
        combos = [(window, theta, 0.10) for window, theta in itertools.product(WINDOWS, THETA_MOTION)]
    else:
        combos = [(window, theta, sep) for window, theta, sep in itertools.product(WINDOWS, THETA_MOTION, THETA_SEP)]
    best = None
    for window, theta, sep in combos:
        false = early = 0; hits = 0; positives = 0; latencies: list[int] = []
        for record in train:
            key_id = str(record["episode_id"])
            cache_key = (key_id, method, window, theta, sep)
            result = replay_cache.get(cache_key)
            if result is None:
                result = replay(record, method, window_ns=window, theta_motion=theta, theta_sep=sep)
                replay_cache[cache_key] = result
            score = score_event(truth=truth_for(record), onset_ns=record.get("physical_loss_onset_ns"), end_ns=result["evidence_end_ns"], first_evidence_ns=result["first_evidence_ns"])
            false += score == "FALSE_LOSS_EVENT"; early += score == "EARLY_LOSS_EVENT"; positives += truth_for(record) == "LOSS"; hits += score == "ON_TIME"
            if score == "ON_TIME": latencies.append(result["first_evidence_ns"] - int(record["physical_loss_onset_ns"]))
        key = (false + early, -(hits / max(1, positives)), max(latencies) if latencies else 10**30, window, theta, sep)
        if best is None or key < best[0]:
            best = (key, (window, theta, sep))
    return best[1] if best else (500_000_000, 0.35, 0.10)


def compare(episodes_path: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    records = _load(episodes_path)
    families = sorted({record["root_family_id"] for record in records})
    methods = {"B1": [], "B2": [], "B3": []}
    predictions: list[dict[str, Any]] = []; splits = []
    replay_cache: dict[tuple[str, str, int, float, float], dict[str, Any]] = {}
    for held in families:
        train = [record for record in records if record["root_family_id"] != held]
        test = [record for record in records if record["root_family_id"] == held]
        splits.append({"held_out_family": held, "train_ids": [r["episode_id"] for r in train], "test_ids": [r["episode_id"] for r in test]})
        for method in methods:
            window, theta, sep = _select(train, method, replay_cache)
            for record in test:
                cache_key = (str(record["episode_id"]), method, window, theta, sep)
                result = replay_cache.get(cache_key)
                if result is None:
                    result = replay(record, method, window_ns=window, theta_motion=theta, theta_sep=sep)
                    replay_cache[cache_key] = result
                score = score_event(truth=truth_for(record), onset_ns=record.get("physical_loss_onset_ns"), end_ns=result["evidence_end_ns"], first_evidence_ns=result["first_evidence_ns"])
                predictions.append({"episode_id": record["episode_id"], "family": held, "method": method, "window_ns": window, "theta_motion": theta, "theta_sep": sep, "first_evidence_ns": result["first_evidence_ns"], "score": score, "truth": truth_for(record), "physical_loss_onset_ns": record.get("physical_loss_onset_ns"), "evidence_end_ns": result["evidence_end_ns"]})
    with (output / "split_manifest.json").open("w", encoding="utf-8") as stream: json.dump({"schema": "l2rar2_r27_grouped_split_v1", "splits": splits}, stream, indent=2, sort_keys=True)
    fields = list(predictions[0]) if predictions else ["episode_id"]
    with (output / "per_episode_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(predictions)
    summary = []
    for method in methods:
        subset = [row for row in predictions if row["method"] == method]
        summary.append({"method": method, "episodes": len(subset), "loss": sum(row["truth"] == "LOSS" for row in subset), "on_time": sum(row["score"] == "ON_TIME" for row in subset), "early": sum(row["score"] == "EARLY_LOSS_EVENT" for row in subset), "false": sum(row["score"] == "FALSE_LOSS_EVENT" for row in subset), "missed": sum(row["score"] in {"MISSED", "RIGHT_CENSORED"} for row in subset)})
    (output / "by_horizon_metrics.csv").write_text("method,horizon_ns,episodes,on_time,early,false,missed\n" + "\n".join(f"{row['method']},{horizon},{row['episodes']},{row['on_time']},{row['early']},{row['false']},{row['missed']}" for row in summary for horizon in (100_000_000,250_000_000,500_000_000,750_000_000,1_000_000_000)) + "\n", encoding="utf-8")
    (output / "by_family_metrics.csv").write_text("method,family,episodes,on_time,early,false\n" + "\n".join(f"{method},{family},{sum(1 for x in predictions if x['method']==method and x['family']==family)},{sum(x['score']=='ON_TIME' for x in predictions if x['method']==method and x['family']==family)},{sum(x['score']=='EARLY_LOSS_EVENT' for x in predictions if x['method']==method and x['family']==family)},{sum(x['score']=='FALSE_LOSS_EVENT' for x in predictions if x['method']==method and x['family']==family)}" for method in methods for family in families) + "\n", encoding="utf-8")
    return {"schema": "l2rar2_r27_outer_validation_v1", "episodes": len(records), "predictions": len(predictions), "methods": summary, "physical_executions": 0, "mujoco_imported": False}
