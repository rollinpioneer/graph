from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


HORIZONS = (100_000_000, 250_000_000, 500_000_000, 750_000_000, 1_000_000_000)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _times(rows: Iterable[dict[str, Any]], valid_only: bool = False) -> list[int]:
    values = []
    for row in rows:
        if valid_only and (row.get("frame_missing") or row.get("detector_error") or row.get("object_centroid") is None or row.get("gripper_centroid") is None):
            continue
        try:
            values.append(int(row["physical_time_ns"]))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(set(values))


def audit_windows(records: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    all_gaps: list[int] = []
    all_visual_gaps: list[int] = []
    for record in records:
        observations = record.get("observations", [])
        times = _times(observations)
        visual_times = _times(observations, valid_only=True)
        trace_times = []
        evidence_end = int(record.get("evidence_end_ns") or 0)
        onset = record.get("physical_loss_onset_ns")
        truth = record.get("reference_state", "UNRESOLVED")
        for row in observations:
            try:
                trace_times.append(int(row["physical_time_ns"]))
            except (KeyError, TypeError, ValueError):
                pass
        gaps = [b - a for a, b in zip(times, times[1:])]
        visual_gaps = [b - a for a, b in zip(visual_times, visual_times[1:])]
        all_gaps.extend(gaps); all_visual_gaps.extend(visual_gaps)
        duplicate_keys: dict[tuple[int, str], int] = {}
        hash_times: dict[str, set[int]] = {}
        for item in observations:
            key = (int(item.get("physical_time_ns", -1)), str(item.get("jpeg_sha256", "")))
            duplicate_keys[key] = duplicate_keys.get(key, 0) + 1
            hash_times.setdefault(key[1], set()).add(key[0])
        duplicate_count = sum(max(0, count - 1) for count in duplicate_keys.values())
        repeated_hash_different_time = sum(1 for key, values in hash_times.items() if key and len(values) > 1)
        duplicate_rows.append({
            "episode_id": record.get("episode_id"),
            "same_time_same_jpeg_duplicate_count": duplicate_count,
            "jpeg_hash_reused_at_different_times": repeated_hash_different_time,
            "same_time_group_count": sum(1 for t in set(times) if sum(1 for x in observations if int(x.get("physical_time_ns", -1)) == t) > 1),
        })
        duration = max(times) - min(times) if times else 0
        pre_action = max(0, evidence_end - int(record.get("episode_start_ns") or 0))
        post_onset = max(0, evidence_end - int(onset)) if onset is not None else None
        can_score: dict[str, bool] = {}
        for horizon in HORIZONS:
            if truth == "LOSS" and onset is not None:
                can_score[str(horizon)] = evidence_end >= int(onset) + horizon
            else:
                can_score[str(horizon)] = pre_action >= horizon
        if truth == "LOSS" and onset is not None and evidence_end < int(onset) + 750_000_000:
            censor_reason = "RIGHT_CENSORED"
        elif any(gap > 100_000_000 for gap in gaps):
            censor_reason = "MISSING_INTERVAL"
        else:
            censor_reason = ""
        row = {
            "episode_id": record.get("episode_id"),
            "root_family_id": record.get("root_family_id"),
            "case_id": record.get("case_id"),
            "arm_id": record.get("arm_id"),
            "reference_state": truth,
            "raw_duration_ns": record.get("last_raw_physics_ns", duration),
            "pre_action_duration_ns": pre_action,
            "post_onset_available_ns": post_onset,
            "actual_physics_step_count": len(trace_times),
            "actual_visual_count": len(visual_times),
            "actual_visual_interval_min_ns": min(visual_gaps) if visual_gaps else None,
            "actual_visual_interval_median_ns": statistics.median(visual_gaps) if visual_gaps else None,
            "actual_visual_interval_max_ns": max(visual_gaps) if visual_gaps else None,
            "same_time_group_count": duplicate_rows[-1]["same_time_group_count"],
            "missing_intervals_gt_100ms": sum(gap > 100_000_000 for gap in gaps),
            "baseline_available": bool(record.get("baseline_ready_ns") is not None),
            "censor_reason": censor_reason,
        }
        row.update({
            f"can_score_{int(h) // 1_000_000}ms": value
            for h, value in can_score.items()
        })
        rows.append(row)
        for horizon in HORIZONS:
            family_rows.append({
                "root_family_id": record.get("root_family_id"),
                "horizon_ns": horizon,
                "episode_id": record.get("episode_id"),
                "can_score": can_score[str(horizon)],
                "visual_count": len(visual_times),
            })
    fields = list(rows[0]) if rows else ["episode_id"]
    with (output / "horizon_availability.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    fields = list(duplicate_rows[0]) if duplicate_rows else ["episode_id"]
    with (output / "duplicate_event_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(duplicate_rows)
    with (output / "window_support_by_family.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(family_rows[0]) if family_rows else ["root_family_id"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(family_rows)
    dump_json(output / "sampling_gap_summary.json", {
        "schema": "l2rar2_r27_sampling_gap_summary_v1",
        "episodes": len(records),
        "all_visual_interval_min_ns": min(all_visual_gaps) if all_visual_gaps else None,
        "all_visual_interval_median_ns": statistics.median(all_visual_gaps) if all_visual_gaps else None,
        "all_visual_interval_max_ns": max(all_visual_gaps) if all_visual_gaps else None,
        "gaps_gt_100ms": sum(gap > 100_000_000 for gap in all_gaps),
        "gaps_gt_100ms_fraction": (sum(gap > 100_000_000 for gap in all_gaps) / len(all_gaps)) if all_gaps else 0.0,
    })
    return {"episodes": len(records), "output_files": 4}


def source_inventory(resources: dict[str, Any], output: Path) -> None:
    rows = []
    for source in resources.get("sources", []):
        root = Path(source["path"])
        if not root.is_dir():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append((source["name"], str(path), path.stat().st_size, digest))
    with output.open("w", encoding="utf-8") as stream:
        stream.write("source\tabsolute_path\tsize_bytes\tsha256\n")
        for row in rows:
            stream.write("\t".join(map(str, row)) + "\n")
