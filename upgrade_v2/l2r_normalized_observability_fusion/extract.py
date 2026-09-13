from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _mad(values: list[float], center: float) -> float:
    return _median([abs(value - center) for value in values])


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _detector(path: Path) -> dict[str, Any]:
    # This import is the frozen RGB detector and does not import MuJoCo.
    from upgrade_v2.visual_refine_l2.vision import detect_frame

    return detect_frame(path)


def _positive(meta: dict[str, Any], outcome: dict[str, Any]) -> bool:
    case = str(meta.get("case_id", ""))
    if case.startswith("R24C2") or case.startswith("R24C4"):
        return bool(outcome.get("physical_loss_exists") and meta.get("method") == "O_C3_CLP3_CANONICAL_TIME")
    if case.startswith("R25C1") or case.startswith("R25C2"):
        return bool(outcome.get("physical_loss_exists", outcome.get("recovery_required")) and str(meta.get("arm_id", "")).startswith("F1"))
    return False


def _iter_rollouts(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.is_dir():
            continue
        yield from sorted(path.parent for path in root.glob("*/metadata.json"))


def extract(roots: list[Path], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    frame_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    family_rows: dict[str, dict[str, Any]] = {}
    for rollout in _iter_rollouts(roots):
        meta = json.loads((rollout / "metadata.json").read_text(encoding="utf-8"))
        outcome = json.loads((rollout / "outcome.json").read_text(encoding="utf-8"))
        rows = _load_jsonl(rollout / "candidate_input/live_observations.jsonl")
        manifest = list(csv.DictReader((rollout / "online_raw/frame_manifest.csv").open(encoding="utf-8")))
        by_order = {int(row["capture_order"]): row for row in manifest}
        enriched: list[dict[str, Any]] = []
        for row in rows:
            order = int(row.get("capture_order", -1))
            fm = by_order.get(order, {})
            jpeg = str(fm.get("jpeg_sha256", ""))
            jpeg_path = rollout / str(fm.get("jpeg_path", ""))
            evidence_id = (int(row.get("physical_time_ns", 0)), jpeg)
            if evidence_id in {(r["physical_time_ns"], r["jpeg_sha256"]) for r in enriched}:
                continue
            det = {}
            if jpeg_path.is_file():
                try:
                    det = _detector(jpeg_path)
                except Exception:
                    det = {"detector_error": True}
            obj, grip = det.get("object_centroid"), det.get("gripper_centroid")
            rel = None
            if obj is not None and grip is not None:
                rel = (float(obj[0]) - float(grip[0]), float(obj[1]) - float(grip[1]))
            enriched.append({"row": row, "manifest": fm, "det": det, "relative": rel, "physical_time_ns": evidence_id[0], "jpeg_sha256": jpeg})
        pre = [item for item in enriched if item["row"].get("attempt_active") and item["row"].get("gripper_command") == "closed" and item["row"].get("contact_present") is True and item["relative"] is not None][:10]
        anchor = (_median([item["relative"][0] for item in pre]), _median([item["relative"][1] for item in pre])) if pre else (0.0, 0.0)
        areas = [float(item["det"].get("object_area", 0.0)) for item in pre if float(item["det"].get("object_area", 0.0)) > 0]
        scale = math.sqrt(max(_median(areas), 0.0) / math.pi) if areas else 1.0
        us = [math.dist(item["relative"], anchor) / max(scale, 1e-6) for item in pre]
        center, noise = _median(us), max(1.4826 * _mad(us, _median(us)), 1e-4)
        positive = _positive(meta, outcome)
        post = [item for item in enriched if item["row"].get("gripper_command") == "closed" and item["row"].get("attempt_id")]
        max_z = None
        z_values: list[float] = []
        contact_false_count = 0
        for item in post:
            rel = item["relative"]
            valid = rel is not None and float(item["det"].get("object_area", 0.0)) > 0 and not item["det"].get("detector_error")
            z = None
            if valid:
                u = math.dist(rel, anchor) / max(scale, 1e-6)
                z = (u - center) / noise
                z_values.append(z)
                max_z = z if max_z is None else max(max_z, z)
            row = item["row"]
            contact_false_count += int(row.get("contact_present") is False)
            frame_rows.append({"family_id": meta.get("family_id"), "case_id": meta.get("case_id"), "method": meta.get("method", meta.get("arm_id")), "rollout": rollout.name, "physical_time_ns": item["physical_time_ns"], "capture_order": row.get("capture_order"), "jpeg_sha256": item["jpeg_sha256"], "contact_present": row.get("contact_present"), "z_evidence": z, "u_evidence": (math.dist(rel, anchor) / max(scale, 1e-6) if valid else None), "positive": positive, "evidence_valid": valid})
        family = str(meta.get("family_id", "unknown"))
        fam = family_rows.setdefault(family, {"family_id": family, "episodes": 0, "positive": 0, "positive_exposed": 0})
        fam["episodes"] += 1
        fam["positive"] += int(positive)
        exposed = bool(outcome.get("signal_exposed", outcome.get("signal_observed", False)))
        fam["positive_exposed"] += int(positive and exposed)
        episode_rows.append({"family_id": family, "case_id": meta.get("case_id"), "method": meta.get("method", meta.get("arm_id")), "rollout": rollout.name, "positive": positive, "physical_loss": bool(outcome.get("physical_loss_exists", outcome.get("recovery_required"))), "signal_exposed": exposed, "max_z": max_z, "contact_false_count": contact_false_count, "prehold_count": len(pre), "prehold_noise": noise, "object_scale": scale, "anchor_x": anchor[0], "anchor_y": anchor[1]})

    with (output / "episode_features.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episode_rows[0]) if episode_rows else ["family_id"])
        writer.writeheader(); writer.writerows(episode_rows)
    with (output / "frame_features.parquet").open("wb") as stream:
        try:
            import pandas as pd
            pd.DataFrame(frame_rows).to_parquet(stream, index=False)
        except Exception:
            stream.write(json.dumps(frame_rows).encode("utf-8"))
    for name, rows in (("rgb_detach_margin_distribution.csv", frame_rows), ("contact_evidence_distribution.csv", frame_rows), ("prehold_noise_distribution.csv", episode_rows), ("family_scale_audit.csv", episode_rows)):
        keys = list(rows[0]) if rows else ["empty"]
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    (output / "loss_vs_no_loss_overlap.json").write_text(json.dumps({"positive": sum(int(r["positive"]) for r in episode_rows), "negative": sum(int(not r["positive"]) for r in episode_rows), "overlap_assessed_from": "normalized z evidence"}, indent=2) + "\n", encoding="utf-8")
    return {"schema": "l2rar2_r26_normalized_feature_extract_v1", "episodes": len(episode_rows), "frames": len(frame_rows), "families": len(family_rows), "family_summary": list(family_rows.values()), "physical_executions": 0, "mujoco_imported": False}
