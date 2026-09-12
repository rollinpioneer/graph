from __future__ import annotations

from pathlib import Path
from typing import Any

from upgrade_v2.l2r_rgb_temporal_confirmation.detector_adapter import detect_rollout as detect_rgb_rollout

from .io_utils import read_csv, read_json, read_jsonl, sha256, write_json, write_jsonl


def detect_rollout(rollout_root: Path) -> dict[str, Any]:
    status = detect_rgb_rollout(rollout_root)
    reference_path = rollout_root / "reference/physical_reference.json"
    reference = read_json(reference_path)
    if reference["case_id"].startswith("T3_"):
        manifest = read_csv(rollout_root / "online_raw/frame_manifest.csv")
        transport = [int(row["capture_order"]) for row in manifest
                     if str(row.get("phase", "")).startswith("transport")]
        if len(transport) < 4:
            raise RuntimeError("T3_TRANSPORT_CAPTURE_4_MISSING")
        target = transport[3]
        path = rollout_root / "candidate_input/contact_proxy.jsonl"
        rows = read_jsonl(path)
        matches = [row for row in rows if int(row["capture_order"]) == target]
        if len(matches) != 1:
            raise RuntimeError("T3_DROPOUT_TARGET_NOT_UNIQUE")
        matches[0]["contact_present"] = False
        write_jsonl(path, rows)
        status["T3_contact_proxy_dropout_capture_order"] = target
        status["contact_proxy_sha256"] = sha256(path)
        write_json(rollout_root / "candidate_input/detector_status.json", status)
    return status


def detect_confirmation(confirmation_root: Path) -> list[dict[str, Any]]:
    return [{"rollout": path.parent.name, **detect_rollout(path.parent)}
            for path in sorted(confirmation_root.glob("*/metadata.json"))]
