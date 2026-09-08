from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

from .inputs import sha256_file, write_json


def select_and_lock(development_root: Path, protocol_path: Path, output_path: Path, run_root: Path) -> dict[str, Any]:
    route = json.loads((development_root / "development_route.json").read_text(encoding="utf-8"))
    candidates = json.loads((development_root / "candidate_registry.json").read_text(encoding="utf-8"))
    lock_files = []
    for path in [protocol_path, development_root / "development_route.json", development_root / "candidate_registry.json", development_root / "candidate_metrics.csv", development_root / "development_gates.csv"]:
        if path.is_file():
            lock_files.append({"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    selected = route.get("selected_candidate_id")
    ready = bool(selected and route.get("status") == "DEVELOPMENT_READY" and route.get("sampling") == "control_tick_20hz")
    lock = {"schema": "pathgraph_l2rar1_selection_lock_v1", "status": "LOCKED_BEFORE_CONFIRMATION" if ready else "DEVELOPMENT_NOT_READY", "selected_candidate_id": selected, "selected_candidate_config": next((row for row in candidates.get("candidates", []) if row.get("candidate_id") == selected), None), "eligible_candidates": route.get("eligible_candidates", []), "ready_for_confirmation": ready, "development_status": route.get("status"), "historical_retained_graph": "G1_predicate_bound", "reference_version": "timestamp_aligned_proxy_hold_v2", "metric_version": "prefix_event_v2", "sampling_lock": "control_tick_20hz", "locked_files": lock_files, "code_package": "upgrade_v2/l2r_hold_evidence", "interpreter": {"executable": sys.executable, "python": platform.python_version()}, "api_calls": 0, "training_jobs": 0, "api_key_read": False}
    lock["selection_sha256"] = hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest()
    write_json(output_path, lock)
    return lock
