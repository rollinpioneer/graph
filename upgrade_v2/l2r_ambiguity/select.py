"""Freeze at most one development candidate before any new confirmation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def lock_candidate(development_root: Path, protocol_path: Path, run_root: Path, frozen_root: Path) -> dict[str, Any]:
    decision = json.loads((development_root / "development_route.json").read_text(encoding="utf-8"))
    selected = decision.get("selected_candidate_id")
    graph_paths = {
        "G0_coarse_direct": frozen_root / "graphs/G0_coarse_direct.json",
        "G1_predicate_bound": frozen_root / "graphs/G1_predicate_bound.json",
        "G2_evidence_refined": frozen_root / "graphs/G2_evidence_refined.json",
    }
    graph_hashes = {key: sha256_file(path) for key, path in graph_paths.items()}
    lock = {
        "schema": "pathgraph_l2ra_selection_lock_v1", "status": "LOCKED_BEFORE_NEW_CONFIRMATION" if selected else "DEVELOPMENT_NOT_READY",
        "selected_candidate_id": selected, "selected_candidate_config": None,
        "candidate_registry_path": str((development_root / "candidate_metrics.csv").resolve()),
        "protocol_path": str(protocol_path.resolve()), "protocol_sha256": sha256_file(protocol_path),
        "predicate_thresholds_path": str((frozen_root / "locks/predicate_thresholds.json").resolve()),
        "predicate_thresholds_sha256": sha256_file(frozen_root / "locks/predicate_thresholds.json"),
        "event_memory_code_path": str(Path(__file__).with_name("event_memory.py").resolve()),
        "event_memory_code_sha256": sha256_file(Path(__file__).with_name("event_memory.py")),
        "graph_sha256_by_id": graph_hashes, "historical_retained_graph": "G1_predicate_bound",
        "historical_l2r_status": "L2R_PARTIAL_KEEP_COARSE_GRAPH", "selection_data": "new development fit/select only",
        "confirmation_consumption": "one-shot; no resampling after STARTED",
        "confirmation_plan": {"standard_families": 24, "standard_rollouts": 96, "challenge_families": 12, "challenge_rollouts": 48},
        "api_calls": 0, "training_jobs": 0, "api_key_read": False,
    }
    if selected:
        if selected.startswith("C2_"):
            parts = selected.rsplit("_h", 1)[1].split("_l")
            lock["selected_candidate_config"] = {"family": "C2_attempt_scoped_event_memory", "hold_confirm_observations": int(parts[0]), "loss_confirm_observations": int(parts[1])}
        elif selected.startswith("C1_"):
            lock["selected_candidate_config"] = {"family": "C1_current_event_exclusion"}
    write_json(run_root / "locks/selection_lock.json", lock)
    return lock
