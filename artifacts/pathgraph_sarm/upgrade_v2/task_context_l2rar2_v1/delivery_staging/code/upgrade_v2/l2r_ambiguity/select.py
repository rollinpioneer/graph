"""Freeze the complete development-selected pipeline before confirmation."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
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


def _locked_file(logical_id: str, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing lock input {logical_id}: {path}")
    return {"logical_id": logical_id, "path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _candidate_config(selected: str | None) -> dict[str, Any] | None:
    if not selected:
        return None
    if selected.startswith("C2_"):
        parts = selected.rsplit("_h", 1)[1].split("_l")
        return {"family": "C2_attempt_scoped_event_memory", "hold_confirm_observations": int(parts[0]), "loss_confirm_observations": int(parts[1])}
    if selected.startswith("C1_"):
        return {"family": "C1_current_event_exclusion"}
    raise ValueError(f"unsupported selected candidate: {selected}")


def lock_candidate(development_root: Path, protocol_path: Path, output_path: Path, frozen_root: Path) -> dict[str, Any]:
    run_root = output_path.parent.parent
    # A final_v1/locks selection lock lives one level below the run root;
    # accept both layouts so the lock records the same data used for scoring.
    if not (run_root / "data/new_development").exists() and (run_root.parent / "data/new_development").exists():
        run_root = run_root.parent
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    decision = json.loads((development_root / "development_route.json").read_text(encoding="utf-8"))
    selected = decision.get("selected_candidate_id")
    config = _candidate_config(selected)
    module_root = Path(__file__).parent
    graph_paths = {name: frozen_root / f"graphs/{name}.json" for name in ("G0_coarse_direct", "G1_predicate_bound", "G2_evidence_refined")}
    locked_paths = {
        "development_route": development_root / "development_route.json",
        "candidate_registry": development_root / "candidate_registry.json",
        "candidate_metrics": development_root / "candidate_metrics.csv",
        "candidate_gate_checks": development_root / "candidate_gate_checks.csv",
        "probe_lock": run_root / "data/new_development/probe_lock.json",
        "probe_manifest": run_root / "data/new_development/probe_manifest.csv",
        "event_reference_contract": run_root / "rounds/l2ra_2_cause_diagnosis/event_reference_contract.json",
        "event_reference_contract_lock": run_root / "locks/event_reference_contract.lock.json",
        "event_memory_code": module_root / "event_memory.py",
        "reference_events_code": module_root / "reference_events.py",
        "evaluation_code": module_root / "evaluate.py",
        "probe_adapter_code": module_root / "probes.py",
        "confirmation_code": module_root / "confirm.py",
        "selection_code": module_root / "select.py",
        "predicate_thresholds": frozen_root / "locks/predicate_thresholds.json",
        **{f"graph:{name}": path for name, path in graph_paths.items()},
    }
    locked_files = [_locked_file(logical_id, path) for logical_id, path in locked_paths.items()]
    candidate_sha = None
    if config:
        material = {
            "candidate_id": selected,
            "config": config,
            "implementation_sha256": {
                item["logical_id"]: item["sha256"]
                for item in locked_files
                if item["logical_id"] in {"event_memory_code", "evaluation_code"}
            },
        }
        candidate_sha = hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    lock = {
        "schema": "pathgraph_l2ra_selection_lock_v2",
        "status": "LOCKED_BEFORE_NEW_CONFIRMATION" if selected else "DEVELOPMENT_NOT_READY",
        "selected_candidate_id": selected,
        "selected_candidate_config": config,
        "candidate_sha256": candidate_sha,
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": sha256_file(protocol_path),
        "locked_files": locked_files,
        "interpreter": {"executable": sys.executable, "python_version": platform.python_version()},
        "reference_label_version": "pathgraph_l2ra_event_reference_contract_v1",
        "metric_contract": {
            "old_standard_thresholds": protocol["old_go_thresholds_on_standard"],
            "new_diagnostic_thresholds": protocol["new_diagnostic_thresholds"],
            "statistics": protocol["statistics"],
            "development_noninferiority": {"branch_accuracy_drop_max": 0.02, "coverage_drop_max": 0.02},
        },
        "confirmation_plan": {
            "standard": protocol["new_standard_confirmation"],
            "challenge": protocol["new_challenge_confirmation"],
        },
        "graph_sha256_by_id": {name: sha256_file(path) for name, path in graph_paths.items()},
        "historical_retained_graph": "G1_predicate_bound",
        "historical_l2r_status": "L2R_PARTIAL_KEEP_COARSE_GRAPH",
        "selection_data": "new development fit/select and legacy dev_select only; legacy confirmation excluded",
        "confirmation_consumption": "one-shot; only the same lock and pre-registered families may resume after STARTED",
        "api_calls": 0,
        "training_jobs": 0,
        "api_key_read": False,
    }
    write_json(output_path, lock)
    return lock
