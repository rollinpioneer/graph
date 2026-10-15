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
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    lock_files = []
    for path in [
        protocol_path,
        development_root / "development_route.json",
        development_root / "candidate_registry.json",
        development_root / "candidate_metrics.csv",
        development_root / "candidate_fit_metrics.csv",
        development_root / "development_gates.csv",
        development_root / "event_decisions.csv",
        development_root / "content_group_manifest.csv",
    ]:
        if path.is_file():
            lock_files.append({"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    selected = route.get("selected_candidate_id")
    ready = bool(selected and route.get("status") == "DEVELOPMENT_READY" and route.get("sampling") == "control_tick_20hz")
    def file_hash(relative: str) -> dict[str, Any]:
        path = run_root / relative
        return {"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} if path.is_file() else {"path": str(path.resolve()), "sha256": None, "size_bytes": None, "status": "MISSING"}

    standard = protocol.get("standard_confirmation", {})
    challenge = protocol.get("challenge_confirmation", {})
    standard_seed = int(protocol.get("new_seeds", {}).get("standard_family", 0))
    challenge_seed = int(protocol.get("new_seeds", {}).get("challenge_family", 0))
    confirmation_families = {
        "standard": {
            "family_count": standard.get("family_count"),
            "families_per_scenario": standard.get("families_per_scenario"),
            "rollouts_per_family": standard.get("rollouts_per_family"),
            "scenarios": standard.get("scenarios", []),
            "families": [
                {"family_id": f"L2RAR1_STD_{scenario}_{index:02d}", "scenario": scenario, "family_seed": standard_seed + scenario_index * 100 + index}
                for scenario_index, scenario in enumerate(standard.get("scenarios", []))
                for index in range(int(standard.get("families_per_scenario", 0)))
            ],
        },
        "challenge": {
            "family_count": challenge.get("family_count"),
            "families_per_stratum": challenge.get("families_per_stratum"),
            "rollouts_per_family": challenge.get("rollouts_per_family"),
            "strata": challenge.get("strata", []),
            "families": [
                {"family_id": f"L2RAR1_CHALLENGE_{stratum}_{index:02d}", "stratum": stratum, "family_seed": challenge_seed + stratum_index * 100 + index}
                for stratum_index, stratum in enumerate(challenge.get("strata", []))
                for index in range(int(challenge.get("families_per_stratum", 0)))
            ],
        },
    }
    denominator_definitions = {
        "primary": "all non-masked events with reference physical_label_status=reference_labeled; family-level unit",
        "analysis": "protocol-judged observable subset reported separately; never substitutes for primary",
        "negative": "one opportunity per reference negative window; any emergency action counts as false emergency",
        "information_insufficient": "history_or_visual_unavailable windows; no evidence-based emergency action allowed",
        "duplicate_content": "content_group_manifest.csv is sensitivity metadata; primary statistics remain root_family_id",
    }
    lock = {"schema": "pathgraph_l2rar1_selection_lock_v1", "status": "LOCKED_BEFORE_CONFIRMATION" if ready else "DEVELOPMENT_NOT_READY", "selected_candidate_id": selected, "selected_candidate_config": next((row for row in candidates.get("candidates", []) if row.get("candidate_id") == selected), None), "eligible_candidates": route.get("eligible_candidates", []), "ready_for_confirmation": ready, "development_status": route.get("status"), "historical_retained_graph": "G1_predicate_bound", "reference_version": "timestamp_aligned_proxy_hold_v2", "metric_version": "prefix_event_v2", "sampling_lock": "control_tick_20hz", "locked_files": lock_files, "code_package": "upgrade_v2/l2r_hold_evidence", "interpreter": {"executable": sys.executable, "python": platform.python_version()}, "api_calls": 0, "training_jobs": 0, "api_key_read": False, "generator_hash": file_hash("data/new_development/generator_lock.json"), "reference_contract_hash": file_hash("data/new_development/reference_contract.json"), "statistics_lock": {"unit": protocol.get("statistics", {}).get("unit"), "bootstrap_resamples": protocol.get("statistics", {}).get("bootstrap_resamples"), "bootstrap_seed": protocol.get("statistics", {}).get("bootstrap_seed"), "thresholds": protocol.get("thresholds", {})}, "interface_hash": file_hash("locks/development_lock.json"), "confirmation_family_registry": confirmation_families, "denominator_definitions": denominator_definitions}
    lock["selection_sha256"] = hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest()
    write_json(output_path, lock)
    return lock
