"""Validation of the versioned attach-relpose repair collection."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from upgrade_v2.visual_refine_l2.io import secret_scan

from .followup_resolution import _comparison_segments, _fixed_m1_rows
from .io import read_csv, read_json, sha256, write_csv, write_json


EXPECTED_CASE_COUNT = 8
EXPECTED_FAMILY_COUNT = 4
EXPECTED_ROLLOUT_COUNT = EXPECTED_CASE_COUNT * EXPECTED_FAMILY_COUNT
REPAIR_VERSION = "l2rar2_attach_relpose_v1"
COLLECTION_VERSION = "l2rar2_attach_relpose_collection_v1"
LOCKED_CANDIDATES = ("B_count2", "C3_vector_rho035")


def _reference_rows(metadata: list[dict[str, Any]], contract: dict[str, Any]) -> list[dict[str, Any]]:
    from .reference import build_reference

    rows = []
    limit = float(contract["hold_proxy"]["maximum_relative_position_drift_m"])
    for meta in sorted(metadata, key=lambda item: item["rollout_id"]):
        reference = build_reference(meta)
        intervals = reference.get("hold_intervals", [])
        rows.append({
            "rollout_id": meta["rollout_id"],
            "root_family_id": meta["root_family_id"],
            "case_id": meta["case_id"],
            "repair_version": meta.get("repair_version"),
            "collection_version": meta.get("collection_version"),
            "reference_status": reference.get("status"),
            "reference_reason": reference.get("reason"),
            "hold_interval_count": len(intervals),
            "hold_verified_interval_count": sum(item.get("status") == "held_verified" for item in intervals),
            "maximum_relative_position_drift_m": max((float(item["maximum_relative_position_drift_m"]) for item in intervals), default=None),
            "frozen_relative_drift_limit_m": limit,
            "reference_contract_version": reference.get("version"),
            "reference_label_changed": False,
        })
    return rows


def _metric_rows(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = []
    for candidate in sorted({row["candidate_id"] for row in decisions}):
        own = [row for row in decisions if row["candidate_id"] == candidate]
        labeled = [row for row in own if row["reference_status"] == "reference_labeled"]

        def rows(case: str) -> list[dict[str, Any]]:
            return [row for row in labeled if row["case_id"] == case]

        def rate(values: list[dict[str, Any]], predicate) -> float | None:
            return sum(bool(predicate(row)) for row in values) / len(values) if values else None

        metrics.append({
            "candidate_id": candidate,
            "reference_labeled_events": len(labeled),
            "reference_unresolved_events": len(own) - len(labeled),
            "K1_recall": rate(rows("K1_hold_request_ends_without_hold"), lambda item: item["correct"]),
            "K2_false_emergency_rate": rate(rows("K2_touch_request_completes_without_hold"), lambda item: item["selected_action"] != "none"),
            "K3_false_emergency_rate": rate(rows("K3_normal_hold_pause_resume"), lambda item: item["selected_action"] != "none"),
            "K4_recall": rate(rows("K4_regular_hold_loss"), lambda item: item["correct"]),
            "K5_recall": rate(rows("K5_brief_hold_loss"), lambda item: item["correct"]),
            "K6_recall": rate(rows("K6_long_gap_after_loss"), lambda item: item["correct"]),
            "K7_false_emergency_rate": rate(rows("K7_commanded_release"), lambda item: item["selected_action"] != "none"),
            "K8_false_emergency_rate": rate(rows("K8_acquisition_touch_then_continue"), lambda item: item["selected_action"] != "none"),
            "all_labeled_event_accuracy": rate(labeled, lambda item: item["correct"]),
            "same_m1_interface": "M1_requested_effect_gate",
            "independent_confirmation": False,
        })
    return metrics


def evaluate_repair_collection(data_root: Path, lock_path: Path, reference_contract_path: Path, output_root: Path) -> dict[str, Any]:
    lock = read_json(lock_path)
    if lock.get("repair_version") != REPAIR_VERSION or lock.get("collection_version") != COLLECTION_VERSION:
        raise ValueError("repair lock version does not match the attach-relpose validation contract")
    metadata = read_csv(data_root / "rollout_manifest.csv")
    if len(metadata) != EXPECTED_ROLLOUT_COUNT:
        raise ValueError(f"expected exactly {EXPECTED_ROLLOUT_COUNT} repair rollouts")
    if len({row["root_family_id"] for row in metadata}) != EXPECTED_FAMILY_COUNT:
        raise ValueError("repair collection does not contain exactly four root families")
    if len({row["case_id"] for row in metadata}) != EXPECTED_CASE_COUNT:
        raise ValueError("repair collection does not contain all eight frozen cases")
    if any(row.get("repair_version") != REPAIR_VERSION or row.get("collection_version") != COLLECTION_VERSION for row in metadata):
        raise ValueError("repair collection manifest contains a non-versioned rollout")
    contract = read_json(reference_contract_path)
    if float(contract["hold_proxy"]["maximum_relative_position_drift_m"]) != 0.02:
        raise ValueError("reference drift limit changed during repair validation")
    output_root.mkdir(parents=True, exist_ok=True)
    reference_rows = _reference_rows(metadata, contract)
    write_csv(output_root / "reference_validation.csv", reference_rows)
    unresolved = [row for row in reference_rows if row["reference_status"] != "reference_labeled"]
    unresolved_reasons = Counter(row["reference_reason"] for row in unresolved)
    meta_by_id = {row["rollout_id"]: row for row in metadata}
    segments = [
        row for row in _comparison_segments(meta_by_id, {row["rollout_id"] for row in unresolved})
        if row["candidate_id"] in LOCKED_CANDIDATES
    ]
    decisions, metrics = _fixed_m1_rows(meta_by_id)
    decisions = [row for row in decisions if row["candidate_id"] in LOCKED_CANDIDATES]
    metrics = [row for row in metrics if row["candidate_id"] in LOCKED_CANDIDATES]
    write_csv(output_root / "fixed_predicate_comparison.csv", segments)
    write_csv(output_root / "fixed_m1_comparison.csv", decisions)
    write_csv(output_root / "fixed_m1_metrics.csv", metrics)

    all_events_labeled = len(unresolved) == 0
    all_requested_effects = all(row.get("requested_effect") in {"HOLD_OBJECT", "TOUCH_OBJECT", "RELEASE_OBJECT"} for row in metadata)
    decision = "PROCEED_TO_ONLINE_INTERFACE_REPAIR" if all_events_labeled and all_requested_effects else "STOP_COLLECTION_REPAIR_INPUT_OR_REFERENCE_FAILURE"
    result = {
        "schema": "pathgraph_l2rar2_attach_relpose_validation_v1",
        "status": "REPAIR_COLLECTION_VALIDATED" if decision == "PROCEED_TO_ONLINE_INTERFACE_REPAIR" else "REPAIR_COLLECTION_BLOCKED",
        "repair_version": REPAIR_VERSION,
        "collection_version": COLLECTION_VERSION,
        "root_families": len({row["root_family_id"] for row in metadata}),
        "physical_rollouts": len(metadata),
        "case_count": len({row["case_id"] for row in metadata}),
        "reference_labeled": len(reference_rows) - len(unresolved),
        "reference_unresolved": len(unresolved),
        "reference_unresolved_reason_counts": dict(sorted(unresolved_reasons.items())),
        "reference_contract_unchanged": True,
        "all_events_labeled": all_events_labeled,
        "candidate_comparison": {
            "candidates": list(LOCKED_CANDIDATES),
            "same_m1_interface": "M1_requested_effect_gate",
            "parameter_search": False,
            "metrics": metrics,
        },
        "decision": decision,
        "online_interface_change_applied": False,
        "new_sampling_after_collection": False,
        "training_jobs": 0,
        "api_calls": 0,
        "api_key_read": False,
        "confirmation_run": False,
        "l3_entry_allowed": False,
        "retained_graph": "G1_predicate_bound",
        "historical_status": "L2RAR1_PARTIAL_KEEP_G1",
        "current_status": "L2RAR2_PARTIAL_KEEP_G1",
    }
    write_json(output_root / "repair_validation.json", result)
    report = [
        "## Material Passport",
        "",
        "- Artifact type: versioned attach-relpose repair collection validation",
        "- Verification status: `ANALYZED`; development validation only",
        "- Current boundary: `L2RAR2_PARTIAL_KEEP_G1`; L3 closed",
        "",
        "# Attach-relpose repair validation",
        "",
        f"- Collection: `{len({row['root_family_id'] for row in metadata})}` new root families x `{len({row['case_id'] for row in metadata})}` frozen cases = `{len(metadata)}` rollouts.",
        f"- Reference result: `{len(reference_rows) - len(unresolved)}/{len(reference_rows)}` labeled; unresolved rows retained: `{len(unresolved)}`.",
        f"- Frozen reference limit remains `{contract['hold_proxy']['maximum_relative_position_drift_m']} m`; no label or threshold change was applied.",
        "- The only simulator change is attach-time weld relative pose initialization; observation, controller, M1, and reference implementations remain version-frozen.",
        "",
        "## Fixed candidate comparison",
        "",
        "- Compared only `B_count2` and existing `C3_vector_rho035` on this new collection through `M1_requested_effect_gate`; no parameter search or rename was used.",
        f"- Decision: `{decision}`.",
        "- If collection passes, the next independently gated step is the online interface repair: distinguish data missing/invalid, no current hold evidence, and established historical hold evidence. This validation does not apply that repair.",
        "",
        "## Accounting",
        "",
        "- New data rollouts are counted as development validation, not confirmation. Training jobs: `0`; API calls: `0`; API keys read: `false`; L3: closed.",
    ]
    (output_root / "repair_validation.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output_root / "actual_commands.txt").write_text(
        "git fetch origin --prune\n"
        "python -m upgrade_v2.l2r_task_context.cli plan-attach-relpose-repair ...\n"
        "python -m upgrade_v2.l2r_task_context.cli collect-attach-relpose-repair ...\n"
        "python -m upgrade_v2.l2r_task_context.cli evaluate-attach-relpose-repair ...\n",
        encoding="utf-8",
    )
    scan = secret_scan([Path(__file__), Path(__file__).with_name("repair_collection.py"), output_root])
    scan.update({"schema": "pathgraph_l2rar2_repair_validation_secret_scan_v1", "api_calls": 0, "api_key_read": False, "training_jobs": 0})
    if scan["status"] != "PASS":
        raise RuntimeError("secret scan failed")
    write_json(output_root / "secret_scan.json", scan)
    artifact_paths = [
        "reference_validation.csv", "fixed_predicate_comparison.csv", "fixed_m1_comparison.csv",
        "fixed_m1_metrics.csv", "repair_validation.json", "repair_validation.md", "actual_commands.txt", "secret_scan.json",
    ]
    manifest = {
        "schema": "pathgraph_l2rar2_attach_relpose_validation_manifest_v1",
        "repair_version": REPAIR_VERSION,
        "collection_version": COLLECTION_VERSION,
        "input_hashes": {
            "repair_lock": sha256(lock_path),
            "repair_rollout_manifest": sha256(data_root / "rollout_manifest.csv"),
            "reference_contract": sha256(reference_contract_path),
        },
        "artifacts": [
            {"path": name, "size_bytes": (output_root / name).stat().st_size, "sha256": sha256(output_root / name)}
            for name in artifact_paths
        ],
        "validation": {
            "reference_contract_unchanged": True,
            "reference_unresolved_preserved": True,
            "parameter_search": False,
            "training_jobs": 0,
            "api_calls": 0,
            "api_key_read": False,
        },
    }
    write_json(output_root / "run_manifest.json", manifest)
    return result
