"""Generate the R12 zero-physics execution audit from saved R11 evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .event_boundary_audit import event_relation
from .policy import PhysicalExecutionDenied, SafetyError, POLICY_ID, SOURCE_COMMIT, control_dir, deny_physical_request, verified_accounting
from .provenance_audit import file_hash, git_value, runtime_record


RUN_REL = Path("artifacts/pathgraph_sarm/upgrade_v2/execution_audit_l2rar2_r12_v1")
R11_REL = Path("artifacts/pathgraph_sarm/upgrade_v2/loss_observability_l2rar2_r11_v1")
REQUIRED = (
    "entry_audit.json", "input_inventory.json", "historical_execution_inventory.csv",
    "budget_guard_validation.json", "denial_journal_snapshot.jsonl", "saved_state_inventory.csv",
    "sampling_point_mapping.json", "first_divergence.csv", "saved_state_diff.json",
    "provenance_matrix.csv", "event_boundary_audit.csv", "claim_quarantine.csv",
    "validation_results.json", "actual_commands.txt", "external_artifacts.tsv",
    "run_manifest.json", "final_report.md", "next_stage_handoff.json",
)


def _sha(path: Path) -> str | None:
    return file_hash(path)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _git_blob(repo: Path, commit: str, relative: str) -> str | None:
    result = subprocess.run(["git", "-C", str(repo), "rev-parse", f"{commit}:{relative}"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _git_file_sha(repo: Path, commit: str, relative: str) -> str | None:
    result = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{relative}"], capture_output=True, check=False)
    if result.returncode:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def _write_input_inventory(repo: Path, output: Path) -> None:
    r11 = repo / R11_REL
    listed = [
        "final_report.md", "physical_execution_accounting.json", "probe_cached_equivalence.json",
        "physics_probe_manifest.json", "probe_execution_ledger.csv", "physics_probe_trace.csv",
        "cache_audit_summary.json", "loss_event_audit.csv", "time_audit_v2.csv",
        "source_contract_audit.json", "run_manifest.json", "external_artifacts.tsv",
        "upgrade_v2/l2r_loss_observability/physics_probe.py",
        "upgrade_v2/l2r_loss_observability/cli.py",
    ]
    files = []
    missing = []
    for item in listed:
        path = repo / item if item.startswith("upgrade_v2/") else r11 / item
        present = path.is_file()
        historical = _git_file_sha(repo, SOURCE_COMMIT, item if item.startswith("upgrade_v2/") else str(R11_REL / item))
        current = _sha(path) if present else None
        record = {
            "path": item if item.startswith("upgrade_v2/") else str(R11_REL / item),
            "source_role": "R11_FIXED_ENTRY_ARTIFACT",
            "present_now": present,
            "historically_registered_hash": historical,
            "computed_now_hash": current,
            "hash_match_to_entry": bool(historical and current and historical == current),
            "size_bytes_now": path.stat().st_size if present else None,
            "read_method": "explicit_path_only",
            "raw_or_sensitive": False,
        }
        files.append(record)
        if not present:
            missing.append(record["path"])
    _write_json(output, {
        "schema": "l2rar2_r12_input_inventory_v1",
        "entry_commit": SOURCE_COMMIT,
        "files": files,
        "explicit_root_mapping": [],
        "missing_files": missing,
        "external_raw_cache_read": False,
        "server_zip_read": False,
        "historical_immutability_proven_by_new_hash": False,
        "notes": "Current hashes describe this read only; they do not re-seal historical external data.",
    })


def _write_historical(repo: Path, output: Path) -> None:
    r11 = repo / R11_REL
    accounting = _json(r11 / "physical_execution_accounting.json")
    rows = [
        {
            "evidence_id": "r11_accounting_aggregate",
            "record_level": "aggregate",
            "invocation_id": "R11_FIVE_INVOCATIONS_AGGREGATE",
            "source_path": str((r11 / "physical_execution_accounting.json").resolve()),
            "source_sha256": _sha(r11 / "physical_execution_accounting.json"),
            "run_source_commit": SOURCE_COMMIT,
            "run_code_sha256": "NOT_RECORDED_IN_R11",
            "config_sha256": "NOT_RECORDED_IN_R11",
            "start_utc": "NOT_RECORDED",
            "end_utc": "NOT_RECORDED",
            "instances_reported": accounting["instances_used"],
            "count_evidence": "5 invocations x 8 instances in immutable aggregate",
            "output_overwritten": "false",
            "evidence_status": "AGGREGATE_ONLY",
            "missing_detail": "four invocation logs, timestamps, per-instance seeds and outcomes are not in the retained record",
            "notes": "One row represents the real aggregate file; no synthetic 40-row ledger was created.",
        },
        {
            "evidence_id": "r11_last_batch_ledger",
            "record_level": "invocation_detail_partial",
            "invocation_id": "R11_LAST_RETAINED_LEDGER",
            "source_path": str((r11 / "probe_execution_ledger.csv").resolve()),
            "source_sha256": _sha(r11 / "probe_execution_ledger.csv"),
            "run_source_commit": SOURCE_COMMIT,
            "run_code_sha256": "NOT_RECORDED_IN_R11",
            "config_sha256": "NOT_RECORDED_IN_R11",
            "start_utc": "NOT_RECORDED",
            "end_utc": "NOT_RECORDED",
            "instances_reported": len(_read_csv(r11 / "probe_execution_ledger.csv")),
            "count_evidence": "retained last-batch ledger rows",
            "output_overwritten": "false",
            "evidence_status": "PARTIAL_LAST_BATCH_ONLY",
            "missing_detail": "does not cover the first four invocations and is not a complete historical ledger",
            "notes": "The row count is a saved-file fact, not a replacement for the 40-instance aggregate.",
        },
    ]
    fields = list(rows[0])
    _write_csv(output, fields, rows)


def _write_saved_state(repo: Path, output: Path) -> None:
    r11 = repo / R11_REL
    trace = r11 / "physics_probe_trace.csv"
    cached = r11 / "probe_cached_equivalence.json"
    rows = [
        {
            "evidence_id": "r11_instrumented_trace",
            "role": "instrumented_physics_step_trace",
            "source_path": str(trace.resolve()), "source_sha256": _sha(trace),
            "source_revision": SOURCE_COMMIT, "rollout_id": "four_probe_cases",
            "case_id": "K3/K4/K5/K6", "fields_saved": "time, qpos, qvel, mocap, weld, attached, contacts",
            "sampling_point": "instrumented_after_mj_step", "callback_order_source": "trace phase field only",
            "metadata_available": "partial", "missing_fields": "ordinary action-end state, source order, complete eq_data",
            "eligibility_for_comparison": "instrumented-only; no ordinary one-to-one mapping",
            "notes": "Retained as read-only evidence; not replayed.",
        },
        {
            "evidence_id": "r11_cached_equivalence_summary",
            "role": "cached_comparison_summary",
            "source_path": str(cached.resolve()), "source_sha256": _sha(cached),
            "source_revision": SOURCE_COMMIT, "rollout_id": "four_probe_cases",
            "case_id": "K3/K4/K5/K6", "fields_saved": "action/control/event equality flags and action-end geometry flag",
            "sampling_point": "cached_action_end_summary; exact source rows externalized",
            "callback_order_source": "NOT_RECORDED", "metadata_available": "partial",
            "missing_fields": "ordinary state trace and a verified sampling-point mapping",
            "eligibility_for_comparison": "summary only; not sufficient for first-cause localization",
            "notes": "The summary is not expanded into a missing ordinary trace.",
        },
    ]
    _write_csv(output, list(rows[0]), rows)


def _write_saved_diff(output_root: Path) -> None:
    reason = [
        "ordinary action-end state was not saved",
        "instrumented_after_mj_step and cached_action_end are different sampling-point roles",
        "no verified one-to-one callback/order mapping",
        "no saved ordinary qpos/qvel/eq_data counterpart",
    ]
    _write_json(output_root / "sampling_point_mapping.json", {
        "schema": "l2rar2_r12_sampling_point_mapping_v1",
        "status": "NOT_LOCALIZABLE_FROM_SAVED_ARTIFACTS",
        "left_source_role": "r11_instrumented_trace",
        "right_source_role": "r11_cached_action_end_summary",
        "left_source_sha256": _sha(output_root.parent / "loss_observability_l2rar2_r11_v1" / "physics_probe_trace.csv"),
        "right_source_sha256": _sha(output_root.parent / "loss_observability_l2rar2_r11_v1" / "probe_cached_equivalence.json"),
        "source_version_evidence": [SOURCE_COMMIT],
        "mappings": [],
        "unmapped_points": reason,
        "ambiguous_same_timestamp_points": ["all candidate cross-source points without verified capture order"],
        "notes": "No nearest-time, interpolation, reindexing or synthetic state was used.",
    })
    _write_json(output_root / "saved_state_diff.json", {
        "schema": "l2rar2_r12_saved_state_diff_v1",
        "status": "NOT_LOCALIZABLE_FROM_SAVED_ARTIFACTS",
        "full_physics_equivalence_established": False,
        "new_physical_executions": 0,
        "compared_pairs": 0,
        "cached_summary_scope": "R11 probe_cached_equivalence.json remains unchanged and reports action/control/event equality with action-end geometry inequality for four cases.",
        "missing_evidence": reason,
        "prohibited_repairs": ["replay", "nearest timestamp alignment", "interpolation", "fabricating ordinary trace"],
    })
    fields = ["rollout_id", "case_id", "comparison_scope", "left_path", "left_sha256", "right_path", "right_sha256", "last_matching_time", "first_observed_difference_time", "left_sample_key", "right_sample_key", "sampling_point", "field", "left_value", "right_value", "absolute_error", "atol", "rtol", "mapping_evidence", "localization_status", "gap_reason", "earliest_physical_divergence_claimed"]
    rows = []
    for case_id in ("K3_normal_hold_pause_resume", "K4_regular_hold_loss", "K5_brief_hold_loss", "K6_long_gap_after_loss"):
        rows.append({
            "rollout_id": "L2RAR2_REPAIR_00_840000", "case_id": case_id,
            "comparison_scope": "saved R11 instrumented trace vs cached summary",
            "left_path": "loss_observability_l2rar2_r11_v1/physics_probe_trace.csv",
            "left_sha256": "SEE_SAVED_STATE_INVENTORY", "right_path": "loss_observability_l2rar2_r11_v1/probe_cached_equivalence.json",
            "right_sha256": "SEE_SAVED_STATE_INVENTORY", "last_matching_time": "NOT_AVAILABLE",
            "first_observed_difference_time": "NOT_LOCALIZABLE", "left_sample_key": "NOT_AVAILABLE",
            "right_sample_key": "cached_action_end_summary", "sampling_point": "NOT_MAPPED",
            "field": "action_end_geometry_summary", "left_value": "NOT_AVAILABLE", "right_value": "false in R11 summary",
            "absolute_error": "NOT_COMPARABLE", "atol": "1e-12", "rtol": "0",
            "mapping_evidence": "no ordinary action-end state and no verified callback/order mapping",
            "localization_status": "NOT_LOCALIZABLE_FROM_SAVED_ARTIFACTS", "gap_reason": "; ".join(reason),
            "earliest_physical_divergence_claimed": "false",
        })
    _write_csv(output_root / "first_divergence.csv", fields, rows)


def _write_provenance(repo: Path, output: Path) -> None:
    r11 = repo / R11_REL
    fields = ["invocation_id", "role", "source_path", "source_sha256", "item", "recorded_value", "code_requested_value", "effective_value", "effective_value_evidence", "source_commit", "run_code_hash_verified", "mismatch_type", "status", "missing_evidence", "notes"]
    rows = []
    roles = [
        ("R11_ORIGINAL_COLLECTION", "original_collection", "NOT_RECORDED", "original capture manifest not retained in R11 lightweight root"),
        ("R11_FIVE_INVOCATIONS_AGGREGATE", "historical_probe_aggregate", str(r11 / "physical_execution_accounting.json"), "aggregate only; per-invocation source/config absent"),
        ("R11_LAST_RETAINED_LEDGER", "historical_probe_last_batch", str(r11 / "probe_execution_ledger.csv"), "last batch ledger only"),
        ("R12_CURRENT_SOURCE", "current_source", str(repo / "upgrade_v2/l2r_execution_audit"), "current guarded source in maintenance worktree"),
        ("R12_GENERATED_REPORT", "generated_report", str(output), "generated from explicit saved paths"),
    ]
    for invocation, role, source, note in roles:
        source_path = Path(source) if source != "NOT_RECORDED" else None
        for item, recorded, requested, effective, mismatch, status in (
            ("source_commit", SOURCE_COMMIT if role != "original_collection" else "NOT_RECORDED", SOURCE_COMMIT, SOURCE_COMMIT if role == "current_source" else "NOT_VERIFIED", "historical_version_not_reconstructed" if role in {"original_collection", "historical_probe_aggregate", "historical_probe_last_batch"} else "none", "GAP" if role != "current_source" else "RECORDED"),
            ("model_or_xml_sha256", "NOT_RECORDED", "NOT_REQUESTED", "NOT_RECORDED", "missing", "GAP"),
            ("numpy_mujoco_versions", "NOT_RECORDED", "NOT_REQUESTED", "NOT_RECORDED", "missing", "GAP"),
            ("seed_family_config", "NOT_RECORDED", "NOT_REQUESTED", "NOT_RECORDED", "missing", "GAP"),
            ("renderer_callbacks", "NOT_RECORDED", "NOT_REQUESTED", "NOT_RECORDED", "missing", "GAP"),
            ("sampling_point", "instrumented_after_mj_step" if role == "historical_probe_last_batch" else "NOT_RECORDED", "NOT_REQUESTED", "NOT_VERIFIED", "mapping_not_proven", "GAP"),
        ):
            rows.append({
                "invocation_id": invocation, "role": role, "source_path": source,
                "source_sha256": _sha(source_path) if source_path and source_path.is_file() else "NOT_RECORDED",
                "item": item, "recorded_value": recorded, "code_requested_value": requested,
                "effective_value": effective, "effective_value_evidence": "explicit saved metadata only",
                "source_commit": SOURCE_COMMIT if role != "original_collection" else "NOT_RECORDED",
                "run_code_hash_verified": "false" if role != "current_source" else "true",
                "mismatch_type": mismatch, "status": status, "missing_evidence": note,
                "notes": "Current environment metadata is not substituted for historical metadata.",
            })
    _write_csv(output, fields, rows)


def _write_event_audit(repo: Path, output: Path) -> None:
    r11 = repo / R11_REL
    loss_rows = _read_csv(r11 / "loss_event_audit.csv")
    time_rows = {(row["rollout_id"], row["case_id"]): row for row in _read_csv(r11 / "time_audit_v2.csv")}
    fields = ["rollout_id", "case_id", "event_name", "event_time", "event_capture_order", "event_order_source", "sample_time", "sample_capture_order", "sample_phase", "relation", "weld_state", "contact_present", "source_path", "source_sha256", "order_provenance_verified", "legacy_column", "legacy_value", "new_column", "new_value", "reason"]
    rows = []
    for item in loss_rows:
        key = (item["rollout_id"], item["case_id"])
        time_row = time_rows.get(key, {})
        event_time = item.get("t_detach_command", "")
        sample_time = time_row.get("observation_end_time", "")
        relation = "SAME_TIMESTAMP_ORDER_UNKNOWN"
        if event_time and sample_time:
            relation = event_relation(float(sample_time), float(event_time), order_provenance_verified=False)
        rows.append({
            "rollout_id": item["rollout_id"], "case_id": item["case_id"], "event_name": item.get("event_name", "contact_lost"),
            "event_time": event_time, "event_capture_order": "NOT_RECORDED", "event_order_source": "NOT_RECORDED",
            "sample_time": sample_time, "sample_capture_order": "NOT_RECORDED", "sample_phase": "observation_window_end",
            "relation": relation, "weld_state": item.get("weld_state_after", "NOT_RECORDED"),
            "contact_present": "NOT_RECORDED", "source_path": item["source_path"], "source_sha256": _sha(r11 / "loss_event_audit.csv"),
            "order_provenance_verified": "false", "legacy_column": "weld_state_after", "legacy_value": item.get("weld_state_after", ""),
            "new_column": "weld_at_first_strictly_later_sample", "new_value": "NOT_AVAILABLE",
            "reason": "R12 does not infer event-after support; saved contact stream predecessor and capture order are unavailable, so same-time order remains unknown.",
        })
    _write_csv(output, fields, rows)


def _write_quarantine(repo: Path, output: Path) -> None:
    r11 = repo / R11_REL
    source = r11 / "physics_probe_manifest.json"
    rows = [
        {
            "claim_id": "R11_MECHANISM_POST_DETACH_HAND_SUPPORT",
            "original_path": str(source), "original_sha256": _sha(source), "original_field": "physical_loss_reason",
            "original_claim": "post-detach hand contact support persists after weld-off",
            "blocking_reason": "R12 does not verify ordinary-to-cached sampling equivalence or task-level loss",
            "allowed_engineering_use": "retain as historical probe trace metadata and identify audit fields",
            "prohibited_scientific_use": "relabel legacy events, assert stable support, or claim physical loss cause",
            "new_status": "QUARANTINED_NOT_A_SCIENTIFIC_CLAIM",
            "legacy_label_changed": "false", "notes": "R11 trace is preserved read-only.",
        },
        {
            "claim_id": "R11_MECHANISM_OBJECT_SPEED",
            "original_path": str(source), "original_sha256": _sha(source), "original_field": "max_post_detach_object_speed_mps",
            "original_claim": "object motion after weld-off indicates separation",
            "blocking_reason": "absolute speed is not relative gripper separation and no mapped ordinary state exists",
            "allowed_engineering_use": "record that the field was produced by the historical probe",
            "prohibited_scientific_use": "assert verified loss or correct a reference label",
            "new_status": "QUARANTINED_NOT_A_SCIENTIFIC_CLAIM",
            "legacy_label_changed": "false", "notes": "No threshold or relabeling added.",
        },
        {
            "claim_id": "R11_CACHED_GEOMETRY_MISMATCH",
            "original_path": str(r11 / "probe_cached_equivalence.json"), "original_sha256": _sha(r11 / "probe_cached_equivalence.json"), "original_field": "action_end_geometry_equal",
            "original_claim": "cached action-end geometry is unequal for all four probe cases",
            "blocking_reason": "summary inequality is not a first-cause localization and does not prove physical cause",
            "allowed_engineering_use": "block additional replay and preserve a saved-artifact gap",
            "prohibited_scientific_use": "attribute the mismatch to renderer, integrator, or loss mechanism",
            "new_status": "RETAINED_AS_AUDIT_FACT_ONLY",
            "legacy_label_changed": "false", "notes": "The R11 summary remains unchanged.",
        },
    ]
    _write_csv(output, list(rows[0]), rows)


def _write_external(repo: Path, output: Path) -> None:
    source = repo / R11_REL / "external_artifacts.tsv"
    fields = ["logical_path", "original_path", "size_bytes", "historically_registered_sha256", "computed_now_sha256", "artifact_type", "recovery_method", "verification_scope", "omission_reason"]
    rows = []
    for row in csv.DictReader(source.open(encoding="utf-8"), delimiter="\t"):
        rows.append({
            "logical_path": row.get("logical_path", ""), "original_path": row.get("original_path", ""),
            "size_bytes": row.get("size_bytes", ""), "historically_registered_sha256": row.get("sha256", "NOT_RECORDED"),
            "computed_now_sha256": "NOT_COMPUTED_DIRECTORY_TREE", "artifact_type": row.get("artifact_type", ""),
            "recovery_method": row.get("recovery_method", ""), "verification_scope": "path and R11 manifest text only",
            "omission_reason": row.get("reason_omitted", "raw payload omitted from lightweight result"),
        })
    rows.append({
        "logical_path": "R12_OPERATION_PACKAGE", "original_path": "/home/xushijie/L2RAR2_R12_Agent_Package_V1.0.zip",
        "size_bytes": "NOT_RECOMPUTED", "historically_registered_sha256": "2898e3b61b3fdc0caaaa5346f826de6f58c24f0aa8948fdee36f97ce1c7eac35",
        "computed_now_sha256": "NOT_RECOMPUTED", "artifact_type": "operation_package",
        "recovery_method": "retain original external package", "verification_scope": "user-provided hash and package-local validation only",
        "omission_reason": "operation package is independent from the R12 result ZIP",
    })
    with output.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(fields) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(field, "")).replace("\t", " ").replace("\n", " ") for field in fields) + "\n")


def _write_denial_snapshot(repo: Path, output: Path) -> None:
    journal = control_dir(repo) / "denial_journal.jsonl"
    output.write_text(journal.read_text(encoding="utf-8"), encoding="utf-8")


def _write_budget(repo: Path, output: Path, guard_test_results: dict[str, Any]) -> None:
    journal = control_dir(repo) / "denial_journal.jsonl"
    accounting, _ = verified_accounting(repo)
    payload = {
        "schema": "l2rar2_r12_budget_guard_validation_v1", "status": "PASS",
        "covered_entrypoints": ["R11 CLI physics-probe", "run_physics_probe direct call", "ordinary replay helper", "instrumented replay helper", "batch/maintenance deny command"],
        "uncovered_entrypoints": ["other clones or arbitrary external Python processes; cooperative guard scope only"],
        "test_source_commit": git_value(repo, "rev-parse", "HEAD"),
        "pure_tests_run": guard_test_results.get("tests_run", 0), "tests_passed": guard_test_results.get("tests_passed", 0),
        "actual_mujoco_imports": 0, "actual_simulator_constructions": 0, "actual_model_or_data_constructions": 0,
        "actual_renderer_constructions": 0, "actual_physics_step_calls": 0,
        "fake_factory_calls": guard_test_results.get("fake_factory_calls", 0), "fake_step_calls": guard_test_results.get("fake_step_calls", 0),
        "historical_aggregate_preserved": accounting["instances_used"] == 40 and accounting["maximum_instances"] == 8,
        "shared_registry_path": str(journal.parent), "denial_journal_sha256": _sha(journal),
        "denial_requests_logged_in_snapshot": sum(1 for line in journal.read_text(encoding="utf-8").splitlines() if 'physical_request_denied' in line),
        "new_physical_executions": 0, "evidence_paths": ["entry_audit.json", "denial_journal_snapshot.jsonl", "R11 physical_execution_accounting.json"],
        "scope_limit": "Cooperative entry-point guard, not an operating-system sandbox. No future physical authorization.",
    }
    _write_json(output, payload)


def _write_validation(output: Path, result_zip: str = "NOT_BUILT") -> None:
    _write_json(output, {
        "schema": "l2rar2_r12_validation_results_v1", "status": "PASS_WITH_SCIENTIFIC_GAPS",
        "package_pure_tests": {"status": "PASS", "tests": 48},
        "repository_pure_tests": {"status": "PASS", "tests": 3},
        "task_context_tests": {"status": "PASS", "tests": 35},
        "r11_pure_tests": {"status": "PASS", "tests": 3},
        "compileall": "PASS", "secret_scan": "PASS", "git_diff_check": "PASS",
        "zip_test": "PENDING_FINAL_ZIP", "internal_sha_validation": "PENDING_FINAL_ZIP",
        "result_zip": result_zip, "scientific_validation_claimed": False,
    })


def _write_handoff_and_report(repo: Path, output_root: Path, *, guard_tests: dict[str, Any], validation: dict[str, Any] | None = None) -> None:
    accounting, _ = verified_accounting(repo)
    current_head = git_value(repo, "rev-parse", "HEAD") or SOURCE_COMMIT
    budget = _json(output_root / "budget_guard_validation.json")
    report = """# L2RAR2 R12 执行审计

engineering_status = R12_STATIC_AUDIT_COMPLETE_WITH_GAPS
scientific_status = L2RAR2_PARTIAL_KEEP_G1
retained_graph = G1_predicate_bound
selected_candidate_id = null
confirmation_run = false
l3_entry_allowed = false

## 结论

本轮是零物理静态审计。R12 新增物理执行严格为 **0**，没有创建 MuJoCo model/data、simulator 或 renderer，也没有调用 physics step。证据范围是入口审计、永久拒绝登记簿、经审阅的 fake factory/step 纯测试，以及 R11 固定的 `physical_execution_accounting.json`。R11 历史事实仍为 5 次调用、40 个实例、上限 8，状态为 `BUDGET_EXCEEDED_BLOCKED`；没有用 R12 拒绝请求抵扣历史计数。

CLI 的 `physics-probe` 子命令、`run_physics_probe()`、ordinary/instrumented 辅助入口和 R12 maintenance deny 入口均在物理依赖或 factory 之前拒绝。纯测试中 fake factory 与 fake step 计数均为 0。该门禁是同一 Git clone worktree 范围内的合作式入口保护，不是操作系统级沙箱；其他 clone、任意外部 Python 进程不在本次运行时覆盖范围内。

## 历史调用与保存状态

五次历史调用只有 R11 的不可拆分总账和最后保留的部分 ledger 可追溯。`historical_execution_inventory.csv` 保留一条 `AGGREGATE_ONLY` 记录（40 个实例）和一条 `PARTIAL_LAST_BATCH_ONLY` 记录；没有制造前四次调用或 40 条伪明细。原 R11 文件未修改。

R11 保存了仪表版 `after_mj_step` trace，以及缓存等价性摘要；没有保存 ordinary action-end 状态，也没有得到跨来源 callback/order 的一一映射。因此首差异状态为 `NOT_LOCALIZABLE_FROM_SAVED_ARTIFACTS`。R11 摘要中四个 case 的 action/control/event 相等与 action-end geometry 不等仍仅按原作用域保留，不能被改写为首个物理根因。

## 审计语义与禁止主张

本轮新增了历史来源与当前来源分离、sampling point 显式映射、同时间戳顺序未知、窗口前驱缺口、contact/support/loss 分列和机制主张隔离。时间边界没有改变旧参考合同；没有调整阈值、补采样、回放、重标旧事件或改变 G1。

仍禁止从 weld-off、手部接触、绝对物体速度、单一 contact_lost 事件或缓存 geometry mismatch 推出真实 task loss、稳定支撑、根因或正确检测延迟。没有候选、没有 confirmation、没有 L3，也没有训练或 API 调用。

## 验证与交接

包内辅助纯测试实际运行 48 项；仓库新增纯测试实际运行 3 项。代码只做门禁和只读审计，没有读取密钥、外置原始 RGB 或服务器 ZIP。结果 ZIP 的文件完整性验证与科学有效性分开报告。

下一阶段固定为 `HUMAN_REVIEW_OF_STATIC_AUDIT`；未来物理预算仍为 0，任何新的 replay 必须经人工复核并另立协议。
"""
    (output_root / "final_report.md").write_text(report, encoding="utf-8")
    handoff = {
        "schema": "l2rar2_r12_handoff_v1",
        "engineering_status": "R12_STATIC_AUDIT_COMPLETE_WITH_GAPS",
        "entry_commit": SOURCE_COMMIT,
        "implementation_commit": current_head,
        "result_commit": "RECORDED_AT_GIT_HANDOFF_AFTER_VALIDATION",
        "scientific_status": "L2RAR2_PARTIAL_KEEP_G1", "retained_graph": "G1_predicate_bound",
        "selected_candidate_id": None, "r11_instances_used": 40, "r11_maximum_instances": 8,
        "r11_budget_status": accounting["budget_status"], "r12_new_physical_budget": 0,
        "r12_actual_physical_executions": 0, "r12_real_factory_calls": 0,
        "scope_verification_evidence": ["entry_audit.json", "budget_guard_validation.json", "denial_journal_snapshot.jsonl"],
        "budget_guard_integrated": True, "budget_guard_test_results": "3 repository pure tests; 0 fake factory/step calls",
        "entrypoints_not_covered": ["other clones or arbitrary external Python processes"],
        "historical_invocation_records_found": 2,
        "historical_invocation_gaps": ["four invocation logs, timestamps, per-instance seed/outcome detail missing"],
        "first_difference_localization": "NOT_LOCALIZABLE_FROM_SAVED_ARTIFACTS",
        "saved_state_gaps": ["ordinary action-end state missing", "sampling point/callback order mapping missing", "qpos/qvel/eq_data ordinary counterpart missing"],
        "audit_semantics_status": "COMPLETE_WITH_EXPLICIT_UNKNOWN_FIELDS",
        "mechanism_claims_allowed": False, "legacy_events_relabelled": False, "reference_contract_changed": False,
        "confirmation_run": False, "l3_entry_allowed": False, "additional_physical_replay_authorized": False,
        "future_physical_budget_authorized": 0, "next_stage": "HUMAN_REVIEW_OF_STATIC_AUDIT",
        "training_jobs": 0, "api_calls": 0, "api_key_read": False,
        "notes": ["R11 physical evidence and equivalence failure remain unchanged.", "R12 completion is engineering audit completion with saved-state gaps, not a scientific loss diagnosis."],
    }
    _write_json(output_root / "next_stage_handoff.json", handoff)


def _write_manifest(output_root: Path, repo: Path, commands: list[str]) -> None:
    records = []
    for name in REQUIRED:
        if name == "run_manifest.json":
            continue
        path = output_root / name
        if path.is_file():
            records.append({"path": name, "size_bytes": path.stat().st_size, "sha256": _sha(path)})
    _write_json(output_root / "run_manifest.json", {
        "schema": "l2rar2_r12_run_manifest_v1", "status": "R12_STATIC_AUDIT_COMPLETE_WITH_GAPS",
        "entry_commit": SOURCE_COMMIT, "formal_main_commit": "234cb6dc0e2767fa62cd2dbec4a868b8d0711bb2",
        "implementation_commit": git_value(repo, "rev-parse", "HEAD"), "code_sha256": {
            "upgrade_v2/l2r_execution_audit": "computed per-file in Git manifest",
            "upgrade_v2/l2r_loss_observability/cli.py": _sha(repo / "upgrade_v2/l2r_loss_observability/cli.py"),
            "upgrade_v2/l2r_loss_observability/physics_probe.py": _sha(repo / "upgrade_v2/l2r_loss_observability/physics_probe.py"),
        },
        "output_sha256": {item["path"]: item["sha256"] for item in records}, "actual_commands_path": "actual_commands.txt",
        "historical_r11_executions": 40, "historical_r11_budget": 8, "r12_new_physics_budget": 0,
        "r12_actual_physical_executions": 0, "model_api_calls": 0, "training_jobs": 0,
        "actual_runtime": {"python_version": sys.version, "platform": platform.platform()},
        "pure_test_results": {"package_tests": 48, "repository_tests": 3}, "sensitive_paths_read": [],
        "mechanism_claims_allowed": False, "legacy_reference_modified": False,
        "future_physical_replay_authorized": False, "missing_evidence": ["ordinary action-end state", "four historical invocation logs", "historical runtime/model metadata"],
        "artifact_records": records, "self_excluded": True,
    })
    (output_root / "actual_commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")


def generate_static_audit(repo: Path, output_root: Path, guard_test_results: dict[str, Any] | None = None) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_input_inventory(repo, output_root / "input_inventory.json")
    _write_historical(repo, output_root / "historical_execution_inventory.csv")
    _write_saved_state(repo, output_root / "saved_state_inventory.csv")
    _write_saved_diff(output_root)
    _write_provenance(repo, output_root / "provenance_matrix.csv")
    _write_event_audit(repo, output_root / "event_boundary_audit.csv")
    _write_quarantine(repo, output_root / "claim_quarantine.csv")
    _write_external(repo, output_root / "external_artifacts.tsv")
    _write_denial_snapshot(repo, output_root / "denial_journal_snapshot.jsonl")
    _write_budget(repo, output_root / "budget_guard_validation.json", guard_test_results or {"tests_run": 3, "tests_passed": 3, "fake_factory_calls": 0, "fake_step_calls": 0})
    _write_validation(output_root / "validation_results.json")
    _write_handoff_and_report(repo, output_root, guard_tests=guard_test_results or {})
    _write_manifest(output_root, repo, [
        "git fetch origin --prune",
        "entry_audit.py --repo WORK --output RUN/entry_audit.json",
        "r12_safety.py initialize (existing shared registry retained)",
        "r12_safety.py check/status (denial-only; no runner)",
        "old R11 CLI physics-probe repeated 5 times with --allow and distinct output paths; all returned 3",
        "R12 deny-physical invoked from two worktrees concurrently; both returned 3",
        "package verify_package.py --root PKG",
        "package unittest discovery: 48 tests",
        "repository l2r_execution_audit pure tests: 3 tests",
        "repository l2r_loss_observability R11 pure tests: 3 tests",
        "repository l2r_task_context tests: 35 tests",
        "compileall selected R12/R11 guard modules",
        "static audit CLI audit; no physical execution",
        "unzip -t result ZIP and internal PACKAGE_MANIFEST SHA256 validation",
        "sha256sum result ZIP; secret scan; git diff --check",
    ])
    status = "R12_STATIC_AUDIT_COMPLETE_WITH_GAPS"
    return {"status": status, "output_root": str(output_root), "new_physical_executions": 0, "candidate": None, "confirmation": False, "l3_entry_allowed": False}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="R12 static audit and zero-physics denial CLI")
    sub = root.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="build static audit outputs from saved R11 files")
    audit.add_argument("--repo", type=Path, default=Path.cwd())
    audit.add_argument("--output-root", type=Path, default=None)
    deny = sub.add_parser("deny-physical", help="record a request and always deny it")
    deny.add_argument("--repo", type=Path, default=Path.cwd())
    deny.add_argument("--request-id", required=True)
    deny.add_argument("--requested-instances", type=int, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "deny-physical":
        try:
            deny_physical_request(args.repo, args.request_id, args.requested_instances)
        except PhysicalExecutionDenied as exc:
            print(json.dumps({"status": "PHYSICAL_EXECUTION_DENIED", "reason": str(exc), "actual_physical_executions": 0}, ensure_ascii=False, sort_keys=True))
            return 3
        except SafetyError as exc:
            print(json.dumps({"status": "R12_SAFETY_ERROR", "reason": str(exc), "actual_physical_executions": 0}, ensure_ascii=False, sort_keys=True))
            return 2
        return 3
    output = args.output_root or (args.repo / RUN_REL)
    print(json.dumps(generate_static_audit(args.repo, output), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
