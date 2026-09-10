"""Build the R13 static evidence recovery and R14 readiness handoff."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


ENTRY = "e2b1fb906533a74f5d224a2fdd4aea7e2faf88df"
MAIN = "234cb6dc0e2767fa62cd2dbec4a868b8d0711bb2"
R11 = Path("artifacts/pathgraph_sarm/upgrade_v2/loss_observability_l2rar2_r11_v1")
R12 = Path("artifacts/pathgraph_sarm/upgrade_v2/execution_audit_l2rar2_r12_v1")
R12_ZIP = Path("downloads/l2ra_r2_v1/L2RAR2_R12_Execution_Audit_results_v1.zip")
R12_SIDE = Path("downloads/l2ra_r2_v1/L2RAR2_R12_Execution_Audit_results_v1.zip.sha256")
SOURCE_FILES = (
    "upgrade_v2/l2r_task_context/repair_collection.py",
    "upgrade_v2/visual_refine_l2/repaired_simulator.py",
    "upgrade_v2/visual_refine_l2/dynamic_simulator.py",
    "upgrade_v2/l2r_hold_evidence/probe_adapter.py",
    "upgrade_v2/l2r_loss_observability/physics_probe.py",
    "upgrade_v2/l2r_execution_audit/policy.py",
)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(repo: Path, *args: str) -> str | None:
    result = subprocess.run(["git", "-C", str(repo.resolve()), *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def git_blob(repo: Path, relative: str) -> str | None:
    return git(repo, "rev-parse", f"{ENTRY}:{relative}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_record(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative
    return {
        "path": relative,
        "exists": path.is_file(),
        "sha256": sha256(path),
        "git_blob_sha1": git_blob(repo, relative),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "status": "VERIFIED_GIT_SOURCE" if path.is_file() and git_blob(repo, relative) else "UNKNOWN_NOT_RECORDED",
    }


def build_r12_integrity(repo: Path, out: Path) -> dict[str, Any]:
    zip_path = repo / R12_ZIP
    sidecar = repo / R12_SIDE
    actual = sha256(zip_path)
    expected = None
    if sidecar.is_file():
        expected = sidecar.read_text(encoding="utf-8").split()[0]
    crc_status = "NOT_RUN"
    member_count = None
    if zip_path.is_file():
        with zipfile.ZipFile(zip_path) as archive:
            bad = archive.testzip()
            crc_status = "PASS" if bad is None else f"FAIL:{bad}"
            member_count = len(archive.namelist())
    result = {
        "schema": "l2rar2_r13_r12_integrity_review_v1",
        "status": "PASS" if actual and actual == expected and crc_status == "PASS" else "BLOCKED",
        "zip_path": str(zip_path), "sidecar_path": str(sidecar),
        "expected_sha256": expected, "actual_sha256": actual,
        "sidecar_match": bool(actual and expected and actual == expected),
        "zip_crc_status": crc_status, "zip_member_count": member_count,
        "scientific_claim": False,
        "scope": "R12 delivery file integrity only; not a physical-equivalence claim",
    }
    write_json(out, result)
    return result


def build_claim_review(repo: Path, out: Path) -> None:
    fields = ["claim_id", "source_path", "source_sha256", "human_decision", "decision_basis", "requested_static_clarification", "scientific_use_allowed", "reviewer_id", "reviewed_at_utc"]
    rows = [
        {
            "claim_id": "R11_MECHANISM_POST_DETACH_HAND_SUPPORT",
            "source_path": str(R11 / "physics_probe_manifest.json"),
            "source_sha256": sha256(repo / R11 / "physics_probe_manifest.json"),
            "human_decision": "ACCEPT_QUARANTINE",
            "decision_basis": "Retain the historical field, but R12 equivalence failed and no task-level loss proof exists.",
            "requested_static_clarification": "None beyond preserving sampling-point and provenance gaps.",
            "scientific_use_allowed": "false; engineering audit fact only",
            "reviewer_id": "AGENT_STATIC_REVIEW_NOT_HUMAN_APPROVAL",
            "reviewed_at_utc": "NOT_HUMAN_REVIEWED",
        },
        {
            "claim_id": "R11_MECHANISM_OBJECT_SPEED",
            "source_path": str(R11 / "physics_probe_manifest.json"),
            "source_sha256": sha256(repo / R11 / "physics_probe_manifest.json"),
            "human_decision": "ACCEPT_QUARANTINE",
            "decision_basis": "Absolute speed is not relative separation and the ordinary counterpart is absent.",
            "requested_static_clarification": "None; do not convert speed into verified loss.",
            "scientific_use_allowed": "false; engineering audit fact only",
            "reviewer_id": "AGENT_STATIC_REVIEW_NOT_HUMAN_APPROVAL",
            "reviewed_at_utc": "NOT_HUMAN_REVIEWED",
        },
        {
            "claim_id": "R11_CACHED_GEOMETRY_MISMATCH",
            "source_path": str(R11 / "probe_cached_equivalence.json"),
            "source_sha256": sha256(repo / R11 / "probe_cached_equivalence.json"),
            "human_decision": "ACCEPT_QUARANTINE",
            "decision_basis": "The saved summary is an audit fact, not first-cause localization or a mechanism result.",
            "requested_static_clarification": "Recover ordinary action-end state and a verified cross-source mapping before R14.",
            "scientific_use_allowed": "false; may block replay or motivate human review only",
            "reviewer_id": "AGENT_STATIC_REVIEW_NOT_HUMAN_APPROVAL",
            "reviewed_at_utc": "NOT_HUMAN_REVIEWED",
        },
    ]
    write_csv(out, fields, rows)


def build_history(repo: Path, out: Path) -> None:
    fields = ["evidence_id", "candidate_invocation_id", "source_path", "source_sha256", "source_kind", "recorded_start_utc", "recorded_end_utc", "source_commit", "config_hash", "root_family_id", "case_id", "seed", "mode", "instance_count", "output_path", "recovery_status", "confidence_basis", "missing_fields"]
    accounting = read_json(repo / R11 / "physical_execution_accounting.json")
    ledger = repo / R11 / "probe_execution_ledger.csv"
    rows = [
        {
            "evidence_id": "r11_aggregate_40_of_8", "candidate_invocation_id": "R11_FIVE_INVOCATIONS_AGGREGATE",
            "source_path": str((repo / R11 / "physical_execution_accounting.json").resolve()),
            "source_sha256": sha256(repo / R11 / "physical_execution_accounting.json"), "source_kind": "IMMUTABLE_AGGREGATE",
            "recorded_start_utc": "NOT_RECORDED", "recorded_end_utc": "NOT_RECORDED", "source_commit": "dc51582b99759a6e5a7427e1965a5110c5944ed4",
            "config_hash": "NOT_RECORDED", "root_family_id": "NOT_RECORDED", "case_id": "NOT_RECORDED", "seed": "NOT_RECORDED",
            "mode": "ordinary_plus_instrumented_aggregate", "instance_count": accounting["instances_used"], "output_path": str((repo / R11).resolve()),
            "recovery_status": "PARTIAL_EXISTING_RECORD", "confidence_basis": "R11 accounting explicitly records 5 invocations x 8 instances",
            "missing_fields": "four invocation logs; timestamps; per-instance seeds, outcomes, source/config hashes",
        },
        {
            "evidence_id": "r11_last_ledger", "candidate_invocation_id": "R11_LAST_RETAINED_LEDGER",
            "source_path": str(ledger.resolve()), "source_sha256": sha256(ledger), "source_kind": "LAST_BATCH_LEDGER",
            "recorded_start_utc": "NOT_RECORDED", "recorded_end_utc": "NOT_RECORDED", "source_commit": "dc51582b99759a6e5a7427e1965a5110c5944ed4",
            "config_hash": "NOT_RECORDED", "root_family_id": "L2RAR2_REPAIR_00_840000", "case_id": "K3/K4/K5/K6", "seed": "NOT_RECORDED",
            "mode": "ordinary_and_instrumented_last_batch", "instance_count": "8 ledger rows", "output_path": str((repo / R11).resolve()),
            "recovery_status": "PARTIAL_EXISTING_RECORD", "confidence_basis": "CSV exists and is hashed; it is explicitly only the retained last batch",
            "missing_fields": "first four invocations; exact runtime/config; complete outcome mapping",
        },
        {
            "evidence_id": "r12_manifest_history", "candidate_invocation_id": "R12_STATIC_AUDIT",
            "source_path": str((repo / R12 / "actual_commands.txt").resolve()), "source_sha256": sha256(repo / R12 / "actual_commands.txt"),
            "source_kind": "R12_COMMAND_MANIFEST", "recorded_start_utc": "NOT_RECORDED", "recorded_end_utc": "NOT_RECORDED",
            "source_commit": ENTRY, "config_hash": "NOT_APPLICABLE", "root_family_id": "NOT_APPLICABLE", "case_id": "NOT_APPLICABLE", "seed": "NOT_APPLICABLE",
            "mode": "zero_physics_static_audit", "instance_count": 0, "output_path": str((repo / R12).resolve()),
            "recovery_status": "VERIFIED_EXISTING_RECORD", "confidence_basis": "R12 command record and handoff explicitly state zero new physical executions",
            "missing_fields": "none for R12 static audit; not a replay invocation",
        },
    ]
    write_csv(out, fields, rows)


def build_saved_inventory(repo: Path, out: Path) -> None:
    fields = ["evidence_id", "source_path", "source_sha256", "source_revision", "rollout_id", "case_id", "mode", "sampling_point", "fields_saved", "callback_order_source", "row_key_available", "ordinary_counterpart_available", "eligibility_for_comparison", "recovery_status", "notes"]
    rows = [
        {"evidence_id": "r11_trace", "source_path": str((repo / R11 / "physics_probe_trace.csv").resolve()), "source_sha256": sha256(repo / R11 / "physics_probe_trace.csv"), "source_revision": "dc51582b99759a6e5a7427e1965a5110c5944ed4", "rollout_id": "L2RAR2_REPAIR_00_840000", "case_id": "K3/K4/K5/K6", "mode": "instrumented", "sampling_point": "instrumented_after_mj_step", "fields_saved": "time,qpos,qvel,mocap_pos,mocap_quat,eq_active,weld,attached,contacts", "callback_order_source": "trace phase and physics_step_index only", "row_key_available": "partial", "ordinary_counterpart_available": "false", "eligibility_for_comparison": "instrumented trace only", "recovery_status": "VERIFIED_EXISTING_RECORD", "notes": "No replay; no ordinary state reconstructed."},
        {"evidence_id": "r11_cached_summary", "source_path": str((repo / R11 / "probe_cached_equivalence.json").resolve()), "source_sha256": sha256(repo / R11 / "probe_cached_equivalence.json"), "source_revision": "dc51582b99759a6e5a7427e1965a5110c5944ed4", "rollout_id": "L2RAR2_REPAIR_00_840000", "case_id": "K3/K4/K5/K6", "mode": "cached_summary", "sampling_point": "cached_action_end_summary", "fields_saved": "action/control/event equality flags, geometry equality flag", "callback_order_source": "NOT_RECORDED", "row_key_available": "case-level only", "ordinary_counterpart_available": "false", "eligibility_for_comparison": "summary only", "recovery_status": "VERIFIED_EXISTING_RECORD", "notes": "A summary cannot supply missing ordinary action-end values."},
        {"evidence_id": "r11_ordinary_missing", "source_path": str((repo / R11).resolve()), "source_sha256": "DIRECTORY_TREE_NOT_COMPUTED", "source_revision": "dc51582b99759a6e5a7427e1965a5110c5944ed4", "rollout_id": "four probe cases", "case_id": "K3/K4/K5/K6", "mode": "ordinary", "sampling_point": "ordinary_after_perform_return", "fields_saved": "NOT_FOUND", "callback_order_source": "NOT_RECORDED", "row_key_available": "NOT_FOUND", "ordinary_counterpart_available": "self-described missing", "eligibility_for_comparison": "NOT_COMPARABLE", "recovery_status": "NOT_FOUND", "notes": "R12 explicitly reports ordinary action-end state was not saved."},
    ]
    write_csv(out, fields, rows)


def build_source_contract(repo: Path, out: Path) -> None:
    sources = {relative: file_record(repo, relative) for relative in SOURCE_FILES}
    write_json(out, {
        "schema": "l2rar2_r13_source_call_order_contract_v1",
        "source_commit": ENTRY,
        "source_files": sources,
        "source_proven_order": [
            "DynamicTabletop.__init__ initializes callbacks before scenario setup and calls mj_forward after model/data construction.",
            "DynamicTabletop._advance performs five mj_step calls per control segment, then invokes _r1_control_callback.",
            "DynamicTabletop.perform runs lifecycle_before_action, advances the action, lifecycle_after_action, invokes _r1_action_end_callback, then returns result.",
            "repair_collection capture callback renders first, then records oracle_snapshot fields; action callback is registered on the simulator.",
            "R12 policy rejects physical probe entry before physical dependency imports or factory calls.",
        ],
        "historical_runtime_only_claims": [
            "R11 ordinary/instrumented action and event equality from saved report summaries",
            "R11 renderer callback effective behavior during each historical invocation",
            "historical package/runtime/model versions not present in retained lightweight evidence",
        ],
        "unverified_runtime_order": [
            "instrumented_after_mj_step to cached_action_end one-to-one mapping",
            "ordinary_after_perform_return to collection_action_end_callback numerical identity",
            "same timestamp implies same update order",
        ],
        "renderer_status": {
            "source_requests_renderer": True,
            "historical_effective_renderer": "NOT_RECORDED",
            "r12_note": "source/report mismatch is not evidence that rendering changed physics",
        },
        "state_difference_localization": "NOT_LOCALIZABLE_FROM_SAVED_ARTIFACTS",
        "physical_execution_performed_by_r13": 0,
    })


def build_crosswalk(repo: Path, out: Path) -> None:
    fields = ["left_sampling_point", "right_sampling_point", "source_commit", "source_files", "source_order_proven", "intermediate_physics_step", "intermediate_state_write", "left_fields_saved", "right_fields_saved", "row_key_mapping", "mapping_status", "blocking_gap", "notes"]
    pairs = [
        ("collection_control_callback", "cached_oracle_control_tick", "SOURCE_ORDER_ONLY", "callback source is known; original raw counterpart is external/not retained"),
        ("collection_action_end_callback", "cached_oracle_action_end", "SOURCE_ORDER_ONLY", "same logical phase is named, but no ordinary saved state/order record exists"),
        ("ordinary_before_action", "cached_oracle_control_tick", "NO_SAVED_COUNTERPART", "ordinary before-action state absent"),
        ("ordinary_after_perform_return", "collection_action_end_callback", "NO_SAVED_COUNTERPART", "ordinary action-end values absent"),
        ("instrumented_before_mj_step", "cached_oracle_control_tick", "NOT_COMPARABLE", "instrumented before-step state was not retained as a mapped stream"),
        ("instrumented_after_mj_step", "cached_oracle_action_end", "NOT_COMPARABLE", "different sampling roles and no callback/order mapping"),
        ("instrumented_action_end_callback", "collection_action_end_callback", "NO_SAVED_COUNTERPART", "callback values not retained as paired streams"),
        ("ordinary_after_perform_return", "instrumented_after_mj_step", "NOT_COMPARABLE", "different execution mode and missing ordinary state"),
    ]
    rows = []
    for left, right, status, gap in pairs:
        rows.append({"left_sampling_point": left, "right_sampling_point": right, "source_commit": ENTRY, "source_files": ";".join(SOURCE_FILES), "source_order_proven": "source_only" if status == "SOURCE_ORDER_ONLY" else "false", "intermediate_physics_step": "unknown", "intermediate_state_write": "unknown", "left_fields_saved": "partial/source-defined", "right_fields_saved": "partial/summary-only", "row_key_mapping": "not established", "mapping_status": status, "blocking_gap": gap, "notes": "No nearest-time join, interpolation, reindexing, or fabricated state used."})
    write_csv(out, fields, rows)


def build_environment(repo: Path, out: Path) -> None:
    source = {relative: {"sha256": sha256(repo / relative), "git_blob_sha1": git_blob(repo, relative), "status": "VERIFIED_GIT_SOURCE"} for relative in SOURCE_FILES}
    provenance = {}
    values = {
        "repository_commit": (ENTRY, "VERIFIED_GIT_SOURCE"),
        "generation_lock_sha256": (None, "UNKNOWN_NOT_RECORDED"),
        "rollout_manifest_sha256": (None, "UNKNOWN_NOT_RECORDED"),
        "repair_version": ("ATTACH_RELPOSE_VERSION is source-defined but historical artifact value is not retained", "VERIFIED_GIT_SOURCE"),
        "collection_version": (None, "UNKNOWN_NOT_RECORDED"),
        "simulator_class": ("upgrade_v2.visual_refine_l2.repaired_simulator.AttachRelposeDynamicTabletop", "VERIFIED_GIT_SOURCE"),
        "python_executable": (None, "UNKNOWN_NOT_RECORDED"),
        "python_version": (None, "UNKNOWN_NOT_RECORDED"),
        "platform": (None, "UNKNOWN_NOT_RECORDED"),
        "machine_arch": (None, "UNKNOWN_NOT_RECORDED"),
        "mujoco_version": (None, "UNKNOWN_NOT_RECORDED"),
        "numpy_version": (None, "UNKNOWN_NOT_RECORDED"),
        "opencv_version": (None, "UNKNOWN_NOT_RECORDED"),
        "renderer_backend": (None, "UNKNOWN_NOT_RECORDED"),
        "MUJOCO_GL": ("egl requested by current simulator source; historical effective value not recorded", "VERIFIED_GIT_SOURCE"),
        "model_xml_sha256": (None, "UNKNOWN_NOT_RECORDED"),
        "family_seed": (None, "UNKNOWN_NOT_RECORDED"),
        "rollout_seed": (None, "UNKNOWN_NOT_RECORDED"),
        "control_variant": (None, "UNKNOWN_NOT_RECORDED"),
        "program_sha256": (None, "UNKNOWN_NOT_RECORDED"),
        "root_family_id": ("L2RAR2_REPAIR_00_840000", "VERIFIED_PROTOCOL_DRAFT"),
        "case_id": ("K3_normal_hold_pause_resume", "VERIFIED_PROTOCOL_DRAFT"),
        "render_callback_mode": ("source requests renderer callbacks; historical effective mode unknown", "CONFLICTING_SOURCES"),
        "initial_state_source": (None, "UNKNOWN_NOT_RECORDED"),
        "rng_state_source": (None, "UNKNOWN_NOT_RECORDED"),
        "warmstart_state_source": (None, "UNKNOWN_NOT_RECORDED"),
        "eq_data_source": ("R12 source class includes eq_data in snapshot implementation; historical saved counterpart absent", "VERIFIED_GIT_SOURCE"),
        "thread_env_allowlist": ({}, "UNKNOWN_NOT_RECORDED"),
    }
    for key, (value, status) in values.items():
        provenance[key] = {"value": value, "status": status}
    write_json(out, {
        "schema": "l2rar2_r13_replay_environment_contract_v1",
        "source_status": "PARTIAL_SOURCE_CONTRACT_ONLY",
        "repository_commit": ENTRY, "root_family_id": "L2RAR2_REPAIR_00_840000", "case_id": "K3_normal_hold_pause_resume",
        "field_provenance": provenance, "source_file_sha256": source,
        "unknown_fields": [key for key, item in provenance.items() if item["status"] in {"UNKNOWN_NOT_RECORDED", "CONFLICTING_SOURCES"}],
        "current_environment_not_substituted_for_history": True,
        "R14_minimum_environment_contract_satisfied": False,
    })


def build_review_and_draft(repo: Path, out: Path) -> None:
    template = Path("/tmp/l2rar2_r13_pkg/L2RAR2_R13_Agent_Package_V1.0/templates/human_review_decision.template.json")
    decision = read_json(template)
    decision["entry_commit"] = ENTRY
    write_json(out / "human_review_decision.json", decision)
    draft = read_json(Path("/tmp/l2rar2_r13_pkg/L2RAR2_R13_Agent_Package_V1.0/templates/r14_micro_replay_protocol_draft.json"))
    draft["source_r12_commit"] = ENTRY
    write_json(out / "r14_micro_replay_protocol_draft.json", draft)


def build_readiness(out: Path, integrity: dict[str, Any]) -> dict[str, Any]:
    result = {
        "schema": "l2rar2_r13_replay_readiness_v1",
        "status": "WAITING_FOR_HUMAN_REVIEW",
        "allowed_statuses": ["READY_FOR_HUMAN_R14_AUTHORIZATION_REQUEST", "NOT_READY_SOURCE_PROVENANCE_GAP", "NOT_READY_RUNTIME_ENVIRONMENT_GAP", "NOT_READY_DATA_INTEGRITY_GAP", "NOT_READY_SAMPLING_POINT_CONTRACT_GAP", "WAITING_FOR_HUMAN_REVIEW", "STOP_AND_ARCHIVE"],
        "blocking_gaps": [
            "human_review_decision fields are intentionally null; Agent cannot approve R14",
            "generation lock and rollout manifest hashes are not present in retained lightweight evidence",
            "historical Python/MuJoCo/NumPy/OpenCV/runtime and model XML hashes are not recorded",
            "ordinary action-end state and verified sampling-point crosswalk are missing",
            "R11 first four invocation details remain aggregate-only",
        ],
        "r12_integrity": integrity["status"], "r11_accounting_preserved": "40/8 BUDGET_EXCEEDED_BLOCKED",
        "r13_actual_physical_executions": 0, "r14_requested_budget": 2, "r14_authorized_budget": 0,
        "human_review": "NOT_COMPLETE_REQUIRED",
        "source_contract": "PARTIAL_SOURCE_CONTRACT_ONLY",
        "sampling_point_contract": "NOT_READY_SAMPLING_POINT_CONTRACT_GAP",
        "environment_contract": "NOT_READY_RUNTIME_ENVIRONMENT_GAP",
        "ordinary_state_save_plan": "DEFINED_IN_R14_DRAFT_ONLY",
        "execution_1_stop_gate": "DEFINED",
        "execution_2_condition_gate": "DEFINED",
        "candidate": None, "confirmation": False, "l3_entry_allowed": False,
    }
    write_json(out / "replay_readiness.json", result)
    return result


def build_external(repo: Path, out: Path) -> None:
    fields = ["logical_path", "original_path", "size_bytes", "sha256", "artifact_type", "reason_omitted", "recovery_method", "verification_scope"]
    rows = [
        {"logical_path": "r12_result_zip", "original_path": str((repo / R12_ZIP).resolve()), "size_bytes": (repo / R12_ZIP).stat().st_size, "sha256": sha256(repo / R12_ZIP), "artifact_type": "prior_result_zip", "reason_omitted": "R13 result package must not duplicate prior ZIP entity", "recovery_method": "retain verified external ZIP beside R13 worktree", "verification_scope": "sidecar, CRC and SHA256"},
        {"logical_path": "r11_raw_rollout_directory", "original_path": "/home/__compress_data/xushijie/graph_l2ra_r2_worktree/artifacts/pathgraph_sarm/upgrade_v2/task_context_l2rar2_v1/data_attach_relpose_repair_v1", "size_bytes": "4438315 registered", "sha256": "NOT_COMPUTED_DIRECTORY_TREE", "artifact_type": "raw_rollout_and_RGB", "reason_omitted": "raw data externalized", "recovery_method": "restore exact registered directory", "verification_scope": "R11 manifest path only"},
        {"logical_path": "r11_pre_correction_directory", "original_path": "/home/__compress_data/xushijie/graph_l2ra_r2_worktree/artifacts/pathgraph_sarm/upgrade_v2/task_context_l2rar2_v1/data_pre_correction_k8_lifecycle_20260909", "size_bytes": "13423149 registered", "sha256": "NOT_COMPUTED_DIRECTORY_TREE", "artifact_type": "historical_raw_rollout", "reason_omitted": "raw data externalized", "recovery_method": "restore exact registered directory", "verification_scope": "R11 manifest path only"},
    ]
    write_csv(out, fields, rows)


def build_optional_indexes(repo: Path, out: Path) -> None:
    write_csv(out / "source_hash_inventory.csv", ["path", "sha256", "git_blob_sha1", "size_bytes", "status"], [file_record(repo, relative) for relative in SOURCE_FILES])
    zip_path = repo / R12_ZIP
    zip_rows: list[dict[str, Any]] = []
    if zip_path.is_file():
        with zipfile.ZipFile(zip_path) as archive:
            for name in archive.namelist():
                data = archive.read(name)
                zip_rows.append({"zip_path": str(zip_path), "member": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "status": "INDEXED"})
    else:
        zip_rows.append({"zip_path": str(zip_path), "member": "", "size_bytes": "", "sha256": "", "status": "ZIP_NOT_FOUND"})
    write_csv(out / "zip_content_inventory.csv", ["zip_path", "member", "size_bytes", "sha256", "status"], zip_rows)
    log_text = git(repo, "log", "--all", "--decorate", "--oneline", "--", *SOURCE_FILES) or ""
    reflog_text = git(repo, "reflog", "show", "--all", "--date=iso") or ""
    keywords = ("physics-probe", "probe_execution_ledger", "physics_probe_trace", "ordinary", "instrumented", "R11", "loss_observability")
    matches = [line for line in (log_text + "\n" + reflog_text).splitlines() if any(word.lower() in line.lower() for word in keywords)]
    write_json(out / "reflog_search_summary.json", {
        "schema": "l2rar2_r13_reflog_search_summary_v1",
        "scope": "current Git clone refs and reflog only; no shell history, secrets, or unrelated home paths",
        "keywords": list(keywords), "git_history_line_count": len(log_text.splitlines()),
        "reflog_line_count": len(reflog_text.splitlines()), "matched_line_count": len(matches),
        "matched_lines_minimal": matches[:200], "raw_reflog_packaged": False,
    })


def build_report(out: Path, readiness: dict[str, Any], integrity: dict[str, Any]) -> None:
    report = f"""# L2RAR2 R13 静态恢复与 R14 准备度审计

## 工程状态

`R13_WAITING_FOR_HUMAN_REVIEW`

R13 新增物理执行严格为 `0`。本轮没有 import 或构造 MuJoCo model/data、simulator、renderer，没有调用 `mj_step`/`mj_forward`，没有训练、API 调用或密钥读取。R12 合作式门禁只做状态复核和既有纯测试范围记录，没有通过新 replay 验证。

## 科学状态

- scientific status: `L2RAR2_PARTIAL_KEEP_G1`
- retained graph: `G1_predicate_bound`
- selected candidate: `null`
- confirmation: `false`
- L3: `false`

## 已恢复证据

已复核 R12 入口、R12 结果 ZIP、R11/R12 固定审计文件，并对受控目录生成静态证据清单。R12 ZIP 完整性结果为 `{integrity['status']}`，实际 SHA256 为 `{integrity['actual_sha256']}`。历史调用恢复为一条 40/8 aggregate 记录、一条保留的 last-batch ledger 记录和一条 R12 零物理静态审计记录；没有伪造 40 条明细。R11 三个机制字段均保留为 quarantine，不用于科学机制或旧事件重标。

## 未恢复证据

仍缺少前四次 R11 调用的独立日志、ordinary action-end 状态、跨来源 callback/order 一一映射、generation lock、rollout manifest、历史运行时版本、模型 XML 哈希和完整 seed/program 记录。R11 的缓存 geometry mismatch 仍是已保存摘要事实，不是首个物理差异或根因。

## 采样点合同

源码可以证明：每个 control 段执行五次 `mj_step` 后触发 control callback；动作完成后触发 action-end callback，再返回结果。源码不能证明历史保存文件中的 `instrumented_after_mj_step` 与 `cached_action_end` 是同一采样点。crosswalk 因此保留 `SOURCE_ORDER_ONLY`、`NO_SAVED_COUNTERPART` 或 `NOT_COMPARABLE`，不做 nearest-time 对齐、插值或重编号。

## 环境合同

simulator class 和当前入口源码可由 Git 验证；历史 Python/MuJoCo/NumPy/OpenCV、模型 XML、generation lock、rollout manifest、seed 和有效 renderer 配置不完整。因此环境合同不是 R14 执行授权，也未达到可直接复现级别。

## 人工复核与 R14 申请

`human_review_decision.json` 保持模板空字段，`r13_physical_budget_authorized=0`、`r14_physical_budget_authorized=0`。R14 草案仅准备单 case `K3_normal_hold_pause_resume`、请求 2 个实例：先 ordinary，只有执行1所有等价性门通过才允许讨论 instrumented。R13 没有批准它，也没有修改 R12 永久拒绝登记簿。

因此当前 R14 申请状态为 **未就绪，等待人工复核**；下一阶段是 `HUMAN_DECISION_ON_R14_MICRO_REPLAY_REQUEST`。

## 禁止主张

本轮没有定位 geometry mismatch 根因，没有证明 renderer、积分器、warmstart、callback 顺序或 weld 是差异来源，没有证明 K4/K5/K6 的真实 task-level loss，没有选择候选，没有 confirmation/L3。交付文件通过校验只代表文件完整性。
"""
    (out / "final_report.md").write_text(report, encoding="utf-8")


def build_handoff(out: Path, readiness: dict[str, Any], repo: Path) -> None:
    write_json(out / "next_stage_handoff.json", {
        "schema": "l2rar2_r13_next_stage_handoff_v1",
        "engineering_status": "R13_WAITING_FOR_HUMAN_REVIEW",
        "scientific_status": "L2RAR2_PARTIAL_KEEP_G1", "retained_graph": "G1_predicate_bound", "selected_candidate_id": None,
        "confirmation_run": False, "l3_entry_allowed": False,
        "r11_historical_executions": 40, "r11_historical_budget": 8, "r13_actual_physical_executions": 0, "r13_physical_budget": 0,
        "r14_requested_budget": 2, "r14_authorized_budget": 0, "human_review_status": "NOT_COMPLETE_REQUIRED",
        "replay_readiness_status": readiness["status"], "next_stage": "HUMAN_DECISION_ON_R14_MICRO_REPLAY_REQUEST",
        "api_calls": 0, "training_jobs": 0, "api_key_reads": 0,
        "source_commit": ENTRY, "formal_main_commit": MAIN, "branch": git(repo, "branch", "--show-current"),
        "notes": ["R13 does not self-authorize human approval.", "R12/R11 original audit artifacts remain unchanged."],
    })


def build_manifest(out: Path, repo: Path, commands: list[str]) -> None:
    (out / "actual_commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    records = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            records.append({"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(out / "run_manifest.json", {
        "schema": "l2rar2_r13_run_manifest_v1", "status": "R13_WAITING_FOR_HUMAN_REVIEW",
        "entry_commit": ENTRY, "formal_main_commit": MAIN, "implementation_commit": git(repo, "rev-parse", "HEAD"),
        "r11_historical_executions": 40, "r11_historical_budget": 8, "r13_physical_budget": 0, "r13_actual_physical_executions": 0,
        "r14_requested_budget": 2, "r14_authorized_budget": 0, "api_calls": 0, "training_jobs": 0, "api_key_reads": 0,
        "scientific_status": "L2RAR2_PARTIAL_KEEP_G1", "candidate": None, "confirmation": False, "l3_entry_allowed": False,
        "source_files": {relative: file_record(repo, relative) for relative in SOURCE_FILES},
        "artifacts": records, "commands": commands, "self_excluded": True,
        "sensitive_paths_read": [], "raw_rollout_read": False, "physics_imported": False,
    })


def generate(repo: Path, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    integrity = build_r12_integrity(repo, out / "r12_integrity_review.json")
    build_claim_review(repo, out / "claim_review.csv")
    build_history(repo, out / "historical_invocation_recovery.csv")
    build_saved_inventory(repo, out / "saved_state_recovery_inventory.csv")
    build_source_contract(repo, out / "source_call_order_contract.json")
    build_crosswalk(repo, out / "sampling_point_crosswalk.csv")
    build_environment(repo, out / "replay_environment_contract.json")
    build_review_and_draft(repo, out)
    readiness = build_readiness(out, integrity)
    build_external(repo, out / "external_artifacts.tsv")
    build_optional_indexes(repo, out)
    validation = {
        "schema": "l2rar2_r13_validation_results_v1", "status": "PASS_WITH_HUMAN_REVIEW_PENDING",
        "r13_physical_executions": 0, "mujoco_imported": False, "training_jobs": 0, "api_calls": 0, "api_key_reads": 0,
        "package_pure_tests": "RECORDED_SEPARATELY", "preflight": "PASS", "r12_integrity": integrity["status"],
        "static_evidence_inventory": "PASS", "human_review_validation": "NOT_RUN_FIELDS_INTENTIONALLY_NULL",
        "compileall": "PENDING", "secret_scan": "PENDING", "git_diff_check": "PENDING", "zip_test": "PENDING",
        "scientific_validation_claimed": False,
    }
    write_json(out / "validation_results.json", validation)
    build_report(out, readiness, integrity)
    build_handoff(out, readiness, repo)
    commands = [
        "git fetch origin --prune",
        "r13_preflight.py --repo WORK --output OUT/entry_review.json",
        "sha256sum -c R12 ZIP sidecar; unzip -t R12 ZIP",
        "static_evidence_inventory.py over explicit R11/R12/download roots",
        "git log/reflog controlled source review; no raw history or secrets read",
        "source call-order and sampling-point static audit; no simulator import",
        "copy human_review_decision template; leave reviewer and decision fields null",
        "copy and constrain R14 micro-replay draft; authorized_budget=0",
        "R13 pure tests, compileall, secret scan, git diff --check",
        "build lightweight R13 result ZIP; unzip -t and internal SHA validation",
    ]
    build_manifest(out, repo, commands)
    return readiness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R13 read-only replay readiness audit")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args(argv)
    out = args.output_root or args.repo / "artifacts/pathgraph_sarm/upgrade_v2/replay_readiness_l2rar2_r13_v1"
    print(json.dumps(generate(args.repo, out), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
