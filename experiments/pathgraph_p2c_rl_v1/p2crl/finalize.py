"""Assemble light artifacts. Does not claim policy utility."""
from __future__ import annotations
from pathlib import Path
from .io_utils import load_json, sha256_file, write_new, write_text_new

def write_historical_scope(path):
    rec = {
        "schema": "P2CRL_HISTORICAL_SCOPE_NOTE_V1",
        "archived_rm_route": "READY_FOR_P2C_RL_PREREGISTRATION",
        "archived_rm_environment_gate": "BENCHMARK_NOT_QUALIFIED_FOR_GRAPH_INCREMENT_TEST",
        "archived_rm_alternative_cost_roots": 29,
        "old_manual_required_roots": 32,
        "action": "RETAIN_BOTH_RECORD_CONFORMANCE_LIMITATION_NO_OLD_RERUN",
        "new_phase": "INDEPENDENT_PROTOCOL_WITH_PROSPECTIVE_RL_DATA_COVERAGE",
        "waiver_of_old_gate": False,
        "old_v1_hit_not_commensurate_with_p2cq": True,
        "candidate_worktree_byte_lock": {
            "n_core_files": 11,
            "n_bytes_match_candidate": 10,
            "mismatch_path": "experiments/pathgraph_p2c_rm_v1/p2crm/masked_smoke.py",
            "candidate_blob_sha1": "7b2b7ec50b6d8950ff67f1fea9527f71d68fc5b5",
            "worktree_and_result_base_blob_sha1": "10dbf5a95095f0a40ee5ee0d9bf8f9eefe95b568",
            "worktree_byte_sha256": "d2bc79b3f4b2e31f82a285bb7de746affa408ebd1148e9b7015713c345b3e6d4",
            "package_preflight_passed": False,
            "rewrote_protected_rm_path": False,
            "new_runner_imports_masked_smoke": False,
            "note": "Base/result commit unwrapped two unused invalid_selected/nonfinite lines from candidate smoke. Restoring candidate bytes would violate only_new_scoped_paths. Record 10/11 match; keep published result bytes.",
        },
        "note": "RM JSON environment gate failed 29/32 ALTERNATIVE_COST roots while RM manual 10.3 required 32. Keep both. Do not rerun 1310xxx. New P2C-RL coverage gates are prospective and do not repair the historical failure.",
    }
    write_new(path, rec)
    return rec

def write_not_released(path, protocol_sha=None, plan_sha=None):
    rec = {
        "schema": "P2CRL_CAMPAIGN_RELEASE_V1",
        "status": "NOT_RELEASED",
        "campaign_id": "P2CRL_V1",
        "protocol_sha256": protocol_sha,
        "preregistration_commit": None,
        "runner_commit": None,
        "source_lock_sha256": None,
        "dataset_manifest_sha256": None,
        "explicit_user_execution_instruction_ref": None,
        "formal_job_limit": 60,
        "formal_step_limit": 31457280,
        "smoke_job_limit": 5,
        "smoke_steps_each": 512,
        "training_release": False,
        "note": "Prepare-only. Do not start smoke or formal jobs.",
        "plan_sha256": plan_sha,
    }
    write_new(path, rec)
    return rec

def write_deviation(path, **fields):
    rec = {
        "schema": "P2CRL_PREREGISTRATION_DEVIATION_V1",
        "id": "CANDIDATE_WORKTREE_UNWRAP_MASKED_SMOKE_FOLLOWS_RESULT_BASE",
        "severity": "DOCUMENTED_SOURCE_LOCK_EXCEPTION",
        "rewrote_history": False,
        "training_release": False,
        "package_preflight_passed": False,
        **fields,
    }
    write_new(path, rec)
    return rec

def write_final_decision(path, status="PREREGISTERED_READY_NOT_STARTED"):
    rec = {
        "schema": "P2CRL_FINAL_DECISION_V1",
        "status": status,
        "completed_jobs": 0,
        "primary_comparisons": {},
        "policy_utility_evidence": "NOT_EVALUATED",
        "claim_scope": "NEW_INSTANCES_WITHIN_FROZEN_MOTIF_TEMPLATES_NOT_UNSEEN_TOPOLOGY",
        "historical_decisions_modified": False,
        "physical_executions": 0,
        "global_confirmation_passed": False,
        "training_release": False,
        "prepare_gradient_updates": 0,
    }
    write_new(path, rec)
    return rec
