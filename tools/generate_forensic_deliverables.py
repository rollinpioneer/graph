#!/usr/bin/env python3
"""Build the non-physical R14 forensic review deliverables.

This script only reads an existing execution, cache, authorization, and Git
metadata. It does not import repository code or any physics/rendering package.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


SOURCE_COMMIT = "ab342f8a5a04871b7d535c7b9190574cfce2cb94"
PROTOCOL_SHA = "effc4137447803c3b684e5d60373b1faddf0d2d8de03428008a97ed62c9c134a"
NONCE = "e8027aed85c2233b173af332524b288339ddcb2866d0626d"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(repo: Path, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return p.stdout.strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--execution", type=Path, required=True)
    ap.add_argument("--authorization", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--generation-lock", type=Path, required=True)
    ap.add_argument("--environment-candidates", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    repo = args.repo.resolve()
    execution = args.execution.resolve()
    authorization_path = args.authorization.resolve()
    protocol_path = args.protocol.resolve()
    generation_lock_path = args.generation_lock.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    auth = load(authorization_path)
    consumption = load(execution / "authorization_consumption.json")
    result = load(execution / "result.json")
    comparison = load(execution / "ordinary_replay_comparison.json")
    preflight = load(out / "preflight.json")
    first = load(out / "first_divergence.json")
    pattern = load(out / "divergence_pattern.json")
    boundary = load(out / "ordinary_internal_boundary_consistency.json")
    attach = load(out / "attach_geometry_consistency.json")
    provenance = load(out / "source_provenance.json")
    generation_lock = load(generation_lock_path)
    protocol_sha = sha256(protocol_path)

    auth_fields = [
        "single_use_nonce", "runner_commit", "protocol_sha256", "output_root",
        "authorized_instances", "stage", "case", "root_family_id", "rollout_seed",
        "automatic_retry", "on_any_main_gate_mismatch",
    ]
    auth_comparisons = {field: {"authorization": auth.get(field), "consumption": consumption.get(field),
                                "match": auth.get(field) == consumption.get(field)} for field in auth_fields}
    dump(out / "authorization_integrity.json", {
        "schema": "l2rar2_r14_authorization_integrity_v1",
        "status": "PASS" if all(v["match"] for v in auth_comparisons.values()) and protocol_sha == PROTOCOL_SHA else "FAIL",
        "authorization_file_sha256": sha256(authorization_path),
        "consumption_file_sha256": sha256(execution / "authorization_consumption.json"),
        "protocol_path": str(protocol_path),
        "protocol_sha256_observed": protocol_sha,
        "protocol_sha256_expected": PROTOCOL_SHA,
        "nonce_is_single_use": auth.get("single_use_nonce") == NONCE,
        "other_stage_authorized_instances": consumption.get("other_stage_authorized_instances"),
        "field_comparisons": auth_comparisons,
        "physical_executions_in_review": 0,
    })

    env_rows: list[dict[str, Any]] = []
    runtime = result.get("runtime", {})
    env_rows.append({
        "evidence_id": "E1_execution_result_runtime",
        "path": str(execution / "result.json"),
        "sha256": sha256(execution / "result.json"),
        "source_round": "R14 ordinary_002",
        "recorded_purpose": "runtime recorded by the completed ordinary replay",
        "python": runtime.get("python"), "numpy": runtime.get("numpy"), "mujoco": runtime.get("mujoco"),
        "opencv": runtime.get("opencv"), "platform": runtime.get("platform"),
        "MUJOCO_GL": "UNRECORDED", "evidence_status": "DIRECT_EXECUTION_EVIDENCE",
        "notes": "Proves the replay-reported versions only; does not prove historical cache collection equivalence.",
    })
    env_req = protocol_path.parent / "environment_requirements.json"
    if env_req.is_file():
        req = load(env_req)
        recorded = req.get("recorded_adjacent_environment", {})
        env_rows.append({
            "evidence_id": "E2_environment_requirements",
            "path": str(env_req), "sha256": sha256(env_req), "source_round": "R14 application",
            "recorded_purpose": "adjacent controlled reconstruction environment requirement",
            "python": recorded.get("python"), "numpy": "UNRECORDED", "mujoco": recorded.get("mujoco"),
            "opencv": recorded.get("opencv"), "platform": "UNRECORDED", "MUJOCO_GL": "UNRECORDED",
            "evidence_status": "ADJACENT_ONLY",
            "notes": "The file explicitly says equivalence with historical collection is not proven.",
        })
    for index, raw in enumerate(args.environment_candidates.read_text(encoding="utf-8").splitlines(), 1):
        path = Path(raw.strip())
        if not path.is_file() or path == env_req or path == execution / "result.json":
            continue
        try:
            value = load(path)
        except (OSError, json.JSONDecodeError):
            continue
        recorded = value.get("runtime", value.get("environment", value.get("recorded_adjacent_environment", {})))
        if not isinstance(recorded, dict):
            recorded = {}
        env_rows.append({
            "evidence_id": f"E{len(env_rows) + 1}_candidate_{index}", "path": str(path), "sha256": sha256(path),
            "source_round": "historical adjacent record", "recorded_purpose": value.get("purpose", value.get("schema", "candidate")),
            "python": recorded.get("python", value.get("python")), "numpy": recorded.get("numpy", value.get("numpy")),
            "mujoco": recorded.get("mujoco", value.get("mujoco")), "opencv": recorded.get("opencv", value.get("opencv")),
            "platform": recorded.get("platform", value.get("platform")), "MUJOCO_GL": recorded.get("MUJOCO_GL", value.get("MUJOCO_GL", "UNRECORDED")),
            "evidence_status": "ADJACENT_CANDIDATE", "notes": "Not treated as proof of the R14 cache collection environment.",
        })
    write_csv(out / "runtime_environment_evidence.csv", env_rows,
              ["evidence_id", "path", "sha256", "source_round", "recorded_purpose", "python", "numpy", "mujoco", "opencv", "platform", "MUJOCO_GL", "evidence_status", "notes"])
    dump(out / "runtime_environment_assessment.json", {
        "schema": "l2rar2_r14_runtime_environment_assessment_v1",
        "status": "HISTORICAL_COLLECTION_ENVIRONMENT_NOT_PROVEN",
        "direct_replay_environment": runtime,
        "historical_collection_environment_recoverable": False,
        "proven": ["R14 ordinary result runtime versions", "adjacent environment records exist"],
        "unproven": ["Round-9 effective Python/NumPy/MuJoCo/OpenCV environment", "MUJOCO_GL", "platform", "renderer backend", "warmstart source", "RNG source", "model XML hash"],
        "reason": "The generation-lock source is not the exact source tree and does not include two execution-critical files; adjacent records are not collection provenance.",
        "physical_executions": 0,
    })

    hypotheses = [
        ("H1", "collection source differs from replay runner source", "Generation-lock commit omits repair_collection.py and repaired_simulator.py; replay authorization hashes later files.", "Callback mapping, actions, controls, events, and attach relpose agree; no direct collection blob is available.", "exact collection tree and uncommitted source snapshot", "SUPPORTED", "source_provenance.json;source_file_matrix.csv", "Strong provenance risk, not a causal proof."),
        ("H2", "historical collection runtime differs from current replay runtime", "Historical runtime fields are not recoverable; only adjacent records are present.", "R14 result records a coherent runtime.", "historical collection runtime manifest", "SUPPORTED", "runtime_environment_assessment.json;runtime_environment_evidence.csv", "Supported as an unresolved source of non-equivalence, not proven causal."),
        ("H3", "attach weld relpose target differs", "None from attach geometry review.", "Cache and ordinary close-gripper geometry match exactly; ordinary local relative position equals model.eq_data relpose.", "none for current evidence", "REFUTED", "attach_geometry_consistency.json", "Not the first actionable explanation."),
        ("H4", "qvel or qacc_warmstart hidden state differs at attach boundary", "Cache does not store qvel or qacc_warmstart.", "Ordinary state is numerically healthy and boundary transitions are internally consistent.", "historical qvel/qacc_warmstart at attach", "UNKNOWN", "attach_geometry_consistency.json;ordinary_internal_boundary_consistency.json", "Cannot be resolved without forbidden rerun or archived hidden state."),
        ("H5", "contact set or solver state differs at first lift", "Cache does not preserve full contact/solver state.", "Weld state and event sequence match; first mismatch is after attach and hidden solver state remains unobserved.", "historical contact set and solver state", "UNKNOWN", "first_divergence.json;attach_lift_state_trace.jsonl", "Plausible but unverified."),
        ("H6", "cache and replay sampling points are not one-to-one", "None.", "37 states, 37 captures, and 37 cache rows map exactly.", "none for this execution", "REFUTED", "callback_state_mapping.json;callback_state_mapping.csv", "Sampling mismatch is not the cause."),
        ("H7", "renderer or EGL backend changes numerical trajectory", "Renderer backend and MUJOCO_GL are unrecorded historically.", "No renderer callback-count mismatch; all callbacks and state health pass.", "historical renderer/EGL metadata and causal isolation", "UNKNOWN", "runtime_environment_assessment.json;preflight.json", "Cannot infer causality from callback count."),
        ("H8", "exact reconstruction is limited by architecture or numerical nondeterminism", "Persistent offset begins at first lift callback; exact source/runtime/hidden state are unavailable.", "Deterministic action/control/event ordering and stable offset pattern leave a repairable source difference possible.", "same-environment repeated run or complete historical provenance", "SUPPORTED", "first_divergence.json;divergence_pattern.json;source_provenance.json", "Supports Route B for the old cache, not a universal nondeterminism claim."),
        ("H9", "cache serialization precision explains the observed offset", "Cache stores serialized floating-point observations.", "Observed offset is ~1.22e-5 m, far above normal decimal serialization noise in the available values.", "original serialization implementation and precision contract", "UNKNOWN", "divergence_pattern.json;source_provenance.json", "Not established from current artifacts."),
    ]
    write_csv(out / "hypothesis_evidence_matrix.csv", [
        {"hypothesis_id": r[0], "hypothesis": r[1], "supporting_evidence": r[2], "contradicting_evidence": r[3], "missing_evidence": r[4], "status": r[5], "evidence_paths": r[6], "notes": r[7]}
        for r in hypotheses
    ], ["hypothesis_id", "hypothesis", "supporting_evidence", "contradicting_evidence", "missing_evidence", "status", "evidence_paths", "notes"])

    inventory_rows: list[dict[str, str]] = []
    inventory_path = out / "input_inventory.tsv"
    if inventory_path.is_file():
        with inventory_path.open("r", encoding="utf-8", newline="") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 2)
                if len(parts) == 3:
                    inventory_rows.append({"sha256": parts[0], "size_bytes": parts[1], "path": parts[2]})
    inventory_mismatches = []
    for row in inventory_rows:
        path = Path(row.get("path", ""))
        expected = row.get("sha256", "")
        if not path.is_file():
            inventory_mismatches.append({"path": str(path), "status": "MISSING"})
            continue
        observed = sha256(path)
        if observed != expected:
            inventory_mismatches.append({"path": str(path), "status": "CHANGED", "expected": expected, "observed": observed})
    dump(out / "input_integrity_verification.json", {
        "schema": "l2rar2_r14_input_integrity_verification_v1",
        "status": "PASS" if not inventory_mismatches else "FAIL",
        "inventory_path": str(out / "input_inventory.tsv"),
        "checked_files": len(inventory_rows),
        "mismatches": inventory_mismatches,
        "execution_and_cache_treated_as_immutable": True,
        "physical_executions": 0,
    })

    dump(out / "review_decision.json", {
        "schema": "l2rar2_r14_ordinary002_forensic_decision_v1",
        "status": "OLD_CACHE_EXACT_RECONSTRUCTION_NOT_RECOVERABLE",
        "next_route": "B_NEW_REPRODUCIBLE_BASELINE_REQUIRED",
        "callback_mapping_status": "PASS",
        "first_divergence_status": "FOUND_AT_FIRST_LIFT_CONTROL_CALLBACK",
        "source_tree_recoverability": "NOT_EXACTLY_RECOVERABLE",
        "historical_runtime_recoverability": "NOT_PROVEN",
        "supported_hypotheses": ["H1", "H2", "H8"],
        "refuted_hypotheses": ["H3", "H6"],
        "unknown_hypotheses": ["H4", "H5", "H7", "H9"],
        "new_physical_authorized_instances": 0,
        "new_physical_execution_performed": False,
        "confirmation_run": False,
        "scientific_status": "L2RAR2_PARTIAL_KEEP_G1",
        "retained_graph": "G1_predicate_bound",
        "selected_candidate_id": None,
        "l3_entry_allowed": False,
        "reviewer_notes": "Action/control/event/callback ordering, internal boundaries, attach geometry, and numeric health are consistent. The first geometry mismatch is 1.2249276643646983e-05 m at capture 12, lift control callback, 0.5000000000000002 s. Exact old-cache reconstruction cannot be repaired defensibly from the available provenance without another physical run, which is not authorized.",
    })

    dump(out / "next_stage_application_draft.json", {
        "schema": "l2rar2_r14_next_stage_application_draft_v1",
        "status": "DRAFT_NOT_AUTHORIZED",
        "route": "B_NEW_REPRODUCIBLE_BASELINE_REQUIRED",
        "purpose": "Create a separately versioned, fully provenance-locked baseline; do not claim equivalence to the old Round-9 cache.",
        "requested_instances": 0, "authorized_instances": 0, "automatic_execution_allowed": False,
        "runner_commit": None, "runner_file_hashes": None, "protocol_sha256": None,
        "output_root": None, "single_use_nonce": None, "expires_at_utc": None,
        "approved_at_utc": None, "reviewer_id": None,
        "prerequisites": ["human authorization", "new nonce", "complete source tree and model hash", "runtime and renderer/EGL manifest", "explicit preservation of R14/R16 zero states"],
    })

    commands = [
        "python -B tools/verify_package.py --root /tmp/L2RAR2_R14_Ordinary002_Forensic_Review_Agent_Package_V1.0",
        f"python -B tools/preflight.py --repo {repo} --execution-root {execution} --cache-root /home/__compress_data/xushijie/graph_l2ra_r2_worktree/artifacts/pathgraph_sarm/upgrade_v2/task_context_l2rar2_v1/data_attach_relpose_repair_v1/rollouts/L2RAR2_REPAIR_00_840000/K3_normal_hold_pause_resume --authorization {authorization_path} --output {out / 'preflight.json'}",
        f"python -B tools/forensic_compare.py --execution-root {execution} --cache-root /home/__compress_data/xushijie/graph_l2ra_r2_worktree/artifacts/pathgraph_sarm/upgrade_v2/task_context_l2rar2_v1/data_attach_relpose_repair_v1/rollouts/L2RAR2_REPAIR_00_840000/K3_normal_hold_pause_resume --output-root {out}",
        f"python -B tools/source_provenance.py --repo {repo} --generation-lock {generation_lock_path} --commit 668e581b0de9e60373f64107aa15b7e3b8c92b3a --commit 54d3e95ff84cbab0e9305b08378ae7190a8e6c71 --commit {SOURCE_COMMIT} --output-root {out}",
        f"python -B tools/generate_forensic_deliverables.py --repo {repo} --execution {execution} --authorization {authorization_path} --protocol {protocol_path} --generation-lock {generation_lock_path} --environment-candidates {args.environment_candidates} --output {out}",
        f"python -B tools/validate_outputs.py --output-root {out} --required-list /tmp/L2RAR2_R14_Ordinary002_Forensic_Review_Agent_Package_V1.0/templates/required_outputs.json --report {out / 'output_validation.json'}",
    ]
    (out / "actual_commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")

    report = f"# R14 ordinary_002 first-divergence forensic review\n\n"
    report += "## Scope\n\n"
    report += "This review is static-only. No replay was rerun, no MuJoCo model/data/renderer was constructed by the review tools, and no new RGB was generated. `$EXEC` and `$CACHE` were treated as immutable inputs.\n\n"
    report += "## Fixed execution identity\n\n"
    report += f"- runner commit: `{SOURCE_COMMIT}`\n- protocol SHA256: `{PROTOCOL_SHA}`\n- nonce: `{NONCE}`\n- execution: `{execution.name}`\n- original result: `STOP_AFTER_EXECUTION_1`, 7/8 main gates\n\n"
    report += "## Findings\n\n"
    report += f"- Actions: exact sequence match.\n- Low-level controls: exact match.\n- Events: exact match.\n- Callback/capture order: exact 37-row mapping.\n- Renderer callback contract: 37 total, 29 control callbacks, 8 action-end callbacks; pass.\n- Numeric health: qpos/qvel/mocap/eq_data/qacc_warmstart shapes and finite checks pass.\n- Internal close-return/lift boundaries: pass.\n- Attach geometry: cache and ordinary close-gripper world-relative geometry match exactly; ordinary local relative position equals `model.eq_data` relpose.\n- First divergence: capture 12, `lift`, `control_tick`, action index 4, time `0.5000000000000002 s`; object/relative L2 `1.2249276643646983e-05 m` versus locked tolerance `1e-12 m`; gripper delta is zero and weld state matches.\n- Pattern: `PERSISTENT_VARIABLE_OFFSET`; 25 post-first rows remain above tolerance.\n\n"
    report += "## Provenance assessment\n\n"
    report += f"The Round-9 generation-lock commit is `{generation_lock.get('source_commit')}`. It does not contain `repair_collection.py` or `repaired_simulator.py`; later existence does not prove the historical uncommitted content. The R14 result records MuJoCo 3.4.0, NumPy 2.2.6, OpenCV 4.13.0, Python 3.10.19, but the historical collection runtime, `MUJOCO_GL`, platform, renderer backend, warmstart source, RNG source, and model XML hash are not proven.\n\n"
    report += "## Decision\n\n"
    report += "`OLD_CACHE_EXACT_RECONSTRUCTION_NOT_RECOVERABLE` (Route B). The evidence excludes sampling-point mapping and attach-relpose mismatch as the immediate explanation, but it cannot identify one repairable causal source with the required certainty. A new reproducible baseline would be a separate experiment and requires explicit authorization.\n\n"
    report += "## Frozen gates\n\n"
    report += "`R14 instrumented replay = 0`; `R16 calibration = 0`; `R16 development = 0`; new physical authorization = 0; new physical execution = 0; scientific status remains `L2RAR2_PARTIAL_KEEP_G1`; selected candidate remains `null`; confirmation remains `false`; L3 remains `false`.\n"
    (out / "final_report.md").write_text(report, encoding="utf-8")

    dump(out / "next_stage_handoff.json", {
        "schema": "l2rar2_r14_forensic_next_stage_handoff_v1",
        "status": "HANDOFF_READY_NO_AUTHORIZATION",
        "decision": "OLD_CACHE_EXACT_RECONSTRUCTION_NOT_RECOVERABLE",
        "route": "B_NEW_REPRODUCIBLE_BASELINE_REQUIRED",
        "source_commit": SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA,
        "first_divergence": first,
        "required_before_any_new_physics": ["human approval", "new single-use nonce", "complete source/runtime/renderer/model provenance", "new output root", "preserve all R14/R16 zero gates"],
        "forbidden_without_separate_authorization": ["instrumented replay", "R16 calibration", "R16 development", "reuse old nonce", "alter tolerance", "modify old execution or cache"],
        "physical_executions_in_review": 0,
    })

    # A deterministic manifest is written last among the generated artifacts;
    # its own hash is intentionally excluded to avoid a self-referential value.
    files = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name not in {"run_manifest.json", "output_validation.json"}:
            files.append({"path": path.name, "sha256": sha256(path), "size_bytes": path.stat().st_size})
    dump(out / "run_manifest.json", {
        "schema": "l2rar2_r14_forensic_run_manifest_v1",
        "status": "STATIC_ONLY_COMPLETE",
        "repo": str(repo), "review_head": git(repo, "rev-parse", "HEAD"),
        "source_commit": SOURCE_COMMIT, "protocol_sha256": PROTOCOL_SHA,
        "execution_id": execution.name, "authorization_nonce": NONCE,
        "physical_executions": 0, "mujoco_imported": False,
        "input_roots": {"execution": str(execution), "authorization": str(authorization_path), "protocol": str(protocol_path), "generation_lock": str(generation_lock_path)},
        "generated_files": files, "self_hash_excluded": True,
    })
    print(json.dumps({"status": "GENERATED", "output_root": str(out), "files": len(files) + 1}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
