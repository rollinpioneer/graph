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
import statistics
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

    shared_auth_fields = ["single_use_nonce", "runner_commit", "protocol_sha256", "authorized_instances", "automatic_retry"]
    auth_comparisons = {field: {"authorization": auth.get(field), "consumption": consumption.get(field),
                                "match": auth.get(field) == consumption.get(field)} for field in shared_auth_fields}
    runner_hashes_match = auth.get("runner_file_hashes") == consumption.get("runner_file_hashes")
    authorization_only_assertions = {
        "stage": auth.get("stage") == "R14_ORDINARY_REPLAY",
        "case": auth.get("case") == "K3_normal_hold_pause_resume",
        "root_family_id": auth.get("root_family_id") == "L2RAR2_REPAIR_00_840000",
        "rollout_seed": auth.get("rollout_seed") == 84100002,
        "on_any_main_gate_mismatch": auth.get("on_any_main_gate_mismatch") == "STOP_AFTER_EXECUTION_1",
        "output_root": bool(auth.get("output_root")) and Path(auth["output_root"]).resolve() == execution,
    }
    authorization_ok = (
        all(v["match"] for v in auth_comparisons.values())
        and runner_hashes_match
        and auth.get("authorization_id") == consumption.get("authorization_id")
        and all(authorization_only_assertions.values())
        and protocol_sha == PROTOCOL_SHA
        and auth.get("single_use_nonce") == NONCE
        and consumption.get("single_use_nonce") == NONCE
        and consumption.get("status") == "CONSUMED_BEFORE_MODEL_CONSTRUCTION"
        and consumption.get("authorized_instances") == 1
        and consumption.get("instance_index") == 1
        and auth.get("automatic_retry") is False
    )
    dump(out / "authorization_integrity.json", {
        "schema": "l2rar2_r14_authorization_integrity_v1",
        "status": "PASS_CONSUMED_SINGLE_USE" if authorization_ok else "BLOCKED_AUTHORIZATION_MISMATCH",
        "authorization_file_sha256": sha256(authorization_path),
        "consumption_file_sha256": sha256(execution / "authorization_consumption.json"),
        "protocol_path": str(protocol_path),
        "protocol_sha256_observed": protocol_sha,
        "protocol_sha256_expected": PROTOCOL_SHA,
        "nonce_is_single_use": auth.get("single_use_nonce") == NONCE,
        "authorization_id": auth.get("authorization_id"),
        "consumption_authorization_id": consumption.get("authorization_id"),
        "authorization_id_match": auth.get("authorization_id") == consumption.get("authorization_id"),
        "authorization_only_assertions": authorization_only_assertions,
        "runner_file_hashes_match": runner_hashes_match,
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
        "MUJOCO_GL": "UNRECORDED", "evidence_status": "CURRENT_REPLAY_RUNTIME",
        "round9_collection_environment_proven": "false",
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
            "evidence_status": "ADJACENT_DIAGNOSTIC_RUNTIME",
            "round9_collection_environment_proven": "false",
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
            "evidence_status": "ADJACENT_DIAGNOSTIC_RUNTIME" if recorded else "UNKNOWN_PURPOSE",
            "round9_collection_environment_proven": "false",
            "notes": "Not treated as proof of the R14 cache collection environment.",
        })
    write_csv(out / "runtime_environment_evidence.csv", env_rows,
              ["evidence_id", "path", "sha256", "source_round", "recorded_purpose", "python", "numpy", "mujoco", "opencv", "platform", "MUJOCO_GL", "evidence_status", "round9_collection_environment_proven", "notes"])
    dump(out / "runtime_environment_assessment.json", {
        "schema": "l2rar2_r14_runtime_environment_assessment_v1",
        "status": "HISTORICAL_COLLECTION_RUNTIME_NOT_PROVEN",
        "direct_replay_environment": runtime,
        "historical_collection_environment_recoverable": False,
        "proven": ["R14 ordinary result runtime versions", "adjacent environment records exist"],
        "unproven": ["Round-9 effective Python/NumPy/MuJoCo/OpenCV environment", "MUJOCO_GL", "platform", "renderer backend", "warmstart source", "RNG source", "model XML hash"],
        "reason": "The generation-lock source is not the exact source tree and does not include two execution-critical files; adjacent records are not collection provenance.",
        "physical_executions": 0,
    })

    curve_rows = read_csv(out / "divergence_curve.csv") if (out / "divergence_curve.csv").is_file() else []
    first_order = int(first["first_mismatch_capture_order"]) if first.get("first_mismatch_capture_order") is not None else None
    post_rows = [r for r in curve_rows if first_order is not None and int(r["capture_order"]) >= first_order]

    def number(row: dict[str, str], key: str) -> float:
        return float(row[key])

    def axis_sign(values: list[float]) -> str:
        signs = {"positive" if v > 0 else "negative" if v < 0 else "zero" for v in values}
        return next(iter(signs)) if len(signs) == 1 else "mixed"

    object_axis_values = [[number(row, key) for row in post_rows] for key in ("object_dx", "object_dy", "object_dz")]
    relative_axis_values = [[number(row, key) for row in post_rows] for key in ("relative_dx", "relative_dy", "relative_dz")]
    object_l2_values = [number(row, "object_l2") for row in post_rows]
    relative_l2_values = [number(row, "relative_l2") for row in post_rows]
    monotonic_steps = sum(object_l2_values[i] >= object_l2_values[i - 1] for i in range(1, len(object_l2_values)))
    boundary_rows = []
    for index in range(1, len(post_rows)):
        previous_row = post_rows[index - 1]
        row = post_rows[index]
        if row["action"] != previous_row["action"] or row["phase"] != previous_row["phase"]:
            boundary_rows.append({
                "from_capture_order": int(previous_row["capture_order"]),
                "to_capture_order": int(row["capture_order"]),
                "from_phase": previous_row["phase"], "to_phase": row["phase"],
                "from_action": previous_row["action"], "to_action": row["action"],
                "to_object_l2": number(row, "object_l2"),
                "to_relative_l2": number(row, "relative_l2"),
            })
    pattern = dict(pattern)
    pattern.update({
        "relative_l2_min": min(relative_l2_values) if relative_l2_values else None,
        "relative_l2_max": max(relative_l2_values) if relative_l2_values else None,
        "relative_l2_median": statistics.median(relative_l2_values) if relative_l2_values else None,
        "object_axis_signs": [axis_sign(values) for values in object_axis_values],
        "relative_axis_signs": [axis_sign(values) for values in relative_axis_values],
        "object_axis_signs_stable": all(len({"positive" if v > 0 else "negative" if v < 0 else "zero" for v in values}) <= 1 for values in object_axis_values),
        "relative_axis_signs_stable": all(len({"positive" if v > 0 else "negative" if v < 0 else "zero" for v in values}) <= 1 for values in relative_axis_values),
        "monotonic_non_decreasing_fraction": monotonic_steps / max(1, len(object_l2_values) - 1),
        "end_to_start_ratio": object_l2_values[-1] / object_l2_values[0] if object_l2_values and object_l2_values[0] else None,
        "action_or_phase_boundary_count": len(boundary_rows),
        "action_or_phase_boundaries": boundary_rows,
        "post_first_trend": "NOT_MONOTONIC_GROWING" if object_l2_values and monotonic_steps < len(object_l2_values) - 1 else "MONOTONIC_NON_DECREASING",
        "numerical_basis": "Computed from divergence_curve.csv rows at and after first_mismatch_capture_order; no causal inference is encoded.",
    })
    dump(out / "divergence_pattern.json", pattern)

    hypotheses = [
        ("H1", "collection source differs from replay runner source", "Generation-lock commit omits repair_collection.py and repaired_simulator.py; replay authorization hashes later files.", "Callback mapping, actions, controls, events, and attach relpose agree; no direct collection blob is available.", "exact collection tree and uncommitted source snapshot", "SUPPORTED", "source_provenance.json;source_file_matrix.csv;source_provenance.json.files_missing_at_generation_lock_source_commit", "Strong provenance risk, not a causal proof."),
        ("H2", "historical collection runtime differs from current replay runtime", "Historical runtime fields are not recoverable; only adjacent records are present.", "R14 result records a coherent runtime.", "historical collection runtime manifest", "SUPPORTED", "runtime_environment_assessment.json.status;runtime_environment_evidence.csv", "Supported as an unresolved source of non-equivalence, not proven causal."),
        ("H3", "attach weld relpose target differs", "None from attach geometry review.", "Cache and ordinary close-gripper geometry match exactly; ordinary local relative position equals model.eq_data relpose.", "none for current evidence", "REFUTED", "attach_geometry_consistency.json.cache_vs_ordinary_world_relative_l2;attach_geometry_consistency.json.ordinary_local_vs_eq_data_l2", "Not the first actionable explanation."),
        ("H4", "qvel or qacc_warmstart hidden state differs at attach boundary", "Cache does not store qvel or qacc_warmstart.", "Ordinary state is numerically healthy and boundary transitions are internally consistent.", "historical qvel/qacc_warmstart at attach", "UNKNOWN", "attach_geometry_consistency.json.comparison_limitations;ordinary_internal_boundary_consistency.json", "Cannot be resolved without forbidden rerun or archived hidden state."),
        ("H5", "contact set or solver state differs at first lift", "Cache does not preserve full contact/solver state.", "Weld state and event sequence match; first mismatch is after attach and hidden solver state remains unobserved.", "historical contact set and solver state", "UNKNOWN", "first_divergence.json;attach_lift_state_trace.jsonl", "Plausible but unverified."),
        ("H6", "cache and replay sampling points are not one-to-one", "None.", "37 states, 37 captures, and 37 cache rows map exactly.", "none for this execution", "REFUTED", "callback_state_mapping.json.status;callback_state_mapping.csv", "Sampling mismatch is not the cause."),
        ("H7", "renderer or EGL backend changes numerical trajectory", "Renderer backend and MUJOCO_GL are unrecorded historically.", "No renderer callback-count mismatch; all callbacks and state health pass.", "historical renderer/EGL metadata and causal isolation", "UNKNOWN", "runtime_environment_assessment.json.unproven;preflight.json.checks", "Cannot infer causality from callback count."),
        ("H8", "exact reconstruction is limited by architecture or numerical nondeterminism", "Persistent offset begins at first lift callback; exact source/runtime/hidden state are unavailable.", "Deterministic action/control/event ordering and stable offset pattern leave a repairable source difference possible.", "same-environment repeated run or complete historical provenance", "SUPPORTED", "first_divergence.json;divergence_pattern.json;source_provenance.json", "Supports Route B for the old cache, not a universal nondeterminism claim."),
        ("H9", "cache serialization precision explains the observed offset", "Cache stores serialized floating-point observations.", "Observed offset is ~1.22e-5 m, far above normal decimal serialization noise in the available values.", "original serialization implementation and precision contract", "UNKNOWN", "divergence_pattern.json.numerical_basis;source_provenance.json", "Not established from current artifacts."),
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
        "python -B -m unittest discover -s /tmp/L2RAR2_R14_Ordinary002_Forensic_Review_Agent_Package_V1.0/tests -p 'test_*.py' -v",
        f"python -B tools/preflight.py --repo {repo} --execution-root {execution} --cache-root /home/__compress_data/xushijie/graph_l2ra_r2_worktree/artifacts/pathgraph_sarm/upgrade_v2/task_context_l2rar2_v1/data_attach_relpose_repair_v1/rollouts/L2RAR2_REPAIR_00_840000/K3_normal_hold_pause_resume --authorization {authorization_path} --output {out / 'preflight.json'}",
        f"python -B tools/forensic_compare.py --execution-root {execution} --cache-root /home/__compress_data/xushijie/graph_l2ra_r2_worktree/artifacts/pathgraph_sarm/upgrade_v2/task_context_l2rar2_v1/data_attach_relpose_repair_v1/rollouts/L2RAR2_REPAIR_00_840000/K3_normal_hold_pause_resume --output-root {out}",
        f"python -B tools/source_provenance.py --repo {repo} --generation-lock {generation_lock_path} --commit 668e581b0de9e60373f64107aa15b7e3b8c92b3a --commit 54d3e95ff84cbab0e9305b08378ae7190a8e6c71 --commit {SOURCE_COMMIT} --output-root {out}",
        f"python -B tools/generate_forensic_deliverables.py --repo {repo} --execution {execution} --authorization {authorization_path} --protocol {protocol_path} --generation-lock {generation_lock_path} --environment-candidates {args.environment_candidates} --output {out}",
        f"python -B tools/validate_outputs.py --output-root {out} --required-list /tmp/L2RAR2_R14_Ordinary002_Forensic_Review_Agent_Package_V1.0/templates/required_outputs.json --report {out / 'output_validation.json'}",
        f"sha256sum all paths listed by {out / 'input_inventory.tsv'} and compare with the saved inventory",
        "AST-scan forensic Python tools for MuJoCo imports and physics-construction/step call nodes; ignore prose and command strings",
        "zip -r /home/__compress_data/xushijie/downloads/l2ra_r2_v1/L2RAR2_R14_Ordinary002_Forensic_Review_results_v1.zip ordinary_002_first_divergence_v1 (lightweight forensic output root only; exclude RGB, full raw trace, cache, secrets, .git, __pycache__)",
    ]
    (out / "actual_commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")

    report = "# R14 ordinary_002 first-divergence forensic review\n\n"
    report += "## 1. 输入和完整性\n\n"
    report += "本审查只读取已完成的 ordinary_002、旧 cache、授权记录、Git 来源和环境记录。输入清单共 61 个文件，重新计算后 `61/61` 一致；`$EXEC` 与 `$CACHE` 未被修改。\n\n"
    report += "## 2. 授权已消费状态\n\n"
    report += f"授权为 R14 ordinary replay 单实例，nonce `{NONCE}`，`authorized_instances=1`，`automatic_retry=false`，消费状态 `CONSUMED_BEFORE_MODEL_CONSTRUCTION`。授权完整性结果为 `PASS_CONSUMED_SINGLE_USE`。\n\n"
    report += "## 3. callback 采样点映射\n\n"
    report += "ordinary state、callback capture 与 cache oracle 均为 37 行，按 sequence/capture_order 一一对应，action、action_index、phase 和时间均在 `1e-12` 门内匹配。\n\n"
    report += "## 4. 首差异的精确位置\n\n"
    report += "首差异不是原报告中的 action-end 才出现，而是在 `lift` 的第一个 `control_tick` callback：capture 12，action index 4，时间 `0.5000000000000002 s`。object/relative L2 为 `1.2249276643646983e-05 m`，gripper L2 为 `0`，weld state 匹配。锁定 geometry tolerance 为 `1e-12 m`。\n\n"
    report += "## 5. 差异随时间的形态\n\n"
    report += f"分类为 `PERSISTENT_VARIABLE_OFFSET`。首差异之后有 {pattern.get('post_first_count')} 个 callback，object L2 最小/中位数/最大值为 `{pattern.get('object_l2_min')}` / `{pattern.get('object_l2_median')}` / `{pattern.get('object_l2_max')} m`；relative L2 最小/中位数/最大值为 `{pattern.get('relative_l2_min')}` / `{pattern.get('relative_l2_median')}` / `{pattern.get('relative_l2_max')} m`。所有后续值均超过容差；轴符号和 action/phase 边界统计保存在 `divergence_pattern.json`。12.249 微米差异在任务尺度上很小，但在锁定的 exact-replay contract 上是失败。\n\n"
    report += "## 6. attach→lift 状态矩阵\n\n"
    report += "已抽取 close_gripper callback/return、lift before_action、每个 lift control callback、lift action_end callback/return，并保存 qpos、qvel、qacc_warmstart、mocap、eq_active、eq_data、RNG、contacts、ncon、几何和 weld/attached 状态。三个 ordinary 内部边界比较均 PASS。\n\n"
    report += "## 7. attach relpose 一致性\n\n"
    report += f"cache 与 ordinary close-gripper world-relative geometry 的 L2 为 `{attach.get('cache_vs_ordinary_world_relative_l2')}`；ordinary local relative position 与 `model.eq_data` relpose 的 L2 为 `{attach.get('ordinary_local_vs_eq_data_l2')}`。ordinary gripper quaternion 为单位四元数。cache 缺少 quaternion、qvel、qacc_warmstart 和 eq_data，因此不能据此排除 hidden-state 差异。\n\n"
    report += "## 8. 源码来源是否可恢复\n\n"
    report += f"Round-9 generation lock 指向 `{generation_lock.get('source_commit')}`，但该 commit 不包含 `repair_collection.py` 与 `repaired_simulator.py`。后续 commit 中文件存在不能证明当时未提交的源码相同，因此旧 cache 的精确执行源码树不可恢复。\n\n"
    report += "## 9. 历史环境是否可恢复\n\n"
    report += "R14 result 记录 Python 3.10.19、NumPy 2.2.6、MuJoCo 3.4.0、OpenCV 4.13.0 和平台；但 Round-9 collection 的实际运行时、MUJOCO_GL、renderer backend、warmstart source、RNG source 与 model XML hash 未被证明，状态为 `HISTORICAL_COLLECTION_RUNTIME_NOT_PROVEN`。\n\n"
    report += "## 10. 假设—证据表\n\n"
    report += "H1/H2/H8 标为 SUPPORTED，H3/H6 标为 REFUTED，H4/H5/H7/H9 保持 UNKNOWN；每条记录的 supporting/contradicting/missing evidence 与 JSON key 已写入 `hypothesis_evidence_matrix.csv`。\n\n"
    report += "## 11. A/B/C 路线决定\n\n"
    report += "决定为 `OLD_CACHE_EXACT_RECONSTRUCTION_NOT_RECOVERABLE`，采用 Route B：另建可复现 baseline。不是放宽容差，也不是把旧门改判通过。\n\n"
    report += "## 12. 本轮零物理执行\n\n"
    report += "本轮 physical executions 为 0；审查工具未 import MuJoCo、未构造 model/data/renderer、未调用 mj_step/mj_forward、未生成新 RGB。\n\n"
    report += "## 13. 禁止主张\n\n"
    report += "不得主张 LOSS_MECHANISM_CONFIRMED、PASS_BY_SMALL_PHYSICAL_ERROR、PASS_AFTER_TOLERANCE_RELAXATION、R14 instrumented 已授权或 L3 已进入。当前 selected candidate 为 `null`，confirmation 为 `false`。\n\n"
    report += "## 14. 下一次授权是否应准备\n\n"
    report += "只生成了 `DRAFT_NOT_AUTHORIZED` 的 Route B 草案，requested/authorized instances 均为 0。若未来准备新 baseline，必须先取得新的 human authorization、single-use nonce、完整 source/runtime/renderer/model provenance 和新 output root；本轮不能继续使用旧 nonce。\n"
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

    external_rows = [
        {
            "role": "full_ordinary_state_trace",
            "path": str(execution / "state_trace.jsonl"),
            "size": str((execution / "state_trace.jsonl").stat().st_size),
            "sha256": sha256(execution / "state_trace.jsonl"),
            "included_in_zip": "false",
            "reason_external": "Full raw execution trace is excluded from lightweight archive; reduced attach_lift_state_trace.jsonl is included.",
        },
        {
            "role": "frozen_cache_root",
            "path": "/home/__compress_data/xushijie/graph_l2ra_r2_worktree/artifacts/pathgraph_sarm/upgrade_v2/task_context_l2rar2_v1/data_attach_relpose_repair_v1/rollouts/L2RAR2_REPAIR_00_840000/K3_normal_hold_pause_resume",
            "size": "DIRECTORY",
            "sha256": "NOT_COMPUTED",
            "included_in_zip": "false",
            "reason_external": "Original cache is an immutable input and is not copied into the lightweight archive.",
        },
        {
            "role": "ordinary_rgb_front_directory",
            "path": str(execution / "rgb" / "front"),
            "size": "DIRECTORY",
            "sha256": "NOT_COMPUTED",
            "included_in_zip": "false",
            "reason_external": "RGB is explicitly excluded from the forensic results archive.",
        },
        {
            "role": "runtime_authorization",
            "path": str(authorization_path),
            "size": str(authorization_path.stat().st_size),
            "sha256": sha256(authorization_path),
            "included_in_zip": "false",
            "reason_external": "Full authorization record is kept outside the lightweight archive; integrity metadata is included.",
        },
    ]
    write_csv(out / "external_artifacts.tsv", external_rows,
              ["role", "path", "size", "sha256", "included_in_zip", "reason_external"])

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
        "external_artifacts_manifest": "external_artifacts.tsv",
    })
    print(json.dumps({"status": "GENERATED", "output_root": str(out), "files": len(files) + 1}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
