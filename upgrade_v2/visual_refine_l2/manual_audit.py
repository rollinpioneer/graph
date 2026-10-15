"""Requirement-level completion audit for the L2R V1.0 handbook."""

from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .confirm import decide_status, verify_confirmation_lock
from .io import (
    find_forbidden_keys,
    now_iso,
    read_csv,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    verify_lock,
    write_json,
)
from .predicates import PREDICATE_NAMES


ENTRY_COMMIT = "5a3aafd6ac6d139ba324ee57110bbac508e17570"
SCENARIOS = {
    "normal_pick_place",
    "already_satisfied_stable",
    "already_on_target_offcenter",
    "missed_grasp_then_retry",
    "slip_then_recover",
    "target_occupied",
    "distractor_object_ambiguity",
    "primary_view_occlusion",
}
ACTION_VOCABULARY = [
    "observe_scene", "approach_object", "grasp_object", "verify_grasp", "lift_object",
    "transport_object", "align_with_target", "place_object", "release_object",
    "verify_goal_relation", "retry_grasp", "recover_object", "clear_target",
    "request_second_view", "request_clarification", "stop_no_action",
]
STATE_VOCABULARY = [
    "scene_unverified", "task_already_satisfied_candidate", "object_localized",
    "target_localized", "object_target_relation_verified", "grasp_candidate",
    "contact_established", "stable_hold_candidate", "object_lifted", "object_in_transport",
    "object_above_target", "object_on_target_candidate", "object_released",
    "goal_stability_candidate", "goal_verified", "grasp_failed", "contact_lost",
    "recovery_required", "target_blocked", "visual_unknown", "failure_terminal_candidate",
]
REQUIRED_MODULES = [
    "__init__.py", "cli.py", "io.py", "source_lock.py", "canonicalize.py",
    "dynamic_simulator.py", "scripted_controller.py", "renderer.py", "dataset.py",
    "vision.py", "tracking.py", "predicates.py", "predicate_dsl.py", "compile_graph.py",
    "execute_graph.py", "refine_graph.py", "evaluate.py", "active_view.py", "confirm.py",
    "handoff.py", "package.py",
]
REQUIRED_TESTS = [
    "test_dynamic_simulator.py", "test_predicate_dsl.py", "test_no_oracle_leakage.py",
    "test_graph_executor.py", "test_split_integrity.py",
]
REQUIRED_ROLLOUT_FILES = [
    "actions.csv", "low_level_controls.jsonl", "contact_sensor.csv",
    "gripper_command.csv", "termination.json", "online_observation.npz",
    "oracle_diagnostic.npz", "events.jsonl", "metadata.json", "frame_manifest.csv",
]
ONLINE_NPZ_KEYS = {"timestamps", "action_code", "gripper_command_closed", "contact_present"}
ORACLE_NPZ_KEYS = {"qpos", "qvel", "weld_state", "future_outcome", "scenario"}
FORBIDDEN_ZIP_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".npz", ".npy", ".pt", ".pth", ".ckpt",
    ".safetensors", ".mp4", ".avi", ".mov",
}


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, check_id: str, passed: bool, evidence: str, details: Any) -> None:
        self.checks.append({
            "id": check_id,
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
            "details": details,
        })

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [row for row in self.checks if row["status"] != "PASS"]


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _strip_g3_fields(graph: dict[str, Any]) -> dict[str, Any]:
    result = dict(graph)
    for key in ("graph_id", "active_second_view", "active_view_trigger", "query_threshold", "topology_same_as_g2"):
        result.pop(key, None)
    return result


def _verify_rollouts(rows: list[dict[str, str]]) -> dict[str, Any]:
    failures: list[str] = []
    total_frames = 0
    total_control_steps = 0
    horizons = 0
    terminal_failures = 0
    for row in rows:
        rollout = Path(row["path"])
        missing = [name for name in REQUIRED_ROLLOUT_FILES if not (rollout / name).is_file()]
        if missing:
            failures.append(f"{row['rollout_id']}: missing {missing}")
            continue
        actions = read_csv(rollout / "actions.csv")
        contacts = read_csv(rollout / "contact_sensor.csv")
        commands = read_csv(rollout / "gripper_command.csv")
        frames = read_csv(rollout / "frame_manifest.csv")
        controls = read_jsonl(rollout / "low_level_controls.jsonl")
        lengths = {len(actions), len(contacts), len(commands), len(frames), len(controls)}
        if len(lengths) != 1:
            failures.append(f"{row['rollout_id']}: aligned stream lengths differ")
        for index, (action, control) in enumerate(zip(actions, controls)):
            if control.get("action") != action.get("action") or int(control.get("action_index", -1)) != index + 1:
                failures.append(f"{row['rollout_id']}: low-level control alignment at {index}")
                break
            sequence = control.get("controls")
            if not isinstance(sequence, list) or not sequence:
                failures.append(f"{row['rollout_id']}: empty low-level sequence at {index}")
                break
            total_control_steps += len(sequence)
        for frame in frames:
            for key, digest_key in (("front_path", "front_sha256"), ("side_path", "side_sha256")):
                path_text = frame.get(key, "")
                if not path_text:
                    if key == "front_path":
                        failures.append(f"{row['rollout_id']}: missing front frame path")
                    continue
                path = Path(path_text)
                if not path.is_file() or sha256_file(path) != frame[digest_key]:
                    failures.append(f"{row['rollout_id']}: frame missing/hash mismatch {path.name}")
                    break
        total_frames += len(frames)
        with np.load(rollout / "online_observation.npz") as online:
            if set(online.files) != ONLINE_NPZ_KEYS:
                failures.append(f"{row['rollout_id']}: online NPZ keys {sorted(online.files)}")
            for key in online.files:
                if len(online[key]) != len(frames) or not np.isfinite(online[key]).all():
                    failures.append(f"{row['rollout_id']}: invalid online array {key}")
        with np.load(rollout / "oracle_diagnostic.npz") as oracle:
            if set(oracle.files) != ORACLE_NPZ_KEYS:
                failures.append(f"{row['rollout_id']}: oracle NPZ keys {sorted(oracle.files)}")
        termination = read_json(rollout / "termination.json")
        horizon = bool(termination["horizon"])
        terminal_failure = bool(termination["terminal_failure"])
        horizons += int(horizon)
        terminal_failures += int(terminal_failure)
        if horizon and (termination["done"] or termination["termination_type"] != "horizon" or terminal_failure):
            failures.append(f"{row['rollout_id']}: horizon/terminal classification")
        if terminal_failure and (not termination["done"] or termination["termination_type"] != "failure_terminal"):
            failures.append(f"{row['rollout_id']}: terminal failure classification")
    return {
        "status": "PASS" if not failures else "FAIL",
        "rollouts": len(rows),
        "frames": total_frames,
        "low_level_control_steps": total_control_steps,
        "horizons": horizons,
        "terminal_failures": terminal_failures,
        "failures": failures[:50],
        "failure_count": len(failures),
    }


def _verify_prediction_manifest(path: Path) -> dict[str, Any]:
    failures: list[str] = []
    rows = read_csv(path)
    frame_count = 0
    allowed_top = {"frame_index", "time", "predicates", "camera", "side_view_used"}
    for row in rows:
        prediction_path = Path(row["prediction_path"])
        if not prediction_path.is_file():
            failures.append(f"missing prediction: {prediction_path}")
            continue
        frames = read_jsonl(prediction_path)
        frame_count += len(frames)
        for frame in frames:
            if set(frame) != allowed_top:
                failures.append(f"{row['rollout_id']}: online fields {sorted(frame)}")
                break
            if set(frame["predicates"]) != set(PREDICATE_NAMES):
                failures.append(f"{row['rollout_id']}: predicate schema mismatch")
                break
            forbidden = find_forbidden_keys(frame)
            if forbidden:
                failures.append(f"{row['rollout_id']}: forbidden {forbidden}")
                break
    return {
        "status": "PASS" if not failures else "FAIL",
        "rollouts": len(rows),
        "frames": frame_count,
        "failures": failures[:50],
        "failure_count": len(failures),
    }


def _verify_zip(path: Path) -> dict[str, Any]:
    failures: list[str] = []
    forbidden: list[str] = []
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            failures.append(f"CRC failure: {bad}")
        names = archive.namelist()
        if "PACKAGE_SHA256SUMS.txt" not in names:
            failures.append("missing PACKAGE_SHA256SUMS.txt")
        else:
            for line in archive.read("PACKAGE_SHA256SUMS.txt").decode("utf-8").splitlines():
                digest, relative = line.split("  ", 1)
                if sha256_bytes(archive.read(relative)) != digest:
                    failures.append(f"internal SHA mismatch: {relative}")
        forbidden = [name for name in names if Path(name).suffix.lower() in FORBIDDEN_ZIP_SUFFIXES]
        if forbidden:
            failures.append(f"raw payloads packaged: {forbidden[:10]}")
    digest = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    recorded = sidecar.read_text(encoding="utf-8").split()[0] if sidecar.is_file() else None
    if recorded != digest:
        failures.append("external SHA mismatch")
    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": digest,
        "forbidden_payloads": forbidden,
        "failures": failures,
    }


def run_audit(repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    root = repo / "artifacts/pathgraph_sarm/upgrade_v2/visual_refine_l2_v1"
    rounds = root / "rounds"
    tools = repo / "upgrade_v2/visual_refine_l2"
    final = root / "final_v1"
    audit = Audit()

    refs = {name: _git(repo, "rev-parse", name) for name in ("HEAD", "main", "origin/main")}
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ENTRY_COMMIT, "HEAD"], cwd=repo,
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    audit.add("git.entry_and_refs", ancestor and len(set(refs.values())) == 1, "git refs/merge-base", refs)
    protected_changes = _git(
        repo, "diff", "--name-only", f"{ENTRY_COMMIT}..HEAD", "--",
        "artifacts/pathgraph_sarm/upgrade_v2/visual_coarse_l1_v1",
        "upgrade_v2/visual_coarse_l1",
    )
    audit.add("git.l1v_immutable", not protected_changes, "git diff entry..HEAD", protected_changes.splitlines())

    required_modules = [str(tools / name) for name in REQUIRED_MODULES]
    required_tests = [str(tools / "tests" / name) for name in REQUIRED_TESTS]
    missing_code = [path for path in required_modules + required_tests if not Path(path).is_file()]
    audit.add("implementation.required_modules", not missing_code, str(tools), {"missing": missing_code})

    protocol_path = root / "protocol_v1/configs/l2r_protocol.json"
    protocol = read_json(protocol_path)
    protocol_ok = (
        protocol["coarse_graph_source"] == "V1"
        and protocol["primary_camera"] == "front"
        and protocol["primary_online_observation"] == ["front_rgb", "action_history", "gripper_command", "contact_sensor"]
        and protocol["new_llm_calls_allowed"] is False
        and protocol["new_model_training_allowed"] is False
        and protocol["fresh_confirmation_used_once"] is True
    )
    audit.add("protocol.scientific_boundary", protocol_ok, str(protocol_path), protocol)

    entry_gate_path = rounds / "l2r_0_entry_and_source_lock/metrics/l2r_entry_gate.json"
    entry_gate = read_json(entry_gate_path)
    source_manifest = rounds / "l2r_0_entry_and_source_lock/manifests/v1_candidate_paths.txt"
    source_paths = [line for line in source_manifest.read_text(encoding="utf-8").splitlines() if line]
    source_lock_path = root / "protocol_v1/locks/l1v_source_lock.json"
    source_lock = verify_lock(source_lock_path)
    entry_ok = (
        entry_gate["status"] == "L2R_ENTRY_AND_SOURCE_LOCKED"
        and entry_gate["scientific_decision"] == "L1V_READY_FOR_REFINEMENT_SINGLE_VIEW"
        and entry_gate["coarse_graph_candidate_source"] == "V1"
        and len(source_paths) == 18
        and source_lock["status"] == "PASS"
    )
    audit.add("l2r0.entry_source_lock", entry_ok, str(entry_gate_path), {"gate": entry_gate, "candidates": len(source_paths), "source_lock": source_lock})

    action_path = root / "coarse_graph_v1/locks/action_vocabulary.json"
    state_path = root / "coarse_graph_v1/locks/state_vocabulary.json"
    canonical_gate_path = rounds / "l2r_1_coarse_graph_canonicalization/metrics/canonicalization_gate.json"
    evidence_path = root / "coarse_graph_v1/canonical/canonicalization_evidence.jsonl"
    canonical_gate = read_json(canonical_gate_path)
    evidence = read_jsonl(evidence_path)
    canonical_ok = (
        read_json(action_path)["values"] == ACTION_VOCABULARY
        and read_json(state_path)["values"] == STATE_VOCABULARY
        and canonical_gate["status"] == "V1_COARSE_GRAPH_CANONICALIZED"
        and canonical_gate["candidate_count"] == 18
        and len(evidence) == canonical_gate["evidence_rows"]
        and all(len(row.get("mapping_reason", "")) >= 12 for row in evidence)
        and all(isinstance(row.get("dynamic_validation_required"), bool) for row in evidence)
    )
    audit.add("l2r1.canonicalization", canonical_ok, str(canonical_gate_path), {"gate": canonical_gate, "evidence_rows": len(evidence)})

    manifest_paths = {
        "pilot": root / "dynamic_dataset_v1/pilot/pilot_rollout_manifest.csv",
        "development": root / "dynamic_dataset_v1/manifests/development_rollout_manifest.csv",
        "fresh": root / "fresh_confirmation_v1/data/rollout_manifest.csv",
    }
    manifest_rows = {name: read_csv(path) for name, path in manifest_paths.items()}
    rollout_audits = {name: _verify_rollouts(rows) for name, rows in manifest_rows.items()}
    family_counts = {name: len({row["root_family_id"] for row in rows}) for name, rows in manifest_rows.items()}
    development_split = read_csv(root / "dynamic_dataset_v1/manifests/development_family_split.csv")
    fit = {row["root_family_id"] for row in development_split if row["split"] == "dev_fit"}
    select = {row["root_family_id"] for row in development_split if row["split"] == "dev_select"}
    dynamic_gate_path = rounds / "l2r_2_dynamic_tabletop_dataset/metrics/dynamic_dataset_gate.json"
    dynamic_gate = read_json(dynamic_gate_path)
    fresh_scenarios = Counter(row["scenario"] for row in manifest_rows["fresh"] if row["rollout_id"].endswith("_r00"))
    dynamic_ok = (
        len(manifest_rows["pilot"]) == 16 and family_counts["pilot"] == 8
        and len(manifest_rows["development"]) == 192 and family_counts["development"] == 48
        and len(fit) == 32 and len(select) == 16 and not fit & select
        and len(manifest_rows["fresh"]) == 96 and family_counts["fresh"] == 24
        and set(fresh_scenarios) == SCENARIOS and set(fresh_scenarios.values()) == {3}
        and dynamic_gate["status"] == "DYNAMIC_TABLETOP_DATASET_READY"
        and dynamic_gate["event_counts"]["failure_occurrences"] >= 20
        and dynamic_gate["event_counts"]["recovery_attempts"] >= 20
        and dynamic_gate["event_counts"]["recovery_achievements"] >= 15
        and dynamic_gate["event_counts"]["stable_goal_confirmations"] >= 20
        and all(row["status"] == "PASS" for row in rollout_audits.values())
    )
    audit.add("l2r2.dynamic_dataset", dynamic_ok, str(dynamic_gate_path), {"families": family_counts, "fresh_scenarios": fresh_scenarios, "rollouts": rollout_audits, "gate": dynamic_gate})

    l1v_families = {row["root_family_id"] for row in read_csv(repo / "artifacts/pathgraph_sarm/upgrade_v2/visual_coarse_l1_v1/final/family_scores.csv")}
    dev_families = {row["root_family_id"] for row in manifest_rows["development"]}
    fresh_families = {row["root_family_id"] for row in manifest_rows["fresh"]}
    overlap = {"l1v": sorted(l1v_families & fresh_families), "development": sorted(dev_families & fresh_families)}
    audit.add("split.fresh_disjoint", not overlap["l1v"] and not overlap["development"], "family manifests", overlap)

    gpu_rounds = [
        "l2r_2_dynamic_tabletop_dataset", "l2r_3_observable_predicates",
        "l2r_4_graph_binding_and_refinement", "l2r_5_fresh_confirmation",
    ]
    gpu_required = ["nvidia_smi_sudo.txt", "nvidia_smi_direct.txt", "gpu_query_mode.txt", "gpu_inventory.csv", "runtime_environment.txt"]
    gpu_missing = {
        round_name: [name for name in gpu_required if not (rounds / round_name / "gpu" / name).is_file()]
        for round_name in gpu_rounds
    }
    gpu_runtime_bad = []
    for round_name in gpu_rounds:
        runtime_path = rounds / round_name / "gpu/runtime_environment.txt"
        if runtime_path.is_file():
            text = runtime_path.read_text(encoding="utf-8")
            if not all(label in text for label in ("python =", "torch =", "mujoco =")):
                gpu_runtime_bad.append(round_name)
    audit.add("environment.gpu_runtime_evidence", not any(gpu_missing.values()) and not gpu_runtime_bad, "round gpu directories", {"missing": gpu_missing, "runtime_incomplete": gpu_runtime_bad})

    threshold_path = root / "observable_predicates_v1/locks/predicate_thresholds.json"
    thresholds = read_json(threshold_path)
    predicate_metrics_path = rounds / "l2r_3_observable_predicates/tables/predicate_metrics.csv"
    predicate_metrics = {row["baseline"]: row for row in read_csv(predicate_metrics_path)}
    b3 = predicate_metrics["B3_single_view_temporal_contact_action"]
    predicate_audits = {
        "dev_select": _verify_prediction_manifest(rounds / "l2r_3_observable_predicates/tables/predicate_prediction_manifest.csv"),
        "fresh": _verify_prediction_manifest(root / "fresh_confirmation_v1/predicates/prediction_manifest.csv"),
    }
    predicate_ok = (
        thresholds["fit_split"] == "dev_fit" and thresholds["dev_select_used"] is False
        and thresholds["fresh_confirmation_used"] is False
        and set(thresholds["fit_families"]) == fit
        and set(predicate_metrics) == {
            "B0_text_coarse_assumption", "B1_current_frame_rgb", "B2_single_view_temporal",
            "B3_single_view_temporal_contact_action", "Oracle_diagnostic_upper_bound",
        }
        and float(b3["goal_f1"]) >= .80 and float(b3["stable_hold_f1"]) >= .70
        and float(b3["failure_f1"]) >= .60 and float(b3["recovery_f1"]) >= .60
        and float(b3["false_ready_rate"]) <= .15 and float(b3["unknown_rate"]) <= .30
        and predicate_audits["dev_select"]["rollouts"] == 64
        and predicate_audits["fresh"]["rollouts"] == 96
        and all(row["status"] == "PASS" for row in predicate_audits.values())
    )
    audit.add("l2r3.observable_predicates", predicate_ok, str(predicate_metrics_path), {"threshold_lock": thresholds, "B3": b3, "predictions": predicate_audits})

    graph_root = root / "refined_graphs_v1"
    graph_paths = {
        "G0_coarse_direct": graph_root / "compiled/G0_coarse_direct.json",
        "G1_predicate_bound": graph_root / "compiled/G1_predicate_bound.json",
        "G2_evidence_refined": graph_root / "candidates/G2_evidence_refined.json",
        "G3_active_second_view": graph_root / "candidates/G3_active_second_view.json",
    }
    graphs = {name: read_json(path) for name, path in graph_paths.items()}
    edit_log = read_csv(graph_root / "edit_logs/G2_edit_log.csv")
    allowed_edits = {f"E{index}" for index in range(1, 9)}
    edit_ok = (
        0 < len(edit_log) <= 8
        and all(row["edit_type"] in allowed_edits for row in edit_log)
        and all(row["selection_split"] == "dev_fit" for row in edit_log)
        and all(int(row["independent_family_count"]) >= 3 for row in edit_log)
        and all(row["dev_fit_error_count"] and row["coarse_elements"] and row["expected_metric"] and row["known_risk"] for row in edit_log)
    )
    topology_ok = _strip_g3_fields(graphs["G2_evidence_refined"]) == _strip_g3_fields(graphs["G3_active_second_view"])
    selection_path = graph_root / "selection/selection_lock.json"
    selection = read_json(selection_path)
    selection_hashes_ok = all(
        sha256_file(Path(selection["graph_source_path_by_id"][graph_id])) == digest
        for graph_id, digest in selection["graph_sha256_by_id"].items()
    )
    pre_selection = root / "fresh_confirmation_v1/locks/selection_lock.pre_confirmation.json"
    lock_same = sha256_file(selection_path) == sha256_file(pre_selection)
    dev_metrics_path = rounds / "l2r_4_graph_binding_and_refinement/tables/dev_select_graph_metrics.csv"
    dev_metrics = {row["graph_id"]: row for row in read_csv(dev_metrics_path)}
    graph_ok = (
        set(dev_metrics) == set(graph_paths)
        and edit_ok and topology_ok and selection_hashes_ok and lock_same
        and selection["selection_split"] == "dev_select"
        and selection["fresh_confirmation_used"] is False
        and selection["selected_graph_id"] == "G2_evidence_refined"
        and float(dev_metrics["G3_active_second_view"]["second_view_query_rate"]) <= .35
    )
    audit.add("l2r4.graph_refinement", graph_ok, str(selection_path), {"edits": len(edit_log), "edit_ok": edit_ok, "g3_topology_same": topology_ok, "lock_same": lock_same, "selection_hashes_ok": selection_hashes_ok, "metrics": dev_metrics})

    family_lock_path = root / "fresh_confirmation_v1/locks/fresh_family_lock.json"
    confirmation_lock = verify_confirmation_lock(selection_path, family_lock_path)
    consumption_path = root / "fresh_confirmation_v1/locks/confirmation_consumption.json"
    consumption = read_json(consumption_path)
    confirmation_path = root / "fresh_confirmation_v1/evaluation/confirmation_metrics.csv"
    paired_path = root / "fresh_confirmation_v1/evaluation/refined_minus_baselines.csv"
    scenario_path = root / "fresh_confirmation_v1/evaluation/metrics_by_scenario.csv"
    output_hashes_ok = all(
        sha256_file(root / "fresh_confirmation_v1/evaluation" / name) == digest
        for name, digest in consumption["output_sha256"].items()
    )
    confirmation_rows = read_csv(confirmation_path)
    status, reasons = decide_status(selection_path, confirmation_path, paired_path, scenario_path)
    fresh_ok = (
        confirmation_lock["status"] == "PASS"
        and consumption["status"] == "CONSUMED_ONCE"
        and consumption["metric_driven_tuning_after_first_open"] is False
        and consumption["post_confirmation_tuning"] is False
        and output_hashes_ok
        and len(confirmation_rows) == 3
        and {row["graph_id"] for row in confirmation_rows} == {"G0_coarse_direct", "G1_predicate_bound", "G2_evidence_refined"}
    )
    audit.add("l2r5.fresh_confirmation", fresh_ok, str(consumption_path), {"lock": confirmation_lock, "consumption": consumption, "output_hashes_ok": output_hashes_ok, "decision": status, "reasons": reasons})

    handoff_path = final / "l3_or_stop_handoff.json"
    handoff = read_json(handoff_path)
    unsupported = read_json(final / "unsupported_claims.json")
    final_manifest = read_json(final / "final_manifest.json")
    manifest_failures = []
    for row in final_manifest["files"]:
        path = final / row["relative_path"]
        if not path.is_file() or path.stat().st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
            manifest_failures.append(row["relative_path"])
    final_ok = (
        handoff["status"] == status == "L2R_PARTIAL_KEEP_COARSE_GRAPH"
        and handoff["graph"]["development_selected_graph_id"] == "G2_evidence_refined"
        and handoff["graph"]["retained_graph_id"] == "G1_predicate_bound"
        and handoff["l3_interface"]["entry_allowed"] is False
        and all(unsupported[key] is False for key in ("reward_gain", "policy_gain", "physical_robot_success", "new_task_generalization"))
        and not manifest_failures
    )
    audit.add("l2r6.final_handoff", final_ok, str(handoff_path), {"status": handoff["status"], "manifest_failures": manifest_failures, "unsupported_claims": unsupported})

    round_manifests = [read_json(path) for path in sorted(rounds.glob("l2r_*/run_manifest.json"))]
    execution_ok = (
        len(round_manifests) == 7
        and all(row["new_api_calls"] == 0 and row["new_training_jobs"] == 0 and row["api_key_read"] is False for row in round_manifests)
    )
    audit.add("execution.no_api_or_training", execution_ok, "round run manifests", {"rounds": len(round_manifests), "statuses": [row["status"] for row in round_manifests]})

    downloads = repo / "downloads/l2r"
    zip_paths = sorted(downloads.glob("*.zip"))
    zip_rows = [_verify_zip(path) for path in zip_paths]
    package_index = read_json(downloads / "package_index.json")
    index_by_name = {row["filename"]: row for row in package_index["packages"]}
    actual_names = {row["filename"] for row in zip_rows}
    indexed_names = set(index_by_name)
    index_ok = all(
        row["filename"] in index_by_name
        and index_by_name[row["filename"]]["sha256"] == row["sha256"]
        and index_by_name[row["filename"]]["size_bytes"] == row["size_bytes"]
        for row in zip_rows
    ) and actual_names == indexed_names and len(zip_rows) == int(package_index.get("package_count", len(zip_rows)))
    package_ok = len(zip_rows) == 8 and all(not row["failures"] for row in zip_rows) and index_ok
    audit.add("delivery.zip_integrity", package_ok, str(downloads / "package_index.json"), {"packages": zip_rows, "index_ok": index_ok})

    legacy = repo / "downloads/upgrade_v2/U0_U1_complete.zip"
    legacy_record = rounds / "l2r_6_final_handoff/checksums/U0_U1_complete_actual.sha256"
    legacy_digest = sha256_file(legacy)
    recorded_legacy = legacy_record.read_text(encoding="utf-8").split()[0]
    audit.add("delivery.legacy_zip_preserved", legacy_digest == recorded_legacy, str(legacy_record), {"sha256": legacy_digest})

    return {
        "schema": "pathgraph_l2r_manual_completion_audit_v1",
        "handbook": "/home/xushijie/PathGraph-SARM_L2R_单视图粗图观测绑定与动态细化_Agent操作手册_V1.0.md",
        "handbook_sha256": "396b58e88f9e8b8c78ad5ed696281c32b1a676bca3b5c1e871ca4d8d15f9deab",
        "audited_at": now_iso(),
        "status": "PASS" if not audit.failures else "FAIL",
        "checks_total": len(audit.checks),
        "checks_passed": len(audit.checks) - len(audit.failures),
        "checks_failed": len(audit.failures),
        "failures": audit.failures,
        "checks": audit.checks,
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# L2R Handbook Completion Audit",
        "",
        f"- Status: `{result['status']}`",
        f"- Checks: `{result['checks_passed']}/{result['checks_total']}` passed",
        f"- Handbook SHA256: `{result['handbook_sha256']}`",
        "",
    ]
    for row in result["checks"]:
        lines.append(f"- `{row['status']}` `{row['id']}`: {row['evidence']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(args.repo)
    write_json(args.output, result)
    write_report(args.report, result)
    print(json.dumps({key: result[key] for key in ("status", "checks_total", "checks_passed", "checks_failed")}, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
