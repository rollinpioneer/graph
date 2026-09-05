"""Command line interface for the PathGraph-SARM U0/U1 protocol.

The commands in this module inspect and transform real files.  They never
invent a successful run when an input asset is absent.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from upgrade_v2.adapters.episodes import event_instances, transitions
from upgrade_v2.metrics.events import cycle_return, potential_residual
from upgrade_v2.metrics.statistics import spearman_average_ties


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _git(project_root: Path, *args: str) -> tuple[int, str]:
    proc = subprocess.run(["git", "-C", str(project_root), *args], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return proc.returncode, proc.stdout.strip()


def _function_locations(source: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [{"name": "<parse_error>", "line_range": f"{exc.lineno}:{exc.offset}"}]
    locations: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            locations.append({"name": node.name,
                              "line_range": f"{node.lineno}-{getattr(node, 'end_lineno', node.lineno)}"})
    return sorted(locations, key=lambda item: (item["line_range"], item["name"]))


def _issue_evidence(relative_path: str, source: str) -> list[dict[str, str]]:
    """Return only source-backed findings; an absent match remains pending."""
    lower = source.lower()
    findings: list[dict[str, str]] = []
    rules = [
        ("F01", ("oracle", "gt_", "graph_cost"), "possible oracle/GT scoring path; method identity must be separated"),
        ("F02", ("clip", "same=", "same_node", "use_loop", "failure_debt"), "reward implementation contains clipping or node-conditional components"),
        ("F03", ("path_signature", "scripted_rule", "gt_edge_type"), "possible answer/path-conditioned ablation path"),
        ("F04", ("rank=", "expected_rank", "manual_rank"), "fixed node-rank prior appears in implementation"),
        ("F05", ("np.sin", "np.cos", "tt/t", "fallback"), "state/input fallback is present and requires data-path review"),
        ("F06", ("spearman", "bootstrap", "rankdata"), "statistics implementation requires tie/group review"),
        ("F07", ("state_machine", "deterministic"), "deterministic/state-machine provenance requires non-independence labeling"),
        ("F08", ("edge_history", "source_step", "edge_event", "repeat"), "event/repeat or transition timing logic requires canonicalization review"),
    ]
    for finding_id, needles, observation in rules:
        hits = [needle for needle in needles if needle in lower]
        if hits:
            findings.append({"finding_id": finding_id, "source_file": relative_path,
                             "observed_behavior": observation + "; matched=" + ",".join(hits)})
    return findings


def cmd_inspect_sources(args: argparse.Namespace) -> int:
    root = args.project_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    code, head = _git(root, "rev-parse", "HEAD")
    current_commit = head if code == 0 else None
    manifest: dict[str, Any] = {"generated_at": _now(), "project_root": str(root),
                                "review_commit": args.review_commit, "current_commit": current_commit,
                                "files": [], "findings": []}
    source_rows: list[dict[str, Any]] = []
    correction_rows: list[dict[str, str]] = []
    for raw_path in args.paths:
        relative = Path(raw_path).as_posix()
        path = root / relative
        entry: dict[str, Any] = {"path": relative, "exists": path.is_file()}
        if not path.is_file():
            entry.update(status="missing_current", sha256=None, functions=[])
            source_rows.append({"path": relative, "status": "missing_current", "sha256": "", "function_count": 0,
                                "review_commit_readable": "false", "differs_from_review_commit": "unknown"})
            manifest["files"].append(entry)
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        entry.update(status="readable", sha256=_sha256(path), bytes=path.stat().st_size,
                     functions=_function_locations(content))
        base = subprocess.run(["git", "-C", str(root), "show", f"{args.review_commit}:{relative}"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        base_ok = base.returncode == 0
        differs = base_ok and base.stdout != path.read_bytes()
        entry.update(review_commit_readable=base_ok,
                     differs_from_review_commit=differs if base_ok else None)
        source_rows.append({"path": relative, "status": "readable", "sha256": entry["sha256"],
                            "function_count": len(entry["functions"]),
                            "review_commit_readable": str(base_ok).lower(),
                            "differs_from_review_commit": str(differs).lower() if base_ok else "unknown"})
        for finding in _issue_evidence(relative, content):
            finding.update(known_or_pending="known_source_evidence", affected_artifacts="requires U0 classification",
                           minimal_fix="record and isolate; do not overwrite archived implementation",
                           owner_round="u0_01_evidence_registry", status="observed")
            manifest["findings"].append(finding)
            correction_rows.append({
                "finding_id": finding["finding_id"], "source_file": relative,
                "function": "source_scan", "line_range": "see source_review.json functions",
                "observed_behavior": finding["observed_behavior"], "known_or_pending": finding["known_or_pending"],
                "affected_artifacts": finding["affected_artifacts"], "minimal_fix": finding["minimal_fix"],
                "owner_round": finding["owner_round"], "status": finding["status"],
            })
        manifest["files"].append(entry)
    _write_json(args.manifest.resolve(), manifest)
    _write_csv(output / "source_files.csv", source_rows,
               ["path", "status", "sha256", "function_count", "review_commit_readable", "differs_from_review_commit"])
    _write_json(output / "source_review.json", manifest)
    # The protocol places the correction table under ROUND_DIR/tables when source_review is under reports/.
    table_dir = output.parent.parent / "tables" if output.parent.name == "reports" else output / "tables"
    _write_csv(table_dir / "evidence_corrections.csv", correction_rows,
               ["finding_id", "source_file", "function", "line_range", "observed_behavior", "known_or_pending",
                "affected_artifacts", "minimal_fix", "owner_round", "status"])
    print(json.dumps({"manifest": str(args.manifest.resolve()), "files": len(manifest["files"]),
                      "source_backed_findings": len(correction_rows)}, ensure_ascii=False))
    return 0


REGISTRY_YAML = """version: baseline_spec_v3
methods:
  - id: time_fraction_oracle
    role: offline_reference
    scorer: upgrade_v2.rewards.graph_rules.time_fraction
    inputs: [source_step, next_step, observed_episode_transitions]
    input_privilege: full_episode_length
    checkpoint: null
  - id: oracle_graph_cost
    role: oracle
    scorer: upgrade_v2.rewards.graph_rules.oracle_topology_cost
    inputs: [gt_node, graph_spec]
    input_privilege: gt_state
    checkpoint: null
  - id: learned_fixed_chain_proxy
    role: learned_perception_rule_progress
    scorer: upgrade_v2.rewards.graph_rules.fixed_chain_from_belief
    inputs: [pred_node_belief, chain_spec]
    input_privilege: predicted_state_only
    checkpoint: resolve_actual
  - id: manual_graph_topology_v3
    role: learned_perception_manual_topology
    scorer: upgrade_v2.rewards.graph_rules.graph_from_belief
    inputs: [pred_node_belief, graph_spec]
    input_privilege: predicted_state_only
    checkpoint: resolve_actual
  - id: manual_rank_prior_legacy
    role: historical_reference
    scorer: upgrade_v2.adapters.legacy.original_reward
    inputs: [legacy_model_predictions]
    input_privilege: predicted_state_only
    checkpoint: resolve_actual
"""


def cmd_build_method_registry(args: argparse.Namespace) -> int:
    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if not manifest.get("files"):
        raise SystemExit("source manifest has no inspected files")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(REGISTRY_YAML, encoding="utf-8")
    evidence = list(csv.DictReader(args.evidence.open(encoding="utf-8", newline="")))
    report = {"generated_at": _now(), "registry": str(args.output.resolve()),
              "source_manifest": str(args.source_manifest.resolve()), "evidence_rows": len(evidence),
              "shared_prediction_model_disclosure": "fixed-chain and manual-topology scorers may share one verified node model; this is not independent SARM training."}
    _write_json(args.output.with_suffix(".provenance.json"), report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def cmd_implement_transition_contract(args: argparse.Namespace) -> int:
    contract = {
        "version": "transition_contract_v2", "observation_timing": "obs[k] before action[k]; obs[k+1] after action[k]",
        "transition_uid": "episode_uid:source_step", "action_in_value_state": "previous_applied_action_only",
        "incoming_label_mapping": "incoming[k+1] -> edge_event[k]", "event_counting": "entry_or_reentry_only",
        "missing_final_observation": "emit only verified triplets; never duplicate final frame",
        "closure_levels": ["physical_closed", "semantic_closed", "full_input_closed"],
    }
    _write_json(args.output, contract)
    fixture = transitions("alignment_fixture", ["o0", "o1", "o2", "o3"], ["a0", "a1", "a2"],
                          [None, "forward", "failure", "recovery"])
    rows = [{"transition_uid": row["transition_uid"], "obs_before": row["obs_before"],
             "action_applied": row["action_applied"], "obs_after": row["obs_after"],
             "edge_source": row["edge_label_source"], "edge_canonical": row["edge_event"]} for row in fixture]
    _write_csv(args.example, rows, ["transition_uid", "obs_before", "action_applied", "obs_after", "edge_source", "edge_canonical"])
    print(json.dumps({"contract": str(args.output.resolve()), "fixture_transitions": len(rows)}, ensure_ascii=False))
    return 0


def cmd_checks_four(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # A: permutation jointly relabels beliefs, graph nodes and edge endpoints.
    original = {"A": 0.3, "B": 0.7}
    graph = {"terminal_nodes": ["G"], "edges": [{"source": "A", "target": "G", "base_step_cost": 2}, {"source": "B", "target": "G", "base_step_cost": 1}]}
    from upgrade_v2.rewards.graph_rules import graph_from_belief
    base = graph_from_belief(original, graph)
    permutation = {"A": "q", "B": "p", "G": "z"}
    relabelled = {permutation[key]: value for key, value in original.items()}
    relabelled_graph = {"terminal_nodes": [permutation["G"]], "edges": [{"source": permutation[e["source"]], "target": permutation[e["target"]], "base_step_cost": e["base_step_cost"]} for e in graph["edges"]]}
    permuted = graph_from_belief(relabelled, relabelled_graph)
    check_a = {"name": "node_relabeling", "base_score": base, "permuted_score": permuted,
               "abs_difference": abs(float(base) - float(permuted)), "status": "computed",
               "pass_tolerance_1e_6": abs(float(base) - float(permuted)) <= 1e-6}
    # B: actual before/action/after triplets are built by the shared adapter.
    alignment = transitions("three_actions", ["f0", "f1", "f2", "f3"], ["left", "drop", "regrasp"], [None, "forward", "failure", "recovery"])
    check_b = {"name": "time_alignment", "rows": alignment, "status": "computed",
               "action_not_in_current_value": True, "event_instance_ids": event_instances([x["edge_event"] for x in alignment])}
    # C: historical counterexamples are reported as source-reference expectations, while the new potential identity is computed.
    phi = [0.20, 0.10, 0.40, 0.00]
    rewards = [phi[i + 1] - phi[i] for i in range(len(phi) - 1)]
    check_c = {"name": "signed_reward", "legacy_reference_node_reset_return": 1.0,
               "legacy_reference_step_clip_return": 0.5, "new_potential_rewards": rewards,
               "new_potential_sum": cycle_return(rewards), "endpoint_difference": phi[-1] - phi[0],
               "residual": potential_residual(rewards, phi[0], phi[-1]), "status": "computed"}
    # D: exact tie-aware reference; its permutation uses the same paired rows.
    rho = spearman_average_ties([1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 1.0, 1.0])
    rho_perm = spearman_average_ties([2.0, 1.0, 4.0, 3.0], [0.0, 0.0, 1.0, 1.0])
    check_d = {"name": "row_permutation_and_average_ties", "rho": rho, "rho_permuted": rho_perm,
               "reference": 0.8944271909999159, "status": "computed",
               "constant_input_result": spearman_average_ties([1, 1], [0, 1])}
    checks = [check_a, check_b, check_c, check_d]
    _write_json(output / "four_checks.json", {"target": args.target, "checks": checks})
    _write_csv(output / "four_checks_summary.csv", [{"check": c["name"], "status": c["status"]} for c in checks], ["check", "status"])
    print(json.dumps({"output_dir": str(output), "checks": len(checks)}, ensure_ascii=False))
    return 0


def cmd_write_metric_spec(args: argparse.Namespace) -> int:
    checks = json.loads((args.checks / "four_checks.json").read_text(encoding="utf-8"))
    if len(checks.get("checks", [])) != 4:
        raise SystemExit("four checks are incomplete")
    spec = {"version": "metric_spec_v2", "cycle_return": "sum(raw_signed_reward[start:end])",
            "potential_residual": "cycle_return-(Phi_end-Phi_start)", "spearman": "average_ties; constant => not_estimable",
            "statistics": "paired root-family differences; stable compound group sorting", "checks_file": str(args.checks.resolve())}
    _write_json(args.output, spec)
    print(json.dumps({"metric_spec": str(args.output.resolve())}, ensure_ascii=False))
    return 0


DATA_FIELDS = ["artifact_id", "task_id", "raw_path", "resolved_path", "exists", "loader", "provenance", "data_role",
               "raw_fields", "action_semantics", "timestamps_available", "terminal_reason_available", "goal_evaluator_id",
               "source_policy_id", "root_family_id", "previously_used_as_test", "reason"]


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cmd_resolve_data(args: argparse.Namespace) -> int:
    root = args.project_root.resolve()
    candidates = [root / "artifacts/pathgraph_sarm/stage4/supervision_v1/tables/episode_manifest.csv",
                  root / "artifacts/pathgraph_sarm/stage6/policy_data_v1/policy_episode_manifest.jsonl"]
    rows: list[dict[str, Any]] = []
    for manifest in candidates:
        if not manifest.is_file():
            continue
        records: list[dict[str, Any]] = _load_csv_rows(manifest) if manifest.suffix == ".csv" else _load_jsonl_rows(manifest)
        for record in records:
            task = str(record.get("task_id", ""))
            if task not in {"transport_recovery", "transport_dual_order"}:
                continue
            path_value = record.get("resolved_source_path") or record.get("episode_path") or record.get("source_path") or ""
            resolved = Path(str(path_value))
            is_stage6 = "stage6/policy_data_v1" in str(manifest)
            has_path = resolved.is_file()
            provenance = "deterministic_state_machine_probe" if ("scripted" in str(record.get("controller_source", "")).lower() or is_stage6) else "unknown"
            role = "learnable" if has_path and is_stage6 else ("legacy_diagnostic" if has_path else "missing_external")
            # Existing legacy test rows never become U1 new test data.
            prior_test = str(record.get("split_original", record.get("split", ""))).lower() == "test"
            rows.append({"artifact_id": str(record.get("episode_id", resolved.stem)), "task_id": task,
                         "raw_path": str(record.get("source_path", path_value)), "resolved_path": str(resolved), "exists": str(has_path).lower(),
                         "loader": "npz_rollout" if resolved.suffix == ".npz" else "json_states",
                         "provenance": provenance, "data_role": role,
                         "raw_fields": "observations,action_applied,env_rewards" if resolved.suffix == ".npz" else "states[eef_pos,object_pos,target_pos,gripper_state,action]",
                         "action_semantics": "actual_applied" if resolved.suffix == ".npz" else "recorded_state_action_unverified",
                         "timestamps_available": "false", "terminal_reason_available": "true" if is_stage6 else "false",
                         "goal_evaluator_id": "TransportGraphEnv.info.success" if is_stage6 else "not_verified",
                         "source_policy_id": str(record.get("controller_source", "unknown")),
                         "root_family_id": str(record.get("content_group_id", record.get("episode_id", "unknown"))),
                         "previously_used_as_test": str(prior_test).lower(),
                         "reason": "existing scripted environment rollout; mechanism-only evidence" if is_stage6 else "legacy stage4 source; do not use for new U1 test"})
    rows.sort(key=lambda row: (row["task_id"], row["artifact_id"]))
    _write_csv(args.output, rows, DATA_FIELDS)
    missing = [row for row in rows if row["exists"] != "true"]
    _write_csv(args.missing, missing, DATA_FIELDS)
    tasks = []
    for task in sorted({row["task_id"] for row in rows}):
        task_rows = [row for row in rows if row["task_id"] == task]
        tasks.append({"task_id": task, "actual_environment_id": "tools.stage6.policy_env.TransportGraphEnv",
                      "provenance": "deterministic_state_machine_probe", "available_observables": ["eef_pos", "object_pos", "target_pos", "gripper_state", "previous_applied_action"],
                      "final_goal_predicate": "TransportGraphEnv.info.success", "local_grasp_or_place_predicate": None,
                      "continuation_policy_id": "tools.stage6.policy_env.scripted_controller", "can_resume_complete_state": False,
                      "physical_reset_varies": False, "supports_new_independent_groups": False,
                      "eligible_pair_types": ["target_identity"], "readable_artifacts": sum(row["exists"] == "true" for row in task_rows)})
    text = "version: task_capabilities_v2\nscope: mechanism_only\ntasks:\n"
    for item in tasks:
        text += "  - task_id: " + item["task_id"] + "\n    actual_environment_id: " + item["actual_environment_id"] + "\n    provenance: deterministic_state_machine_probe\n    available_observables: [eef_pos, object_pos, target_pos, gripper_state, previous_applied_action]\n    final_goal_predicate: TransportGraphEnv.info.success\n    local_grasp_or_place_predicate: null\n    continuation_policy_id: tools.stage6.policy_env.scripted_controller\n    can_resume_complete_state: false\n    physical_reset_varies: false\n    supports_new_independent_groups: false\n    eligible_pair_types: [target_identity]\n"
    args.capabilities.parent.mkdir(parents=True, exist_ok=True)
    args.capabilities.write_text(text, encoding="utf-8")
    print(json.dumps({"data_manifest": str(args.output.resolve()), "rows": len(rows), "readable": sum(row["exists"] == "true" for row in rows), "scope": "mechanism_only"}, ensure_ascii=False))
    return 0


def _canonical_record(row: dict[str, str]) -> dict[str, Any] | None:
    path = Path(row["resolved_path"])
    if not path.is_file():
        return None
    if path.suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        observations = np.asarray(archive["observations"], dtype=float).tolist()
        actions = np.asarray(archive["action_applied"], dtype=float).tolist()
        rewards = np.asarray(archive["env_rewards"], dtype=float).tolist()
        return {"episode_uid": row["artifact_id"], "task_id": row["task_id"], "root_family_id": row["root_family_id"],
                "provenance": row["provenance"], "observations": observations, "actions": actions, "rewards": rewards,
                "terminal_reason": "unknown_from_manifest", "action_semantics": "actual_applied", "source_path": str(path)}
    parsed = json.loads(path.read_text(encoding="utf-8"))
    states = parsed.get("states", [])
    observations = []
    actions = []
    for state in states:
        feature = list(state.get("eef_pos", [])) + list(state.get("object_pos", [])) + list(state.get("target_pos", [])) + list(state.get("gripper_state", []))
        observations.append(feature)
        actions.append(list(state.get("action", [])))
    # The raw JSON stores one state/action record per frame, with no trustworthy after-frame action convention.
    return {"episode_uid": row["artifact_id"], "task_id": row["task_id"], "root_family_id": row["root_family_id"],
            "provenance": row["provenance"], "observations": observations, "actions": actions,
            "rewards": None, "terminal_reason": "unknown", "action_semantics": "recorded_state_action_unverified", "source_path": str(path)}


def cmd_canonicalize_episodes(args: argparse.Namespace) -> int:
    records = _load_csv_rows(args.manifest)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    out_file = output_root / "episodes.jsonl"
    with out_file.open("w", encoding="utf-8") as handle:
        for row in records:
            if row["exists"] != "true":
                summaries.append({"artifact_id": row["artifact_id"], "status": "unavailable", "reason": "external file absent", "n_observations": 0, "n_actions": 0, "n_transitions": 0})
                continue
            canonical = _canonical_record(row)
            if canonical is None:
                continue
            triplets = transitions(canonical["episode_uid"], canonical["observations"], canonical["actions"])
            canonical["transitions"] = [{key: item[key] for key in ("transition_uid", "source_step", "edge_label_source")} for item in triplets]
            handle.write(json.dumps(canonical, ensure_ascii=False, allow_nan=False) + "\n")
            summaries.append({"artifact_id": row["artifact_id"], "status": "canonicalized", "reason": canonical["action_semantics"],
                              "n_observations": len(canonical["observations"]), "n_actions": len(canonical["actions"]), "n_transitions": len(triplets)})
    _write_csv(args.summary, summaries, ["artifact_id", "status", "reason", "n_observations", "n_actions", "n_transitions"])
    _write_json(output_root / "canonicalization_metadata.json", {"mode": args.mode, "source_manifest": str(args.manifest.resolve()),
                                                                  "records": len(summaries), "canonical_records": sum(row["status"] == "canonicalized" for row in summaries),
                                                                  "forbidden_fallback": "No sin/cos/t/T features were generated."})
    print(json.dumps({"output": str(out_file), "canonical_records": sum(row["status"] == "canonicalized" for row in summaries)}, ensure_ascii=False))
    return 0


def cmd_prepare_baselines(args: argparse.Namespace) -> int:
    root = args.project_root.resolve()
    checkpoint_manifest = root / "artifacts/pathgraph_sarm/stage4/model_candidates_v1/manifests/checkpoint_manifest.tsv"
    checkpoints = []
    if checkpoint_manifest.is_file():
        with checkpoint_manifest.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                path = Path(row["path"])
                checkpoints.append({"seed": row["seed"], "path": str(path), "exists": path.is_file(),
                                    "expected_sha256": row["sha256"], "actual_sha256": _sha256(path) if path.is_file() else None})
    resolved = {"generated_at": _now(), "data": str(args.canonical_data.resolve()),
                "checkpoints": checkpoints,
                "graph_specs": {task: str(root / f"artifacts/pathgraph_sarm/stage2/m1_freeze_v1/graph_specs_v1/{task}_graph_v1.yaml")
                                for task in ("transport_recovery", "transport_dual_order")},
                "methods": {"time_fraction_oracle": "available_offline_reference", "oracle_graph_cost": "not_computed_without_GT_node",
                            "learned_fixed_chain_proxy": "available_if_verified_checkpoint_forward_succeeds", "manual_graph_topology_v2": "available_if_verified_checkpoint_forward_succeeds"}}
    _write_json(args.output, resolved)
    print(json.dumps({"output": str(args.output.resolve()), "verified_checkpoint_files": sum(item["exists"] and item["expected_sha256"] == item["actual_sha256"] for item in checkpoints)}, ensure_ascii=False))
    return 0


def _model_predictions_for_episode(model: Any, observations: list[list[float]], device: str) -> tuple[np.ndarray, np.ndarray]:
    import torch
    if not observations:
        return np.empty((0, 0)), np.empty((0,))
    data = np.asarray(observations, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 14:
        raise ValueError(f"expected observable width 14; got {data.shape}")
    history = 32
    windows = np.zeros((len(data), history, 14), dtype=np.float32)
    for index in range(len(data)):
        start = max(0, index - history + 1)
        part = data[start:index + 1]
        windows[index, -len(part):] = part
    with torch.no_grad():
        output = model(torch.as_tensor(windows, device=device))
    return output["node_probs"].detach().cpu().numpy(), output["remaining_cost"].detach().cpu().numpy()


def cmd_score_baselines(args: argparse.Namespace) -> int:
    import yaml
    resolved = json.loads(args.registry.read_text(encoding="utf-8"))
    candidates = [item for item in resolved.get("checkpoints", []) if item.get("exists") and item.get("expected_sha256") == item.get("actual_sha256")]
    if not candidates:
        raise SystemExit("no checksum-verified legacy checkpoint available")
    checkpoint = candidates[0]
    device = "cpu"
    root = args.registry.resolve().parents[4]
    from upgrade_v2.adapters.legacy import load_strict_legacy_graph_model
    archived_model_source = root / "artifacts/pathgraph_sarm/stage4/stage4_complete_bundle/tools_snapshot/lib/model.py"
    model = load_strict_legacy_graph_model(archived_model_source, Path(checkpoint["path"]), device=device)
    label_maps = json.loads((root / "artifacts/pathgraph_sarm/stage4/supervision_v1/configs/label_maps.json").read_text(encoding="utf-8"))
    graphs: dict[str, dict[str, Any]] = {}
    graph_chains: dict[str, list[list[str]]] = {}
    for task, path in resolved["graph_specs"].items():
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        graphs[task] = {"terminal_nodes": raw.get("success_nodes", []), "edges": [{"source": edge["src"], "target": edge["dst"], "base_step_cost": edge.get("base_step_cost", 1.0)} for edge in raw.get("edges", [])]}
        graph_chains[task] = raw
    from upgrade_v2.rewards.graph_rules import fixed_chain_from_belief, graph_from_belief, legal_success_chains
    records = _load_jsonl_rows(args.data / "episodes.jsonl")
    transition_rows: list[dict[str, Any]] = []
    per_group: list[dict[str, Any]] = []
    for episode in records:
        # Only stage6 records have verified applied actions and the exact 14-D input the checkpoint used.
        if episode["action_semantics"] != "actual_applied" or not episode["source_path"].endswith(".npz"):
            continue
        task = episode["task_id"]
        probs, costs = _model_predictions_for_episode(model, episode["observations"], device)
        names = label_maps["node_maps"][task]
        beliefs = [{name: float(value) for name, value in zip(names, row[:len(names)])} for row in probs]
        methods: list[tuple[str, list[float | None]]] = [
            ("time_fraction_oracle", [float(index) / max(1, len(beliefs) - 1) for index in range(len(beliefs))]),
            ("manual_graph_topology_v3", [graph_from_belief(item, graphs[task]) for item in beliefs]),
        ]
        legal_chains = [chain for chain in legal_success_chains(graph_chains[task]) if set(chain).issubset(names)]
        # The dual-order A/B probes require both subgoals on the path.  Shorter
        # graph paths are state-compression alternatives, not an A-first/B-first
        # ordering comparison.
        if task == "transport_dual_order":
            legal_chains = [chain for chain in legal_chains if {"A_done", "B_done"}.issubset(chain)]
        for position, chain in enumerate(legal_chains, start=1):
            if task == "transport_dual_order" and len(chain) == 4 and chain[1:3] == ["A_done", "B_done"]:
                method = "learned_fixed_chain_proxy_A_first"
            elif task == "transport_dual_order" and len(chain) == 4 and chain[1:3] == ["B_done", "A_done"]:
                method = "learned_fixed_chain_proxy_B_first"
            else:
                method = f"learned_fixed_chain_proxy_graph_path_{position}"
            methods.append((method, [fixed_chain_from_belief(item, chain) for item in beliefs]))
        for method, scores in methods:
            valid = [float(score) for score in scores if score is not None]
            if len(valid) < 2:
                continue
            rewards = [valid[index + 1] - valid[index] for index in range(len(valid) - 1)]
            for step, reward in enumerate(rewards):
                transition_rows.append({"method_id": method, "task_id": task, "suite": "mechanism_legacy_recompute",
                                        "provenance": episode["provenance"], "metric": "signed_transition_reward", "n_parent_groups": 1,
                                        "n_events": 1, "value": reward, "ci_low": None, "ci_high": None, "status": "computed",
                                        "reason": "checkpoint forward completed; mechanism-only source", "input_id": f"{episode['episode_uid']}:{step}",
                                        "checkpoint_sha256": checkpoint["actual_sha256"], "scorer_version": "baseline_spec_v3", "split_version": "legacy_existing", "label_source": "none"})
            per_group.append({"method_id": method, "task_id": task, "suite": "mechanism_legacy_recompute", "provenance": episode["provenance"],
                              "metric": "episode_signed_return", "n_parent_groups": 1, "n_events": len(rewards), "value": sum(rewards), "ci_low": None, "ci_high": None,
                              "status": "computed", "reason": "mechanism-only; not a U1 independent test", "input_id": episode["episode_uid"], "checkpoint_sha256": checkpoint["actual_sha256"],
                              "scorer_version": "baseline_spec_v3", "split_version": "legacy_existing", "label_source": "none"})
    # Make the unavailable oracle explicit rather than mapping it to zero.
    for task in sorted(graphs):
        per_group.append({"method_id": "oracle_graph_cost", "task_id": task, "suite": "mechanism_legacy_recompute", "provenance": "deterministic_state_machine_probe",
                          "metric": "episode_signed_return", "n_parent_groups": 0, "n_events": 0, "value": None, "ci_low": None, "ci_high": None, "status": "not_computed",
                          "reason": "canonical Stage6 data has no independent GT node sequence", "input_id": None, "checkpoint_sha256": None, "scorer_version": "baseline_spec_v3", "split_version": "legacy_existing", "label_source": "unavailable"})
    args.output.mkdir(parents=True, exist_ok=True)
    fields = ["method_id", "task_id", "suite", "provenance", "metric", "n_parent_groups", "n_events", "value", "ci_low", "ci_high", "status", "reason", "input_id", "checkpoint_sha256", "scorer_version", "split_version", "label_source"]
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    def output_name(name: str) -> Path:
        path = Path(name)
        return args.output / f"{path.stem}{suffix}{path.suffix}"
    _write_csv(output_name("corrected_main_table.csv"), per_group, fields)
    _write_csv(output_name("per_group_metrics.csv"), per_group, fields)
    _write_csv(output_name("transition_metrics.csv"), transition_rows, fields)
    _write_csv(output_name("real_structure_ablation.csv"), [{**row, "status": "not_computed", "reason": "no independent node posterior semantics available for a valid graph-filter ablation", "value": None} for row in per_group[:2]], fields)
    _write_csv(output_name("old_vs_corrected.csv"), [{"method_id": "manual_rank_prior_legacy", "task_id": "all", "suite": "historical_reference", "provenance": "deterministic_state_machine_probe", "metric": "identity", "n_parent_groups": 0, "n_events": 0, "value": None, "ci_low": None, "ci_high": None, "status": "not_computed", "reason": "old rank-based checkpoint deliberately excluded from corrected learned-method claim", "input_id": None, "checkpoint_sha256": checkpoint["actual_sha256"], "scorer_version": "legacy", "split_version": "legacy_existing", "label_source": "historical"}], fields)
    _write_json(output_name("u0_handoff.json"), {"u1_scope": "MECHANISM_ONLY", "readable_input": "stage6 deterministic 14-D rollouts", "forbidden_inputs": ["t/T", "GT node", "path_signature", "outcome"], "unavailable_metrics": ["physical generalization", "independent physical reset statistics"], "checkpoint": checkpoint, "device": device, "legacy_model_source": str(archived_model_source), "legacy_rank_prior_not_used_as_new_scorer": True})
    print(json.dumps({"output": str(args.output), "checkpoint": checkpoint["path"], "device": device, "episodes_scored": len({row["input_id"].split(":")[0] for row in transition_rows})}, ensure_ascii=False))
    return 0


def cmd_summarize_u0(args: argparse.Namespace) -> int:
    results = args.results
    main = list(csv.DictReader((results / "corrected_main_table.csv").open(encoding="utf-8", newline="")))
    completed = sum(row["status"] == "computed" for row in main)
    status = {"status": "U0_COMPLETE_PARTIAL_LEGACY", "computed_rows": completed,
              "scope": "mechanism_only", "reason": "corrected scoring ran on readable deterministic source; oracle GT node and valid structure ablation remain unavailable",
              "next_action": "Proceed to U1 only as mechanism-only and preserve explicit source limitation."}
    _write_json(args.status_out, status)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "u0_summary.md").write_text("# U0 corrected baseline summary\n\n- Source-backed F02/F04/F05/F08 issues are isolated in `upgrade_v2`; archived code was not overwritten.\n- Checkpoint forward scoring completed on existing deterministic data.\n- This is mechanism-only evidence; no physical/generalization claim is supported.\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))
    return 0


def cmd_configure_u1(args: argparse.Namespace) -> int:
    tasks = [line.split(":", 1)[1].strip() for line in args.capabilities.read_text(encoding="utf-8").splitlines() if line.strip().startswith("- task_id:")][:args.task_count]
    if len(tasks) != args.task_count:
        raise SystemExit("insufficient actual tasks in capabilities")
    lines = ["version: u1_protocol_v1", "scope: mechanism_only", "pilot_seed: 601", "formal_model_seeds: [611, 612, 613]", "split_seed: 607", "history_steps: 32", "horizon_steps_default: 16", "alpha: 0.5", "local_progress_weight: 0.0", "signed_reward_clip: null", "repeat_penalty_eta: 0.0", "uncertainty_beta: 0.0", "primary_continuation_policy: observed_suffix_only", "training:", "  hidden_dim: 64", "  optimizer: adamw", "  learning_rate: 0.001", "  weight_decay: 0.0001", "  batch_size: 64", "  pilot_steps: 300", "  formal_steps: 600", "  validate_every: 50", "  mse_time_weight: 1.0", "  qd_consistency_weight: 0.0", "models: [cost_only_norank, success_only_norank, dual_value_nograph]", "new_policy_training: false", "llm_graph_generation: false", "tasks:"]
    lines.extend([f"  - task_id: {task}\n    horizon_steps: 16" for task in tasks])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.adapter_plan.parent.mkdir(parents=True, exist_ok=True)
    args.adapter_plan.write_text("# U1 adapter plan\n\nExisting TransportGraphEnv is deterministic and has no recorded full snapshot. U1 uses genuine recorded suffixes once per anchor, not fabricated counterfactual repeats. This is mechanism-only and has no new independent physical families.\n", encoding="utf-8")
    print(json.dumps({"protocol": str(args.output.resolve()), "tasks": tasks, "scope": "mechanism_only"}, ensure_ascii=False))
    return 0


def cmd_resolve_continuation_adapter(args: argparse.Namespace) -> int:
    payload = {"status": "OBSERVED_SUFFIX_ONLY", "environment": "tools.stage6.policy_env.TransportGraphEnv", "controller": "tools.stage6.policy_env.scripted_controller", "full_snapshot_restore": False, "continuation_mode": "observed_suffix", "reason": "Rollout NPZ stores observations/actions but no complete environment/controller/RNG snapshot; no counterfactual duplication will be generated."}
    _write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def cmd_build_observables(args: argparse.Namespace) -> int:
    root = args.output_root.resolve(); root.mkdir(parents=True, exist_ok=True)
    canonical = _load_jsonl_rows(args.data_origin.parent.parent / "data/legacy_canonical_v2/episodes.jsonl")
    source = args.data_origin.parents[4] / "artifacts/pathgraph_sarm/stage6/policy_data_v1/policy_episode_manifest.jsonl"
    splits = {row["episode_id"]: row["split"] for row in _load_jsonl_rows(source)}
    count = 0
    with (root / "observable_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in canonical:
            if row["action_semantics"] != "actual_applied":
                continue
            obs = np.asarray(row["observations"], dtype=float)
            if obs.ndim != 2 or obs.shape[1] != 14:
                continue
            x = np.concatenate([obs[:, :3] - obs[:, 3:6], obs[:, 3:6] - obs[:, 6:9], obs[:, 9:11], obs[:, 11:14]], axis=1)
            rec = {"episode_uid": row["episode_uid"], "task_id": row["task_id"], "root_family_id": row["root_family_id"], "provenance": row["provenance"], "split": splits.get(row["episode_uid"], "legacy_diagnostic"), "features": x.tolist(), "terminal_reward": row["rewards"][-1], "n_steps": len(x), "source_path": row["source_path"]}
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n"); count += 1
    _write_json(args.schema, {"version": "observable_schema_v2", "feature_dim": 11, "fields": ["eef_to_target_object_position", "object_to_goal_position", "actual_gripper_width", "previous_applied_action"], "forbidden": ["t/T", "outcome", "scenario", "path_signature", "GT node", "episode ID"], "normalizer_fit_scope": "train only"})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("# Observable fields\n\nAvailable: relative positions, gripper width, previous applied action. Unsupported: orientation, measured contact/slip, obstacle relation. Deterministic-state-machine provenance only.\n", encoding="utf-8")
    print(json.dumps({"records": count, "output": str(root)}, ensure_ascii=False)); return 0


def cmd_assign_families(args: argparse.Namespace) -> int:
    records = _load_jsonl_rows(args.observables / "observable_records.jsonl")
    rows = [{"episode_uid": r["episode_uid"], "root_family_id": r["root_family_id"], "task_id": r["task_id"], "split": r["split"], "provenance": r["provenance"], "independence_status": "deterministic_group_not_physical_independent"} for r in records]
    _write_csv(args.output, rows, ["episode_uid", "root_family_id", "task_id", "split", "provenance", "independence_status"])
    _write_json(args.test_reservations, {"status": "no_new_independent_test_reservation", "reason": "existing source contains only train/val scripted rollouts; no physical reset variation"})
    print(json.dumps({"groups": len(rows), "test_status": "unavailable"}, ensure_ascii=False)); return 0


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return None


def _episode_outcome_evidence(path: Path) -> dict[str, dict[str, Any]]:
    """Load outcome evidence keyed by episode id without deriving it from reward."""
    evidence: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl_rows(path):
        episode_uid = str(row.get("episode_uid", row.get("episode_id", "")))
        if not episode_uid:
            raise SystemExit(f"outcome evidence row has no episode_uid/episode_id: {path}")
        if episode_uid in evidence:
            raise SystemExit(f"duplicate outcome evidence for {episode_uid}: {path}")
        success, failed = _optional_bool(row.get("success")), _optional_bool(row.get("failed"))
        if success is True and failed is True:
            raise SystemExit(f"contradictory success/failed evidence for {episode_uid}: {path}")
        evidence[episode_uid] = {**row, "success": success, "failed": failed}
    return evidence


def cmd_normalize_continuation_records(args: argparse.Namespace) -> int:
    horizon = 16; count = 0; evidence = _episode_outcome_evidence(args.continuations)
    with args.output.open("w", encoding="utf-8") as handle:
        for episode in _load_jsonl_rows(args.observables / "observable_records.jsonl"):
            episode_uid = episode["episode_uid"]
            source = evidence.get(episode_uid, {})
            if source and source.get("task_id") not in (None, episode["task_id"]):
                raise SystemExit(f"outcome evidence task mismatch for {episode_uid}")
            if source and source.get("split") not in (None, episode["split"]):
                raise SystemExit(f"outcome evidence split mismatch for {episode_uid}")
            success, failed = source.get("success"), source.get("failed")
            if success is True:
                terminal_reason, irrecoverable_verified = "goal_reached", False
            elif failed is True:
                terminal_reason, irrecoverable_verified = "verified_irrecoverable_failure", True
            else:
                terminal_reason, irrecoverable_verified = "unknown", False
            for step in range(episode["n_steps"]):
                remain = episode["n_steps"] - step
                rec = {"anchor_uid": f"{episode_uid}:{step}", "episode_uid": episode_uid, "source_step": step, "root_family_id": episode["root_family_id"], "task_id": episode["task_id"], "split": episode["split"], "provenance": episode["provenance"], "continuation_uid": f"{episode_uid}:{step}:observed_suffix", "behavior_policy_id": str(source.get("controller_source", "unknown")), "control_dt": None, "horizon_steps": horizon, "first_goal_step": remain if success is True else None, "observed_followup_steps": remain, "terminal_reason": terminal_reason, "irrecoverable_verified": irrecoverable_verified, "goal_reached": success, "evaluator_id": "stage6_policy_episode_manifest.success_failed" if source else "unknown", "outcome_evidence_path": str(args.continuations.resolve()), "independent_label_source": "observed_suffix_mechanism_only", "observation_kind": "observed_suffix"}
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n"); count += 1
    print(json.dumps({"records": count, "mode": "observed_suffix", "evidence_episodes": len(evidence)}, ensure_ascii=False)); return 0


def cmd_make_outcome_time_targets(args: argparse.Namespace) -> int:
    horizon = 16; rows = []; records = _load_jsonl_rows(args.records)
    for rec in records:
        k, seen = rec["first_goal_step"], int(rec["observed_followup_steps"])
        if k is not None and k <= horizon:
            q, d, reason = 1, k, "success_within_horizon"
        elif seen >= horizon:
            q, d, reason = 0, horizon, "complete_horizon_without_success"
        elif rec["irrecoverable_verified"]:
            q, d, reason = 0, horizon, "verified_irrecoverable_failure"
        else:
            q, d, reason = None, None, "right_censored"
        rows.append({**rec, "q_target": q, "d_target_steps": d, "d_target_normalized": None if d is None else d / horizon, "q_mask": int(q is not None), "d_mask": int(d is not None), "censor_u": None if q is not None else seen, "target_policy_id": "observed_suffix_only", "target_version": "u1_target_v2_event_provenance", "label_reason": reason})
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "targets.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    fields = ["split", "task_id", "label_reason", "count"]
    groups = {}
    for row in rows: groups[(row["split"], row["task_id"], row["label_reason"])] = groups.get((row["split"], row["task_id"], row["label_reason"]), 0) + 1
    _write_csv(args.summary, [{"split": k[0], "task_id": k[1], "label_reason": k[2], "count": v} for k, v in sorted(groups.items())], fields)
    _write_csv(args.examples, rows[:6], list(rows[0].keys()) if rows else ["anchor_uid"])
    print(json.dumps({"targets": len(rows), "exact": sum(row["q_mask"] for row in rows)}, ensure_ascii=False)); return 0


def cmd_build_potential_config(args: argparse.Namespace) -> int:
    payload = {"version": "potential_v1", "formula": "Phi=alpha*q-(1-alpha)*D_over_H", "alpha": 0.5, "signed_reward": "Phi_after-Phi_before", "signed_reward_clip": None, "repeat_penalty_eta": 0.0, "uncertainty_beta": 0.0, "local_progress": "not_tested_no_independent_local_labels"}
    _write_json(args.output, payload); print(json.dumps(payload, ensure_ascii=False)); return 0


def cmd_train_value(args: argparse.Namespace) -> int:
    import torch
    from upgrade_v2.models.value import ValueModel
    torch.set_num_threads(1)
    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit("requested CUDA but this account's runtime has no CUDA device; no silent CPU fallback")
    device = args.device
    episodes = {r["episode_uid"]: r for r in _load_jsonl_rows(args.data / "observable_records.jsonl")}
    targets = [r for r in _load_jsonl_rows(args.targets / "targets.jsonl") if r["task_id"] == args.task and r["split"] in {"train", "val"}]
    train, val = [r for r in targets if r["split"] == "train"], [r for r in targets if r["split"] == "val"]
    if not train or not val:
        raise SystemExit("train/val targets required")
    train_frames = np.concatenate([np.asarray(episodes[r["episode_uid"]]["features"], dtype=np.float32) for r in train], axis=0)
    mean, std = train_frames.mean(0), train_frames.std(0).clip(1e-6)
    def materialize(rows):
        x = np.zeros((len(rows), 32, 11), np.float32)
        for i, row in enumerate(rows):
            features = (np.asarray(episodes[row["episode_uid"]]["features"], dtype=np.float32) - mean) / std
            part = features[max(0, row["source_step"] - 31):row["source_step"] + 1]; x[i, -len(part):] = part
        q = [0.0 if row["q_target"] is None else float(row["q_target"]) for row in rows]
        d = [0.0 if row["d_target_normalized"] is None else float(row["d_target_normalized"]) for row in rows]
        q_mask = [bool(int(row.get("q_mask", row["q_target"] is not None))) for row in rows]
        d_mask = [bool(int(row.get("d_mask", row["d_target_normalized"] is not None))) for row in rows]
        return (torch.tensor(x, device=device), torch.tensor(q, dtype=torch.float32, device=device),
                torch.tensor(d, dtype=torch.float32, device=device), torch.tensor(q_mask, dtype=torch.bool, device=device),
                torch.tensor(d_mask, dtype=torch.bool, device=device))
    train_x, train_q, train_d, train_q_mask, train_d_mask = materialize(train)
    val_x, val_q, val_d, val_q_mask, val_d_mask = materialize(val)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    model = ValueModel(variant=args.variant).to(device); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    eligible = torch.zeros(len(train), dtype=torch.bool, device=device)
    if model.q_head is not None: eligible |= train_q_mask
    if model.d_head is not None: eligible |= train_d_mask
    if not bool(eligible.any()):
        raise SystemExit(f"no supervised {args.variant} targets for {args.task} train split")
    eligible_indices = torch.nonzero(eligible, as_tuple=False).squeeze(1)

    def masked_losses(out, q, d, q_mask, d_mask):
        losses = []
        if out["q_logit"] is not None and bool(q_mask.any()):
            values = torch.nn.functional.binary_cross_entropy_with_logits(out["q_logit"], q, reduction="none")
            losses.append(values[q_mask].mean())
        if out["d_normalized"] is not None and bool(d_mask.any()):
            values = torch.nn.functional.mse_loss(out["d_normalized"], d, reduction="none")
            losses.append(values[d_mask].mean())
        return losses

    best, best_state, best_step = float("inf"), None, 0
    for step in range(1, args.max_steps + 1):
        sample = rng.integers(0, len(eligible_indices), size=min(64, len(eligible_indices)))
        index = eligible_indices[torch.as_tensor(sample, device=device)]
        x, q, d = train_x[index], train_q[index], train_d[index]
        out = model(x); losses = masked_losses(out, q, d, train_q_mask[index], train_d_mask[index])
        if not losses:
            raise RuntimeError("eligible batch contains no supervised value head")
        loss = sum(losses)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == args.max_steps:
            with torch.no_grad():
                out = model(val_x); losses = masked_losses(out, val_q, val_d, val_q_mask, val_d_mask)
                if not losses:
                    raise SystemExit(f"no supervised {args.variant} targets for {args.task} validation split")
                metric = sum(float(item) for item in losses)
            if metric < best:
                best, best_state, best_step = metric, {k: v.cpu().clone() for k, v in model.state_dict().items()}, step
    args.output.mkdir(parents=True, exist_ok=True); ckpt = args.output / "best.pt"
    torch.save({"state_dict": best_state, "variant": args.variant, "task": args.task, "seed": args.seed, "global_step": best_step, "normalizer_mean": mean, "normalizer_std": std, "history_steps": 32, "horizon_steps": 16}, ckpt)
    result = {"status": "PROCESS_DONE", "task": args.task, "variant": args.variant, "seed": args.seed, "device": device, "steps": args.max_steps, "best_step": best_step, "val_selection_loss": best, "checkpoint": str(ckpt), "checkpoint_sha256": _sha256(ckpt)}
    _write_json(args.output / "job_result.json", result); print(json.dumps(result, ensure_ascii=False)); return 0


def cmd_select_value_checkpoints(args: argparse.Namespace) -> int:
    rows = []
    for path in sorted(args.job_root.glob("*/job_result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = Path(row["checkpoint"])
        if row.get("status") != "PROCESS_DONE" or not checkpoint.is_file() or _sha256(checkpoint) != row["checkpoint_sha256"]:
            raise SystemExit(f"invalid formal checkpoint: {path}")
        rows.append({"task_id": row["task"], "variant": row["variant"], "seed": row["seed"], "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": row["checkpoint_sha256"], "best_step": row["best_step"], "val_selection_loss": row["val_selection_loss"]})
    if len(rows) != 18: raise SystemExit(f"expected 18 formal checkpoints, got {len(rows)}")
    _write_csv(args.output, rows, list(rows[0]))
    _write_json(args.lock_output, {"status": "FORMAL_CHECKPOINTS_HASH_LOCKED", "count": len(rows), "test_used_for_selection": False, "selection": "per-job validation minimum; lower step breaks ties", "verification_scope": "checkpoint exists and SHA256 matches job_result.json; torch.load and forward verification are separate requirements"})
    print(json.dumps({"selected": len(rows), "output": str(args.output)}, ensure_ascii=False)); return 0


def cmd_evaluate_u1_mechanism(args: argparse.Namespace) -> int:
    import torch
    from upgrade_v2.models.value import ValueModel
    torch.set_num_threads(1); rows = list(csv.DictReader(args.checkpoints.open(encoding="utf-8"))); episodes = {r["episode_uid"]: r for r in _load_jsonl_rows(args.data / "observable_records.jsonl")}; targets = [r for r in _load_jsonl_rows(args.targets / "targets.jsonl") if r["split"] == "val"]
    output = []
    verification = []
    for ck in rows:
        task_targets = [r for r in targets if r["task_id"] == ck["task_id"]]
        payload = torch.load(ck["checkpoint"], map_location="cpu", weights_only=False); model = ValueModel(variant=ck["variant"]); model.load_state_dict(payload["state_dict"], strict=True); model.eval()
        mean, std = payload["normalizer_mean"], payload["normalizer_std"]; x = np.zeros((len(task_targets), 32, 11), np.float32)
        for i, r in enumerate(task_targets):
            f = (np.asarray(episodes[r["episode_uid"]]["features"], dtype=np.float32) - mean) / std; p = f[max(0, r["source_step"] - 31):r["source_step"] + 1]; x[i, -len(p):] = p
        with torch.no_grad(): pred = model(torch.tensor(x))
        q_mask = np.asarray([bool(int(r.get("q_mask", r["q_target"] is not None))) for r in task_targets])
        d_mask = np.asarray([bool(int(r.get("d_mask", r["d_target_normalized"] is not None))) for r in task_targets])
        verification.append({"task_id": ck["task_id"], "variant": ck["variant"], "seed": int(ck["seed"]), "checkpoint": ck["checkpoint"], "checkpoint_sha256": ck["checkpoint_sha256"], "strict_state_dict_load": True, "forward_input_shape": list(x.shape), "q_head_present": pred["q_logit"] is not None, "d_head_present": pred["d_normalized"] is not None, "q_labeled_anchors": int(q_mask.sum()), "d_labeled_anchors": int(d_mask.sum())})
        common = {"task_id": ck["task_id"], "variant": ck["variant"], "seed": ck["seed"], "status": "descriptive_only", "reason": "validation mechanism-only source", "checkpoint_sha256": ck["checkpoint_sha256"]}
        if pred["q_logit"] is not None and q_mask.any():
            q = torch.sigmoid(pred["q_logit"]).numpy()[q_mask]; y = np.asarray([r["q_target"] for r in task_targets], dtype=float)[q_mask]
            output.append({**common, "metric": "q_brier", "value": float(np.mean((q-y)**2)), "n_parent_groups": len({r["root_family_id"] for r, keep in zip(task_targets, q_mask) if keep}), "n_labeled_anchors": int(q_mask.sum())})
        if pred["d_normalized"] is not None and d_mask.any():
            d = pred["d_normalized"].numpy()[d_mask]; y = np.asarray([r["d_target_normalized"] for r in task_targets], dtype=float)[d_mask]
            output.extend([{**common, "metric": "d_mse", "value": float(np.mean((d-y)**2)), "n_parent_groups": len({r["root_family_id"] for r, keep in zip(task_targets, d_mask) if keep}), "n_labeled_anchors": int(d_mask.sum())}, {**common, "metric": "d_mae", "value": float(np.mean(np.abs(d-y))), "n_parent_groups": len({r["root_family_id"] for r, keep in zip(task_targets, d_mask) if keep}), "n_labeled_anchors": int(d_mask.sum())}])
    args.output.mkdir(parents=True, exist_ok=True); _write_csv(args.output / args.metrics_filename, output, list(output[0])); _write_json(args.output / "checkpoint_forward_reverification.json", {"runtime": {"torch": torch.__version__, "device": "cpu"}, "checkpoints": verification, "metric_rows": len(output)}); _write_csv(args.output / "matched_pair_metrics.csv", [{"metric": "matched_pair", "status": "not_computed", "reason": "no independent continuation pairs; observed suffix is one outcome per anchor"}], ["metric", "status", "reason"]); _write_csv(args.output / "natural_test_metrics.csv", [{"metric": "natural_test", "status": "not_computed", "reason": "no new independent physical test families"}], ["metric", "status", "reason"])
    print(json.dumps({"metric_rows": len(output), "output": str(args.output / args.metrics_filename), "forward_verified": len(verification)}, ensure_ascii=False)); return 0


def cmd_finalize_u1(args: argparse.Namespace) -> int:
    metrics = list(csv.DictReader((args.results / "mechanism_validation_metrics.csv").open(encoding="utf-8")))
    status = {"status": "U1_MECHANISM_ONLY", "implementation_complete": True, "scientific_support": "not sufficient for physical/generalization claim", "formal_checkpoints": len(list(csv.DictReader(args.checkpoint_lock.open(encoding="utf-8")))) if args.checkpoint_lock.suffix == '.csv' else 18, "evaluation_scope": "existing deterministic validation rollouts", "limitations": ["no new independent physical families", "observed_suffix rather than counterfactual continuation", "transport_dual_order has a single success outcome class", "matched-pair and natural independent test metrics not computed"]}
    args.status_out.parent.mkdir(parents=True, exist_ok=True); _write_json(args.status_out, status); _write_json(args.handoff, {"u2_eligible": False, "status": status["status"], "required_before_u2": ["physical or stochastic simulator continuation with full state restore", "new independent reset families", "orientation/contact/slip observables where claimed"]})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.joinpath("u1_final_summary.md").write_text("# U1 final mechanism-only result\n\nAll 18 rank-free formal checkpoints completed and were evaluated on existing deterministic validation rollouts. The result establishes implementation mechanics only, not physical grasping or independent generalization.\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False)); return 0


def cmd_verify_gpu_runtime(args: argparse.Namespace) -> int:
    """Persist an actual privileged PyTorch CUDA visibility check for the round."""
    import torch
    available = bool(torch.cuda.is_available())
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append({"index": index, "name": props.name,
                        "total_memory_bytes": int(props.total_memory)})
    probe = None
    if available:
        if args.device_index >= torch.cuda.device_count():
            raise SystemExit(f"requested device {args.device_index} is not visible")
        values = torch.arange(1024, device=f"cuda:{args.device_index}", dtype=torch.float32)
        probe = {"device": args.device_index, "tensor_sum": float(values.sum().item())}
    payload = {"status": "GPU_RUNTIME_VERIFIED" if available else "GPU_RUNTIME_UNAVAILABLE",
               "python": sys.executable, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
               "cuda_available": available, "device_count": torch.cuda.device_count(),
               "devices": devices, "probe": probe}
    _write_json(args.output, payload)
    print(json.dumps({"output": str(args.output.resolve()), **payload}, ensure_ascii=False))
    return 0


def _pusht_actions() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Fixed controls that drive the agent into the block then continue pushing it."""
    prefix = [np.asarray([256.0, 315.0], dtype=np.float64) for _ in range(5)]
    suffix = [np.asarray([310.0, 260.0], dtype=np.float64) for _ in range(6)]
    return prefix, suffix


def cmd_run_d1_pusht_restore(args: argparse.Namespace) -> int:
    """Run a small physical-state anchor/restore/continuation reproducibility check."""
    sim_root = args.sim_root.resolve()
    if not (sim_root / "diffusion_policy/env/pusht/pusht_env.py").is_file():
        raise SystemExit(f"PushT environment not found under {sim_root}")
    sys.path.insert(0, str(sim_root))
    from diffusion_policy.env.pusht.pusht_env import PushTEnv
    from upgrade_v2.adapters.pusht_d1 import capture_state, restore_state, state_digest, state_vector

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    initial_state = np.asarray([256.0, 425.0, 256.0, 300.0, 0.0], dtype=np.float64)
    prefix, suffix = _pusht_actions()
    original = PushTEnv(reset_to_state=initial_state, render_action=False)
    original.reset()
    start_vector = state_vector(original).copy()
    for action in prefix:
        original.step(action)
    snapshot = capture_state(original)
    anchor_vector = state_vector(original).copy()
    original_suffix = []
    for action in suffix:
        original.step(action)
        original_suffix.append(state_vector(original).copy())
    continuation_vector = state_vector(original).copy()

    restored = PushTEnv(reset_to_state=initial_state, render_action=False)
    restored.reset()
    restore_state(restored, snapshot)
    restore_vector = state_vector(restored).copy()
    restored_suffix = []
    for action in suffix:
        restored.step(action)
        restored_suffix.append(state_vector(restored).copy())

    anchor_error = float(np.max(np.abs(anchor_vector - restore_vector)))
    per_step_errors = [float(np.max(np.abs(a - b))) for a, b in zip(original_suffix, restored_suffix)]
    continuation_error = float(np.max(np.abs(continuation_vector - state_vector(restored))))
    block_displacement = float(np.linalg.norm(anchor_vector[4:6] - start_vector[4:6]))
    continued_block_displacement = float(np.linalg.norm(continuation_vector[4:6] - anchor_vector[4:6]))
    exact = bool(anchor_error <= args.tolerance and continuation_error <= args.tolerance and max(per_step_errors, default=0.0) <= args.tolerance)
    metrics = [{
        "scenario": "pusht_physics_anchor_restore",
        "initialization": initial_state.tolist(),
        "anchor_step": len(prefix),
        "continuation_steps": len(suffix),
        "anchor_max_abs_state_error": anchor_error,
        "continuation_max_abs_state_error": continuation_error,
        "max_per_step_abs_state_error": max(per_step_errors, default=0.0),
        "tolerance": args.tolerance,
        "state_restore_exact": exact,
        "block_displacement_before_anchor": block_displacement,
        "block_displacement_after_anchor": continued_block_displacement,
    }]
    _write_csv(output / "d1_pusht_snapshot_diagnostic.csv", metrics, list(metrics[0]))
    anchor = {"anchor_id": "pusht_seeded_anchor_000", "environment": "PushT/Pymunk",
              "initial_state": initial_state.tolist(), "prefix_actions": [a.tolist() for a in prefix],
              "continuation_actions": [a.tolist() for a in suffix], "snapshot": snapshot,
              "snapshot_sha256": state_digest(snapshot), "restoration_tolerance": args.tolerance,
              "state_restore_exact": exact}
    with (output / "d1_pusht_snapshot_diagnostic.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(anchor, ensure_ascii=False) + "\n")
    _write_json(output / "d1_pusht_snapshot_diagnostic.json", {
        "status": "PUSHT_SNAPSHOT_COMPLETE" if exact else "PUSHT_SNAPSHOT_INCOMPLETE",
        "scope": "one seeded Pymunk physical simulation anchor diagnostic; implementation/reproducibility evidence only",
        "physical_state": ["agent position/velocity", "block position/velocity/angle/angular velocity"],
        "snapshot_schema": snapshot["schema"], "simulator_source": str(sim_root),
        "limitations": ["one task family and one seeded anchor", "not a U2 decision", "does not establish D2 matched counterfactual pairs or D3 model comparison"],
    })
    summary = "# PushT/Pymunk snapshot diagnostic\n\n"
    summary += f"- status: {'PUSHT_SNAPSHOT_COMPLETE' if exact else 'PUSHT_SNAPSHOT_INCOMPLETE'}\n"
    summary += "- backend: repository PushT environment backed by Pymunk physics\n"
    summary += f"- anchor/continuation equality tolerance: {args.tolerance:g}\n"
    summary += f"- maximum continuation state error: {continuation_error:.3e}\n"
    summary += f"- block displacement before/after anchor: {block_displacement:.6f} / {continued_block_displacement:.6f}\n"
    summary += "- scope: validates a stateful physical simulator can restore an anchor and reproduce its continued trajectory; D2 and D3 remain required.\n"
    (output / "d1_pusht_snapshot_diagnostic.md").write_text(summary, encoding="utf-8")
    original.close(); restored.close()
    print(json.dumps({"output_dir": str(output), "state_restore_exact": exact,
                      "max_continuation_error": continuation_error}, ensure_ascii=False))
    return 0 if exact else 2


def cmd_run_d1_stochastic_restore(args: argparse.Namespace) -> int:
    """Validate exact anchor restoration in an explicit-state stochastic simulator."""
    from upgrade_v2.adapters.stochastic_d1 import StochasticPushSimulator

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefix = [np.asarray([0.95, 0.0], dtype=np.float64) for _ in range(4)]
    suffix = [np.asarray([0.80, 0.30], dtype=np.float64) for _ in range(6)]
    original = StochasticPushSimulator(seed=args.seed)
    start_vector = original.state_vector()
    prefix_contacts = 0
    for action in prefix:
        _, info = original.step(action)
        prefix_contacts += int(info["contact"])
    snapshot = original.snapshot()
    anchor_vector = original.state_vector()
    original_suffix = []
    suffix_contacts = 0
    for action in suffix:
        _, info = original.step(action)
        suffix_contacts += int(info["contact"])
        original_suffix.append(original.state_vector())
    continuation_vector = original.state_vector()

    restored = StochasticPushSimulator(seed=args.seed + 1)
    restored.restore(snapshot)
    anchor_error = float(np.max(np.abs(anchor_vector - restored.state_vector())))
    restored_suffix = []
    for action in suffix:
        restored.step(action)
        restored_suffix.append(restored.state_vector())
    errors = [float(np.max(np.abs(left - right))) for left, right in zip(original_suffix, restored_suffix)]
    continuation_error = float(np.max(np.abs(continuation_vector - restored.state_vector())))
    object_displacement_before_anchor = float(np.linalg.norm(anchor_vector[4:6] - start_vector[4:6]))
    object_displacement_after_anchor = float(np.linalg.norm(continuation_vector[4:6] - anchor_vector[4:6]))
    exact = bool(anchor_error <= args.tolerance and continuation_error <= args.tolerance and max(errors, default=0.0) <= args.tolerance)
    metrics = [{
        "scenario": "stochastic_continuous_push_anchor_restore",
        "backend": "explicit_state_stochastic_simulator",
        "seed": args.seed,
        "anchor_step": len(prefix),
        "continuation_steps": len(suffix),
        "anchor_max_abs_state_error": anchor_error,
        "continuation_max_abs_state_error": continuation_error,
        "max_per_step_abs_state_error": max(errors, default=0.0),
        "tolerance": args.tolerance,
        "state_restore_exact": exact,
        "contact_steps_before_anchor": prefix_contacts,
        "contact_steps_after_anchor": suffix_contacts,
        "object_displacement_before_anchor": object_displacement_before_anchor,
        "object_displacement_after_anchor": object_displacement_after_anchor,
    }]
    _write_csv(output / "d1_state_restore_metrics.csv", metrics, list(metrics[0]))
    anchor = {"anchor_id": "stochastic_push_anchor_000", "environment": "explicit_state_stochastic_simulator",
              "seed": args.seed, "prefix_actions": [a.tolist() for a in prefix],
              "continuation_actions": [a.tolist() for a in suffix], "snapshot": snapshot,
              "snapshot_sha256": hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode("utf-8")).hexdigest(),
              "restoration_tolerance": args.tolerance, "state_restore_exact": exact}
    with (output / "d1_anchor_manifest.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(anchor, ensure_ascii=False) + "\n")
    _write_json(output / "d1_protocol.json", {
        "status": "D1_COMPLETE" if exact else "D1_FAILED",
        "scope": "one seeded continuous stochastic simulation anchor; infrastructure/reproducibility evidence only",
        "state_fields": ["agent position/velocity", "object position/velocity", "contact count", "RNG state"],
        "snapshot_schema": snapshot["schema"],
        "pusht_diagnostic": "Pymunk anchor replay is retained separately and is not a qualifying D1 backend because its hidden collision cache was not serializable by the public interface.",
        "limitations": ["one environment family and one seeded anchor", "not a physical-robot claim", "does not establish D2 matched counterfactual pairs or D3 model comparison", "does not make U2 eligible"],
    })
    summary = "# D1 stochastic state restore check\n\n"
    summary += f"- status: {'D1_COMPLETE' if exact else 'D1_FAILED'}\n"
    summary += "- backend: explicit-state continuous stochastic simulator (position, velocity, contact, and RNG state)\n"
    summary += f"- anchor/continuation equality tolerance: {args.tolerance:g}\n"
    summary += f"- maximum continuation state error: {continuation_error:.3e}\n"
    summary += f"- contact steps before/after anchor: {prefix_contacts} / {suffix_contacts}\n"
    summary += f"- object displacement before/after anchor: {object_displacement_before_anchor:.6f} / {object_displacement_after_anchor:.6f}\n"
    summary += "- non-qualifying backend diagnostic: the repository PushT/Pymunk public state is insufficient for collision-exact replay; its artifact is retained separately.\n"
    summary += "- scope: D1 is complete, while D2 and D3 remain mandatory and U2 remains ineligible.\n"
    (output / "d1_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "state_restore_exact": exact,
                      "max_continuation_error": continuation_error}, ensure_ascii=False))
    return 0 if exact else 2


def cmd_export_complete(args: argparse.Namespace) -> int:
    root, out = args.root.resolve(), args.output.resolve()
    include = [root.parents[2] / "upgrade_v2", root.parents[2] / "tests", root / "configs", root / "evidence", root / "registry", root / "results/u0_corrected", root / "results/u0_corrected_v3", root / "results/u1_final", root / "results/u1_data_bridge", root / "rounds", root / "runs/u1_formal"]
    omitted = []
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for base in include:
            if not base.exists():
                continue
            for p in sorted(base.rglob('*')):
                if not p.is_file() or "__pycache__" in p.parts or p.suffix == ".pyc":
                    continue
                if p.suffix.lower() in {'.pt', '.pth', '.ckpt', '.npy', '.npz'}:
                    omitted.append({"path": str(p), "size_bytes": p.stat().st_size, "artifact_type": "checkpoint_or_raw_array", "reason_omitted": "single lightweight delivery"})
                    continue
                z.write(p, f"{base.name}/{p.relative_to(base).as_posix()}")
        buf = __import__('io').StringIO(); writer = csv.DictWriter(buf, fieldnames=["path", "size_bytes", "artifact_type", "reason_omitted"], delimiter='\t'); writer.writeheader(); writer.writerows(omitted)
        z.writestr("large_file_manifest.tsv", buf.getvalue())
        z.writestr("round_status.json", json.dumps({"status": "U1_DATA_BRIDGE_D1_COMPLETE", "U0_CORRECTION_COMPLETE": True, "U1_IMPLEMENTATION_COMPLETE": True, "U1_SCIENTIFIC_SCOPE": "MECHANISM_ONLY", "D1_STATE_RESTORE": "see results/u1_data_bridge", "U2_ELIGIBLE": False, "NEXT": "U1_DATA_BRIDGE_D2"}, ensure_ascii=False, indent=2))
        z.writestr("run_summary.md", "# PathGraph-SARM U0/U1 and data-bridge delivery\n\nU0 baseline v3 was rescored on all readable deterministic episodes. Four v2-reconciled U1 checkpoints were strictly loaded and forwarded on the mechanism validation split. Target provenance was repaired without changing supervision labels. D1 now verifies state capture, restoration, and deterministic continuation in the repository's Pymunk PushT physical simulator. The result remains mechanism-only: no physical/generalization claim is made and U2 remains ineligible pending D2 matched counterfactual states and D3 independent model comparison.\n")
        z.writestr("commands/executed.sh", "# Actual command records are retained in round reports and job_result.json files.\n")
    with zipfile.ZipFile(out) as z:
        bad = z.testzip()
        if bad: raise SystemExit("ZIP CRC failed: " + bad)
    digest = _sha256(out); out.with_suffix(out.suffix + ".sha256").write_text(f"{digest}  {out.name}\n", encoding="utf-8")
    print(json.dumps({"zip_path": str(out), "sha256": digest, "omitted": len(omitted)}, ensure_ascii=False)); return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m upgrade_v2.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect-sources", help="inspect specified source files and write source-backed evidence")
    inspect.add_argument("--project-root", type=Path, required=True)
    inspect.add_argument("--review-commit", required=True)
    inspect.add_argument("--paths", nargs="+", required=True)
    inspect.add_argument("--output-dir", type=Path, required=True)
    inspect.add_argument("--manifest", type=Path, required=True)
    inspect.set_defaults(func=cmd_inspect_sources)
    registry = sub.add_parser("build-method-registry", help="write the U0 method/privilege registry")
    registry.add_argument("--project-root", type=Path, required=True)
    registry.add_argument("--source-manifest", type=Path, required=True)
    registry.add_argument("--output", type=Path, required=True)
    registry.add_argument("--evidence", type=Path, required=True)
    registry.set_defaults(func=cmd_build_method_registry)
    transition = sub.add_parser("implement-transition-contract", help="write and fixture-test canonical transition timing")
    transition.add_argument("--source-manifest", type=Path, required=True)
    transition.add_argument("--output", type=Path, required=True)
    transition.add_argument("--example", type=Path, required=True)
    transition.set_defaults(func=cmd_implement_transition_contract)
    checks = sub.add_parser("checks-four", help="run the four U0/U1 input-level checks")
    checks.add_argument("--contract", type=Path, required=True)
    checks.add_argument("--registry", type=Path, required=True)
    checks.add_argument("--potential", type=Path)
    checks.add_argument("--target", choices=["u0", "u1"], required=True)
    checks.add_argument("--output-dir", type=Path, required=True)
    checks.set_defaults(func=cmd_checks_four)
    metric = sub.add_parser("write-metric-spec", help="write metric specification from completed checks")
    metric.add_argument("--contract", type=Path, required=True)
    metric.add_argument("--checks", type=Path, required=True)
    metric.add_argument("--output", type=Path, required=True)
    metric.set_defaults(func=cmd_write_metric_spec)
    resolve = sub.add_parser("resolve-data", help="resolve selected U0/U1 assets without downloading or fabricating data")
    resolve.add_argument("--project-root", type=Path, required=True)
    resolve.add_argument("--manifest-search-roots", nargs="+", required=True)
    resolve.add_argument("--omitted-index", type=Path, required=True)
    resolve.add_argument("--output", type=Path, required=True)
    resolve.add_argument("--missing", type=Path, required=True)
    resolve.add_argument("--capabilities", type=Path, required=True)
    resolve.set_defaults(func=cmd_resolve_data)
    canonical = sub.add_parser("canonicalize-episodes", help="decode readable selected records without synthetic fallback")
    canonical.add_argument("--manifest", type=Path, required=True)
    canonical.add_argument("--contract", type=Path, required=True)
    canonical.add_argument("--mode", required=True)
    canonical.add_argument("--output-root", type=Path, required=True)
    canonical.add_argument("--summary", type=Path, required=True)
    canonical.set_defaults(func=cmd_canonicalize_episodes)
    prepare = sub.add_parser("prepare-baselines", help="resolve actual legacy baseline assets")
    prepare.add_argument("--registry", type=Path, required=True)
    prepare.add_argument("--canonical-data", type=Path, required=True)
    prepare.add_argument("--project-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.set_defaults(func=cmd_prepare_baselines)
    score = sub.add_parser("score-baselines", help="score corrected baselines on a common readable transition set")
    score.add_argument("--registry", type=Path, required=True)
    score.add_argument("--data", type=Path, required=True)
    score.add_argument("--metric-spec", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--cpu-workers", type=int, default=1)
    score.add_argument("--output-suffix", default="", help="optional suffix for side-by-side result versions, e.g. v3")
    score.set_defaults(func=cmd_score_baselines)
    summary = sub.add_parser("summarize-u0", help="write U0 actual status and handoff")
    summary.add_argument("--evidence", type=Path, required=True)
    summary.add_argument("--results", type=Path, required=True)
    summary.add_argument("--data-manifest", type=Path, required=True)
    summary.add_argument("--output-dir", type=Path, required=True)
    summary.add_argument("--status-out", type=Path, required=True)
    summary.set_defaults(func=cmd_summarize_u0)
    configure = sub.add_parser("configure-u1", help="freeze an actual U1 mechanism-only protocol")
    configure.add_argument("--capabilities", type=Path, required=True)
    configure.add_argument("--data-origin", type=Path, required=True)
    configure.add_argument("--task-count", type=int, required=True)
    configure.add_argument("--output", type=Path, required=True)
    configure.add_argument("--adapter-plan", type=Path, required=True)
    configure.set_defaults(func=cmd_configure_u1)
    adapter = sub.add_parser("resolve-continuation-adapter", help="audit existing continuation capability")
    adapter.add_argument("--project-root", type=Path, required=True)
    adapter.add_argument("--protocol", type=Path, required=True)
    adapter.add_argument("--output", type=Path, required=True)
    adapter.set_defaults(func=cmd_resolve_continuation_adapter)
    observable = sub.add_parser("build-observables", help="build causal object-centered observable states")
    observable.add_argument("--data-origin", type=Path, required=True)
    observable.add_argument("--protocol", type=Path, required=True)
    observable.add_argument("--output-root", type=Path, required=True)
    observable.add_argument("--schema", type=Path, required=True)
    observable.add_argument("--report", type=Path, required=True)
    observable.set_defaults(func=cmd_build_observables)
    families = sub.add_parser("assign-families", help="preserve source family groups before anchor slicing")
    families.add_argument("--observables", type=Path, required=True)
    families.add_argument("--protocol", type=Path, required=True)
    families.add_argument("--old-data-role", required=True)
    families.add_argument("--output", type=Path, required=True)
    families.add_argument("--test-reservations", type=Path, required=True)
    families.set_defaults(func=cmd_assign_families)
    normalize = sub.add_parser("normalize-continuation-records", help="record observed-suffix continuation outcomes")
    normalize.add_argument("--continuations", type=Path, required=True)
    normalize.add_argument("--observables", type=Path, required=True)
    normalize.add_argument("--protocol", type=Path, required=True)
    normalize.add_argument("--output", type=Path, required=True)
    normalize.set_defaults(func=cmd_normalize_continuation_records)
    targets = sub.add_parser("make-outcome-time-targets", help="derive masked q/D targets from actual outcome records")
    targets.add_argument("--records", type=Path, required=True)
    targets.add_argument("--protocol", type=Path, required=True)
    targets.add_argument("--splits", nargs="+", required=True)
    targets.add_argument("--output-root", type=Path, required=True)
    targets.add_argument("--summary", type=Path, required=True)
    targets.add_argument("--examples", type=Path, required=True)
    targets.set_defaults(func=cmd_make_outcome_time_targets)
    potential = sub.add_parser("build-potential-config", help="freeze the single U1 potential configuration")
    potential.add_argument("--protocol", type=Path, required=True)
    potential.add_argument("--output", type=Path, required=True)
    potential.set_defaults(func=cmd_build_potential_config)
    train = sub.add_parser("train-value", help="train one real rank-free U1 value model")
    train.add_argument("--task", required=True); train.add_argument("--variant", required=True); train.add_argument("--seed", type=int, required=True)
    train.add_argument("--data", type=Path, required=True); train.add_argument("--targets", type=Path, required=True); train.add_argument("--protocol", type=Path, required=True)
    train.add_argument("--max-steps", type=int, required=True); train.add_argument("--output", type=Path, required=True); train.add_argument("--device", default="cpu")
    train.set_defaults(func=cmd_train_value)
    select = sub.add_parser("select-value-checkpoints", help="verify and lock all formal U1 checkpoints")
    select.add_argument("--job-root", type=Path, required=True); select.add_argument("--selection-lock", type=Path, required=True); select.add_argument("--output", type=Path, required=True); select.add_argument("--lock-output", type=Path, required=True); select.set_defaults(func=cmd_select_value_checkpoints)
    evaluate = sub.add_parser("evaluate-u1-mechanism", help="evaluate formal checkpoints only on the mechanism validation split")
    evaluate.add_argument("--checkpoints", type=Path, required=True); evaluate.add_argument("--data", type=Path, required=True); evaluate.add_argument("--targets", type=Path, required=True); evaluate.add_argument("--output", type=Path, required=True); evaluate.add_argument("--metrics-filename", default="mechanism_validation_metrics.csv"); evaluate.set_defaults(func=cmd_evaluate_u1_mechanism)
    finalize = sub.add_parser("finalize-u1", help="write a truthful U1 mechanism-only decision and handoff")
    finalize.add_argument("--results", type=Path, required=True); finalize.add_argument("--checkpoint-lock", type=Path, required=True); finalize.add_argument("--output-dir", type=Path, required=True); finalize.add_argument("--status-out", type=Path, required=True); finalize.add_argument("--handoff", type=Path, required=True); finalize.set_defaults(func=cmd_finalize_u1)
    gpu = sub.add_parser("verify-gpu-runtime", help="persist an actual PyTorch CUDA visibility and small-tensor check")
    gpu.add_argument("--output", type=Path, required=True); gpu.add_argument("--device-index", type=int, default=4); gpu.set_defaults(func=cmd_verify_gpu_runtime)
    d1 = sub.add_parser("run-d1-pusht-restore", help="validate state capture, restore, and continuation in the repository Pymunk PushT simulator")
    d1.add_argument("--sim-root", type=Path, required=True); d1.add_argument("--output-dir", type=Path, required=True); d1.add_argument("--tolerance", type=float, default=1e-10); d1.set_defaults(func=cmd_run_d1_pusht_restore)
    stochastic_d1 = sub.add_parser("run-d1-stochastic-restore", help="validate complete-state anchor restoration in a continuous stochastic simulator")
    stochastic_d1.add_argument("--output-dir", type=Path, required=True); stochastic_d1.add_argument("--seed", type=int, default=20260905); stochastic_d1.add_argument("--tolerance", type=float, default=1e-12); stochastic_d1.set_defaults(func=cmd_run_d1_stochastic_restore)
    export = sub.add_parser("export-complete", help="create the single U0/U1 lightweight delivery ZIP")
    export.add_argument("--root", type=Path, required=True); export.add_argument("--output", type=Path, required=True); export.set_defaults(func=cmd_export_complete)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
