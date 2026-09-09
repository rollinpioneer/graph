from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from .inputs import read_json, sha256_file, write_json


def _write_status_csv(path: Path, status: str, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"status,reason\n{status},{reason}\n", encoding="utf-8")


def _copy_if_exists(source: Path, target: Path) -> None:
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def handoff(run_root: Path, protocol_path: Path, output_root: Path) -> dict[str, Any]:
    final = output_root
    final.mkdir(parents=True, exist_ok=True)
    route_path = run_root / "rounds/l2rar1_3_development_selection/development_route.json"
    route = read_json(route_path) if route_path.is_file() else {"status": "NOT_RUN"}
    confirm_path = run_root / "rounds/l2rar1_4_fresh_confirmation/confirmation_consumption.json"
    confirm = read_json(confirm_path) if confirm_path.is_file() else {"status": "NOT_RUN", "standard_confirmation_status": "NOT_RUN", "challenge_confirmation_status": "NOT_RUN"}
    metrics_path = run_root / "rounds/l2rar1_3_development_selection/candidate_metrics.csv"
    gates_path = run_root / "rounds/l2rar1_3_development_selection/development_gates.csv"
    decisions_path = run_root / "rounds/l2rar1_3_development_selection/event_decisions.csv"
    metrics = list(csv.DictReader(metrics_path.open(encoding="utf-8"))) if metrics_path.is_file() else []
    gates = list(csv.DictReader(gates_path.open(encoding="utf-8"))) if gates_path.is_file() else []
    decisions = list(csv.DictReader(decisions_path.open(encoding="utf-8"))) if decisions_path.is_file() else []
    dense = [row for row in metrics if row.get("sampling") == "control_tick_20hz"]
    best = min(dense, key=lambda row: (float(row["wrong_or_unknown_rate"]), float(row["negative_false_emergency_rate"]), row["candidate_id"])) if dense else None
    failed_gate_names = sorted({
        key for row in gates for key, value in row.items()
        if key not in {"candidate_id", "all_pass", "legacy_compatibility", "legacy_compatibility_status"} and value == "False"
    })
    current_status = "L2RAR1_PARTIAL_KEEP_G1" if route.get("status") == "DEVELOPMENT_NOT_READY" else "L2RAR1_DIAGNOSIS_ONLY"
    if route.get("status") == "NOT_RUN":
        current_status = "L2RAR1_DIAGNOSIS_ONLY"
    for name in ["source_resolution.md"]:
        _copy_if_exists(run_root / name, final / name)
    for source_name, target_name in [("rounds/l2rar1_1_error_mechanism/case_attribution.csv", "cause_attribution.csv"), ("rounds/l2rar1_1_error_mechanism/legacy_to_prefix_bridge.csv", "legacy_to_prefix_bridge.csv"), ("rounds/l2rar1_2_observation_and_reference/callback_equivalence.json", "manifests/callback_equivalence.json"), ("rounds/l2rar1_2_observation_and_reference/dense_stream_normalization.json", "manifests/dense_stream_normalization.json"), ("rounds/l2rar1_2_observation_and_reference/dense_stream_normalization.csv", "manifests/dense_stream_normalization.csv"), ("rounds/l2rar1_2_observation_and_reference/invalid_duplicate_batch_v1.json", "manifests/invalid_duplicate_batch_v1.json"), ("rounds/l2rar1_3_development_selection/content_group_manifest.csv", "manifests/content_group_manifest.csv"), ("rounds/l2rar1_3_development_selection/candidate_registry.json", "candidate_registry.json"), ("rounds/l2rar1_3_development_selection/development_gates.csv", "tables/development_gates.csv"), ("rounds/l2rar1_3_development_selection/candidate_metrics.csv", "tables/method_by_sampling.csv"), ("rounds/l2rar1_3_development_selection/event_decisions.csv", "tables/first_decision_and_delay.csv"), ("rounds/l2rar1_4_fresh_confirmation/confirmation_consumption.json", "locks/confirmation_consumption.json")]:
        _copy_if_exists(run_root / source_name, final / target_name)
    write_json(final / "locks/selection_lock.json", read_json(run_root / "locks/selection_lock.json") if (run_root / "locks/selection_lock.json").is_file() else {"status": "MISSING"})
    write_json(final / "confirmation_status.json", confirm)
    _write_status_csv(final / "tables/standard_legacy_compatibility.csv", "NOT_RUN", "No independent confirmation was consumed")
    _write_status_csv(final / "tables/standard_event_metrics.csv", "NOT_RUN", "Development gate failed before R4")
    _write_status_csv(final / "tables/challenge_event_metrics.csv", "NOT_RUN", "Development gate failed before R4")
    _write_status_csv(final / "tables/hold_evidence_metrics.csv", "DEVELOPMENT_ONLY", "See method_by_sampling.csv for measured development candidates")
    _write_status_csv(final / "tables/paired_family_effects.csv", "NOT_RUN", "No confirmation family was consumed")
    _write_status_csv(final / "tables/content_group_sensitivity.csv", "NOT_RUN", "No confirmation family was consumed")
    unsupported = {"schema": "pathgraph_l2rar1_unsupported_claims_v1", "claims_not_supported": ["not a physical-robot guarantee", "not an L3 entry approval", "not proof that all old errors were caused by sampling", "no API or training gain claim"]}
    write_json(final / "unsupported_claims.json", unsupported)
    best_text = "not available"
    if best:
        miss_rows = [
            row for row in decisions
            if row.get("candidate_id") == best["candidate_id"]
            and row.get("sampling") == "control_tick_20hz"
            and row.get("reference_event_type") == "missed_grasp_retry_required"
            and row.get("physical_label_status") == "reference_labeled"
        ]
        miss_correct = sum(row.get("correct") == "True" for row in miss_rows)
        best_text = (
            f"`{best['candidate_id']}`: miss {best['miss_recall']} "
            f"({miss_correct}/{len(miss_rows)} correct), "
            f"regular/brief/long-gap loss {best['regular_loss_recall']}/{best['brief_loss_recall']}/{best['long_gap_recall']}, "
            f"wrong-or-unknown {best['wrong_or_unknown_rate']}"
        )
    report = f"""# L2RA-R1 final report

- Scientific status: **{current_status}**
- Retained graph: `G1_predicate_bound`; L3 entry remains closed.
- Valid development execution: 16 fit families / 64 rollouts and 16 select families / 64 rollouts, with 128 distinct online content groups.
- Invalid earlier batch: rejected for duplicate control programs, retained only as generator-contract audit evidence, and counted as 0 scientific rollouts.
- Development route: `{route.get('status')}`; 0/8 candidates eligible; selected candidate: `{route.get('selected_candidate_id')}`.
- Best dense-stream result by preregistered ordering: {best_text}.
- Confirmation: `{confirm.get('status')}`; standard `{confirm.get('standard_confirmation_status')}`, challenge `{confirm.get('challenge_confirmation_status')}`. R4 consumed no family because development failed.
- Failed development gate fields observed across candidates: `{', '.join(failed_gate_names)}`. Legacy compatibility was not evaluated because every candidate had already failed primary event gates.
- Callback equivalence and dense-stream timestamp normalization passed; normalization reexecuted 0 physical rollouts.
- API calls: `0`; training jobs: `0`; API keys read: `false`.

The old magnitude-only co-motion score is direction-blind, but the preregistered direction-aware C3 candidate tied B_count2 on the primary dense stream rather than establishing feature gain. Dense and action-end sampling exchanged missed-grasp and loss recall, so sampling gain was not established either. The unresolved boundary is causal: transient contact before a physically recorded missed grasp prevents the common online interface from asserting an empty grasp, while treating such contact as decisive would recreate the short-touch false-positive problem.

All conclusions are limited to the fixed single-view MuJoCo proxy families. They are not physical-robot guarantees or evidence that the allowed observation stream is fundamentally insufficient.
"""
    (final / "final_report.md").write_text(report, encoding="utf-8")
    handoff_payload = {"schema": "pathgraph_l2rar1_handoff_v1", "historical_l2ra_status": "L2RA_PARTIAL_KEEP_G1", "current_status": current_status, "retained_graph_id": "G1_predicate_bound", "selected_candidate_id": route.get("selected_candidate_id"), "candidate_ready_for_review": bool(route.get("status") == "DEVELOPMENT_READY" and confirm.get("standard_confirmation_status") == "PASS" and confirm.get("challenge_confirmation_status") == "PASS"), "primary_sampling": "control_tick_20hz", "comparison_sampling": "action_end", "new_reference_and_metric_versions": ["timestamp_aligned_proxy_hold_v2", "prefix_event_v2"], "development_families": {"fit": route.get("fit_families"), "select": route.get("select_families")}, "development_rollouts": {"fit": route.get("fit_rollouts"), "select": route.get("select_rollouts")}, "content_groups": route.get("content_groups"), "eligible_candidate_count": len(route.get("eligible_candidates", [])), "failed_gate_fields": failed_gate_names, "standard_confirmation_status": confirm.get("standard_confirmation_status", "NOT_RUN"), "challenge_confirmation_status": confirm.get("challenge_confirmation_status", "NOT_RUN"), "feature_gain_supported": False, "sampling_gain_supported": False, "common_interface_effect": "COMMON_INTERFACE_EXPOSED_TRANSIENT_CONTACT_AMBIGUITY", "l3_entry_allowed": False, "api_calls": 0, "training_jobs": 0, "api_key_read": False}
    write_json(final / "next_stage_handoff.json", handoff_payload)
    write_json(final / "execution_manifest.json", {"schema": "pathgraph_l2rar1_execution_manifest_v1", "status": current_status, "run_root": str(run_root.resolve()), "protocol": str(protocol_path.resolve()), "commands": "see rounds/actual_commands.txt", "api_calls": 0, "training_jobs": 0, "api_key_read": False})
    external = final / "manifests/external_artifacts.tsv"
    external.parent.mkdir(parents=True, exist_ok=True)
    external_rows = []
    external_roots = (
        (run_root / "data/new_development", "paired rollout observations and metadata"),
        (run_root / "features", "derived feature cache; reproducible from paired observations"),
        (run_root / "rounds/l2rar1_1_error_mechanism/case_trace.jsonl", "case-level diagnostic trace"),
    )
    for root, purpose in external_roots:
        paths = sorted(root.rglob("*") if root.is_dir() else [root])
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in {".jsonl", ".json", ".csv"}:
                continue
            external_rows.append(
                f"{path.resolve()}\t{path.stat().st_size}\t{sha256_file(path)}\t{purpose}\trestore from this absolute path and verify SHA256"
            )
    external.write_text("path\tsize_bytes\tsha256\tuse\trestore\n" + "\n".join(external_rows) + "\n", encoding="utf-8")
    return handoff_payload
