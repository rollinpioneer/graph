from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from .io import read_csv, read_json, sha256, write_csv, write_json


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def build_handoff(run_root: Path, protocol_path: Path, output_root: Path) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    inputs = read_json(run_root / "manifests/resolved_inputs.json")
    contract_status = read_json(run_root / "rounds/l2rar2_2_task_contract/contract_status.json")
    development_path = run_root / "rounds/l2rar2_3_development/select/development_gates.json"
    development = read_json(development_path) if development_path.is_file() else {"status": "NOT_RUN", "all_pass": False, "failed": ["development_not_run"]}
    confirmation_path = run_root / "rounds/l2rar2_4_focused_confirmation/focused_confirmation_status.json"
    if confirmation_path.is_file():
        confirmation = read_json(confirmation_path)
    else:
        confirmation = {
            "schema": "pathgraph_l2rar2_focused_confirmation_status_v1",
            "status": "NOT_RUN_DEVELOPMENT_FAILED" if development.get("status") == "DEVELOPMENT_FAILED" else "NOT_RUN",
            "all_gates_pass": False,
            "failed_gates": development.get("failed", []),
            "root_families": 0,
            "physical_rollouts": 0,
            "events": 0,
            "selected_candidate_id": None,
            "standard_l2r_confirmation_status": "NOT_RUN_OUT_OF_SCOPE",
            "old_r4_status": "NOT_RUN_PRESERVED",
        }
    if contract_status.get("status") != "CONTRACT_LOCKED_FOR_NEW_COLLECTION":
        new_status = "L2RAR2_CONTRACT_UNRESOLVED"
    elif development.get("all_pass") and confirmation.get("status") == "FOCUSED_CONFIRMATION_PASS":
        new_status = "L2RAR2_READY_FOR_FULL_CONFIRMATION_REVIEW"
    elif development_path.is_file():
        new_status = "L2RAR2_PARTIAL_KEEP_G1"
    else:
        new_status = "L2RAR2_DIAGNOSIS_ONLY"

    output_root.mkdir(parents=True, exist_ok=True)
    copies = {
        run_root / "rounds/l2rar2_0_entry/source_resolution.csv": output_root / "source_resolution.csv",
        run_root / "rounds/l2rar2_1_negative_attribution/negative_attribution.csv": output_root / "negative_attribution.csv",
        run_root / "rounds/l2rar2_2_task_contract/controller_context_contract.json": output_root / "controller_context_contract.json",
        run_root / "rounds/l2rar2_2_task_contract/reference_contract.json": output_root / "reference_contract.json",
        run_root / "rounds/l2rar2_2_task_contract/label_contract_diff.csv": output_root / "label_contract_diff.csv",
    }
    development_files = {
        "method_metrics.csv": "method_metrics.csv",
        "event_decisions.csv": "event_decisions.csv",
        "per_case_metrics.csv": "per_case_metrics.csv",
        "development_gates.json": "development_gates.json",
        "reference_unresolved.csv": "reference_unresolved.csv",
        "context_provenance.csv": "context_provenance.csv",
        "masked_context_diagnostics.csv": "masked_context_diagnostics.csv",
    }
    for name, destination in development_files.items():
        source = run_root / "rounds/l2rar2_3_development/select" / name
        if source.is_file():
            copies[source] = output_root / destination
    for source, destination in copies.items():
        if source.is_file():
            _copy(source, destination)
    write_json(output_root / "focused_confirmation_status.json", confirmation)
    paired_source = run_root / "rounds/l2rar2_4_focused_confirmation/paired_effects.csv"
    if paired_source.is_file():
        _copy(paired_source, output_root / "paired_effects.csv")
    else:
        write_csv(output_root / "paired_effects.csv", [{
            "status": confirmation["status"], "reason": "focused confirmation was not run",
            "metric": "NOT_ESTIMABLE", "families": 0,
        }])

    context_rows = []
    for partition in ("dev_fit", "dev_select"):
        manifest = run_root / f"data/{partition}/rollout_manifest.csv"
        if manifest.is_file():
            for row in read_csv(manifest):
                context_rows.append({
                    "rollout_id": row["rollout_id"], "partition": partition,
                    "requested_effect": row["requested_effect"],
                    "source": "controller_dispatch",
                    "issued_before_first_action": row["request_issued_before_first_action"],
                    "outcome_encoded": False,
                })
    confirm_manifest = run_root / "rounds/l2rar2_4_focused_confirmation/data/rollout_manifest.csv"
    if confirm_manifest.is_file():
        for row in read_csv(confirm_manifest):
            context_rows.append({
                "rollout_id": row["rollout_id"], "partition": "confirmation",
                "requested_effect": row["requested_effect"], "source": "controller_dispatch",
                "issued_before_first_action": row["request_issued_before_first_action"], "outcome_encoded": False,
            })
    write_csv(output_root / "context_provenance.csv", context_rows)

    fit_manifest = run_root / "data/dev_fit/rollout_manifest.csv"
    select_manifest = run_root / "data/dev_select/rollout_manifest.csv"
    fit_rows = read_csv(fit_manifest) if fit_manifest.is_file() else []
    select_rows = read_csv(select_manifest) if select_manifest.is_file() else []
    confirmation_rollouts = int(confirmation.get("physical_rollouts", 0))
    execution = {
        "schema": "pathgraph_l2rar2_execution_manifest_v1",
        "formal_main_commit": inputs["formal_main_commit"],
        "maintenance_source_commit": inputs["maintenance_source_commit"],
        "research_branch": "research/l2ra-r2-task-context-v1",
        "fit_root_families": len({row["root_family_id"] for row in fit_rows}),
        "fit_physical_rollouts": len(fit_rows),
        "select_root_families": len({row["root_family_id"] for row in select_rows}),
        "select_physical_rollouts": len(select_rows),
        "confirmation_root_families": int(confirmation.get("root_families", 0)),
        "confirmation_physical_rollouts": confirmation_rollouts,
        "total_new_physical_rollouts": len(fit_rows) + len(select_rows) + confirmation_rollouts,
        "derived_views_physical_credit": 0,
        "api_calls": 0,
        "training_jobs": 0,
        "api_key_read": False,
        "standard_l2r_confirmation_status": "NOT_RUN_OUT_OF_SCOPE",
        "old_r4_status": "NOT_RUN_PRESERVED",
    }
    write_json(output_root / "execution_manifest.json", execution)
    unsupported = {
        "schema": "pathgraph_l2rar2_unsupported_claims_v1",
        "claims": [
            "L3 entry or completion",
            "standard L2R confirmation pass",
            "real-robot grasp reliability",
            "reward or policy improvement",
            "task context as a visual-feature gain",
            "historical R1.7 cache as an online requested-effect validation",
        ],
        "retained_graph": "G1_predicate_bound",
        "l3_entry_allowed": False,
    }
    write_json(output_root / "unsupported_claims.json", unsupported)
    external_rows = []
    for role, path in (
        ("new_dev_fit_raw_rollouts", run_root / "data/dev_fit/rollouts"),
        ("new_dev_select_raw_rollouts", run_root / "data/dev_select/rollouts"),
        ("new_confirmation_raw_rollouts", run_root / "rounds/l2rar2_4_focused_confirmation/data/rollouts"),
        ("historical_r1_7_cache", Path(inputs["historical_cache_root"])),
    ):
        external_rows.append({
            "logical_id": role, "original_path": str(path.resolve(strict=False)),
            "exists": path.exists(), "size_bytes": _directory_size(path),
            "sha256": "NOT_COMPUTED_DIRECTORY_TREE", "purpose": role,
            "recovery": "restore the original directory at this exact path; verify file-level manifests where present",
        })
    write_csv(output_root / "manifests/external_artifacts.tsv", external_rows, fields=list(external_rows[0]),)
    # Rewrite with tabs while preserving the explicit fields.
    tsv = output_root / "manifests/external_artifacts.tsv"
    lines = ["\t".join(external_rows[0].keys())]
    lines.extend("\t".join(str(row[key]) for key in external_rows[0]) for row in external_rows)
    tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for name in ("source_lock.json", "generation_lock.json", "selection_lock.json"):
        source = run_root / "locks" / name
        if source.is_file():
            _copy(source, output_root / "locks" / name)
    for name in ("controller_context_contract.json", "reference_contract.json", "contract_status.json"):
        source = run_root / "rounds/l2rar2_2_task_contract" / name
        if source.is_file():
            _copy(source, output_root / "locks" / name)

    selected = "M1_requested_effect_gate" if confirmation.get("status") == "FOCUSED_CONFIRMATION_PASS" else None
    handoff = {
        "schema": "pathgraph_l2rar2_next_stage_handoff_v1",
        "formal_main_commit": inputs["formal_main_commit"],
        "maintenance_source_commit": inputs["maintenance_source_commit"],
        "new_experiment_commit": None,
        "new_experiment_commit_status": "assigned by Git after artifact freeze; see delivery package index",
        "historical_status": "L2RAR1_PARTIAL_KEEP_G1",
        "new_status": new_status,
        "retained_graph": "G1_predicate_bound",
        "new_candidate_id": selected,
        "requires_controller_requested_effect": True,
        "focused_confirmation_status": confirmation["status"],
        "standard_l2r_confirmation_status": "NOT_RUN_OUT_OF_SCOPE",
        "old_r4_status": "NOT_RUN_PRESERVED",
        "l3_entry_allowed": False,
        "main_updated": False,
        "api_calls": 0,
        "training_jobs": 0,
        "api_key_read": False,
        "development": development,
        "execution": execution,
    }
    write_json(output_root / "next_stage_handoff.json", handoff)
    report = [
        "# L2RA-R2 task-conditioned attempt result",
        "",
        f"- Scientific status: `{new_status}`.",
        "- Formal retained graph: `G1_predicate_bound`; L3 entry remains closed.",
        f"- Development: `{development.get('status')}` on {execution['select_root_families']} root families / {execution['select_physical_rollouts']} physical rollouts.",
        f"- Focused confirmation: `{confirmation['status']}` on {execution['confirmation_root_families']} root families / {execution['confirmation_physical_rollouts']} physical rollouts.",
        "- The new information is a controller-dispatched requested effect issued before action; it is not a visual feature and does not encode outcome.",
        "- Historical R1.7 requested effect was missing and was not reconstructed. Historical replay remains diagnosis only.",
        "- Standard L2R confirmation and old R4 remain not run; this round cannot authorize L3.",
        f"- API calls: 0; training jobs: 0; API key reads: false; total new physical rollouts: {execution['total_new_physical_rollouts']}.",
    ]
    if development.get("failed"):
        report.append(f"- Failed or non-estimable development gates: {', '.join(development['failed'])}.")
    (output_root / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return handoff
