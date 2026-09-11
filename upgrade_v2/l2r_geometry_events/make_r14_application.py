"""Generate the R14 minimal-replay application materials (no execution).

The application is prepared only: every approval field stays empty and the
authorized budget stays 0.  An agent must never fill in the human
authorization block.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

APPLICATION_FILES = (
    "r14_application.json",
    "r14_protocol_lock.json",
    "r14_execution_plan.md",
    "r14_human_authorization.template.json",
    "source_hashes.json",
    "environment_requirements.json",
    "stop_gates.json",
    "budget_accounting_plan.json",
)

HUMAN_AUTHORIZATION_TEMPLATE = {
    "schema": "l2rar2_r14_human_authorization_v1",
    "status": "AWAITING_HUMAN_AUTHORIZATION",
    "authorization_id": None,
    "reviewer_id": None,
    "approved_at_utc": None,
    "authorized_instances": 0,
    "single_use_nonce": None,
    "expiry_utc": None,
    "bound_protocol_sha256": None,
    "bound_runner_file_hashes": None,
    "notes": "This block must be filled by the user, never by an agent.",
}


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="make_r14_application")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--r15b-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--entry-commit", default="62bd073597a13a74af5b5e30818337c3dbe13433")
    parser.add_argument("--main-commit", default="234cb6dc0e2767fa62cd2dbec4a868b8d0711bb2")
    args = parser.parse_args(argv)

    output = Path(args.output_root)
    if output.exists():
        raise SystemExit("REFUSING_EXISTING_OUTPUT_ROOT " + str(output))
    output.mkdir(parents=True)
    data_root = Path(args.data_root)
    r15b = Path(args.r15b_root)

    _write_json(
        output / "r14_application.json",
        {
            "schema": "l2rar2_r14_application_v1",
            "status": "AWAITING_HUMAN_AUTHORIZATION",
            "requested_instances": 2,
            "authorized_instances": 0,
            "purpose": "ordinary replay vs frozen cache, then instrumented replay vs ordinary",
            "not_validating": ["K4/K5/K6 physical loss", "recovery method ranking", "height-generalisation", "confirmation", "L3"],
            "root_family_id": "L2RAR2_REPAIR_00_840000",
            "case_id": "K3_normal_hold_pause_resume",
            "family_seed": 840000,
            "rollout_seed": 84100002,
            "program_sha256": "8d82a7e946317c393d2024be6159e0c12337a63e00a7877a1855beb369b6ca30",
            "entry_commit": args.entry_commit,
            "formal_main_commit": args.main_commit,
            "agent_may_self_approve": False,
            "authorization_template": "r14_human_authorization.template.json",
        },
    )
    _write_json(
        output / "r14_protocol_lock.json",
        {
            "schema": "l2rar2_r14_protocol_lock_v1",
            "entry_commit": args.entry_commit,
            "case_id": "K3_normal_hold_pause_resume",
            "rollout_seed": 84100002,
            "control_variant": "v0_short",
            "tolerances": {"rtol": 0, "time_atol": 1e-12, "geometry_atol": 1e-12, "mocap_atol": 2e-09},
            "stage1_failure_stops_all_physics": True,
            "stage2_requires_stage1_pass": True,
            "replay_naming": "CONTROLLED_RECONSTRUCTION_ATTEMPT",
        },
    )
    (output / "r14_execution_plan.md").write_text(
        "\n".join(
            [
                "# R14 minimal replay execution plan (not authorised)",
                "",
                "## Execution 1 - ordinary replay",
                "",
                "Reuse the locked case program, variant, simulator version and callback",
                "semantics.  Persist state at: initial-after-forward, before-action,",
                "control callback, action-end callback, after-perform-return.  Every",
                "sample keeps its own capture_order and sequence.",
                "",
                "Fields: qpos, qvel, qacc_warmstart, mocap_pos, mocap_quat, eq_active,",
                "model.eq_data, RNG state/hash, contact pair summary, renderer callback",
                "count, action/control/event rows, object/gripper geometry.",
                "",
                "Compare: cache actions/controls/events vs ordinary, and cache",
                "action-end vs ordinary action-end callback state, with rtol=0 and the",
                "frozen atolerances.  Any main-gate failure stops the run",
                "(`STOP_AFTER_EXECUTION_1`); execution 2 is then not run.",
                "",
                "## Execution 2 - instrumented replay",
                "",
                "Only after execution 1 passes and one instance remains.  Same renderer,",
                "actions and callbacks; adds pre-reviewed read-only recording only.",
                "",
                "## Explicit non-goals",
                "",
                "R14 verifies replay consistency only.  It does not verify K4/K5/K6",
                "physical loss, does not rank recovery candidates and does not open L3.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _write_json(output / "r14_human_authorization.template.json", HUMAN_AUTHORIZATION_TEMPLATE)
    _write_json(
        output / "source_hashes.json",
        {
            "schema": "l2rar2_r14_source_hashes_v1",
            "rollout_manifest_sha256": "0708e6408aa95a83c42b8fcd9269577442b62e093ff008e3aab1597a3eb3e50b",
            "round9_generation_lock_sha256": "74b6accfc12b3865b53531ef02b1a0e4ad28cd8d54e5f7ea86c3be47797a059f",
            "round9_validation_manifest_sha256": "0684fe22d4834c96a1ed7f8cfd56a2800ac2f561512dc140cde60171b51a5868",
            "r13_result_zip_sha256": "97364e5bb628a03b4579bbb56524e8f34e6f89828231e5dc7d8186fc0d05adf9",
            "r15a_result_zip_sha256": "8d3c9bc9f32e11af6d49ebf64d5e3de8fc00a6bed1a319b6d7fa28c55689b5e6",
            "r15b_metrics_sha256": _sha256(r15b / "metrics_v2.json"),
            "r15b_decision_sha256": _sha256(r15b / "decision.json"),
            "data_manifest_path": str(data_root / "rollout_manifest.csv"),
        },
    )
    _write_json(
        output / "environment_requirements.json",
        {
            "schema": "l2rar2_r14_environment_requirements_v1",
            "recorded_adjacent_environment": {
                "python": "3.10.19",
                "mujoco": "3.4.0",
                "opencv": "4.13.0",
                "python_path": "/home/xushijie/.conda/envs/lerobot/bin/python",
            },
            "equivalence_with_historical_collection_proven": False,
            "silent_install_or_upgrade_allowed": False,
            "reconstruction_label": "CONTROLLED_RECONSTRUCTION_ATTEMPT",
        },
    )
    _write_json(
        output / "stop_gates.json",
        {
            "schema": "l2rar2_r14_stop_gates_v1",
            "gates": [
                "STOP_AFTER_EXECUTION_1 on any ordinary-vs-cache main-gate mismatch",
                "STOP on instrumentation mismatch (R14_INSTRUMENTATION_MISMATCH)",
                "STOP if any instance budget is exceeded",
                "STOP if the environment cannot be reconstructed as recorded",
                "STOP if the human authorization file is absent or expired",
            ],
            "no_automatic_retry": True,
            "tolerance_adjustment_allowed": False,
        },
    )
    _write_json(
        output / "budget_accounting_plan.json",
        {
            "schema": "l2rar2_r14_budget_accounting_v1",
            "requested_instances": 2,
            "authorized_instances": 0,
            "historical": {
                "r11_executions": 40,
                "r11_budget": 8,
                "r11_status": "BUDGET_EXCEEDED_BLOCKED",
                "r12_denial_journal": "preserved",
                "r13_authorization": 0,
            },
            "journal": "append_only",
            "reset_or_zeroing_of_history_allowed": False,
            "one_instance_counts_from_first_real_model_construction": True,
        },
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
