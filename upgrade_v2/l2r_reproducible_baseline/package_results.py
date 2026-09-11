from __future__ import annotations

import hashlib
import json
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .protocol import canonical_hash


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8")); digest.update(b"\0"); digest.update(path.read_bytes())
    return digest.hexdigest()


def package_manifest(root: Path) -> dict[str, Any]:
    files = [{"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(p for p in root.rglob("*") if p.is_file())]
    return {"schema": "l2rar2_r14b_package_manifest_v1", "root": str(root.resolve()), "files": files, "tree_sha256": tree_sha256(root)}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def finalize_chain(*, repo: Path, baseline_a: Path, repeat_b: Path, instrumented_c: Path, comparison_ab: Path, comparison_bc: Path, output_root: Path) -> dict[str, Any]:
    """Create the final engineering certificate only after all three gates pass."""
    output_root.mkdir(parents=False, exist_ok=False)
    a, b, c = _load(baseline_a / "result.json"), _load(repeat_b / "result.json"), _load(instrumented_c / "result.json")
    ab, bc = _load(comparison_ab / "comparison_summary.json"), _load(comparison_bc / "comparison_summary.json")
    failures: list[str] = []
    for label, result in (("A", a), ("B", b), ("C", c)):
        if not result.get("all_main_gates_passed"):
            failures.append(f"{label}_main_gates")
    if not ab.get("all_main_gates_passed"):
        failures.append("A_vs_B_main_gates")
    if not bc.get("all_main_gates_passed"):
        failures.append("B_vs_C_main_gates")
    integrity = bc.get("instrumentation", {})
    if not integrity.get("passed"):
        failures.append("instrumentation_integrity")
    for field in (
        "runner_commit", "protocol_sha256", "environment_fingerprint_sha256", "model_fingerprint_sha256",
        "source_lock_sha256", "environment_contract_sha256", "generated_model_xml_sha256", "family_seed",
    ):
        if not (a.get(field) == b.get(field) == c.get(field)):
            failures.append(f"consistency.{field}")
    status = "R14B_REPRODUCIBLE_BASELINE_CHAIN_PASS" if not failures else "R14B_CHAIN_BLOCKED"
    decision = {
        "schema": "l2rar2_r14b_final_decision_v1", "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "failures": failures,
        "scientific_status": "L2RAR2_PARTIAL_KEEP_G1", "retained_graph": "G1_predicate_bound",
        "selected_candidate_id": None, "confirmation_run": False, "l3_entry_allowed": False,
        "r16_automatically_authorized": False,
    }
    _write_json(output_root / "decision.json", decision)
    handoff = {
        "schema": "l2rar2_r14b_next_stage_handoff_v1", "status": "READY_FOR_INDEPENDENT_R16_APPLICATION" if not failures else "BLOCKED",
        "r14b_status": status, "failures": failures, "r16_authorization_created": False,
        "scientific_status": decision["scientific_status"], "selected_candidate_id": None,
        "confirmation_run": False, "l3_entry_allowed": False,
    }
    _write_json(output_root / "next_stage_handoff.json", handoff)
    accounting = {
        "schema": "l2rar2_r14b_physical_execution_accounting_v1", "r14b_A": 1 if not failures else 0,
        "r14b_B": 1 if not failures else 0, "r14b_C": 1 if not failures else 0,
        "r14b_total": 3 if not failures else 0, "r11_historical": {"physical_executions": 40, "authorized_instances": 8},
        "r14_old_ordinary_002": {"status": "FAILED_PRESERVED", "included_in_r14b_gates": False},
    }
    _write_json(output_root / "physical_execution_accounting.json", accounting)
    external = [
        ("baseline_A", baseline_a / "result.json"), ("repeat_B", repeat_b / "result.json"),
        ("instrumented_C", instrumented_c / "result.json"), ("ordinary_A_vs_B", comparison_ab / "comparison_summary.json"),
        ("ordinary_B_vs_instrumented_C", comparison_bc / "comparison_summary.json"),
    ]
    with (output_root / "external_artifacts.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("role", "absolute_path", "size_bytes", "sha256", "included_in_zip", "recovery_instructions"))
        for role, path in external:
            writer.writerow((role, str(path.resolve()), path.stat().st_size, _sha(path), "false", "Use the preserved external execution/comparison directory."))
    report = [
        "# R14-B Reproducible Baseline Chain", "", f"Status: {status}", "",
        f"Runner commit: {a.get('runner_commit')}", f"Protocol SHA256: {a.get('protocol_sha256')}",
        "", "This is an execution-infrastructure certificate only.",
        "It does not confirm loss, forced-drop behavior, candidate validity, or L3 entry.",
    ]
    if failures:
        report.extend(("", "Failures:", *[f"- {failure}" for failure in failures]))
    (output_root / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output_root / "actual_commands.txt").write_text("finalize-chain\n", encoding="utf-8")
    _write_json(output_root / "run_manifest.json", {"schema": "l2rar2_r14b_final_run_manifest_v1", "status": status, "baseline_A": str(baseline_a.resolve()), "repeat_B": str(repeat_b.resolve()), "instrumented_C": str(instrumented_c.resolve()), "comparison_AB": str(comparison_ab.resolve()), "comparison_BC": str(comparison_bc.resolve())})
    if not failures:
        certificate = {
            "schema": "l2rar2_r14b_reproducibility_certificate_v1", "status": status,
            "protocol_sha256": a["protocol_sha256"], "runner_commit": a["runner_commit"], "runner_file_hashes": a["runner_file_hashes"],
            "source_lock_sha256": a["source_lock_sha256"], "environment_contract_sha256": a["environment_contract_sha256"],
            "generated_model_xml_sha256": a["generated_model_xml_sha256"], "family_seed": a["family_seed"],
            "environment_fingerprint_sha256": a["environment_fingerprint_sha256"], "model_fingerprint_sha256": a["model_fingerprint_sha256"],
            "baseline_A_result_sha256": _sha(baseline_a / "result.json"), "baseline_A_manifest_sha256": a["artifact_manifest_sha256"],
            "repeat_B_result_sha256": _sha(repeat_b / "result.json"), "repeat_B_manifest_sha256": b["artifact_manifest_sha256"],
            "ordinary_A_vs_B_comparison_sha256": _sha(comparison_ab / "comparison_summary.json"),
            "instrumented_C_result_sha256": _sha(instrumented_c / "result.json"), "instrumented_C_manifest_sha256": c["artifact_manifest_sha256"],
            "ordinary_B_vs_C_comparison_sha256": _sha(comparison_bc / "comparison_summary.json"),
            "physical_instances": {"A": 1, "B": 1, "C": 1, "total": 3},
            "loss_confirmed": False, "r16_automatically_authorized": False,
            "scientific_status": "L2RAR2_PARTIAL_KEEP_G1", "retained_graph": "G1_predicate_bound",
            "selected_candidate_id": None, "confirmation_run": False, "l3_entry_allowed": False,
        }
        _write_json(output_root / "r14b_reproducibility_certificate.json", certificate)
    manifest = package_manifest(output_root)
    _write_json(output_root / "package_manifest.json", manifest)
    return decision
