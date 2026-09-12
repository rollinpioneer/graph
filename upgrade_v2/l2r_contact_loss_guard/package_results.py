from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import read_json, sha256, write_json


def summarize(artifact_root: Path, output_root: Path, external_data_root: Path | None = None) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    development = read_json(artifact_root / "r17_candidate_development_replay_v1/development_gate.json")
    if not development.get("physical_confirmation_allowed"):
        decision = {"schema": "l2rar2_r18_final_decision_v1",
                    "engineering_status": "R18_ZERO_PHYSICS_DEVELOPMENT_COMPLETE",
                    "scientific_status": "R18_CLP1_DEVELOPMENT_FAIL",
                    "adjudication": "GENUINE_NON_RELEASE_CONTACT_PRECURSOR",
                    "r17_raw_B2_parity": bool(development.get("raw_B2_72_of_72")),
                    "r17_raw_C3_parity": bool(development.get("raw_C3_72_of_72")),
                    "r17_CLP1_correct_72_of_72": bool(development.get("CLP1_72_of_72_correct")),
                    "physical_confirmation_run": False, "generator_gate": None,
                    "selected_candidate_id": None, "confirmation_run": False,
                    "l3_entry_ready": False, "l3_entry_allowed": False}
        write_json(output_root / "decision.json", decision)
        diagnostic = read_json(artifact_root / "r17_candidate_development_replay_v1/failure_diagnostics.json")
        failure = diagnostic["failures"][0] if diagnostic.get("failures") else {}
        (output_root / "final_report.md").write_text(
            "# R18 contact-loss persistence development result\n\n"
            "Boundary adjudication: **GENUINE_NON_RELEASE_CONTACT_PRECURSOR**.\n\n"
            "R17 raw parity: O_B2 72/72; O_C3 72/72. O_C3_CLP1 achieved 71/72 and therefore failed the frozen development gate.\n\n"
            f"The sole miss was `{failure.get('rollout_id', 'unknown')}`. The intercepted contact-loss sample was followed by a same-time capture; the frozen `dt <= 0` rule cleared pending, so no recovery was emitted.\n\n"
            "Per protocol, no R18 physical confirmation was started, no guard parameter was searched, and no candidate was selected.\n",
            encoding="utf-8")
        write_json(output_root / "next_stage_handoff.json",
                   {"schema": "l2rar2_r18_next_stage_handoff_v1", "status": "R18_CLP1_DEVELOPMENT_FAIL",
                    "selected_candidate_id": None, "recommendation": "new development study; do not tune on R18 confirmation",
                    "l3_entry_ready": False, "l3_entry_allowed": False})
        gate, metrics, selected = {}, {"methods": []}, None
    else:
        gate = read_json(artifact_root / "confirmation_data_v1/generator_gate.json")
        metrics = read_json(artifact_root / "evaluation_v1/method_metrics.json")
        selected = metrics.get("selected_candidate_id")
        decision = {"schema": "l2rar2_r18_final_decision_v1",
                    "engineering_status": "R18_CONTACT_LOSS_PERSISTENCE_CONFIRMATION_COMPLETE",
                    "scientific_status": "L2RAR2_ONLINE_CANDIDATE_CONFIRMED" if selected else "R18_NO_ONLINE_CANDIDATE_PASS",
                    "generator_gate": bool(gate.get("candidate_evaluation_allowed")),
                    "selected_candidate_id": selected, "confirmation_run": True,
                    "l3_entry_ready": bool(selected), "l3_entry_allowed": False}
        write_json(output_root / "decision.json", decision)
    if development.get("physical_confirmation_allowed"):
        lines = ["# R18 contact-loss persistence confirmation", "",
             f"Generator gate: {'PASS' if gate.get('candidate_evaluation_allowed') else 'FAIL'}", ""]
        for row in metrics["methods"]:
            lines += [f"## {row['method']}", "", f"- temporal accuracy: {row['overall_temporal_accuracy']:.6f}",
                  f"- early actions: {row['early_actions']}",
                  f"- false actions T1/T2/T3/T12: {row['false_actions_T1_T2_T3_T12']}",
                  f"- strong loss correct: {row['strong_loss_correct_in_window']}/36",
                  f"- p90 physical latency: {row['latency_p90_s']}", ""]
        lines += [f"Selected candidate: {selected or 'none'}", "L3 was not started.", ""]
        (output_root / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
        write_json(output_root / "next_stage_handoff.json",
               {"schema": "l2rar2_r18_next_stage_handoff_v1", "status": decision["scientific_status"],
                "selected_candidate_id": selected, "l3_entry_ready": bool(selected), "l3_entry_allowed": False})
    external = []
    if external_data_root and external_data_root.exists():
        size = sum(path.stat().st_size for path in external_data_root.rglob("*") if path.is_file())
        external.append({"logical_path": "r18_confirmation_raw", "original_path": str(external_data_root.resolve()),
                         "size_bytes": size, "reason_omitted": "raw JPEG and physics traces externalized"})
    fields = ("logical_path", "original_path", "size_bytes", "reason_omitted")
    (output_root / "external_artifacts.tsv").write_text(
        "\t".join(fields) + "\n" + "".join("\t".join(str(row.get(key, "")) for key in fields) + "\n" for row in external),
        encoding="utf-8")
    files = []
    for path in sorted(artifact_root.rglob("*")):
        if path.is_file() and path != output_root / "package_manifest.json":
            files.append({"path": str(path.relative_to(artifact_root)), "size_bytes": path.stat().st_size,
                          "sha256": sha256(path)})
    write_json(output_root / "package_manifest.json",
               {"schema": "l2rar2_r18_result_package_manifest_v1", "files": files,
                "file_count": len(files), "selected_candidate_id": selected})
    return decision
