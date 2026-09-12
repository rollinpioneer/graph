from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io_utils import read_json, sha256_file, write_json


def summarize(artifact_root: Path, output_root: Path, external_data_root: Path | None = None) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    gate = read_json(artifact_root / "confirmation_data_v1/generator_gate.json")
    metrics = read_json(artifact_root / "evaluation_v1/method_metrics.json") if (artifact_root / "evaluation_v1/method_metrics.json").is_file() else {"methods": []}
    candidates = [row for row in metrics.get("methods", []) if row.get("method") in {"O_B2", "O_C3"}]
    passed = [row for row in candidates if row.get("confirmation_status") == "CONFIRMATION_PASS"]
    scientific = "L2RAR2_PARTIAL_KEEP_G1" if passed else "R17_CONFIRMATION_NO_ONLINE_CANDIDATE_PASS"
    decision = {
        "schema": "l2rar2_r17_final_decision_v1",
        "engineering_status": "R17_TRUE_RGB_TEMPORAL_CONFIRMATION_COMPLETE",
        "scientific_status": scientific, "confirmation_run": True,
        "generator_gate": bool(gate.get("candidate_evaluation_allowed")),
        "passing_online_candidates": [row["method"] for row in passed],
        "selected_candidate_id": None, "l3_entry_allowed": False,
    }
    write_json(output_root / "decision.json", decision)
    lines = ["# R17 true-RGB temporal confirmation", "", f"Generator gate: {'PASS' if gate.get('candidate_evaluation_allowed') else 'FAIL'}", ""]
    for row in candidates:
        lines.extend([f"## {row['method']}", "",
                      f"- status: {row['confirmation_status']}",
                      f"- temporal accuracy: {row['overall_temporal_accuracy']:.6f}",
                      f"- strong-loss correct in window: {row['strong_loss_correct_in_window']}/24",
                      f"- false actions (C2/C3/C4/C12): {row['false_actions_C2_C3_C4_C12']}",
                      f"- early recoveries: {row['early_recovery_before_physical_onset']}",
                      f"- unknown rate: {row['unknown_rate']:.6f}", ""])
    lines.extend(["No candidate was selected automatically.", ""])
    (output_root / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output_root / "next_stage_handoff.json", {
        "schema": "l2rar2_r17_next_stage_handoff_v1", "status": scientific,
        "selected_candidate_id": None, "recommendation": "independent scientific review",
    })
    external = []
    if external_data_root is not None and external_data_root.exists():
        size = sum(path.stat().st_size for path in external_data_root.rglob("*") if path.is_file())
        external.append({"logical_path": "r17_true_rgb_confirmation_raw", "original_path": str(external_data_root.resolve()),
                         "size_bytes": size, "reason_omitted": "raw JPEG and physics traces externalized"})
    fields = ("logical_path", "original_path", "size_bytes", "reason_omitted")
    text = "\t".join(fields) + "\n" + "".join("\t".join(str(row.get(key, "")) for key in fields) + "\n" for row in external)
    (output_root / "external_artifacts.tsv").write_text(text, encoding="utf-8")
    files = []
    for path in sorted(artifact_root.rglob("*")):
        if path.is_file() and path != output_root / "package_manifest.json":
            files.append({"path": str(path.relative_to(artifact_root)), "size_bytes": path.stat().st_size,
                          "sha256": sha256_file(path)})
    manifest = {"schema": "l2rar2_r17_result_package_manifest_v1", "files": files,
                "file_count": len(files), "selected_candidate_id": None}
    write_json(output_root / "package_manifest.json", manifest)
    return decision
