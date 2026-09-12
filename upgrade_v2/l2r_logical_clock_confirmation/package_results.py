from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from .io_utils import read_json, sha256, write_json


def summarize(artifact_root: Path, output_root: Path, external_data_root: Path,
              package_path: Path | None = None) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    gate = read_json(artifact_root / "confirmation_data_v1/generator_gate.json")
    metrics_path = artifact_root / "evaluation_v1/method_metrics.json"
    metrics = read_json(metrics_path) if metrics_path.is_file() else {"methods": [], "selected_candidate_id": None}
    selected = metrics.get("selected_candidate_id")
    decision = {"schema": "l2rar2_r20_final_decision_v1",
                "engineering_status": "R20_LOGICAL_CLOCK_PHYSICAL_CONFIRMATION_COMPLETE",
                "scientific_status": "L2RAR2_ONLINE_CANDIDATE_CONFIRMED" if selected else "R20_NO_ONLINE_CANDIDATE_PASS",
                "generator_gate": bool(gate.get("candidate_evaluation_allowed")),
                "selected_candidate_id": selected, "confirmation_run": metrics_path.is_file(),
                "l3_entry_ready": bool(selected), "l3_entry_allowed": False}
    write_json(output_root / "decision.json", decision)
    lines = ["# R20 logical-clock physical confirmation", "",
             f"Generator gate: {'PASS' if gate.get('candidate_evaluation_allowed') else 'FAIL'}", ""]
    for row in metrics.get("methods", []):
        lines += [f"## {row['method']}", "", f"- correct: {row['correct']}/{row['resolved']}",
                  f"- early actions: {row['early_actions']}",
                  f"- false actions: {row['false_actions_T1_T2_T3_T4_T12']}",
                  f"- strong correct: {row['strong_T7_T11_correct_in_window']}/30",
                  f"- p90 latency: {row['latency_p90_s']}", ""]
    lines += [f"Selected candidate: {selected or 'none'}", "L3 was not started.", ""]
    (output_root / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output_root / "next_stage_handoff.json",
               {"schema": "l2rar2_r20_next_stage_handoff_v1", "status": decision["scientific_status"],
                "selected_candidate_id": selected, "l3_entry_ready": bool(selected), "l3_entry_allowed": False})
    size = sum(path.stat().st_size for path in external_data_root.rglob("*") if path.is_file())
    fields = ("logical_path", "original_path", "size_bytes", "reason_omitted")
    row = {"logical_path": "r20_confirmation_raw", "original_path": str(external_data_root.resolve()),
           "size_bytes": size, "reason_omitted": "raw JPEG and full physics traces externalized"}
    (output_root / "external_artifacts.tsv").write_text("\t".join(fields) + "\n" + "\t".join(str(row[key]) for key in fields) + "\n", encoding="utf-8")
    files = []
    for path in sorted(artifact_root.rglob("*")):
        if path.is_file() and path != output_root / "package_manifest.json":
            files.append({"path": str(path.relative_to(artifact_root)), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {"schema": "l2rar2_r20_result_package_manifest_v1", "files": files,
                "file_count": len(files), "selected_candidate_id": selected, "external_data_root": str(external_data_root.resolve())}
    write_json(output_root / "package_manifest.json", manifest)
    if package_path is not None:
        package_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in files + [{"path": "final_v1/package_manifest.json"}]:
                path = artifact_root / item["path"]; archive.write(path, arcname=item["path"])
        write_json(package_path.with_suffix(package_path.suffix + ".sha256.json"),
                   {"path": str(package_path), "sha256": sha256(package_path), "size_bytes": package_path.stat().st_size})
    return decision
