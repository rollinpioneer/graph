from __future__ import annotations

from pathlib import Path

from . import CANDIDATE_ID
from .io_utils import read_json, sha256, write_json


def summarize(artifact_root: Path, output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    clocks = read_json(artifact_root / "r17_clock_forensics_v1/clock_forensics_summary.json")
    gate = read_json(artifact_root / "r17_candidate_development_replay_v1/development_gate.json")
    metrics = read_json(artifact_root / "r17_candidate_development_replay_v1/candidate_metrics.json")
    passed = bool(gate.get("r19_confirmation_design_allowed"))
    decision = {"schema": "l2rar2_r19_logical_clock_development_decision_v1",
                "engineering_status": "R19_LOGICAL_OBSERVATION_CLOCK_DEVELOPMENT_COMPLETE",
                "scientific_status": "R19_LOGICAL_CLOCK_CANDIDATE_DEVELOPMENT_PASS" if passed else "R19_LOGICAL_CLOCK_CANDIDATE_DEVELOPMENT_FAIL",
                "candidate_id": CANDIDATE_ID, "r17_development_gate": passed,
                "r17_temporal_correct": metrics["temporal_correct"],
                "same_physical_time_groups": clocks["same_physical_time_groups"],
                "same_time_confirmation_exercised": gate["same_time_confirmation_exercised"],
                "r19_confirmation_design_created": passed, "r19_physical_confirmation_run": False,
                "selected_candidate_id": None, "l3_entry_allowed": False}
    write_json(output_root / "decision.json", decision)
    (output_root / "final_report.md").write_text(
        "# R19 logical-observation-clock development\n\n"
        f"Candidate: `{CANDIDATE_ID}`\n\n"
        f"R17 duplicate physical-time groups: {clocks['same_physical_time_groups']} across {clocks['affected_rollouts']} rollouts.\n\n"
        f"Raw parity: O_B2 {'72/72' if gate['raw_B2_72_of_72'] else 'FAIL'}; O_C3 {'72/72' if gate['raw_C3_72_of_72'] else 'FAIL'}.\n\n"
        f"Candidate temporal result: {metrics['temporal_correct']}/72; early={metrics['early_actions']}; false={metrics['false_actions']}; missed={metrics['missed_actions']}; unknown={metrics['unknown']}.\n\n"
        f"Prefix causality: {'PASS' if gate['prefix_causality'] else 'FAIL'}. Same-time confirmations exercised: {metrics['same_physical_time_confirmations']}.\n\n"
        "The R19 confirmation registry was created only after the zero-physics development gate passed. No R19 physics was run and no confirmation data was used for tuning.\n",
        encoding="utf-8")
    files = []
    for path in sorted(artifact_root.rglob("*")):
        if path.is_file() and path != output_root / "package_manifest.json":
            files.append({"path": str(path.relative_to(artifact_root)), "size_bytes": path.stat().st_size,
                          "sha256": sha256(path)})
    write_json(output_root / "package_manifest.json",
               {"schema": "l2rar2_r19_logical_clock_package_manifest_v1", "files": files,
                "file_count": len(files), "candidate_id": CANDIDATE_ID,
                "r19_physical_confirmation_run": False})
    return decision
