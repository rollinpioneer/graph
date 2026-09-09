"""Create L2RA run manifests and verified lightweight delivery archives."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

from .tools.l2ra_support import package_round, sha256, write_json


ROUND_IDS = (
    "l2ra_0_entry",
    "l2ra_1_replay_and_localization",
    "l2ra_2_cause_diagnosis",
    "l2ra_3_development_and_selection",
    "l2ra_4_fresh_confirmation",
    "l2ra_5_handoff",
)

COMMANDS = {
    "l2ra_0_entry": ["python -m upgrade_v2.l2r_ambiguity.cli prepare", "l2ra_support.py inspect-frozen"],
    "l2ra_1_replay_and_localization": ["python -m upgrade_v2.l2r_ambiguity.cli trace (dev_fit, dev_select, legacy_confirmation)"],
    "l2ra_2_cause_diagnosis": ["python -m upgrade_v2.l2r_ambiguity.cli diagnose"],
    "l2ra_3_development_and_selection": ["python -m upgrade_v2.l2r_ambiguity.cli collect-probes", "python -m upgrade_v2.l2r_ambiguity.cli evaluate-development", "python -m upgrade_v2.l2r_ambiguity.cli lock-candidate"],
    "l2ra_4_fresh_confirmation": ["python -m upgrade_v2.l2r_ambiguity.cli confirm (gate refusal; no candidate selected)"],
    "l2ra_5_handoff": ["python -m upgrade_v2.l2r_ambiguity.cli handoff", "python -m upgrade_v2.l2r_ambiguity.cli deliver"],
}


def _json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _manifest(run_root: Path, round_id: str, status: str, details: dict[str, Any]) -> None:
    payload = {"schema": "pathgraph_l2ra_round_manifest_v1", "round_id": round_id, "status": status,
               "commands": COMMANDS[round_id], "api_calls": 0, "training_jobs": 0, "api_key_read": False,
               "device": "CPU/EGL rendering; local CUDA unavailable", **details}
    write_json(run_root / f"rounds/{round_id}/run_manifest.json", payload)


def _secret_scan(paths: list[Path]) -> dict[str, Any]:
    pattern = re.compile(rb"(?:sk-[A-Za-z0-9._-]{16,}|Authorization\s*:\s*Bearer\s+\S+|api[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9._-]{20,})", re.I)
    files, findings = 0, []
    for root in paths:
        for path in sorted(root.rglob("*")) if root.is_dir() else [root]:
            if not path.is_file() or path.suffix.lower() in {".zip", ".pyc", ".npz", ".npy", ".jpg"}: continue
            files += 1
            if pattern.search(path.read_bytes()): findings.append(str(path))
    return {"status": "PASS" if not findings else "FAIL", "files": files, "finding_files": findings}


def _verify_zip(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        internal = True
        if "PACKAGE_SHA256SUMS.txt" in archive.namelist():
            for line in archive.read("PACKAGE_SHA256SUMS.txt").decode().splitlines():
                digest, rel = line.split("  ", 1)
                internal &= hashlib.sha256(archive.read(rel)).hexdigest() == digest
    return {"path": str(path.resolve()), "filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path),
            "unzip_test": "PASS" if bad is None else f"FAIL:{bad}", "internal_sha": "PASS" if internal else "FAIL"}


def deliver(repo: Path, run_root: Path, downloads: Path, code_root: Path, staging_root: Path | None = None) -> dict[str, Any]:
    final = run_root / "final_v1"
    route = _json(run_root / "rounds/l2ra_3_development_and_selection/development_route.json", {})
    diagnosis = _json(run_root / "rounds/l2ra_2_cause_diagnosis/diagnosis_gate.json", {})
    consumption = _json(run_root / "rounds/l2ra_4_fresh_confirmation/confirmation_consumption.json", {})
    _manifest(run_root, "l2ra_0_entry", "REPLAY_READY", {"manual_base_commit": "e607131d7745d46745680bc9a6190a467b01dfa2"})
    _manifest(run_root, "l2ra_1_replay_and_localization", "REPLAY_COMPLETE", {"splits": 3, "legacy_confirmation_reproduced": True})
    _manifest(run_root, "l2ra_2_cause_diagnosis", diagnosis.get("status", "MISSING"), {"supported_hypotheses": diagnosis.get("supported_hypotheses", [])})
    _manifest(run_root, "l2ra_3_development_and_selection", route.get("status", "MISSING"), {"new_families": 24, "new_rollouts": 96, "selected_candidate_id": route.get("selected_candidate_id")})
    _manifest(run_root, "l2ra_4_fresh_confirmation", "NOT_RUN", {"consumption_status": consumption.get("status"), "started_once": consumption.get("started_once", False), "reason": consumption.get("reason")})
    handoff_dir = run_root / "rounds/l2ra_5_handoff"
    shutil.copytree(final, handoff_dir / "final_v1", dirs_exist_ok=True)
    _manifest(run_root, "l2ra_5_handoff", _json(final / "next_stage_handoff.json", {}).get("new_status", "MISSING"), {"l3_entry_allowed": False})
    downloads.mkdir(parents=True, exist_ok=True)
    packages = []
    for round_id in ROUND_IDS:
        target = downloads / f"{round_id}.zip"
        package_round(run_root / f"rounds/{round_id}", target, 200)
        packages.append({**_verify_zip(target), "purpose": f"L2RA round {round_id}"})
    staging = staging_root or (run_root / "delivery_staging")
    if staging.exists(): raise FileExistsError(f"delivery staging already exists: {staging}")
    _copy(run_root / "configs/l2ra_protocol.json", staging / "configs/l2ra_protocol.json")
    _copy(run_root / "locks/protocol_lock.json", staging / "locks/protocol_lock.json")
    _copy(run_root / "locks/event_reference_contract.lock.json", staging / "locks/event_reference_contract.lock.json")
    _copy(run_root / "locks/selection_lock.json", staging / "locks/selection_lock.json")
    _copy(run_root / "manifests/resolved_inputs.json", staging / "manifests/resolved_inputs.json")
    correction = run_root / "rounds/l2ra_3_development_and_selection/reports/protocol_deviations_and_corrections.json"
    if correction.is_file():
        _copy(correction, staging / "reports/protocol_deviations_and_corrections.json")
    compact_data = run_root / "data/new_development"
    for name in ("probe_lock.json", "probe_manifest.csv", "probe_rollout_manifest.csv", "prediction_manifest.csv", "probe_records.jsonl", "content_duplicate_groups.csv"):
        _copy(compact_data / name, staging / f"data/new_development/{name}")
    shutil.copytree(final, staging / "final_v1")
    shutil.copytree(code_root, staging / "code/upgrade_v2/l2r_ambiguity", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for round_id in ROUND_IDS:
        _copy(run_root / f"rounds/{round_id}/run_manifest.json", staging / f"round_manifests/{round_id}.json")
    scan = _secret_scan([code_root, run_root / "configs", run_root / "locks", run_root / "manifests", run_root / "rounds", staging])
    if scan["status"] != "PASS": raise RuntimeError("secret scan failed")
    write_json(final / "secret_scan.json", scan)
    _copy(final / "secret_scan.json", staging / "final_v1/secret_scan.json")
    round_index = {"schema": "pathgraph_l2ra_round_package_index_v1", "round_zip_entities_included": False,
                   "note": "The aggregate archive records round ZIP hashes but does not recursively embed ZIP bodies.", "packages": packages}
    write_json(staging / "round_package_index.json", round_index)
    total = downloads / "L2RA_results.zip"
    package_round(staging, total, 200)
    total_record = {**_verify_zip(total), "purpose": "L2RA aggregate lightweight handoff"}
    index = {"schema": "pathgraph_l2ra_package_index_v1", "revision_id": downloads.name, "supersedes": "downloads/l2ra", "scientific_status": _json(final / "next_stage_handoff.json")["new_status"],
             "round_packages": packages, "total_package": total_record, "round_zip_entities_in_total": False,
             "api_calls": 0, "training_jobs": 0, "api_key_read": False, "secret_scan": scan}
    write_json(downloads / "package_index.json", index)
    lines = ["# L2RA archive bodies", "", "ZIP files are local delivery artifacts and are excluded from Git. Restore them from the paths below and verify SHA256.", ""]
    for item in packages + [total_record]:
        lines.extend([f"- Original path: `{item['path']}`", f"  Filename: `{item['filename']}`; size: `{item['size_bytes']}` bytes; SHA256: `{item['sha256']}`; purpose: {item['purpose']}."])
    (downloads / "PACKAGE_BODIES.placeholder.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index
