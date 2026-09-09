from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ONLINE_NAMES = ("frame_manifest.csv", "contact_sensor.csv", "gripper_command.csv")
REFERENCE_NAMES = ("actions.csv", "low_level_controls.jsonl", "events.jsonl", "oracle_timeline.csv", "oracle_diagnostic.npz", "termination.json", "metadata.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _resource(path: Path, logical_id: str, usage: str, split: str) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "logical_id": logical_id,
        "source_path": str(path),
        "resolved_path": str(path.resolve()),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": sha256_file(path) if exists else None,
        "usage": usage,
        "split": split,
    }


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def _rollout_resources(root: Path, rollout_id: str, split: str, prediction_path: Path | None) -> list[dict[str, Any]]:
    resources = []
    for name in ONLINE_NAMES:
        resources.append(_resource(root / name, f"{rollout_id}:{name}", "runtime_allowed", split))
    for name in REFERENCE_NAMES:
        resources.append(_resource(root / name, f"{rollout_id}:{name}", "reference_only", split))
    frame_manifest = root / "frame_manifest.csv"
    if frame_manifest.is_file():
        for row in read_csv(frame_manifest):
            value = row.get("front_path")
            if value:
                resources.append(_resource(Path(value), f"{rollout_id}:front:{row['frame_index']}", "runtime_allowed", split))
    if prediction_path:
        resources.append(_resource(prediction_path, f"{rollout_id}:predictions", "runtime_allowed", split))
    return resources


def _old_records(source_repo: Path) -> list[dict[str, Any]]:
    root = source_repo / "artifacts/pathgraph_sarm/upgrade_v2/edge_ambiguity_l2ra_v1/data/new_development"
    manifest = root / "probe_rollout_manifest.csv"
    pred_manifest = root / "prediction_manifest.csv"
    if not manifest.is_file() or not pred_manifest.is_file():
        raise FileNotFoundError(f"old L2RA development cache is incomplete: {root}")
    pred_by_id = {row["rollout_id"]: row for row in read_csv(pred_manifest)}
    records = []
    for row in read_csv(manifest):
        rollout_root = root / "rollouts" / row["root_family_id"] / f"rollout_{int(row['rollout_id'].rsplit('_r', 1)[1]):02d}"
        prediction = Path(pred_by_id[row["rollout_id"]]["prediction_path"])
        records.append({
            "rollout_id": row["rollout_id"],
            "root_family_id": row["root_family_id"],
            "stratum": row["stratum"],
            "split": row["split"],
            "rollout_path": str(rollout_root.resolve()),
            "prediction_path": str(prediction.resolve()),
            "source_kind": row.get("source_kind", "historical_l2ra_dynamic_probe"),
            "observation_intervention": row.get("observation_intervention", "False").lower() == "true",
            "data_role": "runtime_allowed" if row["split"] in {"dev_fit", "dev_select"} else "historical_diagnosis_only",
            "scenario_reference_only": row.get("stratum", "unknown"),
            "resources": _rollout_resources(rollout_root, row["rollout_id"], row["split"], prediction),
        })
    return records


def prepare(source_repo: Path, repo_root: Path, old_l2ra: Path, old_l2r: Path, protocol: Path, output_root: Path) -> dict[str, Any]:
    payload = read_json(protocol)
    if payload.get("api_calls_max") != 0 or payload.get("training_jobs_max") != 0 or payload.get("read_api_keys") is not False:
        raise ValueError("R1 protocol must forbid API calls, training and key reads")
    records = _old_records(source_repo)
    output_root.mkdir(parents=True, exist_ok=True)
    protocol_copy = output_root / "configs/protocol.json"
    protocol_copy.parent.mkdir(parents=True, exist_ok=True)
    if protocol.resolve() != protocol_copy.resolve():
        shutil.copy2(protocol, protocol_copy)
    frozen_names = [
        "graphs/G0_coarse_direct.json", "graphs/G1_predicate_bound.json", "graphs/G2_evidence_refined.json",
        "locks/predicate_thresholds.json", "locks/selection_lock.json", "tables/confirmation_metrics.csv",
        "manifests/fresh_rollout_manifest.csv", "final_v1/next_stage_handoff.json",
    ]
    frozen = []
    for name in frozen_names:
        path = old_l2r / "final_v1" / name if not name.startswith("final_v1/") else old_l2r / name
        frozen.append(_resource(path, f"frozen:{name}", "historical_diagnosis_only", "frozen"))
    old_files = []
    old_names = (
        "final_v1/l2ra_final_report.md", "final_v1/candidate_registry.json", "final_v1/cause_attribution.csv", "final_v1/next_stage_handoff.json",
        "rounds/l2ra_1_replay_and_localization/legacy_dev_fit/legacy_reproduction.json",
        "rounds/l2ra_1_replay_and_localization/legacy_dev_select/legacy_reproduction.json",
        "rounds/l2ra_3_development_and_selection/decision_delay.csv",
        "rounds/l2ra_3_development_and_selection/false_emergency_and_unknown.csv",
        "rounds/l2ra_3_development_and_selection/legacy_dev_select_compatibility.csv",
        "data/new_development/probe_records.jsonl", "data/new_development/probe_manifest.csv",
        "data/new_development/probe_rollout_manifest.csv", "data/new_development/prediction_manifest.csv",
        "data/new_development/content_duplicate_groups.csv",
    )
    for name in old_names:
        old_files.append(_resource(old_l2ra / name, f"old_l2ra:{name}", "historical_diagnosis_only", "old"))
    resolved = {
        "schema": "pathgraph_l2rar1_resolved_inputs_v1",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "repository": {"path": str(repo_root.resolve()), "implementation_commit": _git_commit(repo_root), "source_repository": str(source_repo.resolve()), "source_repository_commit": _git_commit(source_repo), "manual_base_commit": "10b3738b527b53a13e15be8b58ad5d1c174e49a4"},
        "roles": {"runtime_allowed": ["front_rgb", "contact_sensor", "gripper_command", "online_predicates"], "reference_only": ["events", "oracle_timeline", "weld_state", "future_outcome", "scenario", "stratum"], "historical_diagnosis_only": ["legacy reports", "old confirmation"]},
        "datasets": {"legacy_l2ra_development": {"families": len({r["root_family_id"] for r in records}), "rollouts": len(records), "split_counts": {split: sum(r["split"] == split for r in records) for split in sorted({r["split"] for r in records})}}},
        "rollouts": records,
        "frozen_sources": frozen,
        "old_l2ra_sources": old_files,
        "protocol_path": str(protocol_copy.resolve()),
        "api_calls": 0,
        "training_jobs": 0,
        "api_key_read": False,
        "new_paired_recording_available": False,
        "new_recording_note": "Only legacy action-end data was present at entry; R1 dense paired recording remains an explicit execution limitation until collect-development creates control-tick observations.",
    }
    write_json(output_root / "manifests/resolved_inputs.json", resolved)
    lock_files = frozen + old_files + [{"logical_id": "protocol", "path": str(protocol.resolve()), "sha256": sha256_file(protocol)}]
    write_json(output_root / "locks/source_lock.json", {"schema": "pathgraph_l2rar1_source_lock_v1", "status": "LOCKED", "source_commit": resolved["repository"]["manual_base_commit"], "source_repository_commit": resolved["repository"]["source_repository_commit"], "implementation_commit": resolved["repository"]["implementation_commit"], "manual_base_commit": resolved["repository"]["manual_base_commit"], "files": lock_files, "api_calls": 0, "training_jobs": 0, "api_key_read": False})
    write_json(output_root / "locks/old_confirmation_status.json", {"schema": "pathgraph_l2rar1_old_confirmation_status_v1", "historical_status": "L2RA_PARTIAL_KEEP_G1", "old_confirmation_status": "NOT_RUN", "source": str((old_l2ra / "final_v1/next_stage_handoff.json").resolve()), "consumed_by_r1": False})
    source_note = f"""# Source resolution\n\n- Manual base commit: `{resolved['repository']['manual_base_commit']}`\n- R1 implementation commit: `{resolved['repository']['implementation_commit']}`\n- Source repository commit at execution: `{resolved['repository']['source_repository_commit']}`\n- Source repository: `{source_repo.resolve()}`\n- Old L2RA development recordings are read-only external inputs.\n- The source worktree's unrelated uncommitted L1V changes were not touched.\n- The entry inventory contains action-end observations but no pre-existing 20 Hz paired stream; this is recorded as an execution limitation, not inferred as a scientific negative.\n"""
    note_path = output_root / "source_resolution.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(source_note, encoding="utf-8")
    return {"status": "REPLAY_READY", "resolved_inputs": str((output_root / "manifests/resolved_inputs.json").resolve()), "families": len({r["root_family_id"] for r in records}), "rollouts": len(records), "new_paired_recording_available": False}
