"""Resolve and lock L2RA inputs without mixing online and reference data."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ONLINE_FILES = ("frame_manifest.csv", "contact_sensor.csv", "gripper_command.csv", "low_level_controls.jsonl", "actions.csv")
REFERENCE_FILES = ("events.jsonl", "oracle_timeline.csv", "oracle_diagnostic.npz", "termination.json", "metadata.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def _prediction_manifest(prediction_dir: Path, rollout_ids: set[str], split: str, data_rows: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(prediction_dir.glob("*.jsonl")):
        rollout_id = path.stem
        if rollout_id not in rollout_ids:
            continue
        data = data_rows[rollout_id]
        rows.append({"rollout_id": rollout_id, "root_family_id": data["root_family_id"], "split": split,
                     "prediction_path": str(path.resolve()), "prediction_sha256": sha256_file(path),
                     "frames": int(data["frames"])})
    missing = sorted(rollout_ids - {row["rollout_id"] for row in rows})
    if missing:
        raise FileNotFoundError(f"missing prediction cache for {split}: {missing[:5]} ({len(missing)} total)")
    return rows


def _resource(path: Path, logical: str, usage: str, split: str, *, hash_file: bool = True) -> dict[str, Any]:
    return {"logical_id": logical, "source_path": str(path), "resolved_path": str(path.resolve()),
            "exists": path.is_file(), "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() and hash_file else None, "usage": usage, "split": split}


def _rollout_record(row: dict[str, str], split: str, role: str, prediction: dict[str, Any] | None) -> dict[str, Any]:
    root = Path(row["path"])
    resources = []
    for name in ONLINE_FILES:
        resources.append(_resource(root / name, f"{row['rollout_id']}:{name}", "runtime_allowed", split))
    for name in REFERENCE_FILES:
        resources.append(_resource(root / name, f"{row['rollout_id']}:{name}", "reference_only", split))
    if prediction:
        resources.append(_resource(Path(prediction["prediction_path"]), f"{row['rollout_id']}:predictions", "runtime_allowed", split))
    return {"rollout_id": row["rollout_id"], "root_family_id": row["root_family_id"], "scenario_reference_only": row["scenario"],
            "data_role": role, "split": split, "rollout_path": str(root.resolve()), "prediction": prediction,
            "resources": resources}


def resolve_inputs(repo_root: Path, frozen_root: Path, output_root: Path, protocol_path: Path) -> dict[str, Any]:
    l2r_root = repo_root / "artifacts/pathgraph_sarm/upgrade_v2/visual_refine_l2_v1"
    dataset_root = l2r_root / "dynamic_dataset_v1"
    dev_manifest_path = dataset_root / "manifests/development_rollout_manifest.csv"
    dev_split_path = dataset_root / "manifests/development_family_split.csv"
    confirmation_manifest_path = l2r_root / "fresh_confirmation_v1/data/rollout_manifest.csv"
    if not all(path.is_file() for path in (dev_manifest_path, dev_split_path, confirmation_manifest_path)):
        raise FileNotFoundError("one or more frozen rollout manifests are missing")
    dev_rows = read_csv(dev_manifest_path)
    dev_split = read_csv(dev_split_path)
    confirm_rows = read_csv(confirmation_manifest_path)
    dev_by_id = {row["rollout_id"]: row for row in dev_rows}
    confirm_by_id = {row["rollout_id"]: row for row in confirm_rows}
    dev_fit_ids = {row["root_family_id"] for row in dev_split if row["split"] == "dev_fit"}
    dev_select_ids = {row["root_family_id"] for row in dev_split if row["split"] == "dev_select"}
    dev_fit = [row for row in dev_rows if row["root_family_id"] in dev_fit_ids]
    dev_select = [row for row in dev_rows if row["root_family_id"] in dev_select_ids]
    pred_root = l2r_root / "observable_predicates_v1/predictions"
    confirm_pred_root = l2r_root / "fresh_confirmation_v1/predicates"
    prediction_rows = {
        "legacy_dev_fit": _prediction_manifest(pred_root / "dev_fit", {r["rollout_id"] for r in dev_fit}, "dev_fit", dev_by_id),
        "legacy_dev_select": _prediction_manifest(pred_root / "dev_select", {r["rollout_id"] for r in dev_select}, "dev_select", dev_by_id),
        "legacy_confirmation": _prediction_manifest(confirm_pred_root, set(confirm_by_id), "legacy_confirmation", confirm_by_id),
    }
    records = []
    records.extend(_rollout_record(row, "dev_fit", "runtime_allowed", next(p for p in prediction_rows["legacy_dev_fit"] if p["rollout_id"] == row["rollout_id"])) for row in dev_fit)
    records.extend(_rollout_record(row, "dev_select", "runtime_allowed", next(p for p in prediction_rows["legacy_dev_select"] if p["rollout_id"] == row["rollout_id"])) for row in dev_select)
    records.extend(_rollout_record(row, "legacy_confirmation", "historical_diagnosis_only", next(p for p in prediction_rows["legacy_confirmation"] if p["rollout_id"] == row["rollout_id"])) for row in confirm_rows)
    frozen_names = ["l3_or_stop_handoff.json", "third_layer_interface.json", "verification_summary.json", "final_manifest.json",
                    "graphs/G0_coarse_direct.json", "graphs/G1_predicate_bound.json", "graphs/G2_evidence_refined.json",
                    "locks/predicate_thresholds.json", "locks/selection_lock.json", "locks/fresh_family_lock.json",
                    "tables/confirmation_metrics.csv", "tables/metrics_by_scenario.csv", "tables/metrics_by_family.csv",
                    "manifests/fresh_rollout_manifest.csv", "manifests/external_artifacts.tsv"]
    frozen = [_resource(frozen_root / name, f"frozen:{name}", "historical_diagnosis_only", "frozen") for name in frozen_names]
    return {
        "schema": "pathgraph_l2ra_resolved_inputs_v1", "prepared_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "repository": {"path": str(repo_root.resolve()), "current_commit": git_commit(repo_root), "manual_base_commit": "e607131d7745d46745680bc9a6190a467b01dfa2"},
        "roles": {
            "runtime_allowed": ["front_rgb", "contact_sensor", "gripper_command", "executed_low_level_controls", "frozen_observable_predicates"],
            "reference_only": ["scenario", "family_id", "events", "oracle_timeline", "termination_oracle", "qpos", "qvel", "weld_state", "future_outcome"],
            "historical_diagnosis_only": ["legacy_confirmation", "old reports", "old confirmation metrics"]
        },
        "prediction_manifest_status": "reconstructed_from_filenames_and_rollout_manifests",
        "datasets": {
            "legacy_dev_fit": {"families": len(dev_fit_ids), "rollouts": len(dev_fit), "prediction_manifest": prediction_rows["legacy_dev_fit"]},
            "legacy_dev_select": {"families": len(dev_select_ids), "rollouts": len(dev_select), "prediction_manifest": prediction_rows["legacy_dev_select"]},
            "legacy_confirmation": {"families": len({r["root_family_id"] for r in confirm_rows}), "rollouts": len(confirm_rows), "prediction_manifest": prediction_rows["legacy_confirmation"]}
        },
        "rollouts": records, "frozen_sources": frozen,
        "protocol_path": str(protocol_path.resolve()), "api_calls": 0, "training_jobs": 0, "api_key_read": False
    }


def prepare(repo_root: Path, frozen_root: Path, protocol_path: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    if payload.get("api_calls_max") != 0 or payload.get("training_jobs_max") != 0 or payload.get("read_api_keys") is not False:
        raise ValueError("protocol violates zero API/training/key policy")
    resolved = resolve_inputs(repo_root, frozen_root, output_root, protocol_path)
    write_json(output_root / "manifests/resolved_inputs.json", resolved)
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    lock_files = resolved["frozen_sources"] + [{"logical_id": "protocol", "source_path": str(protocol_path), "resolved_path": str(protocol_path.resolve()), "exists": True, "size_bytes": protocol_path.stat().st_size, "sha256": protocol_sha, "usage": "protocol", "split": "lock"}]
    lock = {"schema": "pathgraph_l2ra_protocol_lock_v1", "status": "LOCKED_FOR_DEVELOPMENT", "protocol_sha256": protocol_sha,
            "source_commit": resolved["repository"]["current_commit"], "manual_base_commit": resolved["repository"]["manual_base_commit"],
            "api_calls_max": 0, "training_jobs_max": 0, "read_api_keys": False, "files": lock_files}
    write_json(output_root / "locks/protocol_lock.json", lock)
    source_note = (f"# L2RA source resolution\n\n- Manual frozen source commit: `{resolved['repository']['manual_base_commit']}`\n"
                   f"- Actual repository commit: `{resolved['repository']['current_commit']}`\n"
                   "- The actual repository is a descendant of the manual baseline; no history was rewritten.\n"
                   "- Frozen L2R artifacts are read-only inputs; this run writes a separate edge-ambiguity tree.\n"
                   "- Prediction manifests are reconstructed from filenames and rollout manifests and are not historical claims.\n")
    (output_root / "rounds/l2ra_0_entry/reports/source_resolution.md").parent.mkdir(parents=True, exist_ok=True)
    (output_root / "rounds/l2ra_0_entry/reports/source_resolution.md").write_text(source_note, encoding="utf-8")
    return {"status": "REPLAY_READY", "datasets": resolved["datasets"], "protocol_sha256": protocol_sha, "current_commit": resolved["repository"]["current_commit"]}
