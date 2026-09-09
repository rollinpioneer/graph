from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import file_record, git, read_csv, read_json, sha256, write_csv, write_json


FORMAL_PATHS = (
    "artifacts/pathgraph_sarm/upgrade_v2/hold_evidence_l2rar1_v2/final_v1/final_report.md",
    "artifacts/pathgraph_sarm/upgrade_v2/hold_evidence_l2rar1_v2/final_v1/next_stage_handoff.json",
)
MAINTENANCE_PATHS = (
    "artifacts/pathgraph_sarm/upgrade_v2/hold_evidence_l2rar1_r1_7/final_v1/final_report.md",
    "artifacts/pathgraph_sarm/upgrade_v2/hold_evidence_l2rar1_r1_7/final_v1/next_stage_handoff.json",
    "artifacts/pathgraph_sarm/upgrade_v2/hold_evidence_l2rar1_r1_7/final_v1/tables/candidate_metrics.csv",
    "artifacts/pathgraph_sarm/upgrade_v2/hold_evidence_l2rar1_r1_7/final_v1/tables/development_gates.csv",
)


def _export_git_file(anchor: Path, ref: str, source_path: str, destination: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "-C", str(anchor), "show", f"{ref}:{source_path}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    exists = result.returncode == 0
    if exists:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(result.stdout)
    return {
        "logical_id": f"git:{ref}:{source_path}",
        "source_commit": ref,
        "source_path": source_path,
        "resolved_path": str(destination.resolve(strict=False)),
        "exists": exists,
        "size_bytes": destination.stat().st_size if exists else None,
        "sha256": sha256(destination) if exists else None,
        "data_role": "historical_diagnosis_only",
        "purpose": "formal historical record" if source_path in FORMAL_PATHS else "maintenance diagnostic record",
    }


def _runtime_record() -> dict[str, Any]:
    import importlib

    result: dict[str, Any] = {"python": sys.executable, "python_version": sys.version.split()[0]}
    for name in ("torch", "cv2", "mujoco"):
        try:
            module = importlib.import_module(name)
            result[name] = {"installed": True, "version": getattr(module, "__version__", "unknown")}
            if name == "torch":
                result["cuda_available"] = bool(module.cuda.is_available())
                result["cuda_device_count"] = int(module.cuda.device_count())
        except ImportError:
            result[name] = {"installed": False}
    result["meaning"] = "dependency inventory only; PyTorch and CUDA availability are reported separately"
    return result


def _cache_candidates(anchor: Path) -> list[Path]:
    return [
        Path("/tmp/l2ra_r1_7_full"),
        anchor / "artifacts/pathgraph_sarm/upgrade_v2/hold_evidence_l2rar1_r1_7",
    ]


def prepare(anchor: Path, formal_ref: str, maintenance_ref: str, protocol_path: Path, output_root: Path) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    if protocol.get("api_calls_max") != 0 or protocol.get("training_jobs_max") != 0 or protocol.get("read_api_keys") is not False:
        raise ValueError("protocol must prohibit API calls, training, and key reads")
    actual_main = git(anchor, "rev-parse", "origin/main")
    git(anchor, "cat-file", "-e", f"{formal_ref}^{{commit}}")
    git(anchor, "cat-file", "-e", f"{maintenance_ref}^{{commit}}")
    subprocess.run(["git", "-C", str(anchor), "merge-base", "--is-ancestor", formal_ref, maintenance_ref], check=True)

    source_dir = output_root / "rounds/l2rar2_0_entry/source_tables"
    sources = []
    for source_path in FORMAL_PATHS:
        sources.append(_export_git_file(anchor, formal_ref, source_path, source_dir / "formal" / Path(source_path).name))
    for source_path in MAINTENANCE_PATHS:
        sources.append(_export_git_file(anchor, maintenance_ref, source_path, source_dir / "maintenance" / Path(source_path).name))
    if not all(row["exists"] for row in sources):
        missing = [row["source_path"] for row in sources if not row["exists"]]
        raise FileNotFoundError(f"required Git sources missing: {missing}")

    cache = next((candidate for candidate in _cache_candidates(anchor) if (candidate / "data/new_development/dev_select_rollout_manifest.csv").is_file()), None)
    if cache is None:
        raise FileNotFoundError("authoritative R1.7 cache could not be resolved")
    cache_files = [
        cache / "data/new_development/dev_fit_rollout_manifest.csv",
        cache / "data/new_development/dev_select_rollout_manifest.csv",
        cache / "rounds/l2rar1_3_development_selection/event_decisions.csv",
        cache / "features/feature_manifest.csv",
        cache / "locks/development_lock.json",
    ]
    cache_records = [file_record(path, logical_id=f"r1_7_cache:{path.relative_to(cache)}", source_commit=maintenance_ref, role="historical_r1_7_diagnosis", purpose="read-only T1 replay input") for path in cache_files]
    if not all(row["exists"] for row in cache_records):
        raise FileNotFoundError("resolved R1.7 cache is incomplete")
    select_rows = read_csv(cache_files[1])
    fit_rows = read_csv(cache_files[0])
    resolved = {
        "schema": "pathgraph_l2rar2_resolved_inputs_v1",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "formal_main_commit": formal_ref,
        "maintenance_source_commit": maintenance_ref,
        "origin_main_at_prepare": actual_main,
        "implementation_commit_at_prepare": git(Path.cwd(), "rev-parse", "HEAD"),
        "formal_is_ancestor_of_maintenance": True,
        "protected_worktree": "/home/__compress_data/xushijie/graph_github_upload",
        "historical_cache_root": str(cache.resolve()),
        "historical_cache": {
            "role": "historical_r1_7_diagnosis",
            "dev_fit_rollouts": len(fit_rows),
            "dev_select_rollouts": len(select_rows),
            "dev_select_root_families": len({row["root_family_id"] for row in select_rows}),
            "controller_requested_effect_available": False,
        },
        "sources": sources + cache_records,
        "online_allowed": ["front_rgb", "contact_sensor", "gripper_command", "attempt_lifecycle", "controller_request"],
        "reference_only": ["scenario", "case_id", "root_family_id", "oracle", "weld_state", "events", "future_outcome"],
        "api_calls": 0,
        "training_jobs": 0,
        "api_key_read": False,
    }
    write_json(output_root / "manifests/resolved_inputs.json", resolved)
    write_json(output_root / "locks/source_lock.json", {
        "schema": "pathgraph_l2rar2_source_lock_v1",
        "status": "LOCKED",
        "formal_main_commit": formal_ref,
        "maintenance_source_commit": maintenance_ref,
        "origin_main_at_prepare": actual_main,
        "protocol_sha256": sha256(protocol_path),
        "sources": [{key: row[key] for key in ("logical_id", "source_commit", "resolved_path", "sha256", "data_role") } for row in sources + cache_records],
    })
    write_csv(output_root / "rounds/l2rar2_0_entry/source_resolution.csv", sources + cache_records)
    write_json(output_root / "rounds/l2rar2_0_entry/environment/runtime.json", _runtime_record())
    summary = {
        "schema": "pathgraph_l2rar2_entry_summary_v1",
        "status": "ENTRY_LOCKED",
        "formal_main_commit": formal_ref,
        "maintenance_source_commit": maintenance_ref,
        "origin_main_matches_formal": actual_main == formal_ref,
        "historical_select_rollouts": len(select_rows),
        "historical_select_root_families": len({row["root_family_id"] for row in select_rows}),
        "new_physical_rollouts": 0,
        "api_calls": 0,
        "training_jobs": 0,
        "api_key_read": False,
    }
    write_json(output_root / "rounds/l2rar2_0_entry/entry_summary.json", summary)
    return summary
