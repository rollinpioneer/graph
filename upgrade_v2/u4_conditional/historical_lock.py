"""Freeze the historical U4 B+ inputs and execution environment."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import canonical_sha, environment_audit, git_commit, read_json, sha256_file, write_json


def freeze_history(repo: Path, output: Path, paths: list[Path]) -> dict[str, Any]:
    resolved = []
    for path in paths:
        path = path if path.is_absolute() else repo / path
        if path.is_file():
            resolved.append({"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    payload = {
        "schema": "u4r1_historical_lock_v1",
        "source_commit": git_commit(repo),
        "history_read_only": True,
        "input_count": len(resolved),
        "inputs": resolved,
        "environment": environment_audit(),
        "api_calls": 0,
        "api_key_read": False,
        "training_jobs": 0,
        "prohibited_online_features": ["scenario", "phase", "gold_mode", "future outcome", "future events", "root_family_id", "episode_id"],
    }
    payload["lock_sha256"] = canonical_sha(payload)
    write_json(output, payload)
    return payload


def verify_history(lock_path: Path) -> dict[str, Any]:
    if not lock_path.is_file():
        return {"status": "BLOCKED", "reason": "historical lock missing"}
    lock = read_json(lock_path)
    failures = []
    for item in lock.get("inputs", []):
        path = Path(item["path"])
        if not path.is_file():
            failures.append(f"missing:{path}")
        elif sha256_file(path) != item["sha256"]:
            failures.append(f"hash:{path}")
    return {"status": "PASS" if not failures else "BLOCKED", "failures": failures, "environment": lock.get("environment", {})}
