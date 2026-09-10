"""Small, explicit provenance records for the R12 static audit."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(repo: Path, *args: str) -> str | None:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def runtime_record(repo: Path, *, role: str, source_path: Path | None = None, reported_render_callbacks: Any = None) -> dict[str, Any]:
    return {
        "role": role,
        "git_head": git_value(repo, "rev-parse", "HEAD"),
        "git_worktree_diff_sha256": hashlib.sha256((git_value(repo, "diff", "--no-ext-diff", "--binary") or "").encode()).hexdigest(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": "NOT_RECORDED_BY_R12_STATIC_AUDIT",
        "mujoco_version": "NOT_RECORDED_BY_R12_STATIC_AUDIT",
        "model_or_xml_sha256": None,
        "source_path": str(source_path.resolve()) if source_path else None,
        "source_sha256": file_hash(source_path) if source_path else None,
        "render_callbacks_requested": "NOT_AVAILABLE",
        "render_callbacks_effective": "NOT_AVAILABLE",
        "render_callbacks_reported": reported_render_callbacks,
        "seed_and_family_spec": "NOT_RECORDED",
        "sampling_point": "NOT_RECORDED",
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
