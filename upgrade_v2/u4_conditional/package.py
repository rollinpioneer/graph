"""Create auditable U4R1 round and complete ZIPs without raw large payloads."""
from __future__ import annotations

import hashlib
import zipfile
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import sha256_file, write_csv, write_json


def _zip(root: Path, output: Path, max_file_mb: int = 200, excluded_prefixes: tuple[str, ...] = ()) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    included = []
    omissions = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = str(path.relative_to(root))
            if any(relative == prefix or relative.startswith(prefix.rstrip("/") + "/") for prefix in excluded_prefixes):
                omissions.append({"path": relative, "size_bytes": path.stat().st_size, "reason": "not_selected_round"})
                continue
            reason = None
            if path.suffix.lower() in {".pt", ".pth", ".npz", ".parquet", ".zip", ".pyc"}:
                reason = "sensitive_or_large_binary"
            elif path.name.endswith("occurrences.jsonl") or path.name.endswith("boundaries.jsonl") or "rollouts" in path.parts:
                reason = "raw_large_or_sensitive_evidence"
            elif path.stat().st_size > max_file_mb * 1024 * 1024:
                reason = "file_size_limit"
            if reason:
                omissions.append({"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "reason": reason})
                continue
            archive.write(path, path.relative_to(root).as_posix())
            included.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failure: {bad}")
        members = sorted(archive.namelist())
    return {"path": str(output.resolve()), "sha256": sha256_file(output), "size_bytes": output.stat().st_size, "zip_test": "PASS", "member_count": len(members), "included_files": included, "omissions": omissions}


def package_round(root: Path, output: Path, max_file_mb: int = 200) -> dict[str, Any]:
    manifest = root / "run_manifest.json"
    if not manifest.exists():
        write_json(manifest, {"schema": "u4r1_run_manifest_v1", "round_id": root.name, "generated_at": datetime.now(timezone.utc).isoformat(), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(), "python": sys.executable, "max_file_mb": max_file_mb, "device": "cpu", "torch_policy": "existing_cupid_environment", "api_calls": 0, "training_jobs": 0})
    result = _zip(root, output, max_file_mb)
    write_json(output.with_suffix(".sha256.json"), result)
    write_json(output.with_suffix(".manifest.json"), {"schema": "u4r1_zip_manifest_v2", **result})
    return {"status": "PASS", **result}


def package_complete(root: Path, output: Path, rounds: list[Path], max_file_mb: int = 200) -> dict[str, Any]:
    index = [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in rounds if path.is_file()]
    write_json(root / "round_index.json", {"schema": "u4r1_round_index_v1", "rounds": index})
    selected_names = {path.stem if path.suffix.lower() == ".zip" else path.name for path in rounds}
    excluded = []
    for parent in (root / "rounds", root / "delivery" / "rounds"):
        if parent.is_dir():
            excluded.extend(str(path.relative_to(root)) for path in parent.iterdir() if path.is_dir() and path.name not in selected_names)
    result = _zip(root, output, max_file_mb, tuple(excluded))
    write_json(output.with_suffix(".sha256.json"), {**result, "rounds": index})
    write_json(output.with_suffix(".manifest.json"), {"schema": "u4r1_complete_zip_manifest_v2", **result, "rounds": index})
    return {"status": "PASS", **result, "rounds": index}
