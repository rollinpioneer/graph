"""Deterministic IO, hashing, locks, and audit helpers for L2R."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SECRET_RE = re.compile(
    rb"(?:sk-[A-Za-z0-9._-]{16,}|Authorization\s*:\s*Bearer\s+\S+|api[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9._-]{20,})",
    re.I,
)
FORBIDDEN_ONLINE_KEYS = {
    "scenario",
    "qpos",
    "qvel",
    "geom_id",
    "geom_ids",
    "body_id",
    "body_ids",
    "weld_state",
    "future_outcome",
    "oracle_event",
    "oracle_state",
    "gold_mode",
    "phase",
}
ROUND_DIRS = ("configs", "commands", "gpu", "logs", "metrics", "tables", "figures", "reports", "manifests", "checksums")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{number}")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None, delimiter: str = ",") -> None:
    materialized = list(rows)
    actual = fields or (list(materialized[0]) if materialized else [])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=actual, extrasaction="ignore", delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def git_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def bool_value(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes", "on"}


def ensure_round(round_dir: Path) -> None:
    for name in ROUND_DIRS:
        (round_dir / name).mkdir(parents=True, exist_ok=True)


def lock_record(path: Path, purpose: str, base: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    logical = resolved.relative_to(base.resolve()).as_posix() if base and resolved.is_relative_to(base.resolve()) else str(resolved)
    return {
        "logical_path": logical,
        "original_path": str(resolved),
        "original_filename": resolved.name,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "purpose": purpose,
    }


def verify_lock(lock_path: Path) -> dict[str, Any]:
    payload = read_json(lock_path)
    failures = []
    for item in payload.get("files", []):
        path = Path(item["original_path"])
        if not path.is_file():
            failures.append(f"missing: {path}")
        elif path.stat().st_size != item["size_bytes"]:
            failures.append(f"size mismatch: {path}")
        elif sha256_file(path) != item["sha256"]:
            failures.append(f"hash mismatch: {path}")
    return {"status": "PASS" if not failures else "FAIL", "files": len(payload.get("files", [])), "failures": failures}


def find_forbidden_keys(value: Any, prefix: str = "") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            location = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in FORBIDDEN_ONLINE_KEYS:
                findings.append(location)
            findings.extend(find_forbidden_keys(child, location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(find_forbidden_keys(child, f"{prefix}[{index}]"))
    return findings


def secret_scan(paths: Iterable[Path]) -> dict[str, Any]:
    findings = []
    files = 0
    for root in paths:
        candidates = [root] if root.is_file() else sorted(root.rglob("*")) if root.is_dir() else []
        for path in candidates:
            if not path.is_file() or path.suffix.lower() in {".zip", ".pyc", ".npz", ".npy"}:
                continue
            files += 1
            if SECRET_RE.search(path.read_bytes()):
                findings.append(str(path))
    return {"status": "PASS" if not findings else "FAIL", "files": files, "findings": findings, "scanned_at": now_iso()}


def write_report(path: Path, title: str, rows: Iterable[tuple[str, Any]]) -> None:
    body = [f"# {title}", ""] + [f"- {key}: `{value}`" for key, value in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
