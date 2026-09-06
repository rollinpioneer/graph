"""Portable IO, provenance, resume and delivery helpers."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SECRET_RE = re.compile(rb"(?:sk-[A-Za-z0-9._-]{20,}|Authorization:\s*Bearer\s+\S+|api[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9._-]{20,})", re.I)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root(path: Path | None = None) -> Path:
    current = (path or Path.cwd()).resolve()
    for parent in (current, *current.parents):
        if (parent / ".git").exists():
            return parent
    raise FileNotFoundError("repository root not found")


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    materialized = list(rows)
    if fields is None:
        fields = sorted({key for row in materialized for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def bool_value(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes", "on"}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def ensure_layout(root: Path) -> None:
    for name in ("protocol", "data", "evidence", "queries", "diagnostics", "targeted_repair", "graphs", "evaluation", "final", "rounds", "delivery"):
        (root / name).mkdir(parents=True, exist_ok=True)


def scan_secrets(paths: Iterable[Path]) -> dict[str, Any]:
    findings: list[str] = []
    for root in paths:
        candidates = [root] if root.is_file() else sorted(root.rglob("*")) if root.is_dir() else []
        for path in candidates:
            if not path.is_file() or path.suffix.lower() in {".zip", ".pyc", ".pt", ".pth", ".npz"}:
                continue
            data = path.read_bytes()
            if SECRET_RE.search(data):
                findings.append(str(path))
    return {"status": "PASS" if not findings else "FAIL", "findings": findings, "scanned_at": now_iso()}


def external_inventory(paths: Iterable[Path], root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted({Path(p) for p in paths}, key=str):
        resolved = path.resolve()
        row = {"path": str(resolved), "logical_path": str(resolved.relative_to(root.resolve())) if resolved.is_relative_to(root.resolve()) else str(resolved), "exists": resolved.is_file(), "purpose": "externalized large or sensitive artifact"}
        if resolved.is_file():
            row.update(size_bytes=resolved.stat().st_size, sha256=sha256_file(resolved))
        rows.append(row)
    return rows


def verify_locked_inputs(lock_path: Path, required_input: Path | None = None) -> dict[str, Any]:
    """Verify that a confirmation lock and every frozen input still match."""
    if not lock_path.is_file():
        return {"status": "BLOCKED", "reason": "final pipeline lock is missing", "failures": [str(lock_path)]}
    lock = read_json(lock_path)
    failures: list[str] = []
    if not lock.get("confirmation_locked"):
        failures.append("confirmation_locked is not true")
    hashes = lock.get("input_hashes")
    if not isinstance(hashes, dict) or not hashes:
        failures.append("input_hashes is missing or empty")
        hashes = {}
    normalized = {str(Path(path).resolve()): digest for path, digest in hashes.items()}
    for raw_path, expected in sorted(hashes.items()):
        path = Path(raw_path)
        if not path.is_file():
            failures.append(f"missing locked input: {path}")
        elif sha256_file(path) != expected:
            failures.append(f"hash mismatch: {path}")
    if required_input is not None:
        required = str(required_input.resolve())
        if required not in normalized:
            failures.append(f"required family lock is not frozen: {required_input}")
        elif required_input.is_file() and sha256_file(required_input) != normalized[required]:
            failures.append(f"required family lock hash mismatch: {required_input}")
    return {
        "status": "PASS" if not failures else "BLOCKED",
        "lock_sha256": sha256_file(lock_path),
        "verified_input_count": len(hashes),
        "failures": failures,
    }
