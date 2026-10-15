"""Small, deterministic helpers for the U3 grounding bridge."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SECRET_RE = re.compile(r"sk-[A-Za-z0-9._-]{20,}|Authorization:\s*Bearer|api[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9._-]{20,}", re.I)
ROUND_DIRS = ("configs", "commands", "gpu", "logs", "metrics", "tables", "figures", "reports", "manifests", "checksums")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root(path: Path | None = None) -> Path:
    current = (path or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError("repository root not found")


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


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    fields = sorted({key for row in materialized for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def ensure_round_layout(round_dir: Path) -> None:
    for name in ROUND_DIRS:
        (round_dir / name).mkdir(parents=True, exist_ok=True)


def secret_scan(paths: Iterable[Path]) -> dict[str, Any]:
    findings: list[str] = []
    for path in paths:
        candidates = [path] if path.is_file() else sorted(path.rglob("*")) if path.is_dir() else []
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix in {".pyc", ".zip"}:
                continue
            text = candidate.read_text(encoding="utf-8", errors="ignore")
            if SECRET_RE.search(text):
                findings.append(str(candidate))
    return {"status": "PASS" if not findings else "FAIL", "findings": findings, "scanned_at": now_iso()}


def family_from_id(value: str) -> str:
    match = re.search(r"(u2_family_\d+)", value)
    return match.group(1) if match else value


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
