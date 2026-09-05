"""Small dependency-light helpers shared by the U3 workflow."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9._-]{20,}"),
    re.compile(r"Authorization:\s*Bearer", re.I),
    # Permit the literal template forms used by documentation, but never a
    # concrete assignment.  Single quotes delimit the Python string so the
    # optional JSON/shell quote class does not need escaping.
    re.compile(r'(?:QWEN|DEEPSEEK)_API_KEY\s*=\s*(?!["\']?(?:<NEW_ROTATED_KEY>|\$\{(?:QWEN|DEEPSEEK)_API_KEY\})["\']?)[^\s]+'),
)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def repo_root(path: Path | None = None) -> Path:
    current = (path or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError("repository root not found")


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected object line in {path}")
                rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], delimiter: str = ",") -> None:
    materialized = list(rows)
    fields = sorted({key for row in materialized for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def append_command(path: Path, command: str) -> None:
    """Record a manually supplied, non-secret command for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(command.rstrip() + "\n")


def ensure_round_layout(round_dir: Path) -> None:
    for name in ("configs", "commands", "gpu", "logs", "metrics", "tables", "reports", "manifests", "checksums", "provider", "usage"):
        (round_dir / name).mkdir(parents=True, exist_ok=True)


def write_round_manifest(round_dir: Path, values: dict[str, Any]) -> None:
    lines = ["# U3 run manifest", ""]
    for key, value in values.items():
        lines.append(f"- {key}: `{value}`")
    (round_dir / "run_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def redact_text(value: str) -> str:
    redacted = value
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    # Some providers include a masked tail (for example ``****abcd``) in an
    # authentication error.  Keep even that partial identifier out of durable
    # artifacts because the U3 protocol records only error classes.
    redacted = re.sub(r"\*{2,}[A-Za-z0-9._-]+", "[REDACTED]", redacted)
    return redacted


def scan_paths(paths: Iterable[Path], root: Path, include_git_diff: bool = False) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    for value in paths:
        path = value if value.is_absolute() else root / value
        candidates = [path] if path.is_file() else list(path.rglob("*")) if path.is_dir() else []
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix in {".pyc", ".zip"}:
                continue
            try:
                content = candidate.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for pattern in SECRET_PATTERNS:
                if pattern.search(content):
                    findings.append({"path": relative(candidate, root), "pattern": pattern.pattern})
                    break
    if include_git_diff:
        diff = subprocess.run(["git", "-C", str(root), "diff", "--", "."], text=True, capture_output=True, check=False).stdout
        for pattern in SECRET_PATTERNS:
            if pattern.search(diff):
                findings.append({"path": "git_diff", "pattern": pattern.pattern})
    return {"status": "PASS" if not findings else "FAIL", "findings": findings, "scanned_at": now_iso()}


def environment_check() -> dict[str, Any]:
    result: dict[str, Any] = {"keys_rotated": os.environ.get("U3_KEYS_ROTATED") == "1"}
    for name in ("QWEN_API_KEY", "DEEPSEEK_API_KEY"):
        result[f"{name.lower()}_present"] = len(os.environ.get(name, "")) >= 20
    result["status"] = "PASS" if result["keys_rotated"] and all(result[name] for name in ("qwen_api_key_present", "deepseek_api_key_present")) else "FAIL"
    return result
