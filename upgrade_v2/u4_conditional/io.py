"""Small, deterministic IO helpers used by the U4R1 command line."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    data = list(rows)
    if fields is None:
        fields = sorted({key for row in data for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root(path: Path | None = None) -> Path:
    current = (path or Path.cwd()).resolve()
    for parent in (current, *current.parents):
        if (parent / ".git").exists():
            return parent
    raise FileNotFoundError("repository root not found")


def episode_files(root: Path) -> list[Path]:
    return sorted(root.glob("*.json")) if root.is_dir() else []


def environment_audit() -> dict[str, Any]:
    result: dict[str, Any] = {"python": __import__("sys").executable, "python_version": __import__("platform").python_version()}
    for name in ("torch", "pandas", "sklearn", "pyarrow"):
        try:
            module = __import__(name)
            result[name] = getattr(module, "__version__", "installed")
            if name == "torch":
                result["torch_cuda_build"] = getattr(module.version, "cuda", None)
                result["cuda_available"] = bool(module.cuda.is_available())
                result["device"] = "cuda" if module.cuda.is_available() else "cpu"
        except Exception as exc:  # pragma: no cover - environment dependent
            result[name] = f"unavailable:{type(exc).__name__}"
    return result
