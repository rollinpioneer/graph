"""Secret-safe stage packaging for the grounding bridge."""

from __future__ import annotations

import zipfile
from pathlib import Path

from .common import SECRET_RE, sha256_file

EXCLUDED = {".npz", ".pt", ".pth", ".parquet", ".jsonl", ".log", ".pyc", ".pyo", ".zip"}


def _placeholder(path: Path, root: Path) -> bytes:
    return ("# Placeholder for excluded artifact\n\n" + f"- Original filename: `{path.name}`\n" + f"- Original relative path: `{path.relative_to(root).as_posix()}`\n" + f"- Original size: `{path.stat().st_size}` bytes\n" + "- Reason: raw, heavy, or sensitive payload excluded from the stage ZIP.\n").encode()


def _write_zip(root: Path, output: Path) -> dict:
    entries: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.endswith(".placeholder.md") and path.suffix == ".md":
            pass
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        try:
            data = path.read_bytes()
        except OSError:
            continue
        text = data.decode("utf-8", "ignore")
        excluded = path.suffix.lower() in EXCLUDED or "authorization" in path.name.lower() or ".env" in path.name.lower() or path.stat().st_size > 200 * 1024 * 1024
        if SECRET_RE.search(text):
            raise RuntimeError(f"secret marker found in {rel}")
        entries[rel + ".placeholder.md" if excluded else rel] = _placeholder(path, root) if excluded else data
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(entries.items()): archive.writestr(name, data)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() or len(archive.namelist()) != len(set(archive.namelist())): raise RuntimeError("stage ZIP validation failed")
    return {"status": "PASS", "zip": str(output.resolve()), "sha256": sha256_file(output), "entry_count": len(entries), "zip_test": "PASS"}


def package_round(*, round_dir: Path, output: Path, max_file_mb: int = 200) -> dict:
    return _write_zip(round_dir, output)


def package_complete(*, root: Path, final_root: Path, output: Path, round_zip_dir: Path | None = None, max_file_mb: int = 200) -> dict:
    result = _write_zip(root, output)
    result["single_public_zip_policy"] = True
    return result
