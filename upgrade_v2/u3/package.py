"""Secret-safe ZIP packaging, honoring the user's single-archive policy."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Any

from .common import repo_root, scan_paths, sha256_file, write_json


EXCLUDED_SUFFIXES = {".npz", ".pt", ".pth", ".parquet", ".jsonl", ".log", ".pyc", ".pyo"}
EXCLUDED_NAME_PARTS = {"__pycache__", "secrets"}


def _excluded(path: Path, max_file_mb: int) -> bool:
    lower = path.name.lower()
    return (
        path.suffix.lower() in EXCLUDED_SUFFIXES
        or any(part in EXCLUDED_NAME_PARTS for part in path.parts)
        or ".env" in lower
        or "authorization" in lower
        or "request_headers" in lower
        or path.stat().st_size > max_file_mb * 1024 * 1024
    )


def _skip_without_placeholder(path: Path) -> bool:
    """Bytecode/cache files are neither evidence nor an omitted artifact."""
    return path.suffix.lower() in {".pyc", ".pyo"} or any(part == "__pycache__" for part in path.parts)


def _placeholder(path: Path, root: Path) -> bytes:
    rel = path.relative_to(root).as_posix()
    return ("# Placeholder for excluded artifact\n\n"
            f"- Original filename: `{path.name}`\n"
            f"- Original relative path: `{rel}`\n"
            f"- Original size: {path.stat().st_size} bytes\n"
            "- Reason: raw/heavy/sensitive payload excluded under the single ZIP policy.\n"
            "- Restore: use the repository workspace artifact, not this public delivery ZIP.\n").encode("utf-8")


def _validate_zip(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        duplicates = len(names) - len(set(names))
        bad = [name for name in names if Path(name).suffix.lower() in EXCLUDED_SUFFIXES or "__pycache__" in Path(name).parts]
        crc_error = archive.testzip()
    if duplicates or bad or crc_error:
        raise RuntimeError(f"invalid ZIP duplicates={duplicates}, banned={bad[:3]}, crc={crc_error}")
    return {"entry_count": len(names), "duplicates": duplicates, "banned_payloads": len(bad), "zip_test": "PASS"}


def package_round(*, round_dir: Path, output: Path, max_file_mb: int) -> dict[str, Any]:
    root = repo_root(round_dir)
    scan = scan_paths([round_dir], root)
    if scan["status"] != "PASS":
        raise RuntimeError("secret scan failed; refusing to package")
    entries: dict[str, bytes] = {}
    for path in sorted(round_dir.rglob("*")):
        if not path.is_file():
            continue
        if _skip_without_placeholder(path):
            continue
        name = path.relative_to(root).as_posix()
        if _excluded(path, max_file_mb):
            entries[name + ".placeholder.md"] = _placeholder(path, root)
        else:
            entries[name] = path.read_bytes()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(entries.items()):
            archive.writestr(name, payload)
    result = _validate_zip(output)
    result.update({"zip": str(output.resolve()), "sha256": sha256_file(output), "secret_scan": "PASS"})
    output.with_suffix(output.suffix + ".sha256").write_text(f"{result['sha256']}  {output.name}\n", encoding="utf-8")
    return result


def package_u3_complete(*, u3_root: Path, final_root: Path, output: Path, max_file_mb: int) -> dict[str, Any]:
    """Refresh the one user-authorized cumulative delivery ZIP.

    Round-level manifests are included as archive entries, rather than exposing
    seven additional downloads.  Existing U0/U1/U2 entries remain untouched.
    """
    root = repo_root(u3_root).resolve()
    u3_root = u3_root.resolve()
    final_root = final_root.resolve()
    output = output.resolve()
    scan = scan_paths([root / "upgrade_v2" / "u3", u3_root], root, include_git_diff=True)
    if scan["status"] != "PASS":
        raise RuntimeError("secret scan failed; refusing to package")
    entries: dict[str, bytes] = {}
    if output.is_file():
        with zipfile.ZipFile(output) as previous:
            for name in previous.namelist():
                if name.endswith("/"):
                    continue
                if name.startswith("upgrade_v2/u3/") or name.startswith("artifacts/pathgraph_sarm/upgrade_v2/u3_candidate_graph_v2/"):
                    continue
                entries[name] = previous.read(name)
    selected_roots = [(root / "upgrade_v2" / "u3").resolve(), u3_root]
    placeholders: list[dict[str, Any]] = []
    for source_root in selected_roots:
        for path in sorted(source_root.rglob("*")):
            if not path.is_file():
                continue
            if _skip_without_placeholder(path):
                continue
            name = path.relative_to(root).as_posix()
            if _excluded(path, max_file_mb):
                entries[name + ".placeholder.md"] = _placeholder(path, root)
                placeholders.append({"path": name, "size_bytes": path.stat().st_size, "reason": "excluded_from_single_zip"})
            else:
                entries[name] = path.read_bytes()
    manifest_path = final_root / "manifests" / "u3_single_zip_omissions.json"
    write_json(manifest_path, {"schema": "u3_single_zip_omissions_v1", "entries": placeholders})
    entries[manifest_path.relative_to(root).as_posix()] = manifest_path.read_bytes()
    entries["U3_SINGLE_PACKAGE_POLICY.md"] = (
        "The user requested exactly one public ZIP. U3 round manifests, results, and sanitized candidate artifacts are included in the cumulative U0_U1_complete.zip. No additional U3 download ZIP is published. Raw JSONL/NPZ/PT/PTH/Parquet/log payloads, credentials, headers, and reasoning content are excluded or represented by placeholders.\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(entries.items()):
            archive.writestr(name, payload)
    result = _validate_zip(output)
    result.update({"zip": str(output.resolve()), "sha256": sha256_file(output), "secret_scan": "PASS", "single_public_zip": True})
    output.with_suffix(output.suffix + ".sha256").write_text(f"{result['sha256']}  {output.name}\n", encoding="utf-8")
    return result
