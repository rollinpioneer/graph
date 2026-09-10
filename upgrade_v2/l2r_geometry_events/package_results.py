"""Build the R15-A results ZIP from an explicit whitelist.

Whitelist first, manifest second, ZIP third: every member is hashed and sized
before packing, and the archive must contain exactly the whitelisted files plus
the manifest itself.  Byte-code, nested archives, credentials, models, raw data
and ``.git`` content are never included.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ALLOWED_SUFFIXES = (".py", ".json", ".jsonl", ".csv", ".md")
FORBIDDEN_PARTS = ("__pycache__", ".git")
FORBIDDEN_SUFFIXES = (".pyc", ".zip", ".tar", ".gz", ".pem", ".key", ".pkl", ".pt", ".ckpt")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect(worktree: Path, module_dir: str, artifacts_dir: str, run_dir: str) -> list[Path]:
    files: list[Path] = []
    module_root = worktree / module_dir
    for path in sorted(module_root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            files.append(path)
    run_root = worktree / artifacts_dir / run_dir
    files.append(worktree / artifacts_dir / "resource_resolution.json")
    for path in sorted(run_root.rglob("*")):
        if path.is_file():
            files.append(path)
    return files


def _guard(path: Path, worktree: Path) -> None:
    if path.is_symlink():
        raise SystemExit(f"REFUSING_SYMLINK {path}")
    if any(part in FORBIDDEN_PARTS for part in path.parts):
        raise SystemExit(f"REFUSING_FORBIDDEN_PATH {path}")
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise SystemExit(f"REFUSING_FORBIDDEN_SUFFIX {path}")
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise SystemExit(f"REFUSING_UNLISTED_SUFFIX {path}")
    resolved = path.resolve()
    if not str(resolved).startswith(str(worktree.resolve())):
        raise SystemExit(f"REFUSING_PATH_ESCAPE {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="package_results")
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--module-dir", default="upgrade_v2/l2r_geometry_events")
    parser.add_argument("--artifacts-dir", default="artifacts/pathgraph_sarm/upgrade_v2/geometry_events_l2rar2_v1")
    parser.add_argument("--run-dir", default="cache_geometry_v1")
    args = parser.parse_args(argv)

    worktree = Path(args.worktree).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = worktree / output
    if output.exists():
        raise SystemExit(f"REFUSING_EXISTING_OUTPUT {output}")

    files = _collect(worktree, args.module_dir, args.artifacts_dir, args.run_dir)
    entries = []
    seen = set()
    for path in files:
        _guard(path, worktree)
        relative = str(path.resolve().relative_to(worktree)).replace("\\", "/")
        if relative in seen:
            raise SystemExit(f"REFUSING_DUPLICATE_MEMBER {relative}")
        seen.add(relative)
        entries.append({"path": relative, "sha256": _sha256(path), "size_bytes": path.stat().st_size})

    manifest = {
        "schema": "l2rar2_geometry_events_results_manifest_v1",
        "entry_commit": "93f89d49317a026db09f21acc8686eb82cd5c668",
        "file_count": len(entries),
        "files": entries,
        "r14_physical_executions": 0,
        "model_api_calls": 0,
        "key_reads": 0,
    }
    manifest_relative = f"{args.artifacts_dir}/{args.run_dir}/package_manifest.json"
    manifest_path = worktree / manifest_relative
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_sha = _sha256(manifest_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for entry in entries:
            archive.write(worktree / entry["path"], arcname=entry["path"])
        archive.write(manifest_path, arcname=manifest_relative)

    with zipfile.ZipFile(output) as archive:
        members = sorted(archive.namelist())
    expected = sorted([entry["path"] for entry in entries] + [manifest_relative])
    if members != expected:
        raise SystemExit("MEMBER_SET_MISMATCH")

    zip_sha = _sha256(output)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{zip_sha}  {output.name}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PACKAGED",
                "members": len(members),
                "zip": str(output),
                "zip_sha256": zip_sha,
                "manifest": manifest_relative,
                "manifest_sha256": manifest_sha,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
