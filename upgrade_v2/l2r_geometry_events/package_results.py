"""Build the R15-B results ZIP from an explicit whitelist.

Whitelist first, manifest second, ZIP third: every member is hashed and sized
before packing and the archive must contain exactly the whitelisted files plus
the manifest.  Byte-code, raw rollouts, RGB, keys, nested archives and ``.git``
content are never included.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import zipfile
from pathlib import Path

ALLOWED_SUFFIXES = (".py", ".json", ".jsonl", ".csv", ".md", ".txt", ".tsv")
FORBIDDEN_PARTS = ("__pycache__", ".git", "rollouts", "rgb")
FORBIDDEN_SUFFIXES = (".pyc", ".zip", ".tar", ".gz", ".pem", ".key", ".pkl", ".pt", ".ckpt", ".jpg")

README = """# L2RAR2 R15-B geometry-event fair re-evaluation - results package

Contents: the corrected geometry-event module snapshot, the R15-B cache
re-evaluation outputs (O-layer parity, weld transition audit, drift anchors,
causal audits, metrics), and the prepared R14 application materials.

This package contains no raw rollouts, no RGB frames, no models, no keys and no
physical execution results.  R14 was NOT run: authorisation is absent.

Entry commit: {entry_commit}
Generated at (UTC): {generated}
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect(worktree: Path, includes: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in includes:
        root = worktree / item
        if root.is_file():
            files.append(root)
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                files.append(path)
    return files


def _guard(path: Path, worktree: Path) -> None:
    if path.is_symlink():
        raise SystemExit("REFUSING_SYMLINK " + str(path))
    if any(part in FORBIDDEN_PARTS for part in path.parts):
        raise SystemExit("REFUSING_FORBIDDEN_PATH " + str(path))
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise SystemExit("REFUSING_FORBIDDEN_SUFFIX " + str(path))
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise SystemExit("REFUSING_UNLISTED_SUFFIX " + str(path))
    if not str(path.resolve()).startswith(str(worktree.resolve())):
        raise SystemExit("REFUSING_PATH_ESCAPE " + str(path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="package_results")
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include", action="append", required=True)
    parser.add_argument("--readme-path")
    parser.add_argument("--entry-commit", default="62bd073597a13a74af5b5e30818337c3dbe13433")
    args = parser.parse_args(argv)

    worktree = Path(args.worktree).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = worktree / output
    if output.exists():
        raise SystemExit("REFUSING_EXISTING_OUTPUT " + str(output))

    if args.readme_path:
        readme = worktree / args.readme_path
        readme.write_text(
            README.format(
                entry_commit=args.entry_commit,
                generated=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            ),
            encoding="utf-8",
        )

    entries = []
    seen = set()
    for path in _collect(worktree, args.include):
        _guard(path, worktree)
        relative = str(path.resolve().relative_to(worktree)).replace("\\", "/")
        if relative in seen:
            raise SystemExit("REFUSING_DUPLICATE_MEMBER " + relative)
        seen.add(relative)
        entries.append({"path": relative, "sha256": _sha256(path), "size_bytes": path.stat().st_size})

    manifest = {
        "schema": "l2rar2_r15b_results_manifest_v1",
        "entry_commit": args.entry_commit,
        "file_count": len(entries),
        "files": entries,
        "physical_executions": 0,
        "training_jobs": 0,
        "model_api_calls": 0,
        "api_key_reads": 0,
    }
    manifest_relative = args.include[0].rstrip("/") + "/package_manifest.json"
    manifest_path = worktree / manifest_relative
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

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
    sidecar.write_text("%s  %s\n" % (zip_sha, output.name), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PACKAGED",
                "members": len(members),
                "zip": str(output),
                "zip_sha256": zip_sha,
                "manifest": manifest_relative,
                "manifest_sha256": _sha256(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
