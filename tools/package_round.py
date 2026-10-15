#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


SKIP_SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".ckpt", ".safetensors"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package(source: Path, output: Path, max_file_mb: float) -> None:
    source = source.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing package: {output}")
    files = []
    for path in sorted(source.rglob("*")):
        if "__pycache__" in path.parts or not path.is_file() or path.suffix.lower() in SKIP_SUFFIXES or path.stat().st_size > max_file_mb * 1024 * 1024:
            continue
        files.append(path)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(source))
    if zipfile.ZipFile(output).testzip() is not None:
        raise RuntimeError("ZIP CRC test failed")
    output.with_suffix(output.suffix + ".sha256").write_text(f"{sha256(output)}  {output.name}\n", encoding="utf-8")


def verify(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        names = archive.namelist()
    return {"zip": str(path.resolve()), "crc_ok": bad is None, "files": len(names), "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("package"); p.add_argument("--source", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--max-file-mb", type=float, default=200)
    p = sub.add_parser("verify"); p.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "package":
        package(args.source, args.output, args.max_file_mb); print(json.dumps(verify(args.output), indent=2)); return 0
    print(json.dumps(verify(args.zip), indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
