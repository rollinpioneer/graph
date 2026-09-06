"""Create auditable U4R1 round and complete ZIPs without raw large payloads."""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Any

from .io import sha256_file, write_csv, write_json


def _zip(root: Path, output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() in {".pt", ".pth", ".npz", ".parquet", ".zip", ".pyc"}:
                continue
            if path.name.endswith("occurrences.jsonl") or path.name.endswith("boundaries.jsonl") or "rollouts" in path.parts:
                continue
            if path.stat().st_size > 200 * 1024 * 1024:
                continue
            archive.write(path, path.relative_to(root).as_posix())
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failure: {bad}")
    return {"path": str(output.resolve()), "sha256": sha256_file(output), "size_bytes": output.stat().st_size}


def package_round(root: Path, output: Path) -> dict[str, Any]:
    result = _zip(root, output)
    write_json(output.with_suffix(".sha256.json"), result)
    return {"status": "PASS", **result}


def package_complete(root: Path, output: Path, rounds: list[Path]) -> dict[str, Any]:
    index = [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in rounds if path.is_file()]
    write_json(root / "round_index.json", {"schema": "u4r1_round_index_v1", "rounds": index})
    result = _zip(root, output)
    write_json(output.with_suffix(".sha256.json"), {**result, "rounds": index})
    return {"status": "PASS", **result, "rounds": index}
