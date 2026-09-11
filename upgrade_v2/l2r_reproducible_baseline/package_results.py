from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8")); digest.update(b"\0"); digest.update(path.read_bytes())
    return digest.hexdigest()


def package_manifest(root: Path) -> dict[str, Any]:
    files = [{"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(p for p in root.rglob("*") if p.is_file())]
    return {"schema": "l2rar2_r14b_package_manifest_v1", "root": str(root.resolve()), "files": files, "tree_sha256": tree_sha256(root)}
