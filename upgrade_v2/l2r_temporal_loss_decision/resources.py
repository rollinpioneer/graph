from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _external_path(repo: Path, relative: str) -> Path | None:
    for candidate in (repo / relative, Path(relative)):
        if candidate.exists():
            return candidate.resolve()
    return None


def discover(repo: Path, base: str) -> dict[str, Any]:
    repo = repo.resolve()
    artifact_root = repo / "artifacts/pathgraph_sarm/upgrade_v2"
    roots: dict[str, Path] = {}
    for name, artifact in (("r24", "r24_observation_boundary_closed_loop_v1/results_v1/final_v1/external_data.json"), ("r25", "r25_observability_fusion_recovery_supervisor_v1/results_v1/final_v1/external_data.json")):
        path = artifact_root / artifact
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            discovered = Path(str(value.get("path", "")))
            if discovered.is_dir():
                roots[name] = discovered.resolve()
    records = []
    for name, root in roots.items():
        files = [p for p in root.rglob("*") if p.is_file()]
        records.append({"name": name, "path": str(root), "file_count": len(files), "size_bytes": sum(p.stat().st_size for p in files), "aggregate_sha256": sha256_manifest(root, files)})
    return {"schema": "l2rar2_r27_resources_v1", "base_commit": base, "repository": str(repo), "sources": records, "physical_executions": 0, "mujoco_imported": False}


def sha256_manifest(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: str(item.relative_to(root))):
        digest.update(str(path.relative_to(root)).encode()); digest.update(b"\0"); digest.update(sha256(path).encode()); digest.update(b"\n")
    return digest.hexdigest()
