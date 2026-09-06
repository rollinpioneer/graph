"""Normalize only the two reusable U3 semantic hypotheses."""

from pathlib import Path

from .common import read_csv, read_json, sha256_file, write_csv

ALLOWED = {"qwen:instruction_only_r01", "qwen:instruction_only_r02"}


def normalize_semantic_candidates(*, hard_checks: Path, candidate_root: Path, candidate_ids: list[str], output_dir: Path, manifest: Path) -> dict:
    checks = {row["candidate_id"]: row for row in read_csv(hard_checks)}
    rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for cid in candidate_ids:
        if cid not in ALLOWED or checks.get(cid, {}).get("hard_valid", "").lower() not in {"true", "1"}:
            raise ValueError(f"candidate is not an allowed hard-valid semantic source: {cid}")
        source = candidate_root / (cid.split(":", 1)[1] + ".json")
        if not source.is_file():
            raise FileNotFoundError(source)
        target = output_dir / source.name
        target.write_bytes(source.read_bytes())
        graph = read_json(source)
        rows.append({"source_candidate_id": cid, "normalized_path": str(target), "immutable_source_sha256": sha256_file(source), "node_count": len(graph["nodes"]), "edge_count": len(graph["edges"]), "semantic_content_unchanged": True})
    write_csv(manifest, rows)
    return {"status": "PASS", "source_count": len(rows), "source_candidates": [row["source_candidate_id"] for row in rows]}
