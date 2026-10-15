"""Materialize sibling placeholders and refresh the repository omission index."""
from __future__ import annotations

from pathlib import Path


EXCLUDED_SUFFIXES = {".npz", ".pt", ".pth", ".parquet", ".jsonl", ".log"}
MAX_FILE_BYTES = 20 * 1024 * 1024


def _file_type(path: Path) -> tuple[str, str]:
    if path.suffix == ".npz":
        return "numeric_array", "raw trajectory, event, or boundary-prediction array"
    if path.suffix in {".pt", ".pth"}:
        return "checkpoint_or_model_weight", "trained boundary, correction, or value-model checkpoint"
    if path.suffix == ".parquet":
        return "tabular_embedding", "segment embedding or derived tabular representation"
    if path.suffix == ".jsonl":
        return "event_or_trajectory_records", "event candidates, segment records, or continuation records"
    if path.suffix == ".log":
        return "experiment_log", "experiment or job execution log"
    return "large_generated_artifact", "generated experiment artifact"


def materialize(repo: Path, u2_root: Path, manifest: Path) -> dict[str, int]:
    rows: dict[str, str] = {}
    if manifest.is_file():
        lines = manifest.read_text(encoding="utf-8").splitlines()
        for line in lines[1:]:
            if line.strip():
                rows[line.split("\t", 1)[0]] = line

    created = 0
    candidates = 0
    for path in sorted(u2_root.rglob("*")):
        if not path.is_file() or path.name.endswith(".placeholder.md"):
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.suffix not in EXCLUDED_SUFFIXES and path.stat().st_size <= MAX_FILE_BYTES:
            continue
        candidates += 1
        relative = path.relative_to(repo).as_posix()
        placeholder = path.with_name(path.name + ".placeholder.md")
        file_type, intended_use = _file_type(path)
        placeholder.write_text(
            "# Omitted file placeholder\n\n"
            f"- Original filename: `{path.name}`\n"
            f"- Original relative path: `{relative}`\n"
            f"- File type: `{file_type}`\n"
            f"- Intended use: {intended_use}\n"
            f"- Original size: {path.stat().st_size} bytes\n"
            "- Omission reason: excluded from GitHub snapshot and the single ZIP to keep the delivery lightweight\n"
            f"- Restore: recover the original file from the experiment workspace or external artifact storage, then place it at `{relative}`.\n"
            "- Source of truth: `LARGE_FILES_OMITTED.tsv` at repository root.\n",
            encoding="utf-8",
        )
        rows[relative] = "\t".join(
            [relative, path.name, str(path.stat().st_size), file_type, placeholder.relative_to(repo).as_posix()]
        )
        created += 1

    header = "source_relative_path\toriginal_filename\tsize_bytes\tfile_type\tplaceholder\n"
    manifest.write_text(header + "\n".join(rows[key] for key in sorted(rows)) + "\n", encoding="utf-8")
    return {"u2_omitted_candidates": candidates, "u2_placeholders_written": created, "manifest_rows": len(rows)}


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[2]
    root = repository / "artifacts/pathgraph_sarm/upgrade_v2/u2_stochastic_boundary"
    index = repository / "LARGE_FILES_OMITTED.tsv"
    print(materialize(repository, root, index))
