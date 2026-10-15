"""Deterministic L2R ZIP creation with hashes and external artifact records."""

from __future__ import annotations

import csv
import io
import shutil
import zipfile
from pathlib import Path
from typing import Any

from .io import SECRET_RE, now_iso, sha256_bytes, sha256_file, write_json


EXTERNAL_SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".mp4", ".avi", ".mov", ".zip"}
EXTERNAL_FIELDS = [
    "logical_path", "original_path", "original_filename", "size_bytes", "sha256",
    "artifact_type", "purpose", "reason_omitted", "recovery_method",
]

# These files are generated after packaging or contain hashes of the package
# being built. Keeping them out of the archive avoids a self-referential ZIP
# whose contents can never agree with its recorded digest.
DERIVED_DELIVERY_NAMES = {
    "manual_completion_audit.json",
    "manual_completion_audit.md",
    "zip_internal_verification.json",
    "zip_internal_verification.log",
    "package_round_completion_audit.log",
    "package_complete_completion_audit.log",
}


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _normalized_external_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    normalized = []
    for row in rows:
        original = row.get("original_path") or row.get("path") or row.get("logical_path") or ""
        normalized.append({
            "logical_path": row.get("logical_path") or original,
            "original_path": original,
            "original_filename": row.get("original_filename") or Path(original).name,
            "size_bytes": row.get("size_bytes") or "",
            "sha256": row.get("sha256") or "directory_manifested_by_component_files",
            "artifact_type": row.get("artifact_type") or "external_runtime_artifact",
            "purpose": row.get("purpose") or row.get("artifact_type") or "external runtime artifact",
            "reason_omitted": row.get("reason_omitted") or "externalized_by_protocol",
            "recovery_method": row.get("recovery_method") or "restore at the original path",
        })
    return normalized


def _pack(source: Path, output: Path, max_file_mb: float, exclude_images: bool = True) -> dict[str, Any]:
    source, output = source.resolve(), output.resolve()
    if not source.is_dir() or source in output.parents:
        raise ValueError("invalid package source/output")
    limit = int(max_file_mb * 1024 * 1024)
    entries: list[tuple[str, Path]] = []
    external_manifest = source / "manifests/large_file_manifest.tsv"
    omitted = _normalized_external_rows(external_manifest)
    for path in sorted(source.rglob("*")):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(source).as_posix()
        if relative == "manifests/large_file_manifest.tsv":
            continue
        if "checksums" in path.relative_to(source).parts or path.name in DERIVED_DELIVERY_NAMES:
            continue
        lower_parts = {part.lower() for part in path.relative_to(source).parts}
        if path.name.startswith(".env") or lower_parts.intersection({"secret", "secrets"}):
            raise RuntimeError(f"secret-like path rejected: {relative}")
        reason = None
        if path.suffix.lower() in EXTERNAL_SUFFIXES:
            reason = "binary_or_raw_payload_externalized"
        elif exclude_images and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            reason = "per_frame_image_externalized"
        elif path.stat().st_size > limit:
            reason = "over_size_threshold"
        if reason:
            omitted.append({
                "logical_path": relative, "original_path": str(path), "original_filename": path.name,
                "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "artifact_type": "L2R runtime artifact",
                "purpose": "source artifact omitted from this ZIP", "reason_omitted": reason,
                "recovery_method": "restore at the exact original path or rerun the locked deterministic command",
            })
        else:
            data = path.read_bytes()
            if SECRET_RE.search(data):
                raise RuntimeError(f"possible credential in {relative}")
            entries.append((relative, path))
    if not entries:
        raise ValueError("empty package")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    sums = []
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for relative, path in entries:
                content = path.read_bytes()
                archive.writestr(_zip_info(relative), content)
                sums.append(f"{sha256_bytes(content)}  {relative}")
            table = io.StringIO()
            writer = csv.DictWriter(table, fieldnames=EXTERNAL_FIELDS, delimiter="\t", lineterminator="\n")
            writer.writeheader(); writer.writerows(omitted)
            manifest = table.getvalue().encode("utf-8")
            archive.writestr(_zip_info("manifests/large_file_manifest.tsv"), manifest)
            sums.append(f"{sha256_bytes(manifest)}  manifests/large_file_manifest.tsv")
            archive.writestr(_zip_info("PACKAGE_SHA256SUMS.txt"), "\n".join(sums) + "\n")
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("ZIP CRC validation failed")
            for line in archive.read("PACKAGE_SHA256SUMS.txt").decode("utf-8").splitlines():
                digest, relative = line.split("  ", 1)
                if sha256_bytes(archive.read(relative)) != digest:
                    raise RuntimeError(f"internal SHA mismatch: {relative}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    digest = sha256_file(output)
    output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return {"status": "PASS", "zip": str(output), "sha256": digest, "packaged_files": len(entries), "externalized_files": len(omitted), "crc": "PASS", "internal_sha": "PASS"}


def package_round(round_dir: Path, output: Path, max_file_mb: float) -> dict[str, Any]:
    return _pack(round_dir, output, max_file_mb, exclude_images=True)


def _copy_if_present(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    elif source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def package_complete(root: Path, final_root: Path, round_zip_dir: Path, output: Path, max_file_mb: float) -> dict[str, Any]:
    stage = root / "complete_release"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    repo = root.parents[3]
    selections = (
        (repo / "upgrade_v2/visual_refine_l2", stage / "upgrade_v2/visual_refine_l2"),
        (final_root, stage / "final_v1"),
        (root / "protocol_v1", stage / "protocol_v1"),
        (root / "coarse_graph_v1", stage / "coarse_graph_v1"),
        (root / "dynamic_dataset_v1/configs", stage / "dynamic_dataset_v1/configs"),
        (root / "dynamic_dataset_v1/manifests", stage / "dynamic_dataset_v1/manifests"),
        (root / "dynamic_dataset_v1/reports", stage / "dynamic_dataset_v1/reports"),
        (root / "observable_predicates_v1/configs", stage / "observable_predicates_v1/configs"),
        (root / "observable_predicates_v1/locks", stage / "observable_predicates_v1/locks"),
        (root / "observable_predicates_v1/reports", stage / "observable_predicates_v1/reports"),
        (root / "refined_graphs_v1/compiled", stage / "refined_graphs_v1/compiled"),
        (root / "refined_graphs_v1/candidates/G2_evidence_refined.json", stage / "refined_graphs_v1/candidates/G2_evidence_refined.json"),
        (root / "refined_graphs_v1/candidates/G3_active_second_view.json", stage / "refined_graphs_v1/candidates/G3_active_second_view.json"),
        (root / "refined_graphs_v1/edit_logs", stage / "refined_graphs_v1/edit_logs"),
        (root / "refined_graphs_v1/reports", stage / "refined_graphs_v1/reports"),
        (root / "refined_graphs_v1/selection", stage / "refined_graphs_v1/selection"),
        (root / "fresh_confirmation_v1/data/rollout_manifest.csv", stage / "fresh_confirmation_v1/data/rollout_manifest.csv"),
        (root / "fresh_confirmation_v1/predicates/prediction_manifest.csv", stage / "fresh_confirmation_v1/predicates/prediction_manifest.csv"),
        (root / "fresh_confirmation_v1/evaluation", stage / "fresh_confirmation_v1/evaluation"),
        (root / "fresh_confirmation_v1/locks", stage / "fresh_confirmation_v1/locks"),
        (root / "fresh_confirmation_v1/reports", stage / "fresh_confirmation_v1/reports"),
        (root / "implementation_correction", stage / "implementation_correction"),
    )
    for source, destination in selections:
        _copy_if_present(source, destination)
    for round_dir in sorted((root / "rounds").glob("l2r_*")):
        for name in ("run_manifest.json", "run_manifest.md", "summary.md"):
            _copy_if_present(round_dir / name, stage / "rounds" / round_dir.name / name)
        for manifest in sorted((round_dir / "manifests").glob("*.tsv")):
            _copy_if_present(manifest, stage / "rounds" / round_dir.name / "manifests" / manifest.name)
    packages = []
    for path in sorted(round_zip_dir.glob("l2r_*.zip")):
        if path.name == output.name:
            continue
        packages.append({"filename": path.name, "path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "purpose": "L2R round delivery; restore from local downloads/l2r"})
    write_json(stage / "round_package_index.json", {"schema": "pathgraph_l2r_round_package_index_v1", "created_at": now_iso(), "round_packages": packages})
    result = _pack(stage, output, max_file_mb, exclude_images=True)
    result["round_packages"] = len(packages)
    return result
