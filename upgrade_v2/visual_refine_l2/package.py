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


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _pack(source: Path, output: Path, max_file_mb: float, exclude_images: bool = True) -> dict[str, Any]:
    source, output = source.resolve(), output.resolve()
    if not source.is_dir() or source in output.parents:
        raise ValueError("invalid package source/output")
    limit = int(max_file_mb * 1024 * 1024)
    entries: list[tuple[str, Path]] = []
    omitted = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(source).as_posix()
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
                "reason_omitted": reason, "recovery_method": "restore at the exact original path or rerun the locked deterministic command",
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
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative, path in entries:
            content = path.read_bytes()
            archive.writestr(_zip_info(relative), content)
            sums.append(f"{sha256_bytes(content)}  {relative}")
        table = io.StringIO()
        fields = ["logical_path", "original_path", "original_filename", "size_bytes", "sha256", "artifact_type", "reason_omitted", "recovery_method"]
        writer = csv.DictWriter(table, fieldnames=fields, delimiter="\t", lineterminator="\n")
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
    digest = sha256_file(output)
    output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return {"status": "PASS", "zip": str(output), "sha256": digest, "packaged_files": len(entries), "externalized_files": len(omitted), "crc": "PASS", "internal_sha": "PASS"}


def package_round(round_dir: Path, output: Path, max_file_mb: float) -> dict[str, Any]:
    return _pack(round_dir, output, max_file_mb, exclude_images=True)


def package_complete(root: Path, final_root: Path, round_zip_dir: Path, output: Path, max_file_mb: float) -> dict[str, Any]:
    stage = root / "complete_release"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for source, destination in ((final_root, stage / "final_v1"), (root / "protocol_v1", stage / "protocol_v1"), (root / "coarse_graph_v1", stage / "coarse_graph_v1"), (root / "observable_predicates_v1", stage / "observable_predicates_v1"), (root / "refined_graphs_v1", stage / "refined_graphs_v1"), (root / "fresh_confirmation_v1", stage / "fresh_confirmation_v1")):
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
    packages = []
    for path in sorted(round_zip_dir.glob("l2r_*.zip")):
        if path.name == output.name:
            continue
        packages.append({"filename": path.name, "path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "purpose": "L2R round delivery; restore from local downloads/l2r"})
    write_json(stage / "round_package_index.json", {"schema": "pathgraph_l2r_round_package_index_v1", "created_at": now_iso(), "round_packages": packages})
    result = _pack(stage, output, max_file_mb, exclude_images=True)
    result["round_packages"] = len(packages)
    return result
