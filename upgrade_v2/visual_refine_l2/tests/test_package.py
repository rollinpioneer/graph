import csv
import io
import zipfile

from upgrade_v2.visual_refine_l2.package import package_round


def test_package_preserves_curated_external_artifact_manifest(tmp_path) -> None:
    source = tmp_path / "round"
    manifests = source / "manifests"
    manifests.mkdir(parents=True)
    (source / "summary.md").write_text("ok\n", encoding="utf-8")
    external = tmp_path / "raw_rollout.npz"
    external.write_bytes(b"external")
    (manifests / "large_file_manifest.tsv").write_text(
        "path\tsize_bytes\tartifact_type\tpurpose\treason_omitted\trecovery_method\n"
        f"{external}\t8\traw_rollout\tdynamic evidence\tbinary externalized\trerun locked seed\n",
        encoding="utf-8",
    )
    output = tmp_path / "round.zip"
    package_round(source, output, 200)
    with zipfile.ZipFile(output) as archive:
        text = archive.read("manifests/large_file_manifest.tsv").decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
    assert rows == [{
        "logical_path": str(external),
        "original_path": str(external),
        "original_filename": "raw_rollout.npz",
        "size_bytes": "8",
        "sha256": "directory_manifested_by_component_files",
        "artifact_type": "raw_rollout",
        "purpose": "dynamic evidence",
        "reason_omitted": "binary externalized",
        "recovery_method": "rerun locked seed",
    }]
