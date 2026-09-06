"""Deterministic ZIP and external-artifact tooling for U4 B+ delivery."""
from __future__ import annotations
import argparse, csv, hashlib, io, json, re, zipfile
from datetime import datetime, timezone
from pathlib import Path

BINARY_EXTERNAL = {".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".npz", ".npy", ".parquet", ".mp4", ".avi", ".mov", ".zip", ".gz"}
SECRET_RE = re.compile(rb"(?:sk-[A-Za-z0-9._-]{20,}|Authorization:\s*Bearer\s+\S+)", re.I)
RAW_PATTERNS = (
    "data/development/*.json", "data/confirmation/*.json", "data/reconfirmation/*.json",
    "evidence/*.jsonl", "queries/*continuations.jsonl", "queries/dev_anchors.jsonl", "diagnostics/*.jsonl",
    "torch_recovery_v2/data/confirmation/*.json", "torch_recovery_v2/evidence/*.jsonl",
    "torch_recovery_v2/predictions/*.jsonl", "torch_recovery_v2/queries/*.jsonl",
    "torch_recovery_v2/diagnostics/*.jsonl",
)
ROUND_SPECS = {
    "u4b_0_entry_and_protocol": {"jobs": [2, 0, 0], "denominators": {"families_locked": 36}, "next_action": "build semantic evidence", "commands": ["prepare-entry", "plan-families"]},
    "u4b_1_semantic_evidence": {"jobs": [3, 0, 0], "denominators": {"legacy_occurrences": 2147, "query_classes": 5}, "next_action": "run development diagnosis", "commands": ["build-occurrences", "propose-semantics", "plan-queries"]},
    "u4b_2_development_diagnosis": {"jobs": [5, 0, 0], "denominators": {"families": 24, "base_rollouts": 96, "continuations": 72, "high_impact_events": 514}, "next_action": "continue with fallback and lock graph", "commands": ["collect-rollouts", "map-new-rollouts", "select-anchors", "run-continuations", "build-diagnostic-cases", "decide-development-route"]},
    "u4b_3_graph_lock": {"jobs": [5, 0, 0], "denominators": {"proposals": 4, "accepted_edits": 1, "edit_budget": 6}, "next_action": "open locked confirmation", "commands": ["fit-final-semantics", "propose-edits", "select-edits", "resolve-input-selection", "freeze-final-pipeline"]},
    "u4b_4_confirmation": {"jobs": [4, 0, 0], "denominators": {"families": 12, "base_rollouts": 48, "continuations": 96}, "next_action": "exclude contaminated confirmation and execute separately locked reconfirmation", "commands": ["verify-final-lock", "collect-rollouts", "confirm-claims", "evaluate-final-graphs"], "scientific_use": "excluded_contaminated_implementation_correction"},
    "u4b_4_reconfirmation": {"jobs": [5, 0, 0], "denominators": {"families": 12, "base_rollouts": 48, "mapped_occurrences": 1170, "continuations": 15}, "next_action": "finalize U4 B+", "commands": ["verify-final-lock", "collect-rollouts", "map-new-rollouts", "confirm-claims", "evaluate-final-graphs", "finalize"], "scientific_use": "valid_final_confirmation"},
    "u4b_2b_torch_recovery": {"jobs": [7, 0, 0], "denominators": {"families": 24, "base_rollouts_reused": 96, "transitions_inferred": 2359, "auto_boundary_occurrences": 1074, "diagnostic_cases": 446}, "next_action": "freeze the recovered-boundary pipeline and open a new confirmation extension", "commands": ["infer-auto-boundaries", "diagnose-recovered-boundaries", "decide-development-route", "map-new-rollouts", "fit-final-semantics", "propose-edits", "select-edits"], "scientific_use": "post_confirmation_development_protocol_extension"},
    "u4b_4b_torch_recovery_confirmation": {"jobs": [7, 0, 0], "denominators": {"families": 12, "base_rollouts": 48, "transitions_inferred": 1193, "auto_boundary_occurrences": 528, "continuations": 0}, "next_action": "retain G1 and the negative edit result; do not start another search round", "commands": ["plan-confirmation-extension", "lock-recovered-boundary", "freeze-final-pipeline", "verify-final-lock", "collect-rollouts", "infer-auto-boundaries", "map-new-rollouts", "confirm-claims", "evaluate-final-graphs", "finalize-torch-recovery"], "scientific_use": "valid_torch_recovery_confirmation"},
}
ROUND_FILES = {
    "u4b_0_entry_and_protocol": {
        "protocol/protocol.yaml": "configs/protocol.yaml", "protocol/input_index.json": "configs/input_index.json",
        "protocol/family_split_lock.json": "configs/family_split_lock.json", "protocol/claims.jsonl": "tables/claims.jsonl",
        "data/family_plan.jsonl": "tables/family_plan.jsonl", "graphs/G0_raw_topology.json": "tables/G0_raw_topology.json",
    },
    "u4b_1_semantic_evidence": {
        "evidence/node_roles_train.csv": "tables/node_roles_train.csv", "evidence/edge_semantics_train.csv": "tables/edge_semantics_train.csv",
        "queries/query_plan.jsonl": "tables/query_plan.jsonl", "graphs/G1_pre_diagnosis.json": "tables/G1_pre_diagnosis.json",
        "evidence/legacy_occurrences.jsonl.placeholder.md": "manifests/legacy_occurrences.jsonl.placeholder.md",
    },
    "u4b_2_development_diagnosis": {
        "diagnostics/diagnostic_metrics.json": "metrics/diagnostic_metrics.json", "diagnostics/development_route.json": "metrics/development_route.json",
        "queries/query_plan.jsonl": "tables/query_plan.jsonl", "evidence/dev_occurrences.jsonl.placeholder.md": "manifests/dev_occurrences.jsonl.placeholder.md",
        "queries/dev_continuations.jsonl.placeholder.md": "manifests/dev_continuations.jsonl.placeholder.md",
        "queries/dev_anchors.jsonl.placeholder.md": "manifests/dev_anchors.jsonl.placeholder.md",
        "diagnostics/diagnostic_cases.jsonl.placeholder.md": "manifests/diagnostic_cases.jsonl.placeholder.md",
    },
    "u4b_3_graph_lock": {
        "graphs/G0_raw_topology.json": "tables/G0_raw_topology.json", "graphs/G1_semantic_only.json": "tables/G1_semantic_only.json",
        "graphs/G2_evidence_edited.json": "tables/G2_evidence_edited.json", "graphs/edit_proposals.jsonl": "tables/edit_proposals.jsonl",
        "protocol/selected_input_pipeline.json": "configs/selected_input_pipeline.json", "protocol/final_pipeline_lock.json": "configs/final_pipeline_lock.json",
    },
    "u4b_4_confirmation": {
        "protocol/confirmation_contamination.json": "metrics/confirmation_contamination.json",
        "protocol/reconfirmation_protocol.yaml": "configs/reconfirmation_protocol.yaml",
    },
    "u4b_4_reconfirmation": {
        "protocol/reconfirmation_family_lock.json": "configs/reconfirmation_family_lock.json",
        "protocol/final_pipeline_lock.json": "configs/final_pipeline_lock.json",
        "evaluation/confirmation_metrics.csv": "metrics/confirmation_metrics.csv",
        "evaluation/G2_minus_G1_effects.csv": "metrics/G2_minus_G1_effects.csv",
        "evaluation/confirmation_by_family.csv": "tables/confirmation_by_family.csv",
        "evaluation/claim_confirmation.csv": "tables/claim_confirmation.csv",
        "final/u4b_final_handoff.json": "reports/u4b_final_handoff.json",
        "evidence/confirmation_occurrences.jsonl.placeholder.md": "manifests/confirmation_occurrences.jsonl.placeholder.md",
        "queries/confirmation_continuations.jsonl.placeholder.md": "manifests/confirmation_continuations.jsonl.placeholder.md",
    },
    "u4b_2b_torch_recovery": {
        "torch_recovery_v2/protocol/inference_manifest.json": "configs/inference_manifest.json",
        "torch_recovery_v2/protocol/dev_mapping_manifest.csv": "configs/dev_mapping_manifest.csv",
        "torch_recovery_v2/diagnostics/diagnostic_metrics.json": "metrics/diagnostic_metrics.json",
        "torch_recovery_v2/diagnostics/development_route.json": "metrics/development_route.json",
        "torch_recovery_v2/diagnostics/diagnostic_by_family.csv": "tables/diagnostic_by_family.csv",
        "torch_recovery_v2/graphs/G1_semantic_only.json": "tables/G1_semantic_only.json",
        "torch_recovery_v2/graphs/G2_evidence_edited.json": "tables/G2_evidence_edited.json",
        "torch_recovery_v2/graphs/accepted_rejected_edits.csv": "tables/accepted_rejected_edits.csv",
        "manifests/external_artifacts.tsv": "manifests/external_artifacts.tsv",
    },
    "u4b_4b_torch_recovery_confirmation": {
        "torch_recovery_v2/protocol/protocol.yaml": "configs/protocol.yaml",
        "torch_recovery_v2/protocol/selected_input_pipeline.json": "configs/selected_input_pipeline.json",
        "torch_recovery_v2/protocol/confirmation_family_lock.json": "configs/confirmation_family_lock.json",
        "torch_recovery_v2/protocol/final_pipeline_lock.json": "configs/final_pipeline_lock.json",
        "torch_recovery_v2/protocol/confirmation_inference_manifest.json": "configs/confirmation_inference_manifest.json",
        "torch_recovery_v2/protocol/confirmation_mapping_manifest.csv": "configs/confirmation_mapping_manifest.csv",
        "torch_recovery_v2/evaluation/confirmation_metrics.csv": "metrics/confirmation_metrics.csv",
        "torch_recovery_v2/evaluation/G2_minus_G1_effects.csv": "metrics/G2_minus_G1_effects.csv",
        "torch_recovery_v2/evaluation/confirmation_by_family.csv": "tables/confirmation_by_family.csv",
        "torch_recovery_v2/evaluation/claim_confirmation.csv": "tables/claim_confirmation.csv",
        "torch_recovery_v2/final/u4b_final_handoff.json": "reports/u4b_final_handoff.json",
        "manifests/external_artifacts.tsv": "manifests/external_artifacts.tsv",
    },
}

def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info

def external_inventory(root: Path, output: Path) -> dict:
    root = root.resolve(); rows = []
    candidates = {path.resolve() for pattern in RAW_PATTERNS for path in root.glob(pattern) if path.is_file()}
    for path in sorted(candidates):
        logical = path.relative_to(root).as_posix()
        rows.append({
            "logical_path": logical, "original_path": str(path), "original_filename": path.name,
            "logical_id": path.stem, "size_bytes": path.stat().st_size, "sha256": sha_file(path),
            "artifact_type": "external_raw_payload", "packaged": "false",
            "dependency_versions": "u4_bplus_v1; simulator=upgrade_v2/u2/simulator.py",
            "purpose": "raw rollout, occurrence, continuation, or diagnostic evidence",
            "recovery_method": "restore this exact local path or rerun from the locked family/rollout/continuation seeds",
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["logical_path", "original_path", "original_filename", "logical_id", "size_bytes", "sha256", "artifact_type", "packaged", "dependency_versions", "purpose", "recovery_method"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    return {"status": "PASS", "output": str(output), "external_file_count": len(rows), "external_bytes": sum(row["size_bytes"] for row in rows)}

def round_manifests(root: Path, download_dir: Path, commit: str, round_ids: str | None = None) -> dict:
    rounds = root.resolve() / "rounds"; written = []
    selected = set(round_ids.split(",")) if round_ids else set(ROUND_SPECS)
    for round_id, spec in ROUND_SPECS.items():
        if round_id not in selected: continue
        round_dir = rounds / round_id
        round_dir.mkdir(parents=True, exist_ok=True)
        files = [path for path in round_dir.rglob("*") if path.is_file() and path.name != "run_manifest.json"]
        mtimes = [path.stat().st_mtime for path in files]
        start = datetime.fromtimestamp(min(mtimes), timezone.utc).isoformat() if mtimes else None
        end = datetime.fromtimestamp(max(mtimes), timezone.utc).isoformat() if mtimes else None
        command_dir = round_dir / "commands"; command_dir.mkdir(exist_ok=True)
        command_text = "#!/bin/sh\n# Sanitized command names only; no shell history or credentials.\n" + "\n".join(f"# python -m upgrade_v2.u4_bplus.cli {name} [locked arguments]" for name in spec["commands"]) + "\n"
        command_path = command_dir / "executed_commands.sh"; command_path.write_text(command_text, encoding="utf-8")
        successful, failed, skipped = spec["jobs"]
        manifest = {
            "schema": "u4b_run_manifest_v1", "round_id": round_id,
            "start_time": start, "end_time": end, "time_provenance": "filesystem_mtime_reconstruction",
            "recorded_at": datetime.now(timezone.utc).isoformat(), "commit_at_execution": commit,
            "commands_sha256": sha_file(command_path), "commands": spec["commands"],
            "config_sha256": sha_file(root / "protocol" / "protocol.yaml"),
            "input_version": "u4_bplus_v1", "jobs_run": successful, "jobs_failed": failed,
            "jobs_skipped": skipped, "actual_denominators": spec["denominators"],
            "scientific_scope": "same_explicit_stochastic_simulator_distribution",
            "scientific_use": spec.get("scientific_use", "development_or_lock_only"),
            "output_package": str((download_dir / f"{round_id}.zip").resolve()),
            "package_checksum_record": str((download_dir / f"{round_id}.zip.sha256").resolve()),
            "next_action": spec["next_action"], "manifest_reconstructed_after_execution": True,
        }
        (round_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(round_id)
    return {"status": "PASS", "round_count": len(written), "rounds": written}

def stage_rounds(root: Path) -> dict:
    import shutil
    root = root.resolve(); copied = []
    for round_id, mappings in ROUND_FILES.items():
        for source_name, target_name in mappings.items():
            source = root / source_name
            if not source.is_file():
                raise FileNotFoundError(source)
            target = root / "rounds" / round_id / target_name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target); copied.append(f"{round_id}/{target_name}")
    return {"status": "PASS", "copied_file_count": len(copied), "files": copied}

def package_index(download_dir: Path, output: Path, legacy_zip: Path) -> dict:
    packages = []
    for path in sorted(download_dir.resolve().glob("u4b_*.zip")):
        row = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha_file(path)}
        if path.name == "u4b_4_confirmation.zip":
            row["scientific_use"] = "excluded_contaminated_implementation_correction"
        elif path.name == "u4b_4_reconfirmation.zip":
            row["scientific_use"] = "valid_final_confirmation"
        elif path.name == "u4b_4b_torch_recovery_confirmation.zip":
            row["scientific_use"] = "valid_torch_recovery_confirmation"
        packages.append(row)
    legacy = legacy_zip.resolve()
    payload = {
        "schema": "u4b_round_package_index_v1", "round_packages": packages,
        "legacy_zip": {"path": str(legacy), "size_bytes": legacy.stat().st_size,
                       "sha256": sha_file(legacy), "modified": False},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "PASS", "output": str(output), "round_package_count": len(packages)}

def pack(root: Path, output: Path, max_mb: float = 200) -> dict:
    root, output = root.resolve(), output.resolve()
    if not root.is_dir() or root in output.parents: raise ValueError("invalid package root/output")
    limit = int(max_mb * 1024 * 1024); entries=[]; omitted=[]
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts: continue
        rel=path.relative_to(root); reason=""
        if path.name.startswith(".env") or any(part.lower() in {"secret","secrets"} for part in rel.parts): raise RuntimeError(f"secret-like path: {rel}")
        if path.suffix.lower() in BINARY_EXTERNAL: reason="large_payload_type_externalized"
        elif path.stat().st_size > limit: reason="over_size_threshold"
        if reason: omitted.append({"logical_path":rel.as_posix(),"path":str(path.resolve()),"size_bytes":path.stat().st_size,"sha256":sha_file(path),"reason":reason})
        else: entries.append((rel.as_posix(),path))
    if not entries: raise ValueError("empty package")
    output.parent.mkdir(parents=True,exist_ok=True); temp=output.with_name(output.name+".partial")
    sums=[]
    try:
        with zipfile.ZipFile(temp,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for rel,path in entries:
                content=path.read_bytes()
                if SECRET_RE.search(content): raise RuntimeError(f"possible credential: {rel}")
                z.writestr(zip_info(rel),content); sums.append(f"{hashlib.sha256(content).hexdigest()}  {rel}")
            table=io.StringIO(); w=csv.DictWriter(table,fieldnames=["logical_path","path","size_bytes","sha256","reason"],delimiter="\t",lineterminator="\n"); w.writeheader(); w.writerows(omitted)
            manifest=table.getvalue().encode(); z.writestr(zip_info("package_external_omissions.tsv"),manifest); sums.append(f"{hashlib.sha256(manifest).hexdigest()}  package_external_omissions.tsv"); z.writestr(zip_info("PACKAGE_SHA256SUMS.txt"),"\n".join(sums)+"\n")
        with zipfile.ZipFile(temp) as z:
            if z.testzip() is not None: raise RuntimeError("CRC mismatch")
            for line in z.read("PACKAGE_SHA256SUMS.txt").decode().splitlines():
                digest,rel=line.split("  ",1)
                if hashlib.sha256(z.read(rel)).hexdigest()!=digest: raise RuntimeError(f"internal SHA mismatch: {rel}")
        temp.replace(output)
    finally: temp.unlink(missing_ok=True)
    digest=sha_file(output); output.with_name(output.name+".sha256").write_text(f"{digest}  {output.name}\n",encoding="utf-8")
    return {"status":"PASS","archive":str(output),"sha256":digest,"small_file_count":len(entries),"omitted_count":len(omitted),"crc_ok":True,"internal_sha_ok":True}

def main():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="command",required=True)
    x=s.add_parser("pack"); x.add_argument("--root",type=Path,required=True); x.add_argument("--output",type=Path,required=True); x.add_argument("--max-mb",type=float,default=200); x.set_defaults(func=lambda a: pack(a.root,a.output,a.max_mb))
    x=s.add_parser("inventory"); x.add_argument("--root",type=Path,required=True); x.add_argument("--output",type=Path,required=True); x.set_defaults(func=lambda a: external_inventory(a.root,a.output))
    x=s.add_parser("round-manifests"); x.add_argument("--root",type=Path,required=True); x.add_argument("--download-dir",type=Path,required=True); x.add_argument("--commit",required=True); x.add_argument("--round-ids"); x.set_defaults(func=lambda a: round_manifests(a.root,a.download_dir,a.commit,a.round_ids))
    x=s.add_parser("stage-rounds"); x.add_argument("--root",type=Path,required=True); x.set_defaults(func=lambda a: stage_rounds(a.root))
    x=s.add_parser("package-index"); x.add_argument("--download-dir",type=Path,required=True); x.add_argument("--output",type=Path,required=True); x.add_argument("--legacy-zip",type=Path,required=True); x.set_defaults(func=lambda a: package_index(a.download_dir,a.output,a.legacy_zip))
    a=p.parse_args(); print(json.dumps(a.func(a),ensure_ascii=False,indent=2))
if __name__=="__main__": main()
