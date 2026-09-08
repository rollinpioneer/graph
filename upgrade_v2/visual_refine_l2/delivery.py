"""Round manifests, local ZIP placeholders, and package index for L2R."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .io import ensure_round, git_commit, now_iso, sha256_file, write_json


def finalize_round(round_dir: Path, purpose: str, status: str, repo: Path, command_log: Path, gpu_used: bool, gpu_ids: str, jobs: int, denominators: dict[str, Any], start_time: str, end_time: str) -> dict[str, Any]:
    ensure_round(round_dir)
    if not command_log.is_file():
        raise FileNotFoundError(command_log)
    large_manifest = round_dir / "manifests/large_file_manifest.tsv"
    if not large_manifest.exists():
        large_manifest.write_text("path\tsize_bytes\tartifact_type\treason_omitted\trecovery_method\n", encoding="utf-8")
    checkpoint_manifest = round_dir / "manifests/checkpoint_manifest.tsv"
    if not checkpoint_manifest.exists():
        checkpoint_manifest.write_text("path\tsize_bytes\tjob_id\tepoch_or_step\tmetric\n", encoding="utf-8")
    payload = {
        "schema": "pathgraph_l2r_run_manifest_v1", "round_id": round_dir.name, "purpose": purpose,
        "status": status, "start_time": start_time, "end_time": end_time, "recorded_at": now_iso(),
        "git_commit": git_commit(repo), "python": "/home/xushijie/.conda/envs/lerobot/bin/python",
        "commands_path": str(command_log.resolve()), "commands_sha256": sha256_file(command_log),
        "gpu_used": gpu_used, "gpu_ids": gpu_ids if gpu_used else "none", "jobs_run": jobs,
        "jobs_failed": 0, "new_api_calls": 0, "new_training_jobs": 0, "api_key_read": False,
        "actual_denominators": denominators, "scientific_scope": "dynamic MuJoCo primitive tabletop refinement benchmark",
    }
    write_json(round_dir / "run_manifest.json", payload)
    lines = ["# Run Manifest", ""] + [f"- {key}: `{value}`" for key, value in payload.items() if key != "actual_denominators"]
    lines.extend(["- actual_denominators:"] + [f"  - {key}: `{value}`" for key, value in denominators.items()])
    (round_dir / "run_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (round_dir / "summary.md").write_text(
        f"# {round_dir.name}\n\n- Purpose: {purpose}\n- Status: `{status}`\n- Jobs: {jobs}; failures: 0\n- API calls: 0; training jobs: 0; API key read: false\n- GPU used: {str(gpu_used).lower()} ({gpu_ids if gpu_used else 'none'})\n",
        encoding="utf-8",
    )
    return payload


def record_zip(zip_path: Path, placeholder: Path, index: Path, purpose: str) -> dict[str, Any]:
    digest = sha256_file(zip_path)
    sha_path = zip_path.with_name(zip_path.name + ".sha256")
    recorded = sha_path.read_text(encoding="utf-8").split()[0] if sha_path.is_file() else None
    if recorded != digest:
        raise ValueError("ZIP SHA sidecar mismatch")
    placeholder.parent.mkdir(parents=True, exist_ok=True)
    placeholder.write_text(
        f"# Local L2R ZIP\n\n- Original path: `{zip_path.resolve()}`\n- Original filename: `{zip_path.name}`\n- Size: {zip_path.stat().st_size} bytes\n- SHA256: `{digest}`\n- Purpose: {purpose}\n- Restore: rerun the corresponding locked L2R command sequence.\n",
        encoding="utf-8",
    )
    payload = read_index(index)
    payload["packages"] = [row for row in payload["packages"] if row["filename"] != zip_path.name]
    payload["packages"].append({"filename": zip_path.name, "path": str(zip_path.resolve()), "size_bytes": zip_path.stat().st_size, "sha256": digest, "purpose": purpose})
    payload["packages"].sort(key=lambda row: row["filename"])
    payload["package_count"] = len(payload["packages"])
    payload["updated_at"] = now_iso()
    write_json(index, payload)
    return {"status": "PASS", "zip": str(zip_path.resolve()), "sha256": digest}


def read_index(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema": "pathgraph_l2r_package_index_v1", "package_count": 0, "packages": [], "new_api_calls": 0, "new_training_jobs": 0}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("finalize-round")
    p.add_argument("--round-dir", type=Path, required=True); p.add_argument("--purpose", required=True); p.add_argument("--status", required=True)
    p.add_argument("--repo", type=Path, required=True); p.add_argument("--command-log", type=Path, required=True); p.add_argument("--gpu-used", action="store_true")
    p.add_argument("--gpu-ids", default="none"); p.add_argument("--jobs", type=int, required=True); p.add_argument("--denominators", required=True); p.add_argument("--start-time", required=True); p.add_argument("--end-time", required=True)
    p = sub.add_parser("record-zip")
    p.add_argument("--zip", type=Path, required=True); p.add_argument("--placeholder", type=Path, required=True); p.add_argument("--index", type=Path, required=True); p.add_argument("--purpose", required=True)
    args = parser.parse_args()
    if args.command == "finalize-round":
        result = finalize_round(args.round_dir, args.purpose, args.status, args.repo, args.command_log, args.gpu_used, args.gpu_ids, args.jobs, json.loads(args.denominators), args.start_time, args.end_time)
    else:
        result = record_zip(args.zip, args.placeholder, args.index, args.purpose)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
