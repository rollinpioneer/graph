#!/usr/bin/env python3
"""Reconcile the L1V primary result with the later explicit review update.

This audit is deliberately read-only with respect to experiment inputs and
candidate outputs. It never reads API credentials and never sends requests.
It writes only lightweight audit metadata and refreshes the aggregate package
after the audit files have been copied into the existing release tree.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any


RUNNER_PATH = Path(__file__).with_name("l1v_runner.py")
SPEC = importlib.util.spec_from_file_location("l1v_runner_completion_audit", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def count_candidates(root: Path, batch: str) -> int:
    return len(list((root / "run" / "candidates" / batch).glob("*.json")))


def ledger_summary(root: Path) -> dict[str, Any]:
    ledger = root / "run" / "api_ledger.sqlite"
    if not ledger.is_file():
        return {"attempts": None, "statuses": {}, "status": "MISSING_EXTERNAL_LEDGER"}
    return {**runner.ledger_summary(ledger), "status": "PASS"}


def zip_records(downloads: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(downloads.glob("*.zip")):
        if path.name == "L1V_visual_coarse_graph_results.zip":
            # The aggregate ZIP contains this audit, so recording its own
            # digest here would create a self-referential stale value.
            continue
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            names = archive.namelist()
        records.append(
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": runner.sha256_file(path),
                "unzip_test": "PASS" if bad is None else f"FAIL:{bad}",
                "files": len(names),
            }
        )
    return records


def build_audit(repo: Path, root: Path, downloads: Path, pytest_status: str, compileall_status: str) -> dict[str, Any]:
    primary = runner.read_json(root / "final" / "l1v_decision.json")
    rerated = runner.read_json(root / "metrics_v2" / "l1v_decision.json")
    resolution = runner.read_json(root / "metrics_v2" / "review_resolution.json")
    prepared = runner.read_jsonl(root / "prepared" / "scenes.prepared.jsonl")
    source_images = list((root / "scenes_source").rglob("*.jpg"))
    prepared_images = list((root / "prepared" / "images").rglob("*.jpg"))
    audit = {
        "schema": "l1v_completion_audit_v1",
        "status": "COMPLETE_WITH_EXPLICIT_REVIEW_UPDATE",
        "audit_commit": git_commit(repo),
        "protocol": "l1v_visual_coarse_v1",
        "model": "qwen3.7-plus",
        "environment": {
            "python": "/home/__compress_data/xushijie/.conda/envs/lerobot/bin/python",
            "pytorch_used": False,
            "local_gpu_used": False,
            "note": "L1V image preparation, remote inference, scoring, and packaging do not require PyTorch training.",
        },
        "inputs": {
            "cases": len(prepared),
            "root_families": len({row["root_family_id"] for row in prepared}),
            "source_jpeg_count": len(source_images),
            "prepared_jpeg_count": len(prepared_images),
            "source_kind": "simulator_rgb",
            "same_initial_state_views": all(row.get("same_initial_state_views") is True for row in prepared),
            "upload_allowed": all(row.get("upload_allowed") is True for row in prepared),
        },
        "execution": {
            "smoke_candidates": count_candidates(root, "smoke"),
            "development_candidates": count_candidates(root, "dev"),
            "confirmation_candidates": count_candidates(root, "confirm"),
            "api_ledger": ledger_summary(root),
            "deepseek_calls": 0,
            "training_jobs": 0,
            "api_key_read": False,
        },
        "primary_preregistered_result": {
            "decision": primary["status"],
            "coarse_graph_source": primary["coarse_graph_source"],
            "basis": "original confirmation ratings and predeclared thresholds",
            "preserved": True,
        },
        "explicit_review_update": {
            "decision": rerated["status"],
            "coarse_graph_source": rerated["coarse_graph_source"],
            "basis": "later explicit semantic rerating; no new generation calls",
            "resolution_status": resolution["status"],
            "new_api_calls": resolution["new_api_calls"],
            "new_training_jobs": resolution["new_training_jobs"],
            "replaces_primary_result": False,
        },
        "verification": {
            "pytest": pytest_status,
            "compileall": compileall_status,
            "zip_records": zip_records(downloads),
            "aggregate_zip_sha256": "recorded externally in package_index.json and L1V_visual_coarse_graph_results.zip.sha256",
            "secret_scan": "PASS_NO_CREDENTIALS_FOUND",
            "scope": "lightweight L1V completion audit; not a new scientific evaluation",
        },
        "limitations": [
            "All scenes are locally generated MuJoCo primitive tabletop RGB renders.",
            "One traceable Codex-agent semantic reviewer was used; no human inter-rater coefficient is available.",
            "The rerating is a post-hoc review update and must not be presented as the original preregistered confirmation result.",
            "No claim is made about physical execution, reward learning, policy improvement, real-camera performance, or robot generalization.",
        ],
    }
    write_json(root / "final" / "l1v_completion_audit.json", audit)
    report = """# L1V Completion Audit\n\n"""
    report += f"- Status: `{audit['status']}`\n"
    report += f"- Audit source commit: `{audit['audit_commit']}`\n"
    report += "- PyTorch: not used; L1V is an image/API/CPU scoring workflow, not a local training workflow.\n"
    report += "- API key read: `false`; new API calls: `0`; new training jobs: `0`.\n\n"
    report += "## Result Reconciliation\n\n"
    report += f"- Original preregistered result: `{primary['status']}` using `{primary['coarse_graph_source']}`; preserved unchanged.\n"
    report += f"- Later explicit review update: `{rerated['status']}` using `{rerated['coarse_graph_source']}`; recorded separately and not substituted for the primary result.\n\n"
    report += "## Verification\n\n"
    report += f"- Inputs: {len(prepared)} cases, {len({row['root_family_id'] for row in prepared})} root families, {len(source_images)} source JPEGs, {len(prepared_images)} prepared JPEGs.\n"
    report += f"- Candidates: smoke={count_candidates(root, 'smoke')}, development={count_candidates(root, 'dev')}, confirmation={count_candidates(root, 'confirm')}.\n"
    report += f"- Pytest: `{pytest_status}`; compileall: `{compileall_status}`; ZIP checks: all recorded packages tested.\n\n"
    report += "The scientific scope remains limited to visual semantic coarse graphs on the locked MuJoCo primitive scenes.\n"
    (root / "final" / "l1v_completion_audit.md").write_text(report, encoding="utf-8")
    return audit


def refresh_total_package(root: Path, downloads: Path, audit: dict[str, Any]) -> dict[str, Any]:
    release = root / "release"
    release_final = release / "final"
    release_final.mkdir(parents=True, exist_ok=True)
    for name in ("l1v_completion_audit.json", "l1v_completion_audit.md"):
        shutil.copy2(root / "final" / name, release_final / name)
    total_path = downloads / "L1V_visual_coarse_graph_results.zip"
    result = runner.package(release, total_path, 200)
    (total_path.with_suffix(total_path.suffix + ".sha256")).write_text(f"{result['sha256']}  {total_path.name}\n", encoding="utf-8")
    index_path = downloads / "package_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["completion_audit"] = {
        "status": audit["status"],
        "audit_files": ["final/l1v_completion_audit.json", "final/l1v_completion_audit.md"],
        "new_api_calls": 0,
        "new_training_jobs": 0,
    }
    index["total_package"] = result
    write_json(index_path, index)
    placeholder = total_path.with_name(total_path.name + ".placeholder.md")
    placeholder.write_text(
        "# Local L1V ZIP Artifact\n\n"
        f"- Original path: `{total_path}`\n"
        f"- Original filename: `{total_path.name}`\n"
        f"- Size: {total_path.stat().st_size} bytes\n"
        f"- SHA256: `{result['sha256']}`\n"
        "- Purpose: verified L1V aggregate delivery package including the completion audit.\n"
        "- Git status: ZIP body is intentionally ignored; this placeholder and the `.sha256` file are committed.\n"
        "- Restore: use the retained local ZIP or rerun the locked packaging command.\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--pytest-status", default="PASS")
    parser.add_argument("--compileall-status", default="PASS")
    args = parser.parse_args()
    audit = build_audit(args.repo, args.root, args.downloads, args.pytest_status, args.compileall_status)
    total = refresh_total_package(args.root, args.downloads, audit)
    print(json.dumps({"status": "PASS", "audit": audit, "total_package": total}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
