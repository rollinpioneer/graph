#!/usr/bin/env python3
"""Read-only L2RA static checks and lightweight package creation.

This helper never reads credentials, sends requests, trains models, or treats
its logical truth table as dynamic rollout evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path

BASE = "e607131d7745d46745680bc9a6190a467b01dfa2"
HEAVY = {".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".npz", ".npy", ".mp4", ".avi", ".parquet", ".sqlite"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def logic_probe() -> dict:
    rows = []
    for closed in (False, True):
        for contact in (False, True):
            for prior_contact in (False, True):
                failed = closed and not contact
                slip = prior_contact and closed and not contact
                rows.append({"closed": closed, "contact": contact, "prior_contact_in_window": prior_contact,
                             "grasp_failed_observed": failed, "slip_observed": slip,
                             "both_raw_guards": failed and slip})
    implication = all(not row["slip_observed"] or row["grasp_failed_observed"] for row in rows)
    assert implication and any(row["both_raw_guards"] for row in rows)
    return {"scope": "CODE_DERIVED_LOGICAL_PROBE_NOT_ROLLOUT_EVIDENCE",
            "slip_implies_grasp_failed_under_frozen_rules": implication,
            "truth_table": rows,
            "raw_prediction_replayed": False,
            "note": "A1 replay is required for rollout and event counts."}


def inspect_frozen(repo: Path, output: Path) -> None:
    final = repo / "artifacts/pathgraph_sarm/upgrade_v2/visual_refine_l2_v1/final_v1"
    required = [final / "tables/confirmation_metrics.csv", final / "tables/metrics_by_scenario.csv",
                final / "graphs/G2_evidence_refined.json", final / "third_layer_interface.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen files: " + ", ".join(missing))
    with (required[0]).open(encoding="utf-8-sig", newline="") as handle:
        overall = [row for row in csv.DictReader(handle) if row.get("graph_id") == "G2_evidence_refined"]
    with (required[1]).open(encoding="utf-8-sig", newline="") as handle:
        scenario = [row for row in csv.DictReader(handle) if row.get("graph_id") == "G2_evidence_refined"]
    if len(overall) != 1:
        raise ValueError("expected one frozen G2 summary row")
    n = int(overall[0]["rollouts"])
    rate = float(overall[0]["ambiguous_edge_rate"])
    rows = [{"scenario": row["scenario"], "rollouts": int(row["rollouts"]),
             "families": int(row["families"]), "ambiguous_rate": float(row["ambiguous_edge_rate"]),
             "flagged_count_derived_from_summary": int(round(int(row["rollouts"]) * float(row["ambiguous_edge_rate"]))),
             "branch_accuracy": float(row["branch_accuracy"])} for row in scenario]
    if sum(row["rollouts"] for row in rows) != n:
        raise ValueError("scenario denominator mismatch")
    result = {"source_commit_expected": BASE, "scope": "FROZEN_SUMMARIES_AND_GRAPH_ONLY",
              "rollouts": n, "families": int(overall[0]["families"]),
              "ambiguous_edge_rate": rate, "flagged_rollouts_derived_from_summary": int(round(n * rate)),
              "scenario_breakdown": rows, "raw_prediction_replayed": False,
              "new_confirmation_run": False,
              "sources": [{"path": str(path.relative_to(repo)), "sha256": sha256(path)} for path in required]}
    write_json(output / "frozen_summary_diagnosis.json", result)
    write_json(output / "logical_overlap_probe.json", logic_probe())


def package_round(source: Path, target: Path, max_file_mb: float) -> None:
    if not source.is_dir() or target.exists() or target.with_suffix(target.suffix + ".sha256").exists():
        raise ValueError("source missing or sealed target exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    limit = int(max_file_mb * 1024 * 1024)
    included, omitted = [], []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".zip"}:
            continue
        rel = path.relative_to(source).as_posix()
        size = path.stat().st_size
        if path.suffix.lower() in HEAVY or size > limit:
            omitted.append({"path": rel, "size_bytes": size, "sha256": None, "reason": "external_large_artifact"})
        else:
            included.append((path, rel, sha256(path)))
    if not included:
        raise ValueError("no lightweight files")
    partial = target.with_suffix(target.suffix + ".partial")
    with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, rel, _ in included:
            archive.write(path, rel)
        omitted_bytes = (json.dumps(omitted, ensure_ascii=False, indent=2) + "\n").encode()
        archive.writestr("PACKAGE_OMITTED_FILES.json", omitted_bytes)
        checks = "".join(f"{digest}  {rel}\n" for _, rel, digest in included)
        checks += f"{hashlib.sha256(omitted_bytes).hexdigest()}  PACKAGE_OMITTED_FILES.json\n"
        archive.writestr("PACKAGE_SHA256SUMS.txt", checks)
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip() is not None:
            partial.unlink(missing_ok=True)
            raise ValueError("ZIP CRC failure")
        for line in archive.read("PACKAGE_SHA256SUMS.txt").decode().splitlines():
            digest, rel = line.split("  ", 1)
            if hashlib.sha256(archive.read(rel)).hexdigest() != digest:
                partial.unlink(missing_ok=True)
                raise ValueError("internal checksum mismatch: " + rel)
    partial.rename(target)
    digest = sha256(target)
    target.with_suffix(target.suffix + ".sha256").write_text(f"{digest}  {target.name}\n", encoding="utf-8")
    print(json.dumps({"archive": str(target), "sha256": digest, "included": len(included), "omitted": len(omitted),
                      "crc_and_internal_sha": "PASS", "scientific_result_verified": False}, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    item = sub.add_parser("inspect-frozen"); item.add_argument("--repo", type=Path, required=True); item.add_argument("--out", type=Path, required=True)
    item = sub.add_parser("logic-probe"); item.add_argument("--out", type=Path)
    item = sub.add_parser("package-round"); item.add_argument("--source", type=Path, required=True); item.add_argument("--output", type=Path, required=True); item.add_argument("--max-file-mb", type=float, default=200)
    args = parser.parse_args()
    try:
        if args.command == "inspect-frozen": inspect_frozen(args.repo, args.out)
        elif args.command == "logic-probe":
            result = logic_probe()
            if args.out: write_json(args.out, result)
            else: print(json.dumps(result, ensure_ascii=False, indent=2))
        else: package_round(args.source, args.output, args.max_file_mb)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
