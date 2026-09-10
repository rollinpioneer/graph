"""CLI for the R11 cache audit and explicitly authorized physical probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cache_audit import build_cache_audit
from .physics_probe import run_physics_probe
from .report import build_report, finalize_manifest


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="L2RA-R2 R11 loss observability diagnostics")
    sub = root.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("cache-audit", help="read-only audit of existing 32-rollout cache")
    cache.add_argument("--data-root", type=Path, required=True)
    cache.add_argument("--generation-lock", type=Path, required=True)
    cache.add_argument("--output-root", type=Path, required=True)
    cache.add_argument("--baseline-root", type=Path)
    probe = sub.add_parser("physics-probe", help="bounded ordinary/read-only MuJoCo replay")
    probe.add_argument("--data-root", type=Path, required=True)
    probe.add_argument("--generation-lock", type=Path, required=True)
    probe.add_argument("--output-root", type=Path, required=True)
    probe.add_argument("--budget", type=int, required=True)
    probe.add_argument("--allow-physical-replay", action="store_true")
    probe.add_argument("--root-family-id")
    report = sub.add_parser("build-report", help="write decision, handoff and report")
    report.add_argument("--output-root", type=Path, required=True)
    report.add_argument("--baseline-root", type=Path, required=True)
    report.add_argument("--cache-summary", type=Path, required=True)
    report.add_argument("--source-commit", required=True)
    report.add_argument("--physical-manifest", type=Path)
    report.add_argument("--command", dest="recorded_command", action="append", default=[])
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "cache-audit":
        result = build_cache_audit(args.data_root, args.generation_lock, args.output_root, args.baseline_root)
    elif args.command == "physics-probe":
        result = run_physics_probe(args.data_root, args.output_root, args.budget, args.allow_physical_replay, args.generation_lock, args.root_family_id)
    else:
        result = build_report(args.output_root, args.baseline_root, args.cache_summary, args.source_commit, args.physical_manifest)
        finalize_manifest(args.output_root, args.source_commit, args.recorded_command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
