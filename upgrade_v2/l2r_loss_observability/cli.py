"""CLI for the R11 cache audit and explicitly authorized physical probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


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
    probe.add_argument("--request-id", default="r12_cli_physics_probe")
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
        from .cache_audit import build_cache_audit

        result = build_cache_audit(args.data_root, args.generation_lock, args.output_root, args.baseline_root)
    elif args.command == "physics-probe":
        from upgrade_v2.l2r_execution_audit.policy import PhysicalExecutionDenied, SafetyError, deny_physical_request

        try:
            deny_physical_request(Path.cwd(), args.request_id, args.budget)
        except PhysicalExecutionDenied as exc:
            result = {"status": "PHYSICAL_EXECUTION_DENIED", "reason": str(exc), "actual_physical_executions": 0}
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 3
        except SafetyError as exc:
            result = {"status": "R12_SAFETY_ERROR", "reason": str(exc), "actual_physical_executions": 0}
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
    else:
        from .report import build_report, finalize_manifest

        result = build_report(args.output_root, args.baseline_root, args.cache_summary, args.source_commit, args.physical_manifest)
        finalize_manifest(args.output_root, args.source_commit, args.recorded_command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
