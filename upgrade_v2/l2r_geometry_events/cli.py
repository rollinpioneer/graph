"""Command line entry point for the R15-A cache geometry diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import run_cache_evaluation_v2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="l2r_geometry_events")
    subparsers = parser.add_subparsers(dest="command", required=True)
    cache_v2 = subparsers.add_parser(
        "cache-evaluate-v2",
        help="R15-B fair re-evaluation on the existing repair cache (no physics)",
    )
    cache_v2.add_argument("--data-root", required=True)
    cache_v2.add_argument("--protocol", required=True)
    cache_v2.add_argument("--round10-root", required=True)
    cache_v2.add_argument("--resource-report", required=True)
    cache_v2.add_argument("--output-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "cache-evaluate-v2":
        output_root = Path(args.output_root)
        if output_root.exists():
            print("REFUSING_EXISTING_OUTPUT_ROOT " + str(output_root))
            return 2
        summary = run_cache_evaluation_v2(
            Path(args.data_root),
            Path(args.protocol),
            Path(args.round10_root),
            Path(args.resource_report),
            output_root,
        )
        print(json.dumps({"status": "CACHE_REEVALUATED_V2", **summary}, indent=2, ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
