"""Command line entry point for the R15-A cache geometry diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import run_cache_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="l2r_geometry_events")
    subparsers = parser.add_subparsers(dest="command", required=True)
    cache = subparsers.add_parser(
        "cache-evaluate",
        help="evaluate the fixed geometry methods on the existing repair cache",
    )
    cache.add_argument("--data-root", required=True)
    cache.add_argument("--protocol", required=True)
    cache.add_argument("--resource-report", required=True)
    cache.add_argument("--output-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "cache-evaluate":
        return 1
    output_root = Path(args.output_root)
    if output_root.exists():
        print("REFUSING_EXISTING_OUTPUT_ROOT " + str(output_root))
        return 2
    summary = run_cache_evaluation(
        Path(args.data_root),
        Path(args.protocol),
        Path(args.resource_report),
        output_root,
    )
    print(json.dumps({"status": "CACHE_EVALUATED", **summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
