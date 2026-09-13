from __future__ import annotations

import argparse
import json
from pathlib import Path

from .extract import extract
from .separability import run


def main() -> int:
    parser = argparse.ArgumentParser(description="R26 zero-physics feature and separability tooling")
    sub = parser.add_subparsers(dest="command", required=True)
    ext = sub.add_parser("extract")
    ext.add_argument("--r24", type=Path, required=True)
    ext.add_argument("--r25", type=Path, required=True)
    ext.add_argument("--output", type=Path, required=True)
    sep = sub.add_parser("separability")
    sep.add_argument("--features", type=Path, required=True)
    sep.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "extract":
        result = extract([args.r24, args.r25], args.output)
    else:
        result = run(args.features, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status", "PASS") in ("PASS", "CURRENT_SENSOR_REPRESENTATION_NOT_SEPARABLE") else 2


if __name__ == "__main__":
    raise SystemExit(main())
