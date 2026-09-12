from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    development = sub.add_parser("develop")
    development.add_argument("--r24-root", type=Path, required=True)
    development.add_argument("--output", type=Path, required=True)
    static = sub.add_parser("static")
    static.add_argument("--repo", type=Path, required=True)
    static.add_argument("--output", type=Path, required=True)
    static.add_argument("--commit", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--workers", type=int, default=2)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--root", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "develop":
        from .development import validate_development
        result = validate_development(args.r24_root, args.output)
    elif args.command == "static":
        from .static_lock import write_static
        result = write_static(args.repo, args.output, args.commit)
    elif args.command == "collect":
        from .runner import collect
        result = collect(args.output, args.workers)
    else:
        from .evaluation import evaluate
        result = evaluate(args.root, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__": raise SystemExit(main())
