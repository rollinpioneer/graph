from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    item = sub.add_parser("validate-frozen"); item.add_argument("--repo", type=Path, required=True)
    item = sub.add_parser("write-static"); item.add_argument("--repo", type=Path, required=True); item.add_argument("--output", type=Path, required=True); item.add_argument("--runner-commit", required=True)
    item = sub.add_parser("collect"); item.add_argument("--output", type=Path, required=True); item.add_argument("--workers", type=int, default=2)
    item = sub.add_parser("evaluate"); item.add_argument("--root", type=Path, required=True); item.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate-frozen":
        from .static_lock import validate_frozen; result = validate_frozen(args.repo)
    elif args.command == "write-static":
        from .static_lock import write_static; result = write_static(args.repo, args.output, args.runner_commit)
    elif args.command == "collect":
        from .runner import collect; result = collect(args.output, args.workers)
    else:
        from .evaluation import evaluate; result = evaluate(args.root, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status", "PASS") == "PASS" or result.get("passed") is True else 2


if __name__ == "__main__": raise SystemExit(main())

