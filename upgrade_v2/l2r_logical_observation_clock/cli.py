from __future__ import annotations

import argparse
import json
from pathlib import Path

from .clock_forensics import audit_r17_clocks
from .audits import source_audit
from .io_utils import write_json
from .package_results import summarize
from .protocol import protocol_lock
from .r19_design import build_design
from .replay import replay_r17


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("audit-r17-clocks"); p.add_argument("--r17-data-root", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("audit-source"); p.add_argument("--repo", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("replay-r17"); p.add_argument("--r17-artifact-root", type=Path, required=True); p.add_argument("--r17-data-root", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("build-r19-design"); p.add_argument("--development-root", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("write-protocol"); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("summarize"); p.add_argument("--artifact-root", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit-r17-clocks": result = audit_r17_clocks(args.r17_data_root, args.output_root)
    elif args.command == "audit-source": result = source_audit(args.repo); write_json(args.output, result)
    elif args.command == "replay-r17": result = replay_r17(args.r17_artifact_root, args.r17_data_root, args.output_root)
    elif args.command == "build-r19-design": result = build_design(args.development_root, args.output_root)
    elif args.command == "write-protocol": result = protocol_lock(); write_json(args.output, result)
    elif args.command == "summarize": result = summarize(args.artifact_root, args.output_root)
    else: raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
