from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import PROBLEM_ROLLOUT_ID
from .audits import source_audit
from .collector import collect_confirmation
from .detector import detect_rollout
from .evaluation import evaluate
from .forensics import forensic_r17_boundary
from .io_utils import write_json
from .package_results import summarize
from .protocol import protocol_lock
from .r17_replay import replay_r17
from .reference_builder import build_reference, validate_generator


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("audit"); p.add_argument("--repo", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("forensic-r17-boundary"); p.add_argument("--r17-artifact-root", type=Path, required=True); p.add_argument("--r17-data-root", type=Path, required=True); p.add_argument("--rollout-id", default=PROBLEM_ROLLOUT_ID); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("replay-r17"); p.add_argument("--r17-artifact-root", type=Path, required=True); p.add_argument("--r17-data-root", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("collect-r18"); p.add_argument("--repo", type=Path); p.add_argument("--output-root", type=Path, required=True); p.add_argument("--workers", type=int, default=1)
    p = sub.add_parser("detect-rgb"); p.add_argument("--confirmation-root", type=Path, required=True)
    p = sub.add_parser("build-reference"); p.add_argument("--confirmation-root", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("validate-generator"); p.add_argument("--reference-root", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("evaluate"); p.add_argument("--confirmation-root", type=Path, required=True); p.add_argument("--reference-root", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("summarize"); p.add_argument("--artifact-root", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True); p.add_argument("--external-data-root", type=Path)
    p = sub.add_parser("write-protocol"); p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit": result = source_audit(args.repo); write_json(args.output, result)
    elif args.command == "forensic-r17-boundary": result = forensic_r17_boundary(args.r17_artifact_root, args.r17_data_root, args.rollout_id, args.output_root)
    elif args.command == "replay-r17": result = replay_r17(args.r17_artifact_root, args.r17_data_root, args.output_root)
    elif args.command == "collect-r18":
        rows = collect_confirmation(args.output_root)
        detected = [{"rollout": path.parent.name, **detect_rollout(path.parent)}
                    for path in sorted(args.output_root.glob("*/metadata.json"))]
        result = {"rollouts": len(rows), "detector_pass": sum(row["status"] == "PASS" for row in detected)}
    elif args.command == "detect-rgb":
        result = {"results": [{"rollout": path.parent.name, **detect_rollout(path.parent)}
                              for path in sorted(args.confirmation_root.glob("*/metadata.json"))]}
    elif args.command == "build-reference": result = build_reference(args.confirmation_root, args.output_root)
    elif args.command == "validate-generator": result = validate_generator(args.reference_root); write_json(args.output, result)
    elif args.command == "evaluate": result = evaluate(args.confirmation_root, args.reference_root, args.output_root)
    elif args.command == "summarize": result = summarize(args.artifact_root, args.output_root, args.external_data_root)
    elif args.command == "write-protocol": result = protocol_lock(); write_json(args.output, result)
    else: raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("status") != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
