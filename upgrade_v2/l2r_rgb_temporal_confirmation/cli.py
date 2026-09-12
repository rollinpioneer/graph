from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audits import source_audit
from .detector_adapter import detect_confirmation
from .difficulty_ladder import run_difficulty_calibration, validate_difficulty
from .evaluation import evaluate
from .io_utils import read_json, write_json
from .package_results import summarize
from .protocol import protocol_lock
from .reference_builder import build_reference, validate_generator
from .rgb_capture import collect_confirmation


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("audit"); p.add_argument("--repo", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("run-difficulty-calibration"); p.add_argument("--repo", type=Path); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("validate-difficulty"); p.add_argument("--calibration-root", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("collect-confirmation"); p.add_argument("--repo", type=Path); p.add_argument("--difficulty-lock", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True); p.add_argument("--workers", type=int, default=1)
    p = sub.add_parser("detect-rgb"); p.add_argument("--confirmation-root", type=Path, required=True); p.add_argument("--workers", type=int, default=1)
    p = sub.add_parser("build-reference"); p.add_argument("--confirmation-root", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("validate-generator"); p.add_argument("--reference-root", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("evaluate"); p.add_argument("--confirmation-root", type=Path, required=True); p.add_argument("--reference-root", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("summarize"); p.add_argument("--artifact-root", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True); p.add_argument("--external-data-root", type=Path)
    p = sub.add_parser("write-protocol"); p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "audit": result = source_audit(args.repo); write_json(args.output, result)
    elif args.command == "run-difficulty-calibration": result = run_difficulty_calibration(args.output_root)
    elif args.command == "validate-difficulty": result = validate_difficulty(args.calibration_root); write_json(args.output, result)
    elif args.command == "collect-confirmation": result = {"rollouts": len(collect_confirmation(args.output_root, read_json(args.difficulty_lock)))}
    elif args.command == "detect-rgb": result = {"results": detect_confirmation(args.confirmation_root)}
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
